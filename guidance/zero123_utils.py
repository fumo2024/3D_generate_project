import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.utils import save_image

from diffusers import DDIMScheduler

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

from ldm.util import instantiate_from_config
from ..utils.image_utils import pil_to_tensor, tensor_to_pil
from ..utils.camera_utils import create_camera

def load_model_from_config(config, ckpt, device, vram_O=False, verbose=False):
    pl_sd = torch.load(ckpt, map_location='cpu')
    if 'global_step' in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')
    sd = pl_sd['state_dict']
    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)
    if len(m) > 0 and verbose:
        print('[INFO] missing keys: \n', m)
    if len(u) > 0 and verbose:
        print('[INFO] unexpected keys: \n', u)
    if getattr(model, 'use_ema', False):
        if verbose:
            print('[INFO] loading EMA...')
        model.model_ema.copy_to(model.model)
        del model.model_ema
    if vram_O:
        del model.first_stage_model.decoder
    torch.cuda.empty_cache()
    model.eval().to(device)
    return model

class Zero123(nn.Module):
    def __init__(self, device, fp16,
                 config='./pretrained/zero123/sd-objaverse-finetune-c_concat-256.yaml',
                 ckpt='./pretrained/zero123/zero123-xl.ckpt', vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()
        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt

        self.config = OmegaConf.load(config)
        self.model = load_model_from_config(self.config, ckpt, device=self.device, vram_O=vram_O)
        self.num_train_timesteps = self.config.model.params.timesteps
        self.scheduler = DDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
            beta_schedule='scaled_linear',
            clip_sample=False,
            set_alpha_to_one=False,
            steps_offset=1,
        )
        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device)

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [B, 3, 256, 256] in [0, 1]
        x = x * 2 - 1
        c = [self.model.get_learned_conditioning(xx.unsqueeze(0)) for xx in x]
        v = [self.model.encode_first_stage(xx.unsqueeze(0)).mode() for xx in x]
        return c, v

    @torch.no_grad()
    def generate_view(self, input_image, azimuth, elevation, radius=0, polar=0, scale=3, ddim_steps=50, ddim_eta=1, h=256, w=256):
        """
        input_image: PIL.Image or torch.Tensor [C,H,W] in [0,1]
        azimuth, elevation: float, 角度
        返回: novel_tensor, camera_params
        """
        if isinstance(input_image, torch.Tensor):
            if input_image.dim() == 3:
                input_image = input_image.unsqueeze(0)
            if input_image.max() > 1:
                input_image = input_image / 255.0
        else:
            input_image = pil_to_tensor(input_image, self.device).unsqueeze(0)
        # 归一化到[0,1]
        input_image = input_image.to(self.device).float()
        if input_image.max() > 1:
            input_image = input_image / 255.0

        # 获取embedding
        c, v = self.get_img_embeds(input_image)
        embeddings = {'c_crossattn': c[0], 'c_concat': v[0], 'ref_radii': [radius], 'ref_polars': [polar], 'ref_azimuths': [azimuth], 'zero123_ws': [1.0]}

        # 构造条件
        T = torch.tensor([math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), radius])
        T = T[None, None, :].to(self.device)
        cond = {}
        clip_emb = self.model.cc_projection(torch.cat([embeddings['c_crossattn'], T], dim=-1))
        cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
        cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings['c_concat']).to(self.device), embeddings['c_concat']], dim=0)]

        # 采样
        latents = torch.randn((1, 4, h // 8, w // 8), device=self.device)
        self.scheduler.set_timesteps(ddim_steps)
        for i, t in enumerate(self.scheduler.timesteps):
            x_in = torch.cat([latents] * 2)
            t_in = torch.cat([t.view(1)] * 2).to(self.device)
            noise_pred = self.model.apply_model(x_in, t_in, cond)
            noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
            noise_pred = noise_pred_uncond + scale * (noise_pred_cond - noise_pred_uncond)
            latents = self.scheduler.step(noise_pred, t, latents, eta=ddim_eta)['prev_sample']

        imgs = self.decode_latents(latents)
        novel_tensor = imgs[0]
        camera_params = create_camera(azimuth, elevation, device=self.device)
        return novel_tensor, camera_params

    def decode_latents(self, latents):
        imgs = self.model.decode_first_stage(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)
        return imgs # [B, 3, 256, 256]

    def generate_novel_views(self, input_image, num_views=10, min_angle=-180, max_angle=180):
        """
        生成多个新视角
        """
        views = []
        angles = np.linspace(min_angle, max_angle, num_views)
        for azimuth in angles:
            elevation = np.random.uniform(-30, 30)
            novel_tensor, camera_params = self.generate_view(input_image, azimuth, elevation)
            views.append((novel_tensor, camera_params))
        return views

# 示例用法
if __name__ == '__main__':
    import cv2
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=str)
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--polar', type=float, default=0)
    parser.add_argument('--azimuth', type=float, default=0)
    parser.add_argument('--elevation', type=float, default=0)
    parser.add_argument('--radius', type=float, default=0)
    opt = parser.parse_args()

    device = torch.device('cuda')
    print(f'[INFO] loading image from {opt.input} ...')
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    print(f'[INFO] loading model ...')
    zero123 = Zero123(device, opt.fp16)

    print(f'[INFO] running model ...')
    outputs, _ = zero123.generate_view(image, azimuth=opt.azimuth, elevation=opt.elevation, radius=opt.radius, polar=opt.polar)
    img = outputs.cpu().numpy().transpose(1, 2, 0)
    plt.imshow(img)
    plt.show()