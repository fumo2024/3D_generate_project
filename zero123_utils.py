# borrowed from https://github.com/ashawkey/stable-dreamfusion/blob/main/guidance/zero123_utils.py, 
# modified to fit the current codebase
# author: viotrim
# date: 2025-6-26

import math
import numpy as np
from omegaconf import OmegaConf
from pathlib import Path
import importlib

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import custom_bwd, custom_fwd
from torchvision.utils import save_image

# unusable because of diffusers need python3.9 which is not compatible with the current environment
# from diffusers import DDIMScheduler

import sys
from os import path
sys.path.append(path.dirname(path.dirname(path.abspath(__file__))))

# ldm.utils
def instantiate_from_config(config):
    if not "target" in config:
        if config == '__is_first_stage__':
            return None
        elif config == "__is_unconditional__":
            return None
        raise KeyError("Expected key `target` to instantiate.")
    return get_obj_from_str(config["target"])(**config.get("params", dict()))

def get_obj_from_str(string, reload=False):
    module, cls = string.rsplit(".", 1)
    if reload:
        module_imp = importlib.import_module(module)
        importlib.reload(module_imp)
    return getattr(importlib.import_module(module, package=None), cls)

# load model
def load_model_from_config(config, ckpt, device, vram_O=False, verbose=False):

    pl_sd = torch.load(ckpt, map_location='cpu', weights_only=False)

    if 'global_step' in pl_sd and verbose:
        print(f'[INFO] Global Step: {pl_sd["global_step"]}')

    sd = pl_sd['state_dict']

    model = instantiate_from_config(config.model)
    m, u = model.load_state_dict(sd, strict=False)

    if len(m) > 0 and verbose:
        print('[INFO] missing keys: \n', m)
    if len(u) > 0 and verbose:
        print('[INFO] unexpected keys: \n', u)

    # manually load ema and delete it to save GPU memory
    if model.use_ema:
        if verbose:
            print('[INFO] loading EMA...')
        model.model_ema.copy_to(model.model)
        del model.model_ema

    if vram_O:
        # we don't need decoder
        del model.first_stage_model.decoder

    torch.cuda.empty_cache()

    model.eval().to(device)

    return model

# 替代diffusers的DDIMScheduler
class SimpleDDIMScheduler:
    def __init__(self, timesteps, linear_start, linear_end):
        self.timesteps = timesteps
        self.alphas = self._create_alphas(linear_start, linear_end)
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
    
    def _create_alphas(self, start, end):
        return torch.linspace(start, end, self.timesteps)
    
    def add_noise(self, original, noise, timestep):
        sqrt_alpha = torch.sqrt(self.alphas_cumprod[timestep])
        sqrt_one_minus_alpha = torch.sqrt(1 - self.alphas_cumprod[timestep])
        return sqrt_alpha * original + sqrt_one_minus_alpha * noise
    
    def step(self, noise_pred, timestep, sample, eta=1.0):
        # 简化的DDIM步骤实现
        prev_timestep = timestep - self.timesteps // 50
        alpha_prod_t = self.alphas_cumprod[timestep]
        alpha_prod_t_prev = self.alphas_cumprod[prev_timestep] if prev_timestep >= 0 else 1.0
        
        pred_original = (sample - torch.sqrt(1 - alpha_prod_t) * noise_pred) / torch.sqrt(alpha_prod_t)
        
        variance = (1 - alpha_prod_t_prev) / (1 - alpha_prod_t)
        std_dev = eta * torch.sqrt(variance)
        
        noise = torch.randn_like(sample)
        prev_sample = (
            torch.sqrt(alpha_prod_t_prev) * pred_original +
            torch.sqrt(1 - alpha_prod_t_prev - std_dev**2) * noise_pred +
            std_dev * noise
        )
        
        return {'prev_sample': prev_sample}

class Zero123(nn.Module):
    def __init__(self, device, fp16,
                 config='./pretrained/sd-objaverse-finetune-c_concat-256.yaml',
                 ckpt='./pretrained/zero123-xl.ckpt', vram_O=False, t_range=[0.02, 0.98], opt=None):
        super().__init__()

        self.device = device
        self.fp16 = fp16
        self.vram_O = vram_O
        self.t_range = t_range
        self.opt = opt

        self.config = OmegaConf.load(config)
        # TODO: seems it cannot load into fp16...
        self.model = load_model_from_config(self.config, ckpt, device=self.device, vram_O=vram_O)

        # timesteps: use diffuser for convenience... hope it's alright.
        self.num_train_timesteps = self.config.model.params.timesteps

        self.scheduler = SimpleDDIMScheduler(
            self.num_train_timesteps,
            self.config.model.params.linear_start,
            self.config.model.params.linear_end,
        )

        self.min_step = int(self.num_train_timesteps * t_range[0])
        self.max_step = int(self.num_train_timesteps * t_range[1])
        self.alphas = self.scheduler.alphas_cumprod.to(self.device) # for convenience

    @torch.no_grad()
    def get_img_embeds(self, x):
        # x: image tensor [1, 3, 256, 256] in [0, 1]
        x = x * 2 - 1
        c = [self.model.get_learned_conditioning(xx.unsqueeze(0)) for xx in x] #.tile(n_samples, 1, 1)
        v = [self.model.encode_first_stage(xx.unsqueeze(0)).mode() for xx in x]
        return c,v

    def angle_between(self, sph_v1, sph_v2):
        def sph2cart(sv):
            r, theta, phi = sv[0], sv[1], sv[2]
            return torch.tensor([r * torch.sin(theta) * torch.cos(phi), r * torch.sin(theta) * torch.sin(phi), r * torch.cos(theta)])
        def unit_vector(v):
            return v / torch.linalg.norm(v)
        def angle_between_2_sph(sv1, sv2):
            v1, v2 = sph2cart(sv1), sph2cart(sv2)
            v1_u, v2_u = unit_vector(v1), unit_vector(v2)
            return torch.arccos(torch.clip(torch.dot(v1_u, v2_u), -1.0, 1.0))
        angles = torch.empty(len(sph_v1), len(sph_v2))
        for i, sv1 in enumerate(sph_v1):
            for j, sv2 in enumerate(sph_v2):
                angles[i][j] = angle_between_2_sph(sv1, sv2)
        return angles

    def train_step(self, embeddings, pred_rgb, polar, azimuth, radius, guidance_scale=3, as_latent=False, grad_scale=1, save_guidance_path:Path=None):
        # pred_rgb: tensor [1, 3, H, W] in [0, 1]

        # adjust SDS scale based on how far the novel view is from the known view
        ref_radii = embeddings['ref_radii']
        ref_polars = embeddings['ref_polars']
        ref_azimuths = embeddings['ref_azimuths']
        v1 = torch.stack([radius + ref_radii[0], torch.deg2rad(polar + ref_polars[0]), torch.deg2rad(azimuth + ref_azimuths[0])], dim=-1)   # polar,azimuth,radius are all actually delta wrt default
        v2 = torch.stack([torch.tensor(ref_radii), torch.deg2rad(torch.tensor(ref_polars)), torch.deg2rad(torch.tensor(ref_azimuths))], dim=-1)
        angles = torch.rad2deg(self.angle_between(v1, v2)).to(self.device)
        if self.opt.zero123_grad_scale == 'angle':
            grad_scale = (angles.min(dim=1)[0] / (180/len(ref_azimuths))) * grad_scale  # rethink 180/len(ref_azimuths) # claforte: try inverting grad_scale or just fixing it to 1.0
        elif self.opt.zero123_grad_scale == 'None':
            grad_scale = 1.0 # claforte: I think this might converge faster...?
        else:
            assert False, f'Unrecognized `zero123_grad_scale`: {self.opt.zero123_grad_scale}'
        
        if as_latent:
            latents = F.interpolate(pred_rgb, (32, 32), mode='bilinear', align_corners=False) * 2 - 1
        else:
            pred_rgb_256 = F.interpolate(pred_rgb, (256, 256), mode='bilinear', align_corners=False)
            latents = self.encode_imgs(pred_rgb_256)

        t = torch.randint(self.min_step, self.max_step + 1, (latents.shape[0],), dtype=torch.long, device=self.device)

        # Set weights acc to closeness in angle
        if len(ref_azimuths) > 1:
            inv_angles = 1/angles
            inv_angles[inv_angles > 100] = 100
            inv_angles /= inv_angles.max(dim=-1, keepdim=True)[0]
            inv_angles[inv_angles < 0.1] = 0
        else:
            inv_angles = torch.tensor([1.]).to(self.device)

        # Multiply closeness-weight by user-given weights
        zero123_ws = torch.tensor(embeddings['zero123_ws'])[None, :].to(self.device) * inv_angles
        zero123_ws /= zero123_ws.max(dim=-1, keepdim=True)[0]
        zero123_ws[zero123_ws < 0.1] = 0

        with torch.no_grad():
            noise = torch.randn_like(latents)
            latents_noisy = self.scheduler.add_noise(latents, noise, t)

            x_in = torch.cat([latents_noisy] * 2)
            t_in = torch.cat([t] * 2)

            noise_preds = []
            # Loop through each ref image
            for (zero123_w, c_crossattn, c_concat, ref_polar, ref_azimuth, ref_radius) in zip(zero123_ws.T,
                                                                                              embeddings['c_crossattn'], embeddings['c_concat'],
                                                                                              ref_polars, ref_azimuths, ref_radii):
                # polar,azimuth,radius are all actually delta wrt default
                p = polar + ref_polars[0] - ref_polar
                a = azimuth + ref_azimuths[0] - ref_azimuth
                a[a > 180] -= 360 # range in [-180, 180]
                r = radius + ref_radii[0] - ref_radius
                # T = torch.tensor([math.radians(p), math.sin(math.radians(-a)), math.cos(math.radians(a)), r])
                # T = T[None, None, :].to(self.device)
                T = torch.stack([torch.deg2rad(p), torch.sin(torch.deg2rad(-a)), torch.cos(torch.deg2rad(a)), r], dim=-1)[:, None, :]
                cond = {}
                clip_emb = self.model.cc_projection(torch.cat([c_crossattn.repeat(len(T), 1, 1), T], dim=-1))
                cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
                cond['c_concat'] = [torch.cat([torch.zeros_like(c_concat).repeat(len(T), 1, 1, 1).to(self.device), c_concat.repeat(len(T), 1, 1, 1)], dim=0)]
                noise_pred = self.model.apply_model(x_in, t_in, cond)
                noise_pred_uncond, noise_pred_cond = noise_pred.chunk(2)
                noise_pred = noise_pred_uncond + guidance_scale * (noise_pred_cond - noise_pred_uncond)
                noise_preds.append(zero123_w[:, None, None, None] * noise_pred)

        noise_pred = torch.stack(noise_preds).sum(dim=0) / zero123_ws.sum(dim=-1)[:, None, None, None]

        w = (1 - self.alphas[t])
        grad = (grad_scale * w)[:, None, None, None] * (noise_pred - noise)
        grad = torch.nan_to_num(grad)

        if save_guidance_path:
            with torch.no_grad():
                if as_latent:
                    pred_rgb_256 = self.decode_latents(latents) # claforte: test!

                # visualize predicted denoised image
                result_hopefully_less_noisy_image = self.decode_latents(self.model.predict_start_from_noise(latents_noisy, t, noise_pred))

                # visualize noisier image
                result_noisier_image = self.decode_latents(latents_noisy)

                # TODO: also denoise all-the-way

                # all 3 input images are [1, 3, H, W], e.g. [1, 3, 512, 512]
                viz_images = torch.cat([pred_rgb_256, result_noisier_image, result_hopefully_less_noisy_image],dim=-1)
                save_image(viz_images, save_guidance_path)

        targets = (latents - grad).detach()
        loss = 0.5 * F.mse_loss(latents.float(), targets, reduction='sum') / latents.shape[0]

        return loss

    # verification
    @torch.no_grad()
    def __call__(self,
            image, # image tensor [1, 3, H, W] in [0, 1]
            polar=0, azimuth=0, radius=0, # new view params
            scale=3, ddim_steps=50, ddim_eta=1, h=256, w=256, # diffusion params
            c_crossattn=None, c_concat=None, post_process=True,
        ):

        if c_crossattn is None:
            c,v = self.get_img_embeds(image)
            embeddings = {
                'c_crossattn': torch.stack(c, dim=0).to(self.device),
                'c_concat': torch.stack(v, dim=0).to(self.device)
            }

        T = torch.tensor([math.radians(polar), math.sin(math.radians(azimuth)), math.cos(math.radians(azimuth)), radius])
        T = T[None, None, :].to(self.device)

        cond = {}
        clip_emb = self.model.cc_projection(torch.cat([embeddings['c_crossattn'] if c_crossattn is None else c_crossattn, T], dim=-1))
        cond['c_crossattn'] = [torch.cat([torch.zeros_like(clip_emb).to(self.device), clip_emb], dim=0)]
        cond['c_concat'] = [torch.cat([torch.zeros_like(embeddings['c_concat']).to(self.device), embeddings['c_concat']], dim=0)] if c_concat is None else [torch.cat([torch.zeros_like(c_concat).to(self.device), c_concat], dim=0)]

        # produce latents loop
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
        imgs = imgs.cpu().numpy().transpose(0, 2, 3, 1) if post_process else imgs

        return imgs

    def decode_latents(self, latents):
        # zs: [B, 4, 32, 32] Latent space image
        # with self.model.ema_scope():
        imgs = self.model.decode_first_stage(latents)
        imgs = (imgs / 2 + 0.5).clamp(0, 1)

        return imgs # [B, 3, 256, 256] RGB space image

    def encode_imgs(self, imgs):
        # imgs: [B, 3, 256, 256] RGB space image
        # with self.model.ema_scope():
        imgs = imgs * 2 - 1
        latents = torch.cat([self.model.get_first_stage_encoding(self.model.encode_first_stage(img.unsqueeze(0))) for img in imgs], dim=0)
        return latents # [B, 4, 32, 32] Latent space image


if __name__ == '__main__':
    import cv2
    import argparse
    import matplotlib.pyplot as plt

    parser = argparse.ArgumentParser()

    parser.add_argument('--input', type=str)
    parser.add_argument('--fp16', action='store_true', help="use float16 for training") # no use now, can only run in fp32

    parser.add_argument('--polar', type=float, default=0, help='delta polar angle in [-90, 90]')
    parser.add_argument('--azimuth', type=float, default=0, help='delta azimuth angle in [-180, 180]')
    parser.add_argument('--radius', type=float, default=0, help='delta camera radius multiplier in [-0.5, 0.5]')

    opt = parser.parse_args()

    device = torch.device('cuda')

    print(f'[INFO] loading image from {opt.input} ...')
    image = cv2.imread(opt.input, cv2.IMREAD_UNCHANGED)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (256, 256), interpolation=cv2.INTER_AREA)
    image = image.astype(np.float32) / 255.0
    image = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).contiguous().to(device)

    print(f'[INFO] loading model ...')
    zero123 = Zero123(device, opt.fp16, opt=opt)

    print(f'[INFO] running model ...')
    outputs = zero123(image, polar=opt.polar, azimuth=opt.azimuth, radius=opt.radius)
    plt.imshow(outputs[0])
    plt.show()