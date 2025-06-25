#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#
# Add zero123 to provide SJC loss

import os
import torch
import math
import numpy as np
from random import randint
from utils.loss_utils import l1_loss, ssim
from gaussian_renderer import render, network_gui
import sys
from scene import Scene, GaussianModel
from utils.general_utils import safe_state, get_expon_lr_func
import uuid
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser, Namespace
from arguments import ModelParams, PipelineParams, OptimizationParams
try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_FOUND = True
except ImportError:
    TENSORBOARD_FOUND = False

try:
    from fused_ssim import fused_ssim
    FUSED_SSIM_AVAILABLE = True
except:
    FUSED_SSIM_AVAILABLE = False

try:
    from diff_gaussian_rasterization import SparseGaussianAdam
    SPARSE_ADAM_AVAILABLE = True
except:
    SPARSE_ADAM_AVAILABLE = False

# ========== 从 run_zero123.py 引入的依赖 ==========
from einops import rearrange
from my.config import BaseConf
from adapt import ScoreAdapter
from run_img_sampling import SD, StableDiffusion
from pose import PoseConfig, camera_pose, sample_near_eye
from my3d import get_T, depth_smooth_loss
from misc import torch_samps_to_imgs

device_glb = torch.device("cuda")

# Zero123 模型配置
class Zero123Config(BaseConf):
    sd: SD = SD(
        variant="objaverse",
        scale=100.0
    )
    emptiness_scale: int = 10
    emptiness_weight: int = 0
    emptiness_step: float = 0.5
    emptiness_multiplier: float = 20.0
    depth_smooth_weight: float = 1e5
    near_view_weight: float = 1e5
    view_weight: int = 10000
    var_red: bool = True

# 初始化 Zero123 模型
def init_zero123_model(config):
    model = config.sd.make()
    model = model.to(device_glb)
    return model

# 计算 SJC 损失
def compute_sjc_loss(model, render_image, depth_map, input_im, input_pose, current_pose):
    """
    计算 SJC 损失，包括:
    - 去噪损失
    - 空性损失
    - 深度平滑损失
    - 近视图一致性损失
    """
    # 获取相对变换矩阵 T
    T_target = current_pose[:3, -1]
    T_cond = input_pose[:3, -1]
    T = get_T(T_target, T_cond).to(model.device)
    
    # 准备输入图像
    H, W = render_image.shape[2], render_image.shape[3]
    target_H, target_W = model.data_shape()[1:]
    
    # 调整渲染图像尺寸
    if not isinstance(model, StableDiffusion):
        y = torch.nn.functional.interpolate(render_image, (target_H, target_W), mode='bilinear')
    else:
        y = render_image
    
    # 选择噪声级别
    ts = model.us[30:-10]
    chosen_σs = np.random.choice(ts, 1, replace=False)
    chosen_σs = chosen_σs.reshape(-1, 1, 1, 1)
    chosen_σs = torch.as_tensor(chosen_σs, device=model.device, dtype=torch.float32)
    
    # 添加噪声
    noise = torch.randn(1, *y.shape[1:], device=model.device)
    zs = y + chosen_σs * noise
    
    # 获取条件嵌入
    score_conds = model.img_emb(input_im, conditioning_key='hybrid', T=T)
    
    # 去噪
    Ds = model.denoise_objaverse(zs, chosen_σs, score_conds)
    
    # 计算梯度 (SJC损失的核心)
    if config.var_red:
        grad = (Ds - y) / chosen_σs
    else:
        grad = (Ds - zs) / chosen_σs
    
    # 主要损失项
    sjc_loss = -torch.mean(grad * y)
    
    # 空性损失 (从原始SJC代码中提取)
    # 注意: 3DGS中没有显式权重，需调整
    # emptiness_loss = (torch.log(1 + config.emptiness_scale * ws) * (-1 / 2 * ws)).mean()
    # emptiness_loss = config.emptiness_weight * emptiness_loss
    
    # 深度平滑损失
    depth_smooth_loss_val = depth_smooth_loss(depth_map) * config.depth_smooth_weight
    
    # 近视图一致性损失 (可选)
    # 需渲染额外视图，计算量大
    # near_view_loss = 0
    
    total_loss = sjc_loss + depth_smooth_loss_val
    
    return total_loss, {
        "sjc": sjc_loss.item(),
        "depth_smooth": depth_smooth_loss_val.item()
    }

# 主训练函数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # ========== Zero123 初始化 ==========
    sjc_config = Zero123Config()
    zero123_model = init_zero123_model(sjc_config)
    
    # 加载输入视图 (条件图像)
    # 注意: 需要根据实际数据集调整
    input_im = load_input_view(dataset.source_path)  # 需实现此函数
    input_pose = np.eye(4)  # 需要实际的输入姿态
    
    # 设置模型条件
    with torch.no_grad():
        tforms = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop((256, 256))
        ])
        cond_im = tforms(input_im)
        zero123_model.clip_emb = zero123_model.model.get_learned_conditioning(cond_im.float()).tile(1,1,1).detach()
        zero123_model.vae_emb = zero123_model.model.encode_first_stage(cond_im.float()).mode().detach()
    
    # ========== 原始3DGS初始化 ==========
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        sys.exit(f"Trying to use sparse adam but it is not installed, please install the correct rasterizer using pip install [3dgs_accel].")

    first_iter = 0
    tb_writer = prepare_output_and_logger(dataset)
    gaussians = GaussianModel(dataset.sh_degree, opt.optimizer_type)
    scene = Scene(dataset, gaussians)
    gaussians.training_setup(opt)
    if checkpoint:
        (model_params, first_iter) = torch.load(checkpoint)
        gaussians.restore(model_params, opt)

    bg_color = [1, 1, 1] if dataset.white_background else [0, 0, 0]
    background = torch.tensor(bg_color, dtype=torch.float32, device="cuda")

    iter_start = torch.cuda.Event(enable_timing = True)
    iter_end = torch.cuda.Event(enable_timing = True)

    use_sparse_adam = opt.optimizer_type == "sparse_adam" and SPARSE_ADAM_AVAILABLE 
    depth_l1_weight = get_expon_lr_func(opt.depth_l1_weight_init, opt.depth_l1_weight_final, max_steps=opt.iterations)

    viewpoint_stack = scene.getTrainCameras().copy()
    viewpoint_indices = list(range(len(viewpoint_stack)))
    ema_loss_for_log = 0.0
    ema_Ll1depth_for_log = 0.0

    progress_bar = tqdm(range(first_iter, opt.iterations), desc="Training progress")
    first_iter += 1
    for iteration in range(first_iter, opt.iterations + 1):
        # ... [网络GUI部分保持不变] ...
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        if not viewpoint_stack:
            viewpoint_stack = scene.getTrainCameras().copy()
            viewpoint_indices = list(range(len(viewpoint_stack)))
        rand_idx = randint(0, len(viewpoint_indices) - 1)
        viewpoint_cam = viewpoint_stack.pop(rand_idx)
        vind = viewpoint_indices.pop(rand_idx)

        # Render
        if (iteration - 1) == debug_from:
            pipe.debug = True

        bg = torch.rand((3), device="cuda") if opt.random_background else background

        render_pkg = render(viewpoint_cam, gaussians, pipe, bg, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)
        image, depth, viewspace_point_tensor, visibility_filter, radii = (
            render_pkg["render"], 
            render_pkg["depth"],
            render_pkg["viewspace_points"], 
            render_pkg["visibility_filter"], 
            render_pkg["radii"]
        )
        
        # ========== 替换为SJC损失计算 ==========
        # 获取当前相机姿态 (需要从viewpoint_cam中提取)
        # 注意: 需要实现从viewpoint_cam到4x4姿态矩阵的转换
        current_pose = get_camera_pose(viewpoint_cam)  # 需要实现此函数
        
        # 计算SJC损失
        sjc_loss, loss_components = compute_sjc_loss(
            model=zero123_model,
            render_image=image.unsqueeze(0),  # 添加批次维度
            depth_map=depth,
            input_im=input_im,
            input_pose=input_pose,
            current_pose=current_pose
        )
        
        # 总损失
        loss = sjc_loss
        
        # 反向传播
        loss.backward()

        iter_end.record()

        with torch.no_grad():
            # 更新进度条
            ema_loss_for_log = 0.4 * loss.item() + 0.6 * ema_loss_for_log
            progress_bar.set_postfix({"Loss": f"{ema_loss_for_log:.{7}f}"})
            progress_bar.update(1)
            
            if iteration == opt.iterations:
                progress_bar.close()

            # 日志和保存
            training_report(tb_writer, iteration, loss, iter_start.elapsed_time(iter_end), 
                           testing_iterations, scene, render, (pipe, background, 1., SPARSE_ADAM_AVAILABLE, None, dataset.train_test_exp), 
                           dataset.train_test_exp)
            
            if (iteration in saving_iterations):
                print("\n[ITER {}] Saving Gaussians".format(iteration))
                scene.save(iteration)

            # 致密化
            if iteration < opt.densify_until_iter:
                gaussians.max_radii2D[visibility_filter] = torch.max(gaussians.max_radii2D[visibility_filter], radii[visibility_filter])
                gaussians.add_densification_stats(viewspace_point_tensor, visibility_filter)

                if iteration > opt.densify_from_iter and iteration % opt.densification_interval == 0:
                    size_threshold = 20 if iteration > opt.opacity_reset_interval else None
                    gaussians.densify_and_prune(opt.densify_grad_threshold, 0.005, scene.cameras_extent, size_threshold, radii)
                
                if iteration % opt.opacity_reset_interval == 0 or (dataset.white_background and iteration == opt.densify_from_iter):
                    gaussians.reset_opacity()

            # 优化器步骤
            if iteration < opt.iterations:
                if use_sparse_adam:
                    visible = radii > 0
                    gaussians.optimizer.step(visible, radii.shape[0])
                    gaussians.optimizer.zero_grad(set_to_none = True)
                else:
                    gaussians.optimizer.step()
                    gaussians.optimizer.zero_grad(set_to_none = True)

            if (iteration in checkpoint_iterations):
                print("\n[ITER {}] Saving Checkpoint".format(iteration))
                torch.save((gaussians.capture(), iteration), scene.model_path + "/chkpnt" + str(iteration) + ".pth")

# 需要实现的辅助函数
def load_input_view(source_path):
    """加载输入视图作为条件图像"""
    # 实现取决于数据集结构
    # 示例: 从固定路径加载图像
    from PIL import Image
    import torchvision.transforms as T
    img = Image.open(os.path.join(source_path, "input_view.png")).convert("RGB")
    transform = T.Compose([T.Resize(256), T.CenterCrop(256), T.ToTensor()])
    return transform(img).unsqueeze(0).to(device_glb) * 2 - 1

def get_camera_pose(viewpoint_cam):
    """从ViewpointCamera对象提取4x4姿态矩阵"""
    # 需要根据3DGS的相机表示实现
    # 示例: 假设viewpoint_cam包含旋转和平移
    pose = np.eye(4)
    pose[:3, :3] = viewpoint_cam.R
    pose[:3, 3] = viewpoint_cam.T
    return pose

# ========== 修改训练报告函数 ==========
def training_report(tb_writer, iteration, loss, elapsed, testing_iterations, scene, renderFunc, renderArgs, train_test_exp):
    if tb_writer:
        tb_writer.add_scalar('train_loss_patches/total_loss', loss.item(), iteration)
        tb_writer.add_scalar('iter_time', elapsed, iteration)

    # 测试和训练样本报告
    if iteration in testing_iterations:
        torch.cuda.empty_cache()
        validation_configs = ({'name': 'test', 'cameras': scene.getTestCameras()}, 
                              {'name': 'train', 'cameras': [scene.getTrainCameras()[idx % len(scene.getTrainCameras())] for idx in range(5, 30, 5)]})

        for config in validation_configs:
            if config['cameras'] and len(config['cameras']) > 0:
                psnr_test = 0.0
                for idx, viewpoint in enumerate(config['cameras']):
                    image = torch.clamp(renderFunc(viewpoint, scene.gaussians, *renderArgs)["render"], 0.0, 1.0)
                    gt_image = torch.clamp(viewpoint.original_image.to("cuda"), 0.0, 1.0)
                    if tb_writer and (idx < 5):
                        tb_writer.add_images(config['name'] + "_view_{}/render".format(viewpoint.image_name), image[None], global_step=iteration)
                    psnr_test += psnr(image, gt_image).mean().double()
                psnr_test /= len(config['cameras'])
                print("\n[ITER {}] Evaluating {}: PSNR {}".format(iteration, config['name'], psnr_test))

        if tb_writer:
            tb_writer.add_histogram("scene/opacity_histogram", scene.gaussians.get_opacity, iteration)
            tb_writer.add_scalar('total_points', scene.gaussians.get_xyz.shape[0], iteration)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    # ... [参数解析部分保持不变] ...
    parser = ArgumentParser(description="Training script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--ip', type=str, default="127.0.0.1")
    parser.add_argument('--port', type=int, default=6009)
    parser.add_argument('--debug_from', type=int, default=-1)
    parser.add_argument('--detect_anomaly', action='store_true', default=False)
    parser.add_argument("--test_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--save_iterations", nargs="+", type=int, default=[7_000, 30_000])
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument('--disable_viewer', action='store_true', default=False)
    parser.add_argument("--checkpoint_iterations", nargs="+", type=int, default=[])
    parser.add_argument("--start_checkpoint", type=str, default = None)
    args = parser.parse_args(sys.argv[1:])
    args.save_iterations.append(args.iterations)
    
    print("Optimizing " + args.model_path)

    # 初始化系统状态
    safe_state(args.quiet)

    # 启动GUI并运行训练
    if not args.disable_viewer:
        network_gui.init(args.ip, args.port)
    torch.autograd.set_detect_anomaly(args.detect_anomaly)
    training(lp.extract(args), op.extract(args), pp.extract(args), args.test_iterations, args.save_iterations, args.checkpoint_iterations, args.start_checkpoint, args.debug_from)

    print("\nTraining complete.")