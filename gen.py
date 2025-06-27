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
# Add pretrained zero123 to provide SJC loss

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

# modules about zero123 
import torchvision.transforms as T
from zero123_utils import Zero123, load_model_from_config
import torch.nn.functional as F

# initialize Zero123
def init_zero123_model(device, config_path, ckpt_path, fp16=True, vram_O=False):
    # 创建虚拟的opt对象，包含zero123需要的参数
    class Opt:
        def __init__(self):
            self.zero123_grad_scale = 'angle'
    
    opt = Opt()
    
    zero123_model = Zero123(
        device=device,
        fp16=fp16,
        config=config_path,
        ckpt=ckpt_path,
        vram_O=vram_O,
        t_range=[0.02, 0.98],
        opt=opt
    )
    return zero123_model

# obtain pose parameters for camera
def get_camera_params(viewpoint_cam, input_pose):
    """
    计算当前相机相对于输入视图的delta姿态参数
    返回: (delta_polar, delta_azimuth, delta_radius)
    """
    # 简化实现 - 实际中需要根据相机参数计算
    # 这里假设viewpoint_cam包含我们需要的角度信息
    return (
        viewpoint_cam.polar - input_pose['polar'],
        viewpoint_cam.azimuth - input_pose['azimuth'],
        viewpoint_cam.radius - input_pose['radius']
    )


# 主训练函数
def training(dataset, opt, pipe, testing_iterations, saving_iterations, checkpoint_iterations, checkpoint, debug_from):
    # ========== Zero123 初始化 ==========
    device = torch.device("cuda")
    zero123_config_path = "./pretrained/sd-objaverse-finetune-c_concat-256.yaml"
    zero123_ckpt_path = "./pretrained/zero123-xl.ckpt"
    zero123_model = init_zero123_model(device, zero123_config_path, zero123_ckpt_path)
    
    # 加载输入视图 (条件图像)
    input_im = load_input_view(dataset.source_path)
    
    # 获取输入视图的条件嵌入
    with torch.no_grad():
        c_crossattn, c_concat = zero123_model.get_img_embeds(input_im)
        embeddings = {
            'c_crossattn': c_crossattn,
            'c_concat': c_concat,
            'ref_polars': [0],  # 参考polar角度
            'ref_azimuths': [0], # 参考azimuth角度
            'ref_radii': [0],    # 参考半径
            'zero123_ws': [1.0]  # 权重
        }
    
    # 输入视图的相机参数 (简化)
    input_pose = {
        'polar': 0,
        'azimuth': 0,
        'radius': 0
    }

    # ========== 原始3DGS初始化 ==========
    if not SPARSE_ADAM_AVAILABLE and opt.optimizer_type == "sparse_adam":
        # did not test this, its functionality is due on the original 3DGS code base 
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
        # no change to network-gui part
        if network_gui.conn == None:
            network_gui.try_connect()
        while network_gui.conn != None:
            try:
                net_image_bytes = None
                custom_cam, do_training, pipe.convert_SHs_python, pipe.compute_cov3D_python, keep_alive, scaling_modifer = network_gui.receive()
                if custom_cam != None:
                    net_image = render(custom_cam, gaussians, pipe, background, scaling_modifier=scaling_modifer, use_trained_exp=dataset.train_test_exp, separate_sh=SPARSE_ADAM_AVAILABLE)["render"]
                    net_image_bytes = memoryview((torch.clamp(net_image, min=0, max=1.0) * 255).byte().permute(1, 2, 0).contiguous().cpu().numpy())
                network_gui.send(net_image_bytes, dataset.source_path)
                if do_training and ((iteration < int(opt.iterations)) or not keep_alive):
                    break
            except Exception as e:
                network_gui.conn = None
        
        iter_start.record()

        gaussians.update_learning_rate(iteration)

        # Every 1000 its we increase the levels of SH up to a maximum degree
        if iteration % 1000 == 0:
            gaussians.oneupSHdegree()

        # Pick a random Camera
        #TODO realize information-oriented sampling, maybe powers sampling efficiency
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
        
        # use zero123 to compute guided loss
        #TODO fullfill loss computation
        # obtain camera parameters for the current viewpoint
        delta_polar, delta_azimuth, delta_radius = get_camera_params(viewpoint_cam, input_pose)

        # preprocess the image for zero123
        pred_rgb = image.unsqueeze(0)  # 添加批次维度 [1, 3, H, W]
        pred_rgb = pred_rgb.clamp(0, 1)  # 确保在[0,1]范围

        # compute zero123 loss
        loss = zero123_model.train_step(
            embeddings=embeddings,
            pred_rgb=pred_rgb,
            polar=delta_polar,
            azimuth=delta_azimuth,
            radius=delta_radius,
            guidance_scale=3,
            as_latent=False,
            grad_scale=1
        )
        
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
            #TODO :add steepest 3DGS densification
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

# utils
def load_input_view(source_path):
    """加载输入视图作为条件图像"""
    from PIL import Image
    input_path = os.path.join(source_path, "input.png")
    img = Image.open(input_path).convert("RGB")

    transform = T.Compose([
        T.Resize(256),
        T.CenterCrop(256),
        T.ToTensor()
    ])
    return transform(img).unsqueeze(0).to(device)

def prepare_output_and_logger(args):    
    if not args.model_path:
        if os.getenv('OAR_JOB_ID'):
            unique_str=os.getenv('OAR_JOB_ID')
        else:
            unique_str = str(uuid.uuid4())
        args.model_path = os.path.join("./output/", unique_str[0:10])
        
    # Set up output folder
    print("Output folder: {}".format(args.model_path))
    os.makedirs(args.model_path, exist_ok = True)
    with open(os.path.join(args.model_path, "cfg_args"), 'w') as cfg_log_f:
        cfg_log_f.write(str(Namespace(**vars(args))))

    with open(os.path.join(args.model_path, "train_log.txt"), 'w') as log_f:
        # create & clear the log file
        pass

    # Create Tensorboard writer
    tb_writer = None
    if TENSORBOARD_FOUND:
        tb_writer = SummaryWriter(args.model_path)
    else:
        print("Tensorboard not available: not logging progress")
    return tb_writer

# ========== 训练报告函数 ==========
#TODO : combine loss and other nessary metrics to report
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
    # parse command line arguments
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

    parser.add_argument("--guidance_scale", type=float, default=3.0, help="zero123 guidance scale")
    parser.add_argument("--zero123_grad_scale", type=str, default="angle", help="zero123 gradient scale type, can be 'angle' or 'radius'")
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