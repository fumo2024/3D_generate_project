import torch
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np
import os
import yaml
from guidance.zero123_utils import Zero123
from gaussian_model import GaussianModel
from render.gaussian_renderer import render_gaussian  # 假设存在此函数
from tqdm import tqdm

def load_input_image(path):
    image = Image.open(path).convert("RGB")
    return image

def remove_background(image):
    # TODO: 根据需求实现背景去除或 alpha 通道处理
    return image

def generate_points_from_depth(image, device):
    """
    基于单目深度估计生成初始点云
    :param image: PIL.Image 输入图像
    :param device: 设备
    :return: dict 包含初始化的点参数
    """
    # TODO: 利用预训练单目深度、法线估计模型生成点云坐标和属性
    # 占位示例生成随机点
    n_points = 1024
    points_xyz = torch.randn(n_points,3,device=device)
    points_scale = torch.ones(n_points,1,device=device)
    points_alpha = torch.ones(n_points,1,device=device)
    points_sh = torch.zeros(n_points,9,3,device=device)  # 球谐系数示例
    points_rot = torch.zeros(n_points,3,device=device)
    return {
        "points_xyz": points_xyz,
        "points_scale": points_scale,
        "points_alpha": points_alpha,
        "points_sh": points_sh,
        "points_rot": points_rot,
    }

def initialize_on_sphere(device, num_points=1024):
    """
    简单的球面均匀采样初始化
    :param device: Torch 设备
    :param num_points: 采样点数量
    :return: 初始化参数字典
    """
    points_xyz = torch.randn(num_points, 3, device=device)
    points_xyz /= points_xyz.norm(dim=-1, keepdim=True)
    points_scale = torch.ones(num_points, 1, device=device) * 0.1
    points_alpha = torch.ones(num_points, 1, device=device) * 0.1
    points_sh = torch.zeros(num_points, 9, 3, device=device)
    points_rot = torch.zeros(num_points, 3, device=device)
    return {
        "points_xyz": points_xyz,
        "points_scale": points_scale,
        "points_alpha": points_alpha,
        "points_sh": points_sh,
        "points_rot": points_rot,
    }

def sample_camera_pose(iteration, num_iters):
    """
    训练中采样相机随机视角
    :param iteration: 当前迭代次数
    :param num_iters: 总迭代次数
    :return: 4x4 numpy 相机外参矩阵
    """
    # TODO: 根据训练需求采样相机视角。临时返回单位矩阵。
    return np.eye(4)

def add_noise(image: torch.Tensor, t: torch.Tensor):
    """
    根据扩散时间步向图像添加噪声
    :param image: 形状 [1,C,H,W]
    :param t: 标量时间步 [0,1]
    :return: noisy_image, noise
    """
    noise = torch.randn_like(image)
    noisy_image = (1 - t) * image + t * noise
    return noisy_image, noise

def get_timestep_schedule(max_t=1.0, steps=1000):
    """
    构造时间步调度表
    :param max_t: 最大时间步，float
    :param steps: 总步数，int
    :return: numpy 数组，线性调度
    """
    return np.linspace(0, max_t, steps)

def sample_random_timestep(schedule):
    t = np.random.choice(schedule)
    return torch.tensor(t, dtype=torch.float32)

def train(config):
    device = torch.device(config.get("device", "cuda" if torch.cuda.is_available() else "cpu"))

    # 加载输入图像
    input_image = load_input_image(config["input_image_path"])
    input_image = remove_background(input_image)

    # 初始化 3DGS 点云
    if config.get("use_depth_init", False):
        params_init = generate_points_from_depth(input_image, device)
    else:
        params_init = initialize_on_sphere(device, num_points=config.get("num_init_points", 1024))

    gaussian_model = GaussianModel.create_from_params(params_init, device)

    # 初始化 Zero123 指导模型
    zero123_guidance = Zero123(
        config_path = config["zero123_config_path"],
        checkpoint_path = config["zero123_ckpt_path"],
        device=device,
    )

    optimizer = torch.optim.Adam(gaussian_model.get_params_to_optimize(), lr=config.get("learning_rate", 7e-3))

    timestep_schedule = get_timestep_schedule(steps=1000)
    max_iterations = config.get("max_iterations", 30000)
    densify_interval = config.get("densification_interval", 100)
    densify_warmup = config.get("densification_warm_up", 500)

    for iteration in tqdm(range(max_iterations)):
        # 1) 采样随机时间步 t
        t = sample_random_timestep(timestep_schedule).to(device)

        # 2) 采样随机相机视角
        cam_pose = sample_camera_pose(iteration, max_iterations)

        # 3) 通过 3DGS 渲染得到图像
        render_params = gaussian_model.get_point_data_for_rendering()
        rendered_image = render_gaussian(render_params, cam_pose, device)
        # rendered_image: Tensor [1,C,H,W], 范围[-1,1] 或 [0,1]，与 Zero123 输入对应
        
        # 4) 添加噪声
        noisy_image, noise = add_noise(rendered_image, t)

        # 5) 计算 SDS 损失
        predicted_noise = zero123_guidance.predict_noise(input_image, cam_pose, t, noisy_image)
        sds_loss = F.mse_loss(predicted_noise, noise.detach())

        # 6) 其他损失 (如果有)
        loss = sds_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # 7) 动态优化点云（增密、裁剪）
        if iteration > densify_warmup and iteration % densify_interval == 0:
            gaussian_model.densify_and_prune(optimizer, iteration)

        # 8) 视需要保存日志、检查点等
        if (iteration + 1) % config.get("log_interval", 1000) == 0:
            print(f"[Iter {iteration+1}] SDS Loss: {sds_loss.item():.6f}")

        # TODO: 保存检查点，导出中间结果等

