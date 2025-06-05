import torch
import torch.nn as nn
import torch.optim as optim
from .camera import Camera
from scene import Scene, GaussianModel
from gaussian_renderer import render as gs_render
from utils.general_utils import safe_state
from argparse import Namespace

class GaussianRenderer:
    def __init__(self, device="cuda"):
        self.device = device
        self.gaussians = GaussianModel(0)
        self.scene = None
        self.optimizer = None
        self.bg_color = torch.tensor([0, 0, 0], dtype=torch.float32, device=device)
        
    def initialize_from_image(self, image_tensor, depth=None):
        """
        从输入图像初始化高斯模型
        
        参数:
            image_tensor: torch.Tensor (3, H, W) [0,1]
            depth: torch.Tensor (H, W) [可选]
        """
        H, W = image_tensor.shape[1], image_tensor.shape[2]
        
        # 创建相机坐标网格
        u = torch.linspace(0, 1, W, device=self.device)
        v = torch.linspace(0, 1, H, device=self.device)
        grid_u, grid_v = torch.meshgrid(u, v, indexing='xy')
        
        # 如果没有深度图，则创建简单深度估计
        if depth is None:
            # 使用中心点深度为1，边缘深度为2的简单深度图
            center = torch.tensor([0.5, 0.5], device=self.device)
            dist = torch.sqrt((grid_u - center[0])**2 + (grid_v - center[1])**2)
            depth = 1.0 + dist * 1.0  # 深度范围1.0-2.0
        
        # 将UV坐标转换为3D点
        points = torch.zeros(H*W, 3, device=self.device)
        points[:, 0] = grid_u.flatten() * 2 - 1  # x: [-1,1]
        points[:, 1] = grid_v.flatten() * 2 - 1  # y: [-1,1]
        points[:, 2] = depth.flatten()           # z: 深度值
        
        # 使用图像颜色
        colors = image_tensor.permute(1,2,0).reshape(-1, 3)
        
        # 从点云创建高斯
        self.gaussians.create_from_pcd(points, colors)
        
    def set_optimizer(self, lr=0.01):
        self.optimizer = optim.Adam(self.gaussians.parameters(), lr=lr)
        
    def render(self, camera_dict):
        """
        渲染给定相机视角
        
        参数:
            camera_dict: dict, 包含相机参数
                
        返回:
            render_result: dict, 包含渲染图像和深度
        """
        # 创建相机对象
        camera = Camera.from_dict(camera_dict)
        
        # 渲染
        render_result = gs_render(camera, self.gaussians)
        return {
            "render": render_result["render"],
            "depth": render_result["depth"]
        }
    
    def train_step(self, camera_dict, gt_image):
        """
        单次训练步骤
        
        参数:
            camera_dict: 相机参数
            gt_image: 目标图像 (3, H, W) [0,1]
        """
        self.optimizer.zero_grad()
        
        # 渲染
        render_result = self.render(camera_dict)
        render_image = render_result["render"]
        
        # 计算损失
        loss = self.calculate_loss(render_image, gt_image)
        
        # 反向传播
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def calculate_loss(self, render_image, gt_image):
        """组合损失函数：L1 + SSIM"""
        l1_loss = torch.abs(render_image - gt_image).mean()
        
        # SSIM计算
        ssim_loss = 1 - self.ssim(render_image, gt_image)
        
        return l1_loss + 0.2 * ssim_loss
    
    def ssim(self, img1, img2, window_size=11, size_average=True):
        # 简化版SSIM实现
        # 实际应用中应使用更完整的实现
        C1 = 0.01**2
        C2 = 0.03**2
        
        mu1 = img1.mean()
        mu2 = img2.mean()
        sigma1 = img1.std()
        sigma2 = img2.std()
        sigma12 = ((img1 - mu1) * (img2 - mu2)).mean()
        
        ssim_map = ((2*mu1*mu2 + C1) * (2*sigma12 + C2)) / \
                   ((mu1**2 + mu2**2 + C1) * (sigma1**2 + sigma2**2 + C2))
        return ssim_map.mean()
    
    def save_model(self, path):
        """保存高斯模型为.ply文件"""
        self.scene = Scene(self.gaussians)
        self.scene.save(path)