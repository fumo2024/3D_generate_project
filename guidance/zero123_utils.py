import torch
from diffusers import StableDiffusionInstructPix2PixPipeline
from PIL import Image
import numpy as np
from ..utils.image_utils import pil_to_tensor, tensor_to_pil

class Zero123Generator:
    def __init__(self, device="cuda", torch_dtype=torch.float16):
        self.device = device
        self.torch_dtype = torch_dtype
        self.pipe = StableDiffusionInstructPix2PixPipeline.from_pretrained(
            "bennyguo/zero123-diffusers", 
            torch_dtype=self.torch_dtype
        ).to(self.device)
    
    def generate_novel_views(self, input_image, num_views=10, min_angle=-180, max_angle=180):
        """
        为输入图像生成多个新视角
        
        参数:
            input_image: PIL.Image 或 torch.Tensor (C,H,W) [0,1]
            num_views: 生成视角数量
            min_angle: 最小方位角
            max_angle: 最大方位角
            
        返回:
            views: 包含(num_views)个元组(图像, 相机参数)的列表
        """
        # 如果输入是tensor，转换为PIL
        if isinstance(input_image, torch.Tensor):
            input_image = tensor_to_pil(input_image)
        
        views = []
        angles = np.linspace(min_angle, max_angle, num_views)
        
        for i, azimuth in enumerate(angles):
            # 生成随机的仰角 (-30到30度之间)
            elevation = np.random.uniform(-30, 30)
            
            # 生成图像
            novel_image, camera_params = self.generate_view_from_angle(input_image, azimuth, elevation)
            views.append((novel_image, camera_params))
        
        return views
    
    def generate_view_from_angle(self, input_image, azimuth, elevation):
        """
        从指定角度生成新视角
        
        参数:
            input_image: PIL.Image
            azimuth: 方位角 (度)
            elevation: 仰角 (度)
            
        返回:
            image_tensor: 生成的图像 (C,H,W) [0,1]
            camera_params: 相机参数字典
        """
        if isinstance(input_image, torch.Tensor):
            input_image = tensor_to_pil(input_image)
        
        prompt = f"<img> <new_view> azimuth={azimuth:.1f} elevation={elevation:.1f}"
        novel_image = self.pipe(
            prompt, 
            image=input_image,
            guidance_scale=3.0
        ).images[0]
        
        # 转换为tensor
        novel_tensor = pil_to_tensor(novel_image, self.device)
        
        # 相机参数
        camera_params = create_camera(azimuth, elevation, device=self.device)
        
        return novel_tensor, camera_params