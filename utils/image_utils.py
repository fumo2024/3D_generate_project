import torch
import numpy as np
from PIL import Image
from carvekit.api.high import HiInterface

def remove_background(image_path, device='cuda'):
    """移除图像背景"""
    interface = HiInterface(object_type="object",  # 分割对象
                            batch_size_seg=1,
                            batch_size_matting=1,
                            device=device,
                            seg_mask_size=640,  # 分割掩码大小
                            matting_mask_size=2048,
                            trimap_prob_threshold=231,
                            trimap_dilation=30,
                            trimap_erosion_iters=5,
                            fp16=False)
    
    # 处理图像
    image = Image.open(image_path)
    image = image.convert('RGB')
    result = interface([image])
    return result[0]  # 返回移除背景后的PIL图像

def pil_to_tensor(image, device='cuda'):
    """将PIL图像转换为PyTorch张量"""
    if not isinstance(image, Image.Image):
        raise ValueError("Input must be a PIL Image")
    
    # 转换为numpy数组并归一化
    np_image = np.array(image) / 255.0
    tensor = torch.tensor(np_image, dtype=torch.float32).permute(2, 0, 1)
    return tensor.to(device)

def tensor_to_pil(tensor):
    """将PyTorch张量转换为PIL图像"""
    if tensor.dim() != 3:
        raise ValueError("Input tensor must be 3D (C, H, W)")
    
    # 转换为numpy数组并反归一化
    np_image = tensor.permute(1, 2, 0).cpu().numpy()
    np_image = (np_image * 255).astype(np.uint8)
    return Image.fromarray(np_image)

def resize_image(image, size=(256, 256)):
    """调整图像大小"""
    if isinstance(image, torch.Tensor):
        pil_img = tensor_to_pil(image)
        pil_img = pil_img.resize(size)
        return pil_to_tensor(pil_img, device=image.device)
    elif isinstance(image, Image.Image):
        return image.resize(size)
    else:
        raise TypeError("Unsupported image type")