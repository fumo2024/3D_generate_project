import torch
import numpy as np

def rotation_matrix(azimuth, elevation, device='cuda'):
    """根据方位角和仰角生成旋转矩阵"""
    az = np.radians(azimuth)
    el = np.radians(elevation)
    
    R_az = torch.tensor([
        [np.cos(az), -np.sin(az), 0],
        [np.sin(az), np.cos(az), 0],
        [0, 0, 1]
    ], dtype=torch.float32, device=device)
    
    R_el = torch.tensor([
        [1, 0, 0],
        [0, np.cos(el), -np.sin(el)],
        [0, np.sin(el), np.cos(el)]
    ], dtype=torch.float32, device=device)
    
    return R_el @ R_az

def create_camera(azimuth, elevation, distance=1.5, fov=60, width=256, height=256, device='cuda'):
    """创建相机参数字典"""
    return {
        'azimuth': azimuth,
        'elevation': elevation,
        'distance': distance,
        'fov': fov,
        'width': width,
        'height': height,
        'R': rotation_matrix(azimuth, elevation, device),
        'T': torch.tensor([0, 0, distance], dtype=torch.float32, device=device),
        'znear': 0.01,
        'zfar': 100.0
    }