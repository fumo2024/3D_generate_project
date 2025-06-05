import torch
from dataclasses import dataclass
from ..utils.camera_utils import rotation_matrix

@dataclass
class Camera:
    R: torch.Tensor        # 旋转矩阵 (3,3)
    T: torch.Tensor        # 平移向量 (3,)
    FoVx: float            # 水平视场角 (度)
    FoVy: float            # 垂直视场角 (度)
    image_width: int       # 图像宽度
    image_height: int      # 图像高度
    bg: torch.Tensor = torch.tensor([0, 0, 0], dtype=torch.float32)  # 背景颜色
    znear: float = 0.01    # 近裁剪面
    zfar: float = 100.0    # 远裁剪面
    
    @classmethod
    def from_dict(cls, camera_dict):
        """从字典创建相机对象"""
        return cls(
            R=camera_dict['R'],
            T=camera_dict['T'],
            FoVx=camera_dict['fov'],
            FoVy=camera_dict['fov'],
            image_width=camera_dict['width'],
            image_height=camera_dict['height'],
            znear=camera_dict['znear'],
            zfar=camera_dict['zfar']
        )