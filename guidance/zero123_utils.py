import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import numpy as np

class Zero123:
    def __init__(self, config_path, checkpoint_path, device):
        """
        初始化 Zero123 模型
        :param config_path: 模型配置文件路径
        :param checkpoint_path: 模型权重文件路径
        :param device: 运行设备 ('cuda' 或 'cpu')
        """
        self.device = device
        self.config_path = config_path
        self.checkpoint_path = checkpoint_path
        # TODO: 根据所用的Zero123实现加载模型
        # 例如加载 Diffusers pipeline 或 自定义模型
        # self.model = load_model(config_path, checkpoint_path).to(device)
        # self.model.eval()

    def _preprocess_input(self, image, pose_matrix):
        """
        预处理输入图片和相机姿态，转换为模型输入格式
        :param image: PIL.Image 或 Tensor，RGB 图像
        :param pose_matrix: 4x4 numpy 数组或 Tensor 表示相机姿态
        :return: 模型输入字典
        """
        # 假设 image 是 PIL.Image，转换为 Tensor 并归一化到 [-1, 1]
        if isinstance(image, Image.Image):
            image = T.ToTensor()(image).unsqueeze(0)  # [1,C,H,W]
        elif isinstance(image, np.ndarray):
            image = torch.from_numpy(image).permute(2,0,1).unsqueeze(0).float() / 255.0
        image = 2*image - 1  # Normalize to [-1, 1]

        pose = torch.tensor(pose_matrix, dtype=torch.float32, device=self.device).unsqueeze(0)  # [1,4,4]

        return {
            "image": image.to(self.device),
            "pose": pose,
        }

    def _get_raw_noise_prediction(self, inputs, noise_scale):
        """
        根据带噪声输入，使用 Zero123 模型预测噪声
        :param inputs: 预处理后的输入字典
        :param noise_scale: 噪声强度标量 Tensor
        :return: 预测噪声 Tensor，形状等同 inputs["image"]
        """
        # TODO: 调用 Zero123 模型的推理接口，预测噪声
        # 这里是占位符，返回和输入图像相同尺寸的零张量
        batch_size, c, h, w = inputs["image"].shape
        predicted_noise = torch.zeros_like(inputs["image"])
        return predicted_noise

    @torch.no_grad()
    def predict_noise(self, input_image, pose_matrix, noise_level, noisy_image):
        """
        对带噪声的图像调用模型预测噪声（用于SDS）
        :param input_image: 输入的原始图像 (PIL.Image 或 numpy)
        :param pose_matrix: 相机姿态矩阵 4x4
        :param noise_level: 当前时间步噪声强度（float或Tensor）
        :param noisy_image: 添加噪声后的图像 Tensor，形状 (1,C,H,W)
        :return: 预测的噪声 Tensor，形状与 noisy_image 相同
        """
        inputs = self._preprocess_input(input_image, pose_matrix)
        # 将 inputs 中的 image 替换为 noisy_image
        inputs['image'] = noisy_image.to(self.device)
        predicted_noise = self._get_raw_noise_prediction(inputs, noise_level)
        return predicted_noise