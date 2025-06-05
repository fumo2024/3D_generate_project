import argparse
import torch
import numpy as np
from render.gaussian_renderer import GaussianRenderer
from utils.camera_utils import create_camera
from utils.image_utils import tensor_to_pil
import os
# from PIL import Image

def main():
    parser = argparse.ArgumentParser(description="3D Reconstruction from Single Image")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--mode", choices=["train", "render"], default="train", help="Run mode")
    parser.add_argument("--model_path", type=str, help="Path to trained model (for render mode)")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # 确保输出目录存在
    os.makedirs(args.output_dir, exist_ok=True)
    
    if args.mode == "train":
        # 训练模式 - 调用训练脚本
        from train import train
        train(
            input_image_path=args.input_image,
            output_dir=args.output_dir,
            device=args.device
        )
    elif args.mode == "render":
        # 渲染模式 - 加载模型并渲染
        if not args.model_path:
            raise ValueError("Model path is required for render mode")
        
        # 加载模型
        renderer = GaussianRenderer(device=args.device)
        # 注意: 实际加载模型需要实现相应方法
        # renderer.load_model(args.model_path)
        
        # 创建相机轨迹
        angles = np.linspace(0, 360, 36)
        
        # 渲染360度动画
        animation_dir = os.path.join(args.output_dir, "animation")
        os.makedirs(animation_dir, exist_ok=True)
        
        print("Rendering 360 degree animation...")
        for i, angle in enumerate(angles):
            camera = create_camera(angle, 0, device=args.device)
            render_result = renderer.render(camera)
            render_image = render_result["render"]
            
            img = tensor_to_pil(render_image)
            img.save(os.path.join(animation_dir, f"frame_{i:03d}.png"))
        
        print(f"Animation frames saved to {animation_dir}")

if __name__ == "__main__":
    main()