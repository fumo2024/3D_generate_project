import torch
import argparse
import os
from tqdm import tqdm
import numpy as np
from render.gaussian_renderer import GaussianRenderer
from guidance.zero123_utils import Zero123Generator
from utils.image_utils import remove_background, pil_to_tensor, resize_image
from utils.camera_utils import create_camera

def train(input_image_path, output_dir, iterations=1000, num_novel_views=10, device="cuda"):
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 初始化模型
    zero123 = Zero123Generator(device=device)
    gs_renderer = GaussianRenderer(device=device)
    
    # 加载输入图像并移除背景
    print("Removing background...")
    bg_removed = remove_background(input_image_path, device=device)
    bg_removed_path = os.path.join(output_dir, "bg_removed.png")
    bg_removed.save(bg_removed_path)
    print(f"Background removed image saved to {bg_removed_path}")
    
    # 转换为tensor并调整大小
    input_tensor = pil_to_tensor(bg_removed, device=device)
    input_tensor = resize_image(input_tensor, (256, 256))
    
    # 使用输入图像初始化3DGS
    print("Initializing 3D Gaussian model...")
    gs_renderer.initialize_from_image(input_tensor)
    gs_renderer.set_optimizer(lr=0.01)
    
    # 初始视角相机参数
    initial_camera = create_camera(0, 0, device=device)
    
    # 生成新视角
    print(f"Generating {num_novel_views} novel views...")
    novel_views = zero123.generate_novel_views(
        bg_removed, 
        num_views=num_novel_views,
        min_angle=-150,
        max_angle=150
    )
    
    # 添加初始视角到训练集
    training_views = [(input_tensor, initial_camera)] + novel_views
    
    # 保存生成的视角
    os.makedirs(os.path.join(output_dir, "generated_views"), exist_ok=True)
    for i, (image_tensor, camera) in enumerate(novel_views):
        img = tensor_to_pil(image_tensor)
        img.save(os.path.join(output_dir, "generated_views", f"view_{i}.png"))
    
    # 训练循环
    print("Starting training...")
    loss_history = []
    
    for i in tqdm(range(iterations), desc="Training"):
        # 随机选择一个视角进行训练
        idx = np.random.randint(0, len(training_views))
        gt_image, camera = training_views[idx]
        
        # 执行训练步骤
        loss = gs_renderer.train_step(camera, gt_image)
        loss_history.append(loss)
        
        # 定期保存进度
        if i % 100 == 0:
            tqdm.write(f"Iteration {i}/{iterations}, Loss: {loss:.6f}")
            
            # 渲染输入视角用于可视化
            render_result = gs_renderer.render(initial_camera)
            render_image = render_result["render"]
            
            # 保存渲染结果
            progress_path = os.path.join(output_dir, f"progress_{i}.png")
            tensor_to_pil(render_image).save(progress_path)
    
    # 保存最终模型
    model_path = os.path.join(output_dir, "reconstructed_model.ply")
    gs_renderer.save_model(model_path)
    print(f"Saved reconstructed model to {model_path}")
    
    return loss_history

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train 3D Gaussian model from single image")
    parser.add_argument("--input_image", type=str, required=True, help="Path to input image")
    parser.add_argument("--output_dir", type=str, default="output", help="Output directory")
    parser.add_argument("--iterations", type=int, default=1000, help="Number of training iterations")
    parser.add_argument("--num_novel_views", type=int, default=10, help="Number of novel views to generate")
    parser.add_argument("--device", type=str, default="cuda", choices=["cuda", "cpu"], help="Device to use")
    
    args = parser.parse_args()
    
    # 运行训练
    train(
        input_image_path=args.input_image,
        output_dir=args.output_dir,
        iterations=args.iterations,
        num_novel_views=args.num_novel_views,
        device=args.device
    )