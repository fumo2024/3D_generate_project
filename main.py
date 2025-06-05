import argparse
import yaml
import torch
from train import train

def main():
    parser = argparse.ArgumentParser(description="3D Generate Project")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    # 加载 YAML 配置
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # 选择设备
    if "device" not in config:
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Using device: {config['device']}")

    # 启动训练
    train(config)

if __name__ == "__main__":
    main()
