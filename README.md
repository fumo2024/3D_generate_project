# 3D assets generation
这个项目的目的是将zero123预训练权重用于蒸馏优化3DGS模型，用途为课程项目。

This repository is built based on the [official repository of 3DGS](https://github.com/graphdeco-inria/gaussian-splatting/).

我们在3DGS官方代码的基础上，将有关zero123调用的模块加入，实现了基于zero123的蒸馏模型。

## result 展示


## Get Started
### Cloning the Repository
first clone this repository, unlike the original 3DGS repository, the `diff-gaussian-rasterization` and `simple-knn` libraries are already in the repo, not as git submodules.
```plain
submodules/diff-gaussian-rasterization
submodules/simple-knn
```
### Install Dependencies
running the code needs some dependencies, you can install them with the following command:
```bash
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
conda install nvidia/label/cuda-11.8.0::cuda # optional, for nvcc toolkits
```
You also need to install two customized packages `diff-gaussian-rasterization`, `simple-knn`, `latent-diffusion` and `taming-transformers`:
```bash
# remember to specify the cuda library path if some cuda header is missing
cd submodules/diff-gaussian-rasterization
pip install -e .

# remember to specify the cuda library path if some cuda header is missing
cd submodules/simple-knn
pip install -e .

# install latent-diffusion, which is used for setting zero123 model
cd submodules/latent-diffusion
pip install -e .

# install taming-transformers, which is used for implementing VQ-VAE (modified original code because it's too large)
cd submodules/taming-transformers
pip install -e .
```
### Install Zero123
you also need to install the zero123 checkpoints, which can be downloaded from the following sources:
```bash
https://huggingface.co/cvlab/zero123-weights/tree/main
wget https://cv.cs.columbia.edu/zero123/assets/$iteration.ckpt    # iteration = [105000, 165000, 230000, 300000]
```
I use the 105000 iteration checkpoint, under the observations that checkpoints trained longer tend to overfit to training data and suffer in zero-shot generalization. We need the model to perform zero-shot generalization to unseen objects.

### Data Preparation
The data downloading and processing are the same with the original 3DGS. Please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#running) for more details. If you want to run SteepGS on your own dataset, please refer to [here](https://github.com/graphdeco-inria/gaussian-splatting?tab=readme-ov-file#processing-your-own-scenes) for the instructions.

## Running 
after the data preparation and environment setup, if everything works right, the following command should give you the 3DGS model result:
```bash
python train.py -s <path to COLMAP dataset>
```

## Visualization
I have planned to set up a web interface for visualization, like in [this project](https://github.com/camenduru/gaussian-splatting-colab), it's not done anyway.

## Acknowledgements
This project is built based on many excellent open-source projects, which not only provide the code, but also the inspiration and ideas, they include:
| name | link |
| ---- | ---- |
| 3DGS | [https://github.com/graphdeco-inria/gaussian-splatting](https://github.com/graphdeco-inria/gaussian-splatting) |
| SteepGS | [https://github.com/facebookresearch/SteepGS](https://github.com/facebookresearch/SteepGS) |
| Zero123 | [https://github.com/cvlab-columbia/zero123](https://github.com/cvlab-columbia/zero123) |
| Gaussian Splatting Colab | [https://github.com/camenduru/gaussian-splatting-colab](https://github.com/camenduru/gaussian-splatting-colab) |
| Stable DreamFusion | [https://github.com/ashawkey/stable-dreamfusion](https://github.com/ashawkey/stable-dreamfusion) |
| Diffusers | [https://github.com/huggingface/diffusers](https://github.com/huggingface/diffusers) |
|... | ... |
