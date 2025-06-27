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
You also need to install two customized packages `diff-gaussian-rasterization` and `simple-knn`:
```bash
# remember to specify the cuda library path if some cuda header is missing
cd submodules/diff-gaussian-rasterization
pip install -e .

# remember to specify the cuda library path if some cuda header is missing
cd submodules/simple-knn
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
the overalla 

