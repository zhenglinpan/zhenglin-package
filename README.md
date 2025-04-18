# pip install zhenglin
*@ Author: zhenglin*
*@version: 2.0*

This package contains some off-the-shelves deep-learning networks implemented with [![](https://img.shields.io/badge/Pytorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/).

use
```bash
pip install zhenglin
```

to install this package.

## Updates
+ Apr 18 2025: `Version: 2.0`. Add init version of `supercv(cv)` class, a universal wrapper for cv2, PIL.Image and Torch.tensor.
+ Nov 13 2024: `Version: 1.21`. Package overhaul. Add `huggingface accelerate`, `pytorch-lightning`, and `pytorch-ddp` support.
+ Oct 26 2023: `Version: 1.20.19` Add `TransUNet` and `SwinUNet`
+ Aug 13 2023: `Version: 1.18.15` Add basic quantization support.


## Introduction

[zhenglin](https://pypi.org/project/zhenglin/) package was motivated by eriklindernoren and his [repo](https://github.com/eriklindernoren/PyTorch-GAN) which provides many **grab-and-run** implementation of DL variants. It is the level ground for zhenglin's deep learning projects.

Specifically, this package provides
+ universal structure-template:
    - `zhenglin.accelerate.train`
    - `zhenglin.lightning.train`
    - `zhenglin.pytorch.train`
+ loss functions and metrics
    - `zhenglin.utils.loss`
    - `zhenglin.utils.metrics`
+ utils
    - `zhenglin.utils.utils`
+ 20 modular deep-learning networks
    - `zhenglin.networks.*`

### Networks(Alphabetically):
- [cycleGAN](https://github.com/aitorzip/PyTorch-CycleGAN)
- [DDPM](https://github.com/dome272/Diffusion-Models-pytorch)
- [DeblurGAN](https://github.com/fourson/DeblurGAN-pytorch/tree/master)
- [EDSR](https://github.com/twtygqyy/pytorch-edsr/blob/master/edsr.py)
- [ESRGAN](https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/esrgan/esrgan.py)
- [Noise2Void](https://github.com/JohnYKiyo/Noise2Void/blob/master/02_training_test_Noise2Void.ipynb)
- [Pix2Pix](https://github.com/mrzhu-cool/pix2pix-pytorch)
- [RCAN](https://github.com/yjn870/RCAN-pytorch)
- ResNet
- [Restormer](https://github.com/leftthomas/Restormer)
- RRDBNet
- [SqueezeNet](https://github.com/gsp-27/pytorch_Squeezenet/tree/master)
- [SRDRM](https://github.com/xahidbuffon/SRDRM/tree/master)
- SRGAN
- [SwinIR](https://github.com/JingyunLiang/SwinIR)
- [SwinUNet](https://github.com/HuCaoFighting/Swin-Unet/tree/main)
- [TransUNet](https://github.com/Beckschen/TransUNet)
- U2Net
- UNet
- [Attention-UNet](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master)
