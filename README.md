# pip install zhenglin
*@ Author: zhenglin*
*@version: 1.18.14*

This package contains some off-the-shelves deep-learning networks implemented with [![](https://img.shields.io/badge/Pytorch-ee4c2c?style=flat-square&logo=pytorch&logoColor=white)](https://pytorch.org/).

use
```bash
pip install zhenglin
```

to install this package.

[zhenglin](https://pypi.org/project/zhenglin/) package is mainly motivated by eriklindernoren and his [repo](https://github.com/eriklindernoren/PyTorch-GAN) which provides many **super clean and easy-to-read** implementation of GAN variants. It is friendly to beginners and also a good reference for advanced users, especially for DL developpers.

Specifically, this package provides
+ A universal structure under `zhenglin.dl.template.v1.*`
+ Loss functions under `zhenglin.dl.losses`
+ Metrics under `zhenglin.dl.metrics`
+ 16 highly modular and easy-to-use implementation of deep-learning networks under `zhenglin.dl.networks.*`
which includes(from a to z)
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
- [SWinIR](https://github.com/JingyunLiang/SwinIR)
- U2Net
- UNet
- [Attention-UNet](https://github.com/Andy-zhujunwen/UNET-ZOO/blob/master)

## News and Updates
+ Aug 13 2021: `Version: 1.18.15` Add basic quantization support.