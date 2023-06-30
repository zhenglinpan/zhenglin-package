"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if argsions are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 esrgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from esrgan.network import *
from datasets import HDF5Data

import torch.nn as nn
import torch.nn.functional as F
import torch
from matplotlib import pyplot as plt

import wandb
from tqdm import tqdm
from UNet import UNet

# wandb.init(project="")

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=451, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.9, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=192, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=192, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")
parser.add_argument("--checkpoint_interval", type=int, default=5000, help="batch interval between model checkpoints")
parser.add_argument("--residual_blocks", type=int, default=9, help="number of residual blocks in the generator")
parser.add_argument("--warmup_batches", type=int, default=0, help="number of batches with pixel-wise loss only")
parser.add_argument("--lambda_adv", type=float, default=5e-3, help="adversarial loss weight")
parser.add_argument("--lambda_pixel", type=float, default=1e-2, help="pixel-wise loss weight")

parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--model_dir', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument('--version', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument("--step2save", type=int, default=50, help="Number of epoches to save a model")
parser.add_argument('--resume', action="store_true", help="train from latest checkpoints")
args = parser.parse_args()
print(str(args))

DEVICE = 0

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

save_info = ""
with open(model_save_path + '/info.txt', 'w') as f:
    f.write(save_info)
    f.write(str(args))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

hr_shape = (args.hr_height, args.hr_width)

# Initialize generator and discriminator
generator = GeneratorRRDB(args.channels, filters=64, num_res_blocks=9, num_upsample=2)  # recommand: 64, 9, 2
discriminator = Discriminator(input_shape=(args.channels, *hr_shape))
feature_extractor = UNet(args.channels, args.channels, 0)
# feature_extractor = RRDBNet(args.channels, args.channels, 64, 1)

if args.resume is True:
    # Load pretrained models
    generator.load_state_dict(torch.load('', map_location=torch.device(DEVICE)))
    discriminator.load_state_dict(torch.load('', map_location=torch.device(DEVICE)))

# generator = nn.DataParallel(generator).to(DEVICE)
# discriminator = nn.DataParallel(discriminator).to(DEVICE)
# feature_extractor = nn.DataParallel(feature_extractor).to(DEVICE)

generator = generator.to(DEVICE)
discriminator = discriminator.to(DEVICE)
feature_extractor = feature_extractor.to(DEVICE)

# Set feature extractor to inference mode
feature_extractor.eval()

# Losses
criterion_GAN = torch.nn.BCEWithLogitsLoss()
criterion_content = torch.nn.L1Loss()
criterion_pixel = torch.nn.L1Loss()

# argsimizers
argsimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
argsimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

train_set = HDF5Data(args.dataroot)
dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

# ----------
#  Training
# ----------
pre_img_lr = None
pre_img_hr = None
for epoch in tqdm(range(args.epoch, args.n_epochs + 1)):
    for i, imgs in enumerate(dataloader):
        batches_done = epoch * len(dataloader) + i
        img_lr, img_hr, attack = imgs['A'], imgs['B'], imgs['attack']
        
        if not attack:
            pre_img_lr = img_lr
            pre_img_hr = img_hr
            img_lr = Variable(img_lr.type(Tensor)).to(DEVICE)
            img_hr = Variable(img_hr.type(Tensor)).to(DEVICE)
        else:
            print("circumventing attacks...")
            img_lr = Variable(pre_img_lr.type(Tensor)).to(DEVICE)
            img_hr = Variable(pre_img_hr.type(Tensor)).to(DEVICE)

        # print('img_lr', img_lr.shape)
        # print('img_hr', img_hr.shape)
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((img_lr.size(0), 1, *hr_shape))), requires_grad=False).to(DEVICE)
        fake = Variable(Tensor(np.zeros((img_lr.size(0), 1, *hr_shape))), requires_grad=False).to(DEVICE)
    
        # print("valid", valid.shape)
        # print("fake", fake.shape)
        
        # ------------------
        #  Train Generators
        # ------------------

        argsimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(img_lr)
        # print('gen_hr', gen_hr.shape)
        
        # Measure pixel-wise loss against ground truth
        loss_pixel = criterion_pixel(gen_hr, img_hr)

        if batches_done < args.warmup_batches:
            # Warm-up (pixel-wise loss only)
            loss_pixel.backward()
            argsimizer_G.step()
            # print(
            #     "[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
            #     % (epoch, args.n_epochs, i, len(dataloader), loss_pixel.item())
            # )
            continue

        # Extract validity predictions from discriminator
        pred_real = discriminator(img_hr).detach()
        pred_fake = discriminator(gen_hr)

        # print('pred_fake', pred_fake.shape) #  torch.Size([1, 1, 12, 12])
        # print('pred_real', pred_real.shape) #  torch.Size([1, 1, 12, 12])
        
        tmp = pred_fake - pred_real
        # print('temp', tmp.shape)
        
        # Adversarial loss (relativistic average GAN)
        loss_GAN = criterion_GAN(tmp, valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(img_hr).detach()
        loss_content = criterion_content(gen_features, real_features)

        # print('gen_hr', pred_fake.shape)
        # print('img_hr', pred_fake.shape)

        # Total generator loss
        loss_G = loss_content + args.lambda_adv * loss_GAN + args.lambda_pixel * loss_pixel

        loss_G.backward()
        argsimizer_G.step()
 
        # ---------------------
        #  Train Discriminator
        # ---------------------

        argsimizer_D.zero_grad()

        pred_real = discriminator(img_hr)
        pred_fake = discriminator(gen_hr.detach())

        # Adversarial loss for real and fake images (relativistic average GAN)
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), valid)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), fake)
 
        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        argsimizer_D.step()
        
        # wandb.log({"loss G": loss_G.item(), 
        #            "loss D": loss_D.item(), 
        #            "loss content": loss_content.item(), 
        #            "loss_pixel": loss_pixel, 
        #            "loss_GAN": loss_GAN})
        
    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "fake_B_circumvent.jpg", gen_hr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "real_A_circumvent.jpg", img_lr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(generator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_discriminator.pth'))