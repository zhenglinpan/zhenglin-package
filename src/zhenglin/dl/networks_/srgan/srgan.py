"""
Super-resolution of CelebA using Generative Adversarial Networks.
The dataset can be downloaded from: https://www.dropbox.com/sh/8oqt9vytwxb3s4r/AADIKlz8PR9zr6Y20qbkunrba/Img/img_align_celeba.zip?dl=0
(if not available there see if options are listed at http://mmlab.ie.cuhk.edu.hk/projects/CelebA.html)
Instrustion on running the script:
1. Download the dataset from the provided link
2. Save the folder 'img_align_celeba' to '../../data/'
4. Run the sript using command 'python3 srgan.py'
"""

import argparse
import os
import numpy as np
import math
import itertools
import sys
from matplotlib import pyplot as plt
import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torch.autograd import Variable

from network import GeneratorResNet, Discriminator

import wandb
from tqdm import tqdm
# wandb.init(project="")

import torch.nn as nn
import torch.nn.functional as F
import torch

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--hr_height", type=int, default=192, help="high res. image height")
parser.add_argument("--hr_width", type=int, default=192, help="high res. image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving image samples")

parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--model_dir', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument('--version', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument("--step2save", type=int, default=20, help="Number of epoches to save a model")
parser.add_argument('--resume', action="store_true", help="train from latest checkpoints")
args = parser.parse_args()
print(args)

DEVICE = 1

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

save_info = ""
with open(model_save_path + '/info.txt', 'w') as f:
    f.write(save_info)
    f.write(str(args))

hr_shape = (args.hr_height, args.hr_width)

# Initialize generator and discriminator
generator = GeneratorResNet(in_channels=args.channels, out_channels=args.channels, n_residual_blocks=6).to(DEVICE)
discriminator = Discriminator(input_shape=(args.channels, *hr_shape)).to(DEVICE)
feature_extractor = UNet(args.channels, args.channels, 0).to(DEVICE)
# feature_extractor = RRDBNet(1, 1, 64, 1).to(DEVICE)

# Set feature extractor to inference mode
feature_extractor.train()

# Losses
criterion_GAN = torch.nn.MSELoss().to(DEVICE)
criterion_content = torch.nn.L1Loss().to(DEVICE)

if args.resume is True:
    generator.load_state_dict(torch.load(''))
    discriminator.load_state_dict(torch.load(''))

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr, betas=(args.b1, args.b2))

Tensor = torch.cuda.FloatTensor
train_set = HDF5Data(args.dataroot)
dataloader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

# ----------
#  Training
# ----------

for epoch in tqdm(range(args.epoch, args.n_epochs + 1)):
    for i, imgs in enumerate(dataloader):

        # Configure model input
        imgs_lr = Variable(imgs['A'].type(Tensor)).to(DEVICE)
        imgs_hr = Variable(imgs['B'].type(Tensor)).to(DEVICE)
        # print('imgs_lr', imgs_lr.shape)
        # print('imgs_hr', imgs_hr.shape)
        
        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((args.batch_size, imgs_lr.size(0), *hr_shape))), requires_grad=False).to(DEVICE)
        fake = Variable(Tensor(np.zeros((args.batch_size, imgs_lr.size(0), *hr_shape))), requires_grad=False).to(DEVICE)
        # print("valid", valid.shape)
        # print("fake", fake.shape)

        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()

        # Generate a high resolution image from low resolution input
        gen_hr = generator(imgs_lr)
        # print('gen_hr', gen_hr.shape)

        # Adversarial loss
        loss_GAN = criterion_GAN(discriminator(gen_hr), valid)

        # Content loss
        gen_features = feature_extractor(gen_hr)
        real_features = feature_extractor(imgs_hr)
        loss_content = criterion_content(gen_features, real_features.detach())

        # Total loss
        loss_G = loss_content + 1e-3 * loss_GAN

        loss_G.backward()
        optimizer_G.step()

        # ---------------------
        #  Train Discriminator
        # ---------------------

        optimizer_D.zero_grad()

        # Loss of real and fake images
        loss_real = criterion_GAN(discriminator(imgs_hr), valid)
        loss_fake = criterion_GAN(discriminator(gen_hr.detach()), fake)

        # Total loss
        loss_D = (loss_real + loss_fake) / 2

        loss_D.backward()
        optimizer_D.step()
        
        # wandb.log({"loss G": loss_G.item(), 
        #            "loss D": loss_D.item(), 
        #            "loss content": loss_content.item(), 
        #            "loss_GAN": loss_GAN})
        
    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "fake_B.jpg", gen_hr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "real_A.jpg", imgs_lr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(generator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_discriminator.pth'))
