import argparse
import os
import numpy as np
import math
import itertools
import datetime
import time

import torchvision.transforms as transforms
from torchvision.utils import save_image, make_grid

from torch.utils.data import DataLoader
from torchvision import datasets
from torch.autograd import Variable

from network import *
from datasets import *
from utils import *

import torch.nn as nn
import torch.nn.functional as F
import torch

import wandb
from tqdm import tqdm
wandb.init(project="")

from matplotlib import pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument("--epoch", type=int, default=0, help="epoch to start training from")
parser.add_argument("--n_epochs", type=int, default=500, help="number of epochs of training")
parser.add_argument("--dataset_name", type=str, default="monet2photo", help="name of the dataset")
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5, help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999, help="adam: decay of first order momentum of gradient")
parser.add_argument("--decay_epoch", type=int, default=100, help="epoch from which to start lr decay")
parser.add_argument("--n_cpu", type=int, default=8, help="number of cpu threads to use during batch generation")
parser.add_argument("--img_height", type=int, default=192, help="size of image height")
parser.add_argument("--img_width", type=int, default=192, help="size of image width")
parser.add_argument("--channels", type=int, default=1, help="number of image channels")
parser.add_argument("--sample_interval", type=int, default=100, help="interval between saving generator outputs")
parser.add_argument("--n_residual_blocks", type=int, default=9, help="number of residual blocks in generator")
parser.add_argument("--lambda_cyc", type=float, default=10.0, help="cycle loss weight")
parser.add_argument("--lambda_id", type=float, default=5.0, help="identity loss weight")

parser.add_argument('--dataroot', type=str, default='')
parser.add_argument('--model_dir', type=str, default='')
parser.add_argument('--version', type=str, default='')
parser.add_argument("--step2save", type=int, default=20)
parser.add_argument('--resume', action="store_true")
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
    
# Losses
criterion_GAN = torch.nn.MSELoss().to(DEVICE)
criterion_cycle = torch.nn.L1Loss().to(DEVICE)
criterion_identity = torch.nn.L1Loss().to(DEVICE)

input_shape = (args.channels, args.img_height, args.img_width)

# Initialize generator and discriminator
G_AB = GeneratorResNet(input_shape, args.n_residual_blocks).to(DEVICE)
G_BA = GeneratorResNet(input_shape, args.n_residual_blocks).to(DEVICE)
D_A = Discriminator(input_shape).to(DEVICE)
D_B = Discriminator(input_shape).to(DEVICE)

if args.resume is True:
    # Load pretrained models
    G_AB.load_state_dict(torch.load("saved_models/%s/G_AB_%d.pth" % (args.dataset_name, args.epoch)))
    G_BA.load_state_dict(torch.load("saved_models/%s/G_BA_%d.pth" % (args.dataset_name, args.epoch)))
    D_A.load_state_dict(torch.load("saved_models/%s/D_A_%d.pth" % (args.dataset_name, args.epoch)))
    D_B.load_state_dict(torch.load("saved_models/%s/D_B_%d.pth" % (args.dataset_name, args.epoch)))
else:
    # Initialize weights
    G_AB.apply(weights_init_normal)
    G_BA.apply(weights_init_normal)
    D_A.apply(weights_init_normal)
    D_B.apply(weights_init_normal)



# Optimizers
optimizer_G = torch.optim.Adam(
    itertools.chain(G_AB.parameters(), G_BA.parameters()), lr=args.lr, betas=(args.b1, args.b2)
)
optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.lr, betas=(args.b1, args.b2))
optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.lr, betas=(args.b1, args.b2))

# Learning rate update schedulers
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(
    optimizer_G, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_A, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(
    optimizer_D_B, lr_lambda=LambdaLR(args.n_epochs, args.epoch, args.decay_epoch).step
)

Tensor = torch.cuda.FloatTensor

# Buffers of previously generated samples
fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

train_set = HDF5Data(args.dataroot)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.n_cpu, batch_size=args.batch_size, shuffle=True)

# ----------
#  Training
# ----------
prev_time = time.time()
for epoch in tqdm(range(args.epoch, args.n_epochs)):
    for i, batch in enumerate(training_data_loader):

        # Set model input
        real_A = Variable(batch["A"].type(Tensor)).to(DEVICE)
        real_B = Variable(batch["B"].type(Tensor)).to(DEVICE)

        # Adversarial ground truths
        valid = Variable(Tensor(np.ones((real_A.size(0), *D_A.output_shape))), requires_grad=False).to(DEVICE)
        fake = Variable(Tensor(np.zeros((real_A.size(0), *D_A.output_shape))), requires_grad=False).to(DEVICE)

        # ------------------
        #  Train Generators
        # ------------------

        G_AB.train()
        G_BA.train()

        optimizer_G.zero_grad()

        # Identity loss
        loss_id_A = criterion_identity(G_BA(real_A), real_A)
        loss_id_B = criterion_identity(G_AB(real_B), real_B)

        loss_identity = (loss_id_A + loss_id_B) / 2

        # GAN loss
        fake_B = G_AB(real_A)
        loss_GAN_AB = criterion_GAN(D_B(fake_B), valid)
        fake_A = G_BA(real_B)
        loss_GAN_BA = criterion_GAN(D_A(fake_A), valid)

        loss_GAN = (loss_GAN_AB + loss_GAN_BA) / 2

        # Cycle loss
        recov_A = G_BA(fake_B)
        loss_cycle_A = criterion_cycle(recov_A, real_A)
        recov_B = G_AB(fake_A)
        loss_cycle_B = criterion_cycle(recov_B, real_B)

        loss_cycle = (loss_cycle_A + loss_cycle_B) / 2

        # Total loss
        loss_G = loss_GAN + args.lambda_cyc * loss_cycle + args.lambda_id * loss_identity

        loss_G.backward()
        optimizer_G.step()

        # -----------------------
        #  Train Discriminator A
        # -----------------------

        optimizer_D_A.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_A(real_A), valid)
        # Fake loss (on batch of previously generated samples)
        fake_A_ = fake_A_buffer.push_and_pop(fake_A)
        loss_fake = criterion_GAN(D_A(fake_A_.detach()), fake)
        # Total loss
        loss_D_A = (loss_real + loss_fake) / 2

        loss_D_A.backward()
        optimizer_D_A.step()

        # -----------------------
        #  Train Discriminator B
        # -----------------------

        optimizer_D_B.zero_grad()

        # Real loss
        loss_real = criterion_GAN(D_B(real_B), valid)
        # Fake loss (on batch of previously generated samples)
        fake_B_ = fake_B_buffer.push_and_pop(fake_B)
        loss_fake = criterion_GAN(D_B(fake_B_.detach()), fake)
        # Total loss
        loss_D_B = (loss_real + loss_fake) / 2

        loss_D_B.backward()
        optimizer_D_B.step()

        loss_D = (loss_D_A + loss_D_B) / 2

        wandb.log({'loss_GAN_AB': loss_GAN_AB.item(), 'loss_GAN_BA': loss_GAN_BA.item(), 'loss_D_A':loss_D_A.item(), 'loss_D_B': loss_D_B.item()})

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "_real_A.jpg", real_A[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "_real_B.jpg", real_B[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "_fake_A.jpg", fake_A[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "_fake_B.jpg", fake_B[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0:
        torch.save(G_AB.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_generator.pth'))
