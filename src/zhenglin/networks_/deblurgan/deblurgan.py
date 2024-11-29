import argparse, os, sys
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from torch.autograd import Variable

from model import ResNetGenerator, NLayerDiscriminator
from model import Generator, Discriminator
from utils import PerceptualLoss_l1

import wandb
import numpy as np
from tqdm import tqdm
from matplotlib import pyplot as plt
from datasets import HDF5Data

# wandb.init(project="")

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--num_rg', type=int, default=10)
parser.add_argument('--num_rcab', type=int, default=20)
parser.add_argument('--reduction', type=int, default=16)
parser.add_argument('--patch_size', type=int, default=128)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=0.0003)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--in_channel', type=int, default=1)
parser.add_argument('--use_fast_loader', action='store_true')

parser.add_argument('--dataroot', type=str, default='')
parser.add_argument('--model_dir', type=str, default='')
parser.add_argument('--version', type=str, default='v1_0_0_1')
parser.add_argument("--step2save", type=int, default=20)
parser.add_argument('--resume', action="store_true")

parser.add_argument('--upscale', type=int, default=2)
parser.add_argument('--tile_overlap', type=int, default=32, help='Overlapping of different tiles')
args = parser.parse_args()

DEVICE = 0

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

save_info = ""
with open(model_save_path + '/info.txt', 'w') as f:
    f.write(save_info)
    f.write(str(args))

generator = ResNetGenerator(1, 1).to(DEVICE)
discriminator = NLayerDiscriminator(1).to(DEVICE)

generator = Generator().to(DEVICE)
discriminator = Discriminator().to(DEVICE)

criterion_content = nn.L1Loss()
criterion_GAN = nn.BCEWithLogitsLoss()
criterion_perceptual = PerceptualLoss_l1(device=DEVICE)

optimizer_G = optim.AdamW(generator.parameters(), lr=args.lr, weight_decay=1e-4)
optimizer_D = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.9, 0.999))

Tensor = torch.cuda.FloatTensor
target_real = Variable(Tensor(np.ones((args.batch_size, args.in_channel, args.patch_size, args.patch_size))), requires_grad=False).to(DEVICE)
target_fake = Variable(Tensor(np.ones((args.batch_size, args.in_channel, args.patch_size, args.patch_size))), requires_grad=False).to(DEVICE)

train_set = HDF5Data(args.dataroot, patch_size=args.patch_size)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

pre_img_lr = None
pre_img_hr = None
for epoch in tqdm(range(args.num_epochs + 1)):
    for data in training_data_loader:
        img_lr, img_hr, attack = data['A'], data['B'], data['attack']
        if not attack:
            pre_img_lr = img_lr
            pre_img_hr = img_hr
            img_lr = Variable(img_lr.type(Tensor)).to(DEVICE)
            img_hr = Variable(img_hr.type(Tensor)).to(DEVICE)
        else:
            print("circumventing attacks...")
            img_lr = Variable(pre_img_lr.type(Tensor)).to(DEVICE)
            img_hr = Variable(pre_img_hr.type(Tensor)).to(DEVICE)
        
        optimizer_G.zero_grad()

        gen_hr = generator(img_lr)
        pred_real = discriminator(img_hr)
        pred_fake = discriminator(gen_hr.detach())
        
        loss_content = criterion_content(gen_hr, img_hr)
        loss_GAN = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), target_real)
        loss_perceptual = criterion_perceptual(gen_hr, img_hr)
        
        loss_G = loss_content + loss_GAN*5e-3 + loss_perceptual*1e-2
        
        loss_G.backward()
        optimizer_G.step()

        optimizer_D.zero_grad()
        
        pred_fake = discriminator(gen_hr.detach())
        pred_real = discriminator(img_hr)
        
        loss_real = criterion_GAN(pred_real - pred_fake.mean(0, keepdim=True), target_real)
        loss_fake = criterion_GAN(pred_fake - pred_real.mean(0, keepdim=True), target_fake)
        loss_D = loss_real + loss_fake
        
        loss_D.backward()
        optimizer_D.step()
        
        # wandb.log({'loss D': loss_D.item(), 'loss_G': loss_G.item()})
    
    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "fake.jpg", gen_hr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "real.jpg", img_lr[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(generator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_generator.pth'))
        torch.save(discriminator.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_discriminator.pth'))