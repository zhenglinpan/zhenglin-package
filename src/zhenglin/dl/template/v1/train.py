import os, sys
import argparse
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms

from PIL import Image

from network import Generator
from dataset import MyDataset
from utils import weights_init_normal, LinearLambdaLR

import wandb
# wandb.init("")

parser = argparse.ArgumentParser()
### dataset args
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--patch_size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
### training args
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--resume', action="store_true", help='continue training from a checkpoint')
### advanced args
parser.add_argument('--precision', type=str, default='fp32', choices=['fp32', 'fp16'], help='precision of model')
args = parser.parse_args()

### set gpu device
DEVICE = 0

### Networks
model = Generator().to(DEVICE)
if args.precision == 'fp16':
    model = Generator().half().to(DEVICE)
model.apply(weights_init_normal)

if args.resume:
    model.load_state_dict(torch.load('./models/model_20.pth', map_location=torch.device(DEVICE)))

### if rich
# model = nn.DataParallel(model, device_ids=[0, 1])

### Lossess
criterion_GAN = torch.nn.MSELoss().to(DEVICE)

### argsimizers & LR schedulers
optimizer_G = torch.optim.Adam(model.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=LinearLambdaLR(args.start_epoch, args.end_epoch, args.decay_epoch).step)

### Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if args.precision=='fp32' else torch.cuda.HalfTensor
input_A = Tensor(args.batch_size, args.input_nc, args.size, args.size)
input_B = Tensor(args.batch_size, args.output_nc, args.size, args.size)
target_real = Variable(Tensor(args.batch_size).fill_(1.0), requires_grad=False)
target_fake = Variable(Tensor(args.batch_size).fill_(0.0), requires_grad=False)

### Dataset loader
transforms_ = [ transforms.Resize(int(args.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(args.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataset = MyDataset(args.dataroot, transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

###### Training ######
for epoch in range(args.start_epoch, args.end_epoch + 1):
    for i, batch in enumerate(dataloader):

        # Input = Variable(input_A.copy_(batch))    ### style 1
        Input = Variable(batch.type(Tensor)).to(DEVICE)   ### style 2

        ###### Generators A2B and B2A ######
        optimizer_G.zero_grad()
        
        Pred = model(Input)
        
        loss_G = criterion_GAN(Input, Pred)

        loss_G.backward()
        optimizer_G.step()
        
        wandb.log({"loss_G": loss_G.item()})
        
    # Update learning rates
    lr_scheduler_G.step()

    if epoch % 10 == 0:
        save_image(Input, f'./imgs/{epoch}_real.jpg')
        save_image(Pred, f'./imgs/{epoch}_fake.jpg')
    
    if epoch % 20 == 0:
        torch.save(model.state_dict(), f'output/model_{epoch}.pth')
###################################