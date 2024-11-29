import os, sys
import argparse
import itertools

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.utils import save_image
import torchvision.transforms as transforms
from torch2trt import TRTModule

from PIL import Image
import numpy as np

from network import Generator
from dataset import MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--end_epoch', type=int, default=200, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=256, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--resume', action="store_true", help='continue training from a checkpoint')
args = parser.parse_args()
print(args)

# set gpu device
DEVICE = 0

# Networks
model_trt = TRTModule()
model_trt.load_state_dict(torch.load('./models/model_20.trt', map_location=torch.device(DEVICE)))

model_trt.eval()

# Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor if args.cuda else torch.Tensor
input_A = Tensor(args.batch_size, args.input_nc, args.size, args.size)

# Dataset loader
transforms_ = [ transforms.Resize(int(args.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(args.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataset = MyDataset(args.dataroot, transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.n_cpu)

###### infer ######
for epoch in range(args.start_epoch, args.end_epoch + 1):
    for i, batch in enumerate(dataloader):
        
        Input = Variable(input_A.copy_(batch))
        Pred = model_trt(Input)

        save_image(Input, f'./imgs/{np.random.random()}_real.jpg')
        save_image(Pred, f'./imgs/{np.random.random()}_fake.jpg')