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
import numpy as np

from network import Generator
from dataset import MyDataset

parser = argparse.ArgumentParser()
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--model_path', type=str, default='./data/test', help='path of trained model')
parser.add_argument('--infer_epoch', type=int, default=0, help='epoch num of infer')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--patch_size', type=int, default=512, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=3, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=3, help='number of channels of output data')
parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
args = parser.parse_args()
print(args)

### set gpu device
DEVICE = 0

### Networks
model = Generator().to(DEVICE)
model.load_state_dict(torch.load(args.model_path, map_location=torch.device(DEVICE)))

### if rich
# model = nn.DataParallel(model, device_ids=[0, 1])

model.eval()

### Inputs & targets memory allocation
Tensor = torch.cuda.FloatTensor
input_A = Tensor(args.batch_size, args.input_nc, args.patch_size, args.patch_size)

### Dataset loader
transforms_ = [ transforms.Resize(int(args.size*1.12), Image.BICUBIC), 
                transforms.RandomCrop(args.size), 
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.5,0.5,0.5), (0.5,0.5,0.5)) ]
dataset = MyDataset(args.dataroot, transforms_=transforms_, unaligned=True)
dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

######## infer ######
for epoch in range(args.infer_epoch):
    for i, batch in enumerate(dataloader):
        
        Input = Variable(input_A.copy_(batch))
        Pred = model(Input)

        save_image(Input, f'./imgs/{np.random.random()}_real.jpg')
        save_image(Pred, f'./imgs/{np.random.random()}_fake.jpg')