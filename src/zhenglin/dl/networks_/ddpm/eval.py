import os
import torch
from matplotlib import pyplot as plt
from network import UNet, CTUNet
from .diffusion import CTDiffusion
from torchvision.transforms import transforms
import argparse
from torch.utils.data import DataLoader
from utils import dissect, stitch
import time

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='starting epoch')
parser.add_argument('--n_epochs', type=int, default=500, help='number of epochs of training')
parser.add_argument('--batch_size', type=int, default=1, help='size of the batches')
parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
parser.add_argument('--decay_epoch', type=int, default=200, help='epoch to start linearly decaying the learning rate to 0')
parser.add_argument('--size', type=int, default=64, help='size of the data crop (squared assumed)')
parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
parser.add_argument('--cuda', type=bool, default=True, help='use GPU computation')
parser.add_argument('--n_cpu', type=int, default=8, help='number of cpu threads to use during batch generation')
parser.add_argument('--model_dir', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument('--version', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument("--step2save", type=int, default=10, help="Number of epoches to save a model")
parser.add_argument('--resume', action="store_true", help="train from latest checkpoints")

parser.add_argument('--aclr_factor', type=float, default=1, help='input image reshape size')
parser.add_argument('--infer_shape', type=int, default=64, help='input image reshape size')
parser.add_argument("--best_size",default=True,action='store_true', help="resize image to the the most suitable size ") 

args = parser.parse_args()
print(args)

device = 1
torch.cuda.device(device)

model = CTUNet().to(device)
model.load_state_dict(torch.load(""))
diffusion = CTDiffusion(img_size=64, device=device)  # 64 fixed by UNet model size

def infer(args, data_in:torch.cuda.FloatTensor, batch_n, i):
    patches = dissect(data_in, int((data_in.shape[2]/args.size)**2))
    xs = list()
    for j, patch in enumerate(patches):
        print(f"patch processing============================>{j+1}/{int((data_in.shape[2]/args.size)**2)}, {i+1}th image")
        t = diffusion.sample_timesteps(patch.shape[0]).to(device)
        t = torch.Tensor([19]).long().to(device)
        if j == 0: diffusion.save_progressive_noised_image(patch, t)
        x_t, e = diffusion.noise_images(patch, t)
        x = diffusion.sample(model, batch_n, x_t, t)
        # plt.imsave(f'./results/generated/{j}_patch_'+args.version+'.jpg', x[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        xs.append(x)
    X = stitch(xs)
    
    return X