import os, sys, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim

import numpy as np
import matplotlib.pyplot as plt

import wandb
from tqdm import tqdm

wandb.init(project="")

from network import UNet

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_decay_step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 0")
parser.add_argument("--patch_size", default=64, type=int, help="patch size")

parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--model_dir', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument('--version', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument("--step2save", type=int, default=50, help="Number of epoches to save a model")
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

# model = UNet(1,1,depth=4,batch_norm=False,up_mode='upsample',padding=True).to(DEVICE)
# model = GeneratorRRDB(1, 64, 6, 2).to(DEVICE)
# model = RRDBNet(1, 1, 64, 1).to(DEVICE)
model = UNet_pby(1, 1, 0).to(DEVICE)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100], gamma=0.1)

def pixel_mse_loss(predictions, targets, pixel_pos):
    mask = torch.zeros(targets.shape).to(targets.device)
    mask[0,:,pixel_pos[0],pixel_pos[1]] = 1.
    return F.mse_loss(predictions*mask, targets*mask)*1000

train_set = HDF5Data(args.dataroot, patch_size=args.patch_size, use_blindspot_mask=True)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

Input = torch.cuda.FloatTensor(1, 1, args.patch_size, args.patch_size).to(DEVICE)

model.train()
for epoch in tqdm(range(args.nEpochs + 1)):
    for i, batch in enumerate(training_data_loader):
        shift = torch.autograd.Variable(Input.copy_(batch['A']))
        original = torch.autograd.Variable(Input.copy_(batch['B']))
        blind_loc = batch['loc']
        
        model.zero_grad()
        recover = model(shift)
        loss = pixel_mse_loss(recover, original, blind_loc)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        wandb.log({'loss': loss.item()})
    
    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "recover.jpg", recover[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "src.jpg", original[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_model.pth'))