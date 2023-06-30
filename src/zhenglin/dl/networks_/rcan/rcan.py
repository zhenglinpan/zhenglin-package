import argparse, os, sys
import torch
from torch import nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
from torch.utils.data.dataloader import DataLoader
from tqdm import tqdm
from network import RCAN

import wandb
from tqdm import tqdm

from matplotlib import pyplot as plt

DEVICE = 1
cudnn.benchmark = True
wandb.init(project="")

parser = argparse.ArgumentParser()
parser.add_argument('--scale', type=int, default=2)
parser.add_argument('--num_features', type=int, default=64)
parser.add_argument('--num_rg', type=int, default=10)
parser.add_argument('--num_rcab', type=int, default=20)
parser.add_argument('--reduction', type=int, default=16)
parser.add_argument('--patch_size', type=int, default=48)
parser.add_argument('--batch_size', type=int, default=1)
parser.add_argument('--num_epochs', type=int, default=500)
parser.add_argument('--lr', type=float, default=1e-5)
parser.add_argument('--threads', type=int, default=8)
parser.add_argument('--seed', type=int, default=1145141919)
parser.add_argument('--use_fast_loader', action='store_true')

parser.add_argument('--dataroot', type=str, default='')
parser.add_argument('--model_dir', type=str, default='')
parser.add_argument('--version', type=str, default='')
parser.add_argument("--step2save", type=int, default=20)
parser.add_argument('--resume', action="store_true")
args = parser.parse_args()

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

save_info = ""
with open(model_save_path + '/info.txt', 'w') as f:
    f.write(save_info)
    f.write(str(args))

torch.manual_seed(args.seed)

model = RCAN(scale=args.scale, 
             num_features=args.num_features, 
             num_rg=args.num_rg, 
             num_rcab=args.num_rcab, 
             reduction=args.reduction).to(DEVICE)
criterion = nn.L1Loss()
optimizer = optim.Adam(model.parameters(), lr=args.lr)

Tensor = torch.cuda.FloatTensor

train_set = HDF5Data(args.dataroot)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

for epoch in range(args.num_epochs + 1):
    with tqdm(total=(len(train_set) - len(train_set) % args.batch_size)) as _tqdm:
        _tqdm.set_description('epoch: {}/{}'.format(epoch + 1, args.num_epochs))
        for data in training_data_loader:
            inputs, labels = data['A'], data['B']
            inputs = torch.autograd.Variable(inputs.type(Tensor)).to(DEVICE)
            labels = torch.autograd.Variable(labels.type(Tensor)).to(DEVICE)
            preds = model(inputs)
            loss = criterion(preds, labels)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            _tqdm.update(len(inputs))

    wandb.log({'loss': loss.item()})
    
    if epoch%(20) == 0:
        plt.imsave("./imgs/" + str(epoch) + "fake.jpg", preds[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "real.jpg", inputs[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch) + '_generator.pth'))
