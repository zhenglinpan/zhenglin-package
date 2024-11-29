import argparse, os, sys
import torch
import math, random
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from network import EDSR

from matplotlib import pyplot as plt

import wandb
from tqdm import tqdm

wandb.init(project="")

# Training settings
parser = argparse.ArgumentParser(description="PyTorch EDSR")
parser.add_argument("--batch_size", type=int, default=1, help="training batch size")
parser.add_argument("--nEpochs", type=int, default=500, help="number of epochs to train for")
parser.add_argument("--lr", type=float, default=1e-4, help="Learning Rate. Default=1e-4")
parser.add_argument("--lr_decay_step", type=int, default=200, help="Sets the learning rate to the initial LR decayed by momentum every n epochs, Default: n=10")
parser.add_argument("--start-epoch", default=1, type=int, help="manual epoch number (useful on restarts)")
parser.add_argument("--threads", type=int, default=8, help="number of threads for data loader to use")
parser.add_argument("--momentum", default=0.9, type=float, help="momentum")
parser.add_argument("--weight-decay", "--wd", default=1e-4, type=float, help="weight decay, Default: 0")

parser.add_argument('--dataroot', type=str, default='', help='root directory of the dataset')
parser.add_argument('--model_dir', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument('--version', type=str, default='', help='number of cpu threads to use during batch generation')
parser.add_argument("--step2save", type=int, default=50, help="Number of epoches to save a model")
parser.add_argument('--resume', action="store_true", help="train from latest checkpoints")
args = parser.parse_args()
print(args)

DEVICE = 0

model_save_path = os.path.join(args.model_dir, args.version)
if not os.path.exists(model_save_path):
    os.mkdir(model_save_path)

save_info = ""
with open(model_save_path + '/info.txt', 'w') as f:
    f.write(save_info)
    f.write(str(args))

# if args.resume:
#     if os.path.isfile(args.resume):
#         print("=> loading checkpoint '{}'".format(args.resume))
#         checkpoint = torch.load(args.resume)
#         args.start_epoch = checkpoint["epoch"] + 1
#         model.load_state_dict(checkpoint["model"].state_dict())
#     else:
#         print("=> no checkpoint found at '{}'".format(args.resume))

args.seed = random.randint(1, 10000)
torch.manual_seed(args.seed)
torch.cuda.manual_seed(args.seed)
cudnn.benchmark = True

train_set = HDF5Data(args.dataroot)
training_data_loader = DataLoader(dataset=train_set, num_workers=args.threads, batch_size=args.batch_size, shuffle=True)

model = EDSR().to(DEVICE)
criterion = nn.L1Loss(size_average=False).to(DEVICE)
optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=args.weight_decay, betas = (0.9, 0.999), eps=1e-08)

lr = args.lr
for epoch in tqdm(range(args.start_epoch, args.nEpochs + 1)): 
    if epoch % args.lr_decay_step == 0:
        lr *= 0.1
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr 
    print("Epoch={}, lr={}".format(epoch, optimizer.param_groups[0]["lr"]))
    
    model.train()

    for iteration, batch in enumerate(training_data_loader, 1):
        input, target = Variable(batch['A']), Variable(batch['B'], requires_grad=False)
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        generated = model(input)
        loss = criterion(generated, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    wandb.log({'loss': loss.item()})
    
    if epoch%(10) == 0:
        plt.imsave("./imgs/" + str(epoch) + "fake.jpg", generated[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
        plt.imsave("./imgs/" + str(epoch) + "real.jpg", input[0, 0, :, :].cpu().detach().numpy(), cmap=plt.cm.bone)
    
    if epoch%(args.step2save) == 0 and epoch != 0:
        torch.save(model.state_dict(), os.path.join(args.model_dir, args.version + '/' + str(epoch + args.start_epoch) + '_generator.pth'))
