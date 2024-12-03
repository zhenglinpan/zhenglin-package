import os
import sys
import logging
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader
from torch.cuda.amp import GradScaler, autocast
from torchvision.utils import save_image, make_grid

from dataset import MNIST
from networks.unet import UNet

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import LinearLambdaLR

import wandb
wandb.init(project="pytorch-template")

logging.basicConfig(level=logging.INFO)

def main():
    parser = argparse.ArgumentParser()
    ### dataset args
    parser.add_argument('--exp_root', type=str, default='expriments', help='root directory of the dataset')
    parser.add_argument('--exp_name', type=str, default='', help='root directory of the dataset')
    parser.add_argument('--patch_size', type=int, default=256, help='size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='number of channels of input data')
    parser.add_argument('--output_nc', type=int, default=1, help='number of channels of output data')
    parser.add_argument('--num_workers', type=int, default=8, help='number of cpu threads to use during batch generation')
    parser.add_argument('--save_interval_epoch', type=int, default=1, help='interval epoch to save the model')
    ### training args
    parser.add_argument('--start_epoch', type=int, default=0, help='starting epoch')
    parser.add_argument('--end_epoch', type=int, default=10, help='number of epochs of training')
    parser.add_argument('--decay_epoch', type=int, default=5, help='epoch to start linearly decaying the learning rate to 0')
    parser.add_argument('--batch_size', type=int, default=64, help='size of the batches')
    parser.add_argument('--lr', type=float, default=0.0002, help='initial learning rate')
    parser.add_argument('--resume', type=str, default='', help='continue training from a checkpoint')
    parser.add_argument('--fp16', action="store_true", help='mixed precision of model')
    parser.add_argument('--device', type=str, default='0', help='gpu index used')
    args = parser.parse_args()

    DEVICE = 0
    
    ### save checkpoints
    if args.exp_name:
        dir_save = os.path.join(args.exp_root, args.exp_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_save = os.path.join(args.exp_root, timestamp)

    ### Dataset & DataLoader
    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)

    ### Networks
    unet = UNet(chan_in=args.input_nc, chan_out=args.output_nc, long_skip=True).to(DEVICE)
    
    if args.resume:
        unet.load_state_dict(torch.load(args.resume, map_location=torch.device(DEVICE)))

    ### Lossess
    criterion = torch.nn.MSELoss().to(DEVICE)

    ### argsimizers & LR schedulers
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LinearLambdaLR(args.end_epoch * len(dataloader), 
                                            args.start_epoch * len(dataloader), 
                                            args.decay_epoch * len(dataloader)).step)
    grad_scaler = GradScaler()

    pbar = tqdm(total=len(dataloader) * (args.end_epoch - args.start_epoch))
    
    ###### Training ######
    for epoch in range(args.start_epoch, args.end_epoch):
        for i, batch in enumerate(dataloader):
            image = batch['image'].to(DEVICE)
            contour = batch['contour'].to(DEVICE)
            
            with autocast(enabled=args.fp16):
                pred = unet(image)
                
                loss = criterion(pred, contour)
                wandb.log({"train loss": loss.item(), "learning rate": float(lr_scheduler.get_lr()[-1])})

                grad_scaler.scale(loss).backward()
                grad_scaler.step(optimizer)
                grad_scaler.update()
                lr_scheduler.step()
                optimizer.zero_grad()
                pbar.update(1)

        if epoch % args.save_interval_epoch == 0 or epoch == args.end_epoch - 1:
            os.makedirs(os.path.join(dir_save, 'imgs'), exist_ok=True)
            save_image(make_grid(pred, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_pred.png'))
            save_image(make_grid(image, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_gt.png'))
            save_image(make_grid(contour, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_cond.png'))
        
            os.makedirs(os.path.join(dir_save, 'weights', f'epoch_{epoch}'), exist_ok=True)
            torch.save(unet.state_dict(), os.path.join(dir_save, 'weights', f'epoch_{epoch}', 'unet.pth'))
            logging.info(f"Epoch {epoch} is saved to {os.path.join(dir_save, 'weights', f'epoch_{epoch}', 'unet.pth')}")


if __name__=="__main__":
    main()