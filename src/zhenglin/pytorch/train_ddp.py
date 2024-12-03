import os
import sys
import logging
import datetime
import argparse
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader, DistributedSampler
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
    ### Dataset arguments
    parser.add_argument('--exp_root', type=str, default='experiments', help='Root directory for saving results')
    parser.add_argument('--exp_name', type=str, default='', help='Experiment name')
    parser.add_argument('--patch_size', type=int, default=256, help='Size of the data crop (squared assumed)')
    parser.add_argument('--input_nc', type=int, default=1, help='Number of input channels')
    parser.add_argument('--output_nc', type=int, default=1, help='Number of output channels')
    parser.add_argument('--num_workers', type=int, default=8, help='Number of CPU threads for data loading')
    parser.add_argument('--save_interval_epoch', type=int, default=1, help='Interval to save the model')
    ### Training arguments
    parser.add_argument('--start_epoch', type=int, default=0, help='Starting epoch')
    parser.add_argument('--end_epoch', type=int, default=10, help='Number of epochs')
    parser.add_argument('--decay_epoch', type=int, default=5, help='Epoch to start decaying learning rate')
    parser.add_argument('--batch_size', type=int, default=64, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0002, help='Initial learning rate')
    parser.add_argument('--resume', type=str, default='/root/zhenglin/zhenglin-package/src/zhenglin/pytorch/experiments/2024-11-29_18-31-16/weights/epoch_0/unet.pth', help='Checkpoint path to resume training')
    parser.add_argument('--fp16', action="store_true", help='Use mixed precision')
    parser.add_argument('--device', type=str, default='0', help='Comma-separated GPU indices to use')
    parser.add_argument('--backend', type=str, default='nccl', help='DDP backend (default: nccl)')
    args = parser.parse_args()

    ### Initialize DDP
    torch.distributed.init_process_group(backend=args.backend)
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    DEVICE = torch.device(f'cuda:{local_rank}')

    ### Save directory
    if args.exp_name:
        dir_save = os.path.join(args.exp_root, args.exp_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_save = os.path.join(args.exp_root, timestamp)

    ### Dataset & DataLoader
    dataset = MNIST()
    sampler = DistributedSampler(dataset)
    dataloader = DataLoader(dataset, batch_size=args.batch_size, sampler=sampler, num_workers=args.num_workers)

    ### Network
    unet = UNet(chan_in=args.input_nc, chan_out=args.output_nc, long_skip=True).to(DEVICE)
    unet = torch.nn.parallel.DistributedDataParallel(unet, device_ids=[local_rank])

    if args.resume:
        unet.module.load_state_dict(torch.load(args.resume, map_location=DEVICE))

    ### Losses
    criterion = torch.nn.MSELoss().to(DEVICE)

    ### Optimizers & LR schedulers
    optimizer = torch.optim.Adam(unet.parameters(), lr=args.lr, betas=(0.9, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LinearLambdaLR(args.end_epoch * len(dataloader), 
                                            args.start_epoch * len(dataloader), 
                                            args.decay_epoch * len(dataloader)).step)
    grad_scaler = GradScaler()

    pbar = tqdm(total=len(dataloader) * (args.end_epoch - args.start_epoch), disable=(local_rank != 0))

    for epoch in range(args.start_epoch, args.end_epoch):
        sampler.set_epoch(epoch)  # Shuffle data per epoch
        for i, batch in enumerate(dataloader):
            image = batch['image'].to(DEVICE)
            contour = batch['contour'].to(DEVICE)

            with autocast(enabled=args.fp16):
                pred = unet(image)
                loss = criterion(pred, contour)

            optimizer.zero_grad()
            grad_scaler.scale(loss).backward()
            grad_scaler.step(optimizer)
            grad_scaler.update()
            lr_scheduler.step()
            
            if local_rank == 0:
                wandb.log({"train loss": loss.item(), "learning rate": float(lr_scheduler.get_last_lr()[-1])})
                pbar.update(1)

        if local_rank == 0 and (epoch % args.save_interval_epoch == 0 or epoch == args.end_epoch - 1):
            os.makedirs(os.path.join(dir_save, 'imgs'), exist_ok=True)
            save_image(make_grid(pred, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_pred.png'))
            save_image(make_grid(image, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_gt.png'))
            save_image(make_grid(contour, nrow=int(args.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_cond.png'))
        
            os.makedirs(os.path.join(dir_save, 'weights', f'epoch_{epoch}'), exist_ok=True)
            torch.save(unet.module.state_dict(), os.path.join(dir_save, 'weights', f'epoch_{epoch}', 'unet.pth'))
            logging.info(f"Epoch {epoch} saved to {os.path.join(dir_save, 'weights', f'epoch_{epoch}', 'unet.pth')}")

if __name__ == "__main__":
    '''
        Run with the code:
            torchrun --nproc_per_node=1 train_ddp.py --device 0
    '''
    main()
