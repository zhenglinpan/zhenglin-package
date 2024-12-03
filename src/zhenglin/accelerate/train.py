import os
import sys
import logging
import datetime
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader
from torchvision.utils import save_image, make_grid
from pytorch_lightning import seed_everything

from safetensors.torch import load_file
from accelerate import Accelerator
accelerator = Accelerator(log_with="wandb")
accelerator.init_trackers(project_name="accelerate-template")

### for multi-GPU training
# from accelerate import DistributedDataParallelKwargs
# ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=True)
# accelerator = Accelerator(kwargs_handlers=[ddp_kwargs], log_with="wandb")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dataset import MNIST
from networks.unet import UNet
from utils.utils import LinearLambdaLR

logging.basicConfig(level=logging.INFO)


def main():
    config = OmegaConf.load('config.yaml')
    
    seed_everything(config['seed'])
    
    ### save checkpoints
    if config.exp_name:
        dir_save = os.path.join(config.exp_root, config.exp_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_save = os.path.join(config.exp_root, timestamp)
    
    ### Networks
    unet = UNet(chan_in=config.unet_input_nc, chan_out=config.unet_output_nc, long_skip=config.unet_long_skip)

    ### resume from a checkpoint
    if config.resume:
        if config.resume.endswith('.safetensors'):  # in case load the weights directly
            unet.load_state_dict(load_file(config.resume))
        else:
            accelerator.load_state(config.resume)
        logging.info(f"Resume from {config.resume}")
    
    ### dataset & dataloader
    dataset = MNIST()
    dataloader = DataLoader(dataset, config.batch_size, config.shuffle, num_workers=config.num_workers)

    ### Lossess
    criterion_GAN = torch.nn.MSELoss()

    ### optimizers & LR schedulers
    optimizer = torch.optim.Adam(unet.parameters(), lr=config.lr, betas=(0.5, 0.999))
    lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=LinearLambdaLR(config.end_epoch * len(dataloader), 
                                            config.start_epoch * len(dataloader), 
                                            config.decay_epoch * len(dataloader)).step)

    ### accelerate prepare
    unet, optimizer, lr_scheduler, dataloader = accelerator.prepare(unet, optimizer, lr_scheduler, dataloader)

    pbar = tqdm(total=len(dataloader) * (config.end_epoch - config.start_epoch), disable=(not accelerator.is_local_main_process))
    
    ###### Training ######
    for epoch in range(config.start_epoch, config.end_epoch):
        for it, batch in enumerate(dataloader):
            image = batch['image']
            contour = batch['contour']
            
            pred = unet(image)
            loss = criterion_GAN(pred, contour)

            accelerator.backward(loss)
            accelerator.log(values={"train loss": loss, "learning rate": float(lr_scheduler.get_lr()[-1])}, 
                            step=(epoch * len(dataloader) + it))
            
            optimizer.step()
            lr_scheduler.step()
            optimizer.zero_grad()
            pbar.update(1)
            
        if epoch % config.save_interval_epoch == 0 or epoch == config.end_epoch - 1:
            if accelerator.is_local_main_process:
                os.makedirs(os.path.join(dir_save, 'imgs'), exist_ok=True)
                save_image(make_grid(pred, nrow=int(config.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_pred.png'))
                save_image(make_grid(image, nrow=int(config.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_gt.png'))
                save_image(make_grid(contour, nrow=int(config.batch_size ** 0.5)), os.path.join(dir_save, 'imgs', f'{epoch}_cond.png'))
            
                os.makedirs(os.path.join(dir_save, 'weights', f'epoch_{epoch}'), exist_ok=True)
                accelerator.save_state(os.path.join(dir_save, 'weights', f'epoch_{epoch}'))
                logging.info(f"Epoch {epoch} is saved to {os.path.join(dir_save, 'weights', f'epoch_{epoch}')}")
                
        
    
if __name__ == '__main__':
    '''
    Run the code with:
        accelerate launch --config_file config_accelerate.yaml train.py
    '''
    main()