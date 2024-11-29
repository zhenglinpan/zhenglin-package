import os
import sys
import logging
import datetime
import argparse
from tqdm import tqdm
from omegaconf import OmegaConf

import torch
from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning import seed_everything
from pytorch_lightning import loggers as pl_loggers
from pytorch_lightning.callbacks import ModelCheckpoint

from dataset import MNIST
from networks.unet import UNetLightning

logging.basicConfig(level=logging.INFO)


def main():
    config = OmegaConf.load('config.yaml')
    seed_everything(config['seed'])

    ### project directory
    if config.exp_name:
        dir_save = os.path.join(config.exp_root, config.exp_name)
    else:
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        dir_save = os.path.join(config.exp_root, timestamp)

    ### callback funtions
    checkpoint_callback = ModelCheckpoint(
        dirpath=os.path.join(dir_save, 'weights'),
        monitor='train_loss',
        save_top_k=-1,
        mode='min',
    )

    ### Trainer
    trainer = pl.Trainer(
        max_steps=-1,
        max_epochs=config.end_epoch,
        precision=16 if config.fp16 else 32,
        callbacks=[checkpoint_callback],
        resume_from_checkpoint=config.resume,
        logger=pl_loggers.WandbLogger(project='lightning-template'),
        gpus=config.gpus,
        fast_dev_run=config.fast_dev_run,
    )

    ### Dataset & DataLoader
    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)

    ### Model
    unet = UNetLightning(input_nc=config.unet_input_nc, 
                         output_nc=config.unet_output_nc, 
                         long_skip=config.unet_long_skip,
                         len_dataloader=len(dataloader), 
                         dir_save=dir_save,
                         **config)
    
    ### Train
    trainer.fit(unet, train_dataloaders=dataloader)

if __name__ == "__main__":
    main()