import os
import sys
from math import sqrt

import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchvision.utils import save_image, make_grid

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from utils.utils import LinearLambdaLR


class UNet(nn.Module):
    """
    A tity version of UNet. See unet_vanilla for original one.
    Original UNet: depth-5, starting filtersize-64, parameters-31M
    This UNet: depth-3, starting filtersize-32, parameters-0.5M
    https://aidenpan.notion.site/UNet-Structure-1fed5bd8a40b484093c0da6838cc4f96?pvs=4
    """
    def __init__(self,chan_in, chan_out, long_skip,nf=32):
        super(UNet, self).__init__()
        self.long_skip = long_skip
        self.chan_in = chan_in
        self.chan_out = chan_out
        self.relu = nn.ReLU()
        self.with_bn = True
        self.conv1_1 = nn.Conv2d(self.chan_in, nf, (3,3),(1,1),(1,1))
        self.bn1_1   = nn.BatchNorm2d(nf)#,affine=False)# input of (n,n,1), output of (n-2,n-2,64)
        self.conv1_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn1_2   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv2_1 = nn.Conv2d(nf, nf*2, 3,1,1)
        self.bn2_1   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv2_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn2_2   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv3_1 = nn.Conv2d(nf*2, nf*4, 3,1,1)
        self.bn3_1   = nn.BatchNorm2d(nf*4)#,affine=False)
        self.conv3_2 = nn.Conv2d(nf*4, nf*4, 3,1,1)
        self.bn3_2   = nn.BatchNorm2d(nf*4)#,affine=False)
        
        self.dc2     =nn.ConvTranspose2d(nf*4, nf*2, 4, stride=2, padding=1,bias=False)

        self.conv4_1 = nn.Conv2d(nf*4, nf*2, 3,1,1)
        self.bn4_1   = nn.BatchNorm2d(nf*2)#,affine=False)
        self.conv4_2 = nn.Conv2d(nf*2, nf*2, 3,1,1)
        self.bn4_2   = nn.BatchNorm2d(nf*2)#,affine=False)
        
        self.dc1     =nn.ConvTranspose2d(nf*2, nf, 4, stride=2, padding=1,bias=False)
        
        self.conv5_1 = nn.Conv2d(nf*2, nf, 3,1,1)
        self.bn5_1   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv5_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn5_2   = nn.BatchNorm2d(nf)#,affine=False)
        self.conv5_3 = nn.Conv2d(nf, self.chan_out, 3,1,1)


        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()
        print('initialization weights is done')

    def forward(self, x1):
        if self.with_bn:
            x1_ = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x1))))))
            x2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(self.maxpool(x1_)))))))
            x3 = self.relu(self.bn3_2(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(self.maxpool(x2)))))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.bn4_2(self.conv4_2(self.relu(self.bn4_1(self.conv4_1(x4_2))))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.bn5_2(self.conv5_2(self.relu(self.bn5_1(self.conv5_1(x6_1))))))
        else:
            x1_ = self.relu(self.conv1_2(self.relu(self.conv1_1(x1))))
            x2 = self.relu(self.conv2_2(self.relu(self.conv2_1(self.maxpool(x1_)))))
            x3 = self.relu(self.conv3_2(self.relu(self.conv3_1(self.maxpool(x2)))))
            x4 = self.relu(self.dc2(x3))  
            x4_2 = torch.cat((x4, x2), 1)
            x5 = self.relu(self.conv4_2(self.relu(self.conv4_1(x4_2))))
            x6 = self.relu(self.dc1(x5))  
            x6_1 = torch.cat((x6, x1_), 1)
            x7 = self.relu(self.conv5_2(self.relu(self.conv5_1(x6_1))))
        x8 = self.conv5_3(x7)
        if self.long_skip == True:        
            return x8 + x1[:,0:self.chan_out,:,:]
        else:
            return x8

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()

            elif isinstance(m,nn.ConvTranspose2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, sqrt(2. / n))
                # m.weight.data.normal_(0.0, 0.02)
                if m.bias is not None:
                    m.bias.data.zero_()


class UNetLightning(pl.LightningModule):
    def __init__(self, input_nc, output_nc, long_skip, 
                 len_dataloader=None, dir_save=None, **kwargs):
        super().__init__()
        self.model = UNet(chan_in=input_nc, chan_out=output_nc, long_skip=long_skip)
        self.len_dataloader = len_dataloader
        self.dir_save = dir_save
        self.lr = kwargs['lr']
        self.batch_size = kwargs['batch_size']
        self.start_epoch = kwargs['start_epoch']
        self.end_epoch = kwargs['end_epoch']
        self.decay_epoch = kwargs['decay_epoch']
        self.save_interval_epoch = kwargs['save_interval_epoch']
        self.criterion = torch.nn.MSELoss()
        
        self.pred = None
        self.image = None
        self.contour = None
    
    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        self.image = batch['image'].to(self.device)
        self.contour = batch['contour'].to(self.device)

        self.pred = self(self.image)
        loss = self.criterion(self.pred, self.contour)
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.lr, betas=(0.9, 0.999))
        lr_scheduler = {
            'scheduler': torch.optim.lr_scheduler.LambdaLR(
                optimizer, lr_lambda=LinearLambdaLR(
                    self.end_epoch * self.len_dataloader, 
                    self.start_epoch * self.len_dataloader, 
                    self.decay_epoch * self.len_dataloader
                ).step
            ),
            'interval': 'step',
        }
        return [optimizer], [lr_scheduler]
    
    def on_epoch_end(self):
        dir_save = os.path.join(self.dir_save, 'imgs')
        os.makedirs(dir_save, exist_ok=True)
        
        epoch = self.current_epoch
        if epoch % self.save_interval_epoch == 0 or epoch == self.trainer.max_epochs - 1:
            save_image(make_grid(self.pred, nrow=int(self.batch_size ** 0.5)), os.path.join(dir_save, f'epoch_{epoch}_pred.png'))
            save_image(make_grid(self.image, nrow=int(self.batch_size ** 0.5)), os.path.join(dir_save, f'epoch_{epoch}_gt.png'))
            save_image(make_grid(self.contour, nrow=int(self.batch_size ** 0.5)), os.path.join(dir_save, f'epoch_{epoch}_cond.png'))
    
    ### in case iter-based    
    # def on_train_batch_end(self):
    #     if self.global_step % self.save_interval_step == 0:
    #         pass
        