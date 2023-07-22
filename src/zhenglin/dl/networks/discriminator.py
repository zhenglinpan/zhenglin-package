import numpy as np
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels, patch_scale=1):
        super(Discriminator, self).__init__()
        if patch_scale & (patch_scale - 1) != 0:
            raise ValueError("Patch scale should be a power of 2")

        def discriminator_block(in_filters, out_filters, first_block=False, downsample=False):
            layers = []
            layers.append(nn.Conv2d(in_filters, out_filters, kernel_size=3, stride=1, padding=1))
            if not first_block:
                layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            if downsample:
                layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=2, padding=1))
            else:
                layers.append(nn.Conv2d(out_filters, out_filters, kernel_size=3, stride=1, padding=1))
            layers.append(nn.BatchNorm2d(out_filters))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
            return layers

        down = [0, 0, 0, 0] # at which layer do downsample
        for i in range(np.log2(patch_scale).astype(int)):
            down[i] = 1
            
        layers = []
        in_filters = channels
        for i, out_filters in enumerate([64, 128, 256, 512]):
            layers.extend(discriminator_block(in_filters, out_filters, first_block=(i == 0), downsample=down[i]))
            in_filters = out_filters

        layers.append(nn.Conv2d(out_filters, 1, kernel_size=3, stride=1, padding=1))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)