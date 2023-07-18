import torch
import torch.nn as nn
import torch.nn.functional as F

class SRDRM_gen(nn.Module):
    """ Proposed SR model using Residual Multipliers """
    def __init__(self):
        super(SRDRM_gen, self).__init__()
        self.n_residual_blocks = 8
        self.gf = 64

        self.residual_blocks = nn.ModuleList([self.make_residual_block(self.gf) for _ in range(self.n_residual_blocks)])

        self.conv1 = nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn = nn.BatchNorm2d(64, momentum=0.8)
        self.up = nn.Upsample(scale_factor=2, mode='nearest')
        self.conv3 = nn.Conv2d(64, 256, kernel_size=4, stride=2, padding=1)
        self.conv4 = nn.Conv2d(256, 1, kernel_size=5, stride=1, padding=2)
        
    def make_residual_block(self, channels):
        return nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels, momentum=0.5),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(channels, momentum=0.5)
        )

    def forward(self, x):
        l1 = self.conv1(x)
        l1 = F.relu(l1)
        # Propagate through residual blocks
        r = l1
        for residual_block in self.residual_blocks:
            r = residual_block(r)
        # Post-residual block
        l2 = self.conv2(r)
        l2 = self.bn(l2)
        l2 = torch.add(l2, l1)
        # Upsampling
        layer_2x = self.up(l2)
        layer_2x = self.conv3(layer_2x)
        layer_2x = F.relu(layer_2x)
        # Generate high-resolution output
        out = self.conv4(layer_2x)
        out = torch.tanh(out)
        return out
