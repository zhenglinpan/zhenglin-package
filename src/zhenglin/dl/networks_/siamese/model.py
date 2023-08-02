"""
An implementation from https://github.com/adambielski/siamese-triplet
"""

import torch.nn as nn
import torch.nn.functional as F


class HalfUNet(nn.Module):
    """The downsampling part of an U-Net is taken as an encoder
    2 linear layers are used for embedding. 
    **HACK: Mannually set the output size for linear layers**

    """
    def __init__(self,chan_in,nf=32):
        super().__init__()
        self.chan_in = chan_in
        self.relu = nn.ReLU()
        self.with_bn = True
        self.conv1_1 = nn.Conv2d(self.chan_in, nf, (3,3),(1,1),(1,1))
        self.bn1_1   = nn.BatchNorm2d(nf)
        self.conv1_2 = nn.Conv2d(nf, nf, 3,1,1)
        self.bn1_2   = nn.BatchNorm2d(nf)
        self.conv2_1 = nn.Conv2d(nf, nf*2, 3,1,1)
        self.bn2_1   = nn.BatchNorm2d(nf*2)
        self.conv2_2 = nn.Conv2d(nf*2, nf*4, 3,1,1)
        self.bn2_2   = nn.BatchNorm2d(nf*4)
        self.conv3_1 = nn.Conv2d(nf*4, nf*2, 3,1,1)
        self.bn3_1   = nn.BatchNorm2d(nf*2)
        self.conv3_2 = nn.Conv2d(nf*2, 1, 3,1,1)
        
        self.linear_out1 = nn.Linear(48*48, 128)    ### set the output size mannually
        self.linear_out2 = nn.Linear(128, 64)
        self.maxpool = nn.MaxPool2d(2, stride=2, return_indices=False, ceil_mode=False)
        self.upsample = nn.UpsamplingBilinear2d(scale_factor=2)
        self._initialize_weights()
        print('initialization weights is done')

    def forward(self, x1):
        if self.with_bn:
            x1_ = self.relu(self.bn1_2(self.conv1_2(self.relu(self.bn1_1(self.conv1_1(x1))))))
            x2 = self.relu(self.bn2_2(self.conv2_2(self.relu(self.bn2_1(self.conv2_1(self.maxpool(x1_)))))))
            x3 = self.relu(self.conv3_2(self.relu(self.bn3_1(self.conv3_1(self.maxpool(x2))))))
            out = self.linear_out2(self.relu(self.linear_out1(x3.view(x3.size(0), -1))))
        return out


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


class SiameseNet(nn.Module):
    def __init__(self):
        super(SiameseNet, self).__init__()
        self.embedding_net = HalfUNet()     ### any embedding net

    def forward(self, x1, x2):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        return output1, output2

    def get_embedding(self, x):
        return self.embedding_net(x)


class TripletNet(nn.Module):
    def __init__(self, embedding_net):
        super(TripletNet, self).__init__()
        self.embedding_net = embedding_net

    def forward(self, x1, x2, x3):
        output1 = self.embedding_net(x1)
        output2 = self.embedding_net(x2)
        output3 = self.embedding_net(x3)
        return output1, output2, output3

    def get_embedding(self, x):
        return self.embedding_net(x)