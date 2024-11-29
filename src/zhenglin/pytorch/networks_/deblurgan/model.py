import torch
import torch.nn as nn
import torch.nn.functional as F
import functools
from layer_utils import get_norm_layer, ResNetBlock, MinibatchDiscrimination

def dense_block(inputs, dilation_factor=None):
    x = nn.LeakyReLU(negative_slope=0.2)(inputs)
    x = nn.Conv2d(256, 256, 1, 1, 0)(x)
    x = nn.BatchNorm2d(x)
    x = nn.LeakyReLU(negative_slope=0.2)(x)
    if dilation_factor is not None:
        x = nn.Conv2d(256, 64, 3, 1, 1, dilation=dilation_factor)(x)
    else:
        x = nn.Conv2d(256, 64, 3, 1, 1)(x)
    x = nn.BatchNorm2d(x)
    x = nn.Dropout2d(p=0.5)(x)
    return x


class ResNetGenerator(nn.Module):
    """Define a generator using ResNet"""

    def __init__(self, input_nc, output_nc, ngf=64, n_blocks=9, norm_type='instance', padding_type='reflect',
                 use_dropout=True, learn_residual=True):
        super(ResNetGenerator, self).__init__()

        self.learn_residual = learn_residual

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        sequence = [
            nn.ReflectionPad2d(3),
            nn.Conv2d(input_nc, ngf, kernel_size=7, stride=1, padding=0, bias=use_bias),
            norm_layer(ngf),
            nn.ReLU(True)
        ]

        n_downsampling = 2
        for i in range(n_downsampling):  # downsample the feature map
            mult = 2 ** i
            sequence += [
                nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3, stride=2, padding=1, bias=use_bias),
                norm_layer(ngf * mult * 2),
                nn.ReLU(True)
            ]

        for i in range(n_blocks):  # ResNet
            sequence += [
                ResNetBlock(ngf * 2 ** n_downsampling, norm_layer, padding_type, use_dropout, use_bias)
            ]

        for i in range(n_downsampling):  # upsample the feature map
            mult = 2 ** (n_downsampling - i)
            sequence += [
                nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                   output_padding=1, bias=use_bias),
                norm_layer(int(ngf * mult / 2)),
                nn.ReLU(True)
            ]

        sequence += [
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, output_nc, kernel_size=7, stride=1, padding=0),
            nn.Tanh()
        ]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        if self.learn_residual:
            out = x + out
            out = torch.clamp(out, min=-1, max=1)  # clamp to [-1,1] according to normalization(mean=0.5, var=0.5)
        return out

class NLayerDiscriminator(nn.Module):
    """Define a PatchGAN discriminator"""

    def __init__(self, input_nc, ndf=64, n_layers=3, norm_type='instance', use_sigmoid=False,
                 use_minibatch_discrimination=False):
        super(NLayerDiscriminator, self).__init__()

        self.use_minibatch_discrimination = use_minibatch_discrimination

        norm_layer = get_norm_layer(norm_type)
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func != nn.BatchNorm2d
        else:
            use_bias = norm_layer != nn.BatchNorm2d

        kernel_size = 3
        padding = 1
        sequence = [
            nn.Conv2d(input_nc, ndf, kernel_size=kernel_size, stride=1, padding=padding),
            nn.LeakyReLU(0.2, True)
        ]

        nf_mult = 1
        for n in range(1, n_layers):  # gradually increase the number of filters
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,
                          bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, True)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kernel_size, stride=1, padding=padding,
                      bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [
            nn.Conv2d(ndf * nf_mult, 1, kernel_size=kernel_size, stride=1, padding=padding)
        ]  # output 1 channel prediction map

        if use_sigmoid:
            sequence += [nn.Sigmoid()]

        self.model = nn.Sequential(*sequence)

    def forward(self, x):
        out = self.model(x)
        if self.use_minibatch_discrimination:
            out = out.view(out.size(0), -1)
            a = out.size(1)
            out = MinibatchDiscrimination(a, a, 3)(out)
        return out


import torch
import torch.nn as nn
import torch.nn.functional as F

# the paper defined hyper-parameter:chr
channel_rate = 64

# Dense Block
class DenseBlock(nn.Module):
    def __init__(self, in_channels, dilation_factor=None):
        super(DenseBlock, self).__init__()
        self.relu = nn.LeakyReLU(negative_slope=0.2)
        self.conv1 = nn.Conv2d(in_channels, 4 * channel_rate, kernel_size=1, padding=0)
        self.batch_norm1 = nn.BatchNorm2d(4 * channel_rate)
        self.conv2 = nn.Conv2d(4 * channel_rate, channel_rate, kernel_size=3, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(channel_rate)
        self.dropout = nn.Dropout(p=0.5)
        self.dilation_factor = dilation_factor

    def forward(self, inputs):
        x = self.relu(inputs)
        x = self.conv1(x)
        x = self.batch_norm1(x)
        x = self.relu(x)
        if self.dilation_factor is not None:
            x = F.conv2d(x, self.conv2.weight, padding=self.dilation_factor, dilation=self.dilation_factor)
        else:
            x = self.conv2(x)
        x = self.batch_norm2(x)
        x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.head = nn.Conv2d(1, 4 * channel_rate, kernel_size=3, padding=1)
        self.dense1 = DenseBlock(4 * channel_rate)
        self.dense2 = DenseBlock(5 * channel_rate, dilation_factor=(1, 1))
        self.dense3 = DenseBlock(6 * channel_rate)
        self.dense4 = DenseBlock(7 * channel_rate, dilation_factor=(2, 2))
        self.dense5 = DenseBlock(8 * channel_rate)
        self.dense6 = DenseBlock(9 * channel_rate, dilation_factor=(3, 3))
        self.dense7 = DenseBlock(10 * channel_rate)
        self.dense8 = DenseBlock(11 * channel_rate, dilation_factor=(2, 2))
        self.dense9 = DenseBlock(12 * channel_rate)
        self.dense10 = DenseBlock(13 * channel_rate, dilation_factor=(1, 1))
        self.tail_conv1 = nn.Conv2d(13 * channel_rate, 4 * channel_rate, kernel_size=1, padding=0)
        self.tail_batch_norm = nn.BatchNorm2d(4 * channel_rate)
        self.global_skip_conv = nn.Conv2d(4 * channel_rate + 4 * channel_rate, channel_rate, kernel_size=3, padding=1)
        self.global_skip_relu = nn.LeakyReLU(negative_slope=0.2)
        self.output_conv = nn.Conv2d(channel_rate, 1, kernel_size=3, padding=1)
        self.tanh = nn.Tanh()

    def forward(self, inputs):
        h = self.head(inputs)
        # print("1", h.shape)
        d_1 = self.dense1(h)
        # print("2", d_1.shape)
        x = torch.cat((h, d_1), dim=1)
        # print("3", x.shape)
        d_2 = self.dense2(x)
        # print("4", d_2.shape)
        x = torch.cat((x, d_2), dim=1)
        # print("5", x.shape)
        d_3 = self.dense3(x)
        # print("6", d_3.shape)
        x = torch.cat((x, d_3), dim=1)
        # print("7", x.shape)
        d_4 = self.dense4(x)
        # print("8", d_4.shape)
        x = torch.cat((x, d_4), dim=1)
        # print("9", x.shape)
        d_5 = self.dense5(x)
        # print("10", d_5.shape)
        x = torch.cat((x, d_5), dim=1)
        d_6 = self.dense6(x)
        # print("11", d_6.shape)
        x = torch.cat((x, d_6), dim=1)
        # print("12", x.shape)
        d_7 = self.dense7(x)
        # print("13", d_7.shape)
        x = torch.cat((x, d_7), dim=1)
        # print("14", x.shape)
        d_8 = self.dense8(x)
        # print("15", d_8.shape)
        x = torch.cat((x, d_8), dim=1)
        # print("16", x.shape)
        d_9 = self.dense9(x)
        # print("17", d_9.shape)
        x = torch.cat((x, d_9), dim=1)
        # print("18", x.shape)
        d_10 = self.dense10(x)
        # print("19", d_10.shape)

        x = self.tail_batch_norm(self.tail_conv1(x))
        # print("20", x.shape)

        x = torch.cat((h, x), dim=1)
        # print("21", x.shape)
        
        x = self.global_skip_relu(self.global_skip_conv(x))
        # print("22", x.shape)
        
        outputs = self.tanh(self.output_conv(x))
        # print("23", outputs.shape)
        
        return outputs


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.conv1 = nn.Conv2d(1, channel_rate, kernel_size=3, stride=1, padding=1)
        self.batch_norm1 = nn.BatchNorm2d(channel_rate)
        self.conv2 = nn.Conv2d(channel_rate, 2 * channel_rate, kernel_size=3, stride=1, padding=1)
        self.batch_norm2 = nn.BatchNorm2d(2 * channel_rate)
        self.conv3 = nn.Conv2d(2 * channel_rate, 4 * channel_rate, kernel_size=3, stride=1, padding=1)
        self.batch_norm3 = nn.BatchNorm2d(4 * channel_rate)
        self.conv4 = nn.Conv2d(4 * channel_rate, 1, kernel_size=3, stride=1, padding=1)
        # self.batch_norm4 = nn.BatchNorm2d(4 * channel_rate)

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.batch_norm1(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv2(x)
        x = self.batch_norm2(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv3(x)
        x = self.batch_norm3(x)
        x = F.leaky_relu(x, negative_slope=0.2)

        x = self.conv4(x)

        return x
