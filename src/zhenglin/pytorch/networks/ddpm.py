import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

class CTDiffusion:
    def __init__(self, noise_steps=20, beta_start=5e-6, beta_end=1e-2, img_size=64, device=1):
        self.noise_steps = noise_steps
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.img_size = img_size
        self.device = device

        self.beta = self.prepare_noise_schedule(type="softmax").to(device)
        self.alpha = 1. - self.beta
        self.alpha_hat = torch.cumprod(self.alpha, dim=0)
        
    def prepare_noise_schedule(self, type):
        t = torch.linspace(0, 1, self.noise_steps)
        if type == 'linear':
            beta = torch.linspace(self.beta_start, self.beta_end, self.noise_steps)
        elif type == 'sinusodial':
            beta = torch.sin(t * (np.pi / 2)) * (self.beta_end - self.beta_start) + self.beta_start
        elif type == 'softmax':
            beta = torch.sigmoid((t - 0.5) * 10) * (self.beta_end - self.beta_start) + self.beta_start
        else:
            raise NotImplementedError
        return beta
        
    def save_progressive_noised_image(self, x, t:torch.Tensor):  
        for i in range(0, int(t)):
            it = torch.Tensor([i]).long().to(self.device)
            noise_image = self.noise_images(x, it)[0]
            plt.imsave(f'./results/adding_noise/{i}_noised.jpg', noise_image[0, 0, :, :].cpu().detach().np(), cmap=plt.cm.bone)
    
    def noise_images(self, x, t):
        sqrt_alpha_hat = torch.sqrt(self.alpha_hat[t])[:, None, None, None]
        sqrt_one_minus_alpha_hat = torch.sqrt(1 - self.alpha_hat[t])[:, None, None, None]
        Ɛ = torch.randn_like(x)
        return sqrt_alpha_hat * x + sqrt_one_minus_alpha_hat * Ɛ, Ɛ

    def sample_timesteps(self, n):
        return torch.randint(low=1, high=self.noise_steps, size=(n,))   # t range [0, 120)

    def sample(self, model, n, img_in=None, t_=None):
        model.eval()
        with torch.no_grad():
            x = torch.randn((n, 1, self.img_size, self.img_size)).to(self.device)
            if img_in is not None:
                x = torch.Tensor(img_in).to(self.device)
            if t_ is None:
                t_ = torch.tensor([self.noise_steps - 1]).to(self.device)
            for i in tqdm(reversed(range(1, t_+1)), position=0):  
                t = (torch.ones(n) * i).long().to(self.device)
                predicted_noise = model(x, t)   # no t+1 since t starts from 0
                alpha = self.alpha[t][:, None, None, None]
                alpha_hat = self.alpha_hat[t][:, None, None, None]
                beta = self.beta[t][:, None, None, None]
                noise = torch.randn_like(x) if i > 1 else torch.zeros_like(x)
                x = 1 / torch.sqrt(alpha) * (x - ((1 - alpha) / (torch.sqrt(1 - alpha_hat))) * predicted_noise) + torch.sqrt(beta) * noise
                # if img_in is not None:
                #     if i % 100 == 0 or (i <= 50 and i % 10 == 0):
                #         plt.imsave(f'./results/by_step/step_{i}.jpg', x[0, 0, :, :].cpu().detach().np(), cmap=plt.cm.bone)
                #     if i < 20:
                #         plt.imsave(f'./results/by_step/step_{i}.jpg', x[0, 0, :, :].cpu().detach().np(), cmap=plt.cm.bone)
        model.train()
        return x


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta
        self.step = 0

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new

    def step_ema(self, ema_model, model, step_start_ema=2000):
        if self.step < step_start_ema:
            self.reset_parameters(ema_model, model)
            self.step += 1
            return
        self.update_model_average(ema_model, model)
        self.step += 1

    def reset_parameters(self, ema_model, model):
        ema_model.load_state_dict(model.state_dict())


class SelfAttention(nn.Module):
    def __init__(self, channels, size): # 128, 32
        super(SelfAttention, self).__init__()
        self.channels = channels
        self.size = size
        self.mha = nn.MultiheadAttention(embed_dim=channels, num_heads=4, batch_first=True)
        self.ln = nn.LayerNorm([channels])
        self.ff_self = nn.Sequential(
            nn.LayerNorm([channels]),
            nn.Linear(channels, channels),
            nn.GELU(),
            nn.Linear(channels, channels),
        )

    def forward(self, x):
        # print("self-attention x input shape", x.shape)  # torch.Size([4, 128, 96, 96])
        x = x.view(-1, self.channels, self.size * self.size).swapaxes(1, 2)
        # print("self-attention x viewed shape", x.shape) # torch.Size([4, 96*96, 128])
        x_ln = self.ln(x)   #! How to use layer norm? [channel] + .view()!
        # print("self-attention x ln shape", x_ln.shape)   # torch.Size([4, 96*96, 128])
        attention_value, _ = self.mha(x_ln, x_ln, x_ln)
        # print("self-attention x mha shape", attention_value.shape)   # torch.Size([4, 96*96, 128])
        attention_value = attention_value + x
        # print("self-attention x attention_value shape", attention_value.shape)   # torch.Size([4, 96*96, 128])
        attention_value = self.ff_self(attention_value) + attention_value
        # print("self-attention x attention_value shape", attention_value.shape)   # torch.Size([4, 96*96, 128])
        out = attention_value.swapaxes(2, 1).view(-1, self.channels, self.size, self.size)
        # print("self-attention x out shape", out.shape)   # torch.Size([4, 128, 96, 96])
        return out

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels, mid_channels=None, residual=False):
        super().__init__()
        self.residual = residual
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, mid_channels),
            nn.GELU(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(1, out_channels),
        )

    def forward(self, x):
        if self.residual:
            return F.gelu(x + self.double_conv(x))
        else:
            return self.double_conv(x)


class Down(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(emb_dim, out_channels),
        )

    def forward(self, x, t):
        x = self.maxpool_conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb
    
class PureDown(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels),
        )

    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    def __init__(self, in_channels, out_channels, emb_dim=256):
        super().__init__()

        self.up = nn.Upsample(scale_factor=2, mode="bilinear", align_corners=True)
        self.conv = nn.Sequential(
            DoubleConv(in_channels, in_channels, residual=True),
            DoubleConv(in_channels, out_channels, in_channels // 2),
        )

        self.emb_layer = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                emb_dim,
                out_channels
            ),
        )

    def forward(self, x, skip_x, t):
        x = self.up(x)
        x = torch.cat([skip_x, x], dim=1)
        x = self.conv(x)
        emb = self.emb_layer(t)[:, :, None, None].repeat(1, 1, x.shape[-2], x.shape[-1])
        return x + emb


class UNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, size=64, time_dim=256, device=1):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        p = size // 64
        self.inc = DoubleConv(c_in, 64)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32*p)   # if input_size is 192, should be SelfAttention(128, 32*3) 
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16*p)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8*p)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16*p)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32*p)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64*p)
        self.outc = nn.Conv2d(64, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)
        t = self.pos_encoding(t, self.time_dim)
        # print('x.shape', x.shape)   # x.shape torch.Size([4, 1, 192, 192])
        x1 = self.inc(x)
        # print('x1.shape', x1.shape) # x1.shape torch.Size([4, 64, 192, 192])
        x2 = self.down1(x1, t)
        # print('x2.shape', x2.shape) # x2.shape torch.Size([4, 128, 96, 96])
        x2 = self.sa1(x2)
        # print('x2.shape', x2.shape) # x2.shape torch.Size([36, 128, 32, 32])
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        output = self.outc(x)
        return output


class DownAdaptor(nn.Module):
    def __init__(self, size):
        super(DownAdaptor, self).__init__()
        down_adaptor = []
        for _ in range(int(np.log2(size // 64))):
            down_adaptor += [PureDown(64, 64),
                             nn.InstanceNorm2d(64),
                             nn.ReLU(1)]    # sequential() only accept SISO, so PureDown is constructed
        self.down_adaptor = nn.Sequential(*down_adaptor)
        
    def forward(self, x):
        return self.down_adaptor(x)

class UpAdaptor(nn.Module):
    def __init__(self, size):
        super(UpAdaptor, self).__init__()
        up_adaptor = []
        for _ in range(int(np.log2(size // 64))):
            up_adaptor += [nn.ConvTranspose2d(64, 64, 4, 2, 1), 
                           nn.InstanceNorm2d(64),
                           nn.ReLU(1)]
        self.up_adaptor = nn.Sequential(*up_adaptor)
        
    def forward(self, x):
        # for layer in self.up_adaptor: # renders not-same-device error 
        #     x = layer(x)
        return self.up_adaptor(x)

class CTUNet(nn.Module):
    def __init__(self, c_in=1, c_out=1, size=256, time_dim=256, device=1):
        super().__init__()
        self.device = device
        self.time_dim = time_dim
        if size not in [64, 128, 256, 512]:
            raise Exception("image size must be either 64, 128, 256, 512.")
        else:
            self.size = size
        self.inc = DoubleConv(c_in, out_channels=64) 
        self.down_adaptor = DownAdaptor(self.size)
        self.down1 = Down(64, 128)
        self.sa1 = SelfAttention(128, 32)   # if input_size is 192, should be SelfAttention(128, 32*3), attention is fixed with size
        self.down2 = Down(128, 256)
        self.sa2 = SelfAttention(256, 16)
        self.down3 = Down(256, 256)
        self.sa3 = SelfAttention(256, 8)

        self.bot1 = DoubleConv(256, 512)
        self.bot2 = DoubleConv(512, 512)
        self.bot3 = DoubleConv(512, 256)

        self.up1 = Up(512, 128)
        self.sa4 = SelfAttention(128, 16)
        self.up2 = Up(256, 64)
        self.sa5 = SelfAttention(64, 32)
        self.up3 = Up(128, 64)
        self.sa6 = SelfAttention(64, 64)
        self.up_adaptor = UpAdaptor(self.size)
        self.outc1 = DoubleConv(64, 128)
        self.outc2 = DoubleConv(128, 256)
        self.outc3 = DoubleConv(256, 128)
        self.outc4 = nn.Conv2d(128, c_out, kernel_size=1)

    def pos_encoding(self, t, channels):
        inv_freq = 1.0 / (
            10000
            ** (torch.arange(0, channels, 2, device=self.device).float() / channels)
        )
        pos_enc_a = torch.sin(t.repeat(1, channels // 2) * inv_freq)
        pos_enc_b = torch.cos(t.repeat(1, channels // 2) * inv_freq)
        pos_enc = torch.cat([pos_enc_a, pos_enc_b], dim=-1)
        return pos_enc

    def forward(self, x, t):
        t = t.unsqueeze(-1).type(torch.float)   # add dim to last pos
        t = self.pos_encoding(t, self.time_dim)
        x1 = self.inc(x)
        if self.size > 64:
            x1 = self.down_adaptor(x1)
        x2 = self.down1(x1, t)
        x2 = self.sa1(x2)
        x3 = self.down2(x2, t)
        x3 = self.sa2(x3)
        x4 = self.down3(x3, t)
        x4 = self.sa3(x4)

        x4 = self.bot1(x4)
        x4 = self.bot2(x4)
        x4 = self.bot3(x4)

        x = self.up1(x4, x3, t)
        x = self.sa4(x)
        x = self.up2(x, x2, t)
        x = self.sa5(x)
        x = self.up3(x, x1, t)
        x = self.sa6(x)
        if self.size > 64:
            x = self.up_adaptor(x)
        x = self.outc1(x)
        x = self.outc2(x)
        x = self.outc3(x)
        output = self.outc4(x)
        return output
