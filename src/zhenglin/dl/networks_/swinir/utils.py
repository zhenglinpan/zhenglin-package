import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x):
        value=torch.sqrt(torch.pow(x,2)+self.epsilon2)
        return torch.mean(value)

class MeanShift(nn.Conv2d):
    def __init__(
            self, rgb_range,
            rgb_mean=(0.2, 0.2, 0.2), rgb_std=(1, 1, 1), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class PerceptualLoss_l1(nn.Module):
    def __init__(self, rgb_range=1, device=0):
        super(PerceptualLoss_l1, self).__init__()
        model = models.vgg19(weights='IMAGENET1K_V1').to(device)
        vgg_features = model.features
        modules = [m for m in vgg_features]

        self.vgg = nn.Sequential(*modules[:35])
        self.vgg.eval()
        vgg_mean = (0.2, 0.2, 0.2)
        vgg_std = (0.157 * rgb_range, 0.157 * rgb_range, 0.157 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.to(self.device)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, hr, sr):
        sr = torch.cat([sr, sr, sr], axis=1).to(self.device)
        hr = torch.cat([hr, hr, hr], axis=1).to(self.device)

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr).detach()
        loss = F.l1_loss(vgg_sr, vgg_hr)
        # loss = F.mse_loss(vgg_sr, vgg_hr)
        return loss