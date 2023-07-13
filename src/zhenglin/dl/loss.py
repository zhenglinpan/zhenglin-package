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


class PerceptualLoss(nn.Module):
    def __init__(self, criterion='l1', rgb_range=1):
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        model = models.vgg19(weights='IMAGENET1K_V1')
        vgg_features = model.features
        modules = [m for m in vgg_features]

        self.vgg = nn.Sequential(*modules[:35])
        self.vgg.eval()
        vgg_mean = (0.2, 0.2, 0.2)
        vgg_std = (0.157 * rgb_range, 0.157 * rgb_range, 0.157 * rgb_range)
        self.sub_mean = MeanShift(rgb_range, vgg_mean, vgg_std)

        for p in self.parameters():
            p.requires_grad = False

    def forward(self, hr, sr):
        sr = torch.cat([sr, sr, sr], axis=1)
        hr = torch.cat([hr, hr, hr], axis=1)

        def _forward(x):
            x = self.sub_mean(x)
            x = self.vgg(x)
            return x

        vgg_sr = _forward(sr)
        with torch.no_grad():
            vgg_hr = _forward(hr)
            
        if self.criterion.lower() == 'l1':
            loss = F.l1_loss(vgg_sr, vgg_hr)
        elif self.criterion.lower() == 'l2' or self.criterion.lower() == 'mse':
            loss = F.mse_loss(vgg_sr, vgg_hr)
        else:
            raise NotImplementedError('Loss type {} is not implemented'.format(self.criterion))
        
        return loss