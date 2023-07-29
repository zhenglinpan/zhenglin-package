import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision

class CharbonnierLoss(nn.Module):
    def __init__(self,epsilon=1e-3):
        super(CharbonnierLoss, self).__init__()
        self.epsilon2=epsilon*epsilon

    def forward(self,x):
        value=torch.sqrt(torch.pow(x,2)+self.epsilon2)
        return torch.mean(value)


class MeanShift(nn.Conv2d):
    """
        Normalizing the color channels of an image.
    """
    def __init__(self, rgb_range, rgb_mean=(0.2, 0.2, 0.2), rgb_std=(1, 1, 1), sign=-1):
        super(MeanShift, self).__init__(3, 3, kernel_size=1)
        std = torch.Tensor(rgb_std)
        self.weight.data = torch.eye(3).view(3, 3, 1, 1) / std.view(3, 1, 1, 1)
        self.bias.data = sign * rgb_range * torch.Tensor(rgb_mean) / std
        for p in self.parameters():
            p.requires_grad = False


class PerceptualLoss(nn.Module):
    def __init__(self, criterion='l1', model_type='vgg19', rgb_range=1):
        """
            :criterion: loss function, 'l1' or 'l2'
            :model_type: 'vgg19' or 'resnet50' or 'mix'(ESRGAN-DP)
            :rgb_range: MAKE SURE INPUT RANGE IS [0, 1]
        """
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.model_type = model_type
        self.sub_mean = MeanShift(rgb_range, (0.2, 0.2, 0.2), (0.157 * rgb_range, 0.157 * rgb_range, 0.157 * rgb_range))
        
        models = []
        if model_type == 'vgg19':
            models += [torchvision.models.vgg19(weights='DEFAULT')]
        elif model_type == 'resnet50':
            models += [torchvision.models.resnet50(weights='DEFAULT')]
        elif model_type == 'mix':    # as per paper "ESRGAN-DP"
            models += [torchvision.models.vgg19(weights='DEFAULT'),
                       torchvision.models.resnet50(weights='DEFAULT')]
        else:
            raise NotImplementedError('Model {} is not implemented'.format(model))

        self.models = models
        
        ### fix model
        for model in models:
            for p in model.parameters():
                p.requires_grad = False
            model.eval()

    def foward(self, hr, sr):
        """
            :hr: high resolution image, torchsize([n, c, h, w])
            :sr: super resolution image, torchsize([n, c, h, w])
        """
        if self.model_type == 'vgg19':
            loss = self.loss(self.models[0], 'vgg19', hr, sr)
        elif self.model_type == 'resnet50':
            loss = self.loss(self.models[0], 'resnet50', hr, sr)
        elif self.model_type == 'mix':
            loss_vgg = self.loss(self.models[0], 'vgg19', hr, sr)
            loss_res = self.loss(self.models[1], 'resnet50', hr, sr, depth=8)
            lam = ((loss_vgg.item() + 1e-6) / (loss_res.item() + 1e-6))
            loss = loss_vgg + (mu := 1) * lam * loss_res    # mu range: [0.2, inf)
        return loss
        
    def loss(self, model, model_type, hr, sr):
        if model_type == 'vgg19':
            model_feature = model.features
            modules = [m for m in model_feature]
            model = nn.Sequential(*modules[:35])
        if model_type == 'resnet50':
            modules = list(model.children())
            model = nn.Sequential(*modules[:4], modules[5][:2]) # beta_1,2 as per paper "ESRGAN-DP"
        
        if sr.shape[1] == 1:
            sr = torch.cat([sr, sr, sr], axis=1)
            hr = torch.cat([hr, hr, hr], axis=1)           
            # self.sub_mean(x)  ### not sure if this is needed

        fm_sr = model(sr)
        with torch.no_grad():
            fm_hr = model(hr)
            
        if self.criterion.lower() == 'l1':
            loss = F.l1_loss(fm_sr, fm_hr)
        elif self.criterion.lower() == 'l2' or self.criterion.lower() == 'mse':
            loss = F.mse_loss(fm_sr, fm_hr)
        else:
            raise NotImplementedError('Loss type {} is not implemented'.format(self.criterion))
        
        return loss