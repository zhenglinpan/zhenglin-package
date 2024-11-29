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
            See https://github.com/Xinzhe99/Perceptual-Loss-for-pytorch/blob/main/main.py
            for alternative implementation with Perception loss on multiple layers: 
             
            :criterion: loss function, 'l1' or 'l2'
            :model_type: 'vgg16', 'vgg19' or 'resnet50' or 'mix'
            :rgb_range: MAKE SURE INPUT RANGE IS BETWEEN [0, 1]
        """
        super(PerceptualLoss, self).__init__()
        self.criterion = criterion
        self.model_type = model_type
        self.sub_mean = MeanShift(rgb_range, (0.2, 0.2, 0.2), (0.157 * rgb_range, 0.157 * rgb_range, 0.157 * rgb_range))
        
        models = []
        if model_type == 'vgg16':
            models += [torchvision.models.vgg16(weights='DEFAULT')]
        elif model_type == 'vgg19':
            models += [torchvision.models.vgg19(weights='DEFAULT')]
        elif model_type == 'resnet50':
            models += [torchvision.models.resnet50(weights='DEFAULT')]
        elif model_type == 'mix':    # as per paper "Song et al. Dual Perceptual Loss for Single Image Super-Resolution Using ESRGAN"
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
    
    def to(self, device):
        super(PerceptualLoss, self).to(device)
        for model in self.models:
            model.to(device)

        return self

    def forward(self, hr, sr):
        """
            :hr: high resolution image, torchsize([n, c, h, w])
            :sr: super resolution image, torchsize([n, c, h, w])
        """
        if self.model_type == 'vgg16':
            loss = self.loss(self.models[0], 'vgg16', hr, sr)
        elif self.model_type == 'vgg19':
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
        if model_type == 'vgg16':  
            model_feature = model.features
            modules = [m for m in model_feature]
            model = nn.Sequential(*modules[:22])
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
            # self.sub_mean(x)  ### not sure if needed

        f_sr = model(sr)
        with torch.no_grad():
            f_hr = model(hr)
            
        if self.criterion.lower() == 'l1':
            loss = F.l1_loss(f_hr, f_sr)
        elif self.criterion.lower() == 'l2' or self.criterion.lower() == 'mse':
            loss = F.mse_loss(f_hr, f_sr)
        else:
            raise NotImplementedError('Loss type {} is not implemented'.format(self.criterion))
        
        return loss

    
class ContrastiveLoss(nn.Module):
    """
    Contrastive loss
    Takes embeddings of two samples and a target label == 1 if samples are from the same class and label == 0 otherwise
    
    An implementation from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin):
        super(ContrastiveLoss, self).__init__()
        self.margin = margin
        self.eps = 1e-9

    def forward(self, output1, output2, target, size_average=True):
        distances = (output2 - output1).pow(2).sum(1)  # squared distances
        losses = 0.5 * (target.float() * distances + (1 - target).float() * F.relu(self.margin - (distances + self.eps).sqrt()).pow(2))
        return losses.mean() if size_average else losses.sum()


class TripletLoss(nn.Module):
    """
    Triplet loss
    Takes embeddings of an anchor sample, a positive sample and a negative sample
    
    An implementation from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin):
        super(TripletLoss, self).__init__()
        self.margin = margin

    def forward(self, anchor, positive, negative, size_average=True):
        distance_positive = (anchor - positive).pow(2).sum(1)  # .pow(.5)
        distance_negative = (anchor - negative).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(distance_positive - distance_negative + self.margin)
        return losses.mean() if size_average else losses.sum()


class OnlineContrastiveLoss(nn.Module):
    """
    Online Contrastive loss
    Takes a batch of embeddings and corresponding labels.
    Pairs are generated using pair_selector object that take embeddings and targets and return indices of positive
    and negative pairs
    
    An implementation from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin, pair_selector):
        super(OnlineContrastiveLoss, self).__init__()
        self.margin = margin
        self.pair_selector = pair_selector

    def forward(self, embeddings, target):
        positive_pairs, negative_pairs = self.pair_selector.get_pairs(embeddings, target)
        if embeddings.is_cuda:
            positive_pairs = positive_pairs.cuda()
            negative_pairs = negative_pairs.cuda()
        positive_loss = (embeddings[positive_pairs[:, 0]] - embeddings[positive_pairs[:, 1]]).pow(2).sum(1)
        negative_loss = F.relu(
            self.margin - (embeddings[negative_pairs[:, 0]] - embeddings[negative_pairs[:, 1]]).pow(2).sum(
                1).sqrt()).pow(2)
        loss = torch.cat([positive_loss, negative_loss], dim=0)
        return loss.mean()


class OnlineTripletLoss(nn.Module):
    """
    Online Triplets loss
    Takes a batch of embeddings and corresponding labels.
    Triplets are generated using triplet_selector object that take embeddings and targets and return indices of
    triplets
    
    An implementation from https://github.com/adambielski/siamese-triplet
    """

    def __init__(self, margin, triplet_selector):
        super(OnlineTripletLoss, self).__init__()
        self.margin = margin
        self.triplet_selector = triplet_selector

    def forward(self, embeddings, target):

        triplets = self.triplet_selector.get_triplets(embeddings, target)

        if embeddings.is_cuda:
            triplets = triplets.cuda()

        ap_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 1]]).pow(2).sum(1)  # .pow(.5)
        an_distances = (embeddings[triplets[:, 0]] - embeddings[triplets[:, 2]]).pow(2).sum(1)  # .pow(.5)
        losses = F.relu(ap_distances - an_distances + self.margin)

        return losses.mean(), len(triplets)


class IoULoss(nn.Module):
    """ Calculate Intersection over Union (IoU) loss between predicted and true masks.
        
        Args:
        - mask_pred (torch.Tensor): Predicted masks, shape: (N, 1, H, W), range [0, 1].
        - mask_true (torch.Tensor): True masks, shape: (N, 1, H, W), range [0, 1].
        
    """
    def __init__(self, smooth=1e-6):
        super(IoULoss, self).__init__()
        self.smooth = smooth
        
    def forward(self, target:torch.Tensor, pred:torch.Tensor):
        assert target.shape == pred.shape, \
        f'target and pred must have the same shape, got target {target.shape} and pred {pred.shape}'
        
        pred = pred.view(-1)
        target = target.view(-1)
        
        intersection = (pred * target).sum()
        union = pred.sum() + target.sum() - intersection + self.smooth
        
        return 1 - (intersection / union)


class DiceLoss(nn.Module):
    """ Calculate Dice Loss.
        
        Args:
        - mask_pred (torch.Tensor): Predicted masks, shape (N, 1, H, W), range [0, 1].
        - mask_true (torch.Tensor): True masks, shape (N, 1, H, W), range [0, 1].
        
    """
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth
    
    def forward(self, target:torch.Tensor, pred:torch.Tensor):
        assert target.shape == pred.shape, \
        f'target and pred must have the same shape, got target {target.shape} and pred {pred.shape}'
        
        pred = pred.contiguous()
        target = target.contiguous()
        
        intersection = (pred * target).sum(dim=(2, 3))
        dice = (2. * intersection + self.smooth) / (pred.sum(dim=(2, 3)) + target.sum(dim=(2, 3)) + self.smooth)
        dice = dice.mean()
        
        return 1 - dice


class PrecisionLoss(nn.Module):
    def __init__(self, epsilon=1e-6):
        super().__init__()
        self.epsilon = epsilon
    
    def forward(self, target:torch.Tensor, pred:torch.Tensor):
        assert target.shape == pred.shape, \
        f'target and pred must have the same shape, got target {target.shape} and pred {pred.shape}'
        
        pred = pred.contiguous().view(-1)
        target = target.contiguous().view(-1)
        
        tp = (pred * target).sum()
        fp = (pred * (1 - target)).sum()
        pre = tp / (tp + fp + self.epsilon)
        
        return 1 - pre