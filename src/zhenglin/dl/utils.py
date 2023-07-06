import torch
from networks.cyclegan import ReplayBuffer as ReplayBuffer_
import numpy as np
import copy

def summary(model):
    print(model)
    parameter_number = 0
    for layer in list(model.parameters()):
        parameter_number += torch.prod(torch.tensor(layer.size()))
    print('Total Parameter numbers:{:,}'.format(int(parameter_number)))


def fix_model(model):
    for param in model.parameters():
        param.requires_grad = False

    model.eval()
    
    return model

class ReplayBuffer(ReplayBuffer_):
    pass

class EasyReplayBuffer:
    def __init__(self, max_size=10):
        """
            An easier implementation of replaybuffer with peek()
            TODO: using list could be slow, use tensor concate instead
        """
        assert max_size > 0
        self.max_size = max_size
        self.data = []
    
    def push_and_pop(self, data):
        if len(self.data) < self.max_size:
            self.data.append(data)
        else:
            self.data.pop(0)
            self.data.append(data)
        return self.data[-1]
    
    def peek(self):
        assert len(self.data) != 0
        return self.data[0]


def norm_min_max(img_in, norm_type=None, perc_min = 0.01, perc_max = 0.99, min_ = -80000., max_ = 80000.,cut_off=True): 
    '''
    normalize image to (0-1), three types:
    self:by its min and max, 
    sort:by the range(0.01,0.99) to exclude the outlier,
    all:by the choosed min and max value,
    return normalized image, minimum value, maximum value
    '''
    if norm_type == 'self':
        img_min = np.min(img_in)
        img_max = np.max(img_in)
        try:
            img_out = (img_in - img_min)/(img_max - img_min)
        except:
            img_out = img_in
    elif norm_type == 'sort': 
        sort_img = copy.deepcopy(img_in)
        sort_img = np.reshape(sort_img,(1,-1))
        sort_img.sort()
        pixel_num = sort_img.shape[1]

        img_min = sort_img[0,round(perc_min * pixel_num)].item()
        img_max = sort_img[0,round(perc_max * pixel_num)].item()
        img_out = (img_in - img_min)/(img_max - img_min)
        # print('****')
        # print('the max is {}, min is {}'.format(img_max,img_min))
    elif norm_type == 'sort_ct': 
        sort_img = copy.deepcopy(img_in)
        sort_img = np.reshape(sort_img,(1,-1))
        sort_img.sort()
        pixel_num = sort_img.shape[1]

        img_min = sort_img[0,round(perc_min * pixel_num)].item()
        img_max = min(sort_img[0,round(perc_max * pixel_num)].item(),1800)
        # img_in[img_in < img_min] = img_min
        # img_in[img_in > img_max] = img_max
        img_out = (img_in - img_min)/(img_max - img_min)
        
        # print('****')
        # print('the max is {}, min is {}'.format(img_max,img_min))
    elif norm_type == 'no_norm': # uncheck
        img_out = img_in
        img_min = np.min(img_in)
        img_max = np.max(img_in)
    elif norm_type == 'all': # uncheck
        img_min = min_
        img_max = max_
        img_out = np.array((img_in - min_)/(max_ - min_), dtype=np.float32)#.astype(np.float32)
        
    else:
        img_min = np.min(img_in)
        img_max = np.max(img_in)
        img_out = (img_in - img_min)/(img_max - img_min)
    if cut_off:
        img_out[img_out<0]=0
        img_out[img_out>1]=1
    return img_out,img_min,img_max

def denorm_min_max(img_in,img_min,img_max): 
    '''
    denormalize image to original range:
    '''
    img_out = img_in*(img_max-img_min) + img_min

    return img_out