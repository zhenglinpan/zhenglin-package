import os
import sys
import torch
import numpy as np
import copy

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from networks.cyclegan import ReplayBuffer
from networks.cyclegan import LinearLambdaLR
from networks.cyclegan import weights_init_normal
from networks.unets_lite import dw_conv

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


def save_keys(path_model, path_json_out=None):
    '''save the keys of a model weight(.pt, ckpt,) file to a json file'''
    import json
    from collections import defaultdict
    weight = torch.load(path_model, map_location='cpu')
    
    if 'state_dict' in weight.keys():
        writebuff = defaultdict(list)
        for key in weight.keys():
            for subkey in weight[key].keys():
                writebuff[key].append(subkey)
    else:
        writebuff = {'state_dict': list(weight.keys())}
            
    path_json_out = './keys.json' if path_json_out else path_json_out
    with open(path_json_out, 'w') as f:
        json.dump(writebuff, f, indent=4)
    print(f"Keys saved to {path_json_out}")


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
    
    def rand_pick(self):
        assert len(self.data) != 0
        return np.random.choice(self.data)


def norm_min_max(img_in, norm_type='self', perc_lb=0.01, perc_ub=0.99, min_=None, max_=None, cut_off=True): 
    '''
    normalize image to (0-1), three types:
    self: by its min and max, 
    quartile: by the range(0.01,0.99) to exclude the outlier,
    cross: by the given min and max values,
    return normalized image, minimum value, maximum value
    '''
    if min_ is not None and max_ is not None:
        norm_type = 'cross'
    
    if norm_type == 'self':
        img_min = np.min(img_in)
        img_max = np.max(img_in)
        try:
            img_out = (img_in - img_min)/(img_max - img_min)
        except:
            img_out = img_in
    elif norm_type == 'quartile': 
        img_flat = np.reshape(copy.deepcopy(img_in),(1,-1))
        img_flat.sort()
        pixel_num = img_flat.shape[1]
        img_min = img_flat[0,round(perc_lb * pixel_num)].item()
        img_max = img_flat[0,round(perc_ub * pixel_num)].item()
        img_out = (img_in - img_min)/(img_max - img_min)
    elif norm_type == 'cross':
        img_min = min_
        img_max = max_
        img_out = (img_in - img_min)/(img_max - img_min)
    else:
        raise NotImplementedError
    
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

def center_crop(mat: np.ndarray, size):
    """
    Center crop a matrix without transposing, stupid torchvision.transforms
    :param mat: ndarray to be cropped, expect shape [H, W] or [C, H, W]
    :param size: size of the cropped matrix
    :return: cropped matrix
    """
    assert size <= mat.shape[-1]

    if size == mat.shape[-1]:
        return mat

    x = mat.shape[-2] // 2
    y = mat.shape[-1] // 2
    r = size // 2

    if len(mat.shape) == 2:
        return mat[x - r: x + r, y - r: y + r]
    elif len(mat.shape) == 3:
        return mat[:, x - r: x + r, y - r: y + r]
    else:
        raise Exception("Unsupported shape")


def receptive_field(k, s, d):
    """
    Calculate the receptive field of a network
    :param k: kernel size, [9, 3, 3, 7, 3, 3, 5, 3, 3, 3]
    :param s: stride, [1, 1, 2, 1, 1, 2, 1, 1, 1, 1]
    :param d: dilation, [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    
    Reference: https://zhuanlan.zhihu.com/p/113285797
    """
    reg = 0    # register, i.e. rf_i-1
    for i in range(len(k)):
        s_i = np.prod(s[:i+1])
        k_i = k[i] + (k[i] - 1)*(d[i] - 1)
        rf_i = reg + (k_i - 1) * s_i    # rf_i+1 = reg + (k_i - 1) * s_i
        reg = rf_i 
        
    print(rf_i)