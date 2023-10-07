import torch
import torch.nn as nn
import numpy as np
from skimage.metrics import structural_similarity
from skimage.transform import resize
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity
from torchmetrics.image.fid import FrechetInceptionDistance
from collections import OrderedDict
from .utils import norm_min_max
from PIL import Image
import imagehash
import matplotlib.pyplot as plt

def SNR(original:np.ndarray, generated):
    snr = 0
    for c in range(ch := original.shape[0]):
        noise = original[c] - generated[c]
        snr += 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr / ch

def PSNR(original:np.ndarray, generated:np.ndarray, norm=False, aclr_factor=1):
    if norm:
        original = np.array(norm_min_max(original, norm_type='self')[0]*255).astype(np.uint8)
        generated = np.array(norm_min_max(generated, norm_type='self')[0]*255).astype(np.uint8)
    
    assert original.shape == generated.shape
    
    generated = generated * aclr_factor
    
    psnr = 0
    for c in range(ch := original.shape[0]):
        ori = original[c]
        gen = generated[c]
        mse = np.mean((ori - gen) ** 2)
        if(mse == 0): 
            raise Exception("PSNR MSE is 0. This means two inputs are the same!")
        max_pixel = np.max(ori)
        psnr += 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr / ch

def SSIM(mat1:np.ndarray, mat2:np.ndarray, norm=True):
    """
        :mat1, mat2 paras: np.ndarray, shape=(ch, h, w)
        :norm para: whether to normalize the input matrix to [0, 255]
    """
    assert len(mat1.shape) == len(mat2.shape) == 3
    
    if norm:
        mat1 = np.array(norm_min_max(mat1, norm_type='self')[0]*255).astype(np.uint8)
        mat2 = np.array(norm_min_max(mat2, norm_type='self')[0]*255).astype(np.uint8)
    else:
        mat1 = mat1.astype(np.uint8)
        mat2 = mat2.astype(np.uint8)
    
    ssim = 0
    for c in range(ch := mat1.shape[0]):
        ssim += structural_similarity(mat1[c], mat2[c])
    return ssim / ch

def FID(real:np.ndarray, fake:np.ndarray, norm=True):
    """
        :real, fake para: np.ndarray, shape=(n, 1, h, w), should be a batch(n>1) of pictures
    """
    assert len(real.shape) == len(fake.shape) == 4
    if norm:
        for i in range(real.shape[0]):
            real[i] = np.array(norm_min_max(real[i], norm_type='self')[0]*255).astype(np.uint8)
            fake[i] = np.array(norm_min_max(fake[i], norm_type='self')[0]*255).astype(np.uint8)
        
    real = real.astype(np.uint8)
    fake = fake.astype(np.uint8)
    
    real_3c = torch.tensor(np.concatenate((real, real, real), axis=1))  # size[patch_num, 3, 299, 299]
    fake_3c = torch.tensor(np.concatenate((fake, fake, fake), axis=1))
    
    fid = FrechetInceptionDistance(feature=64)  
    fid.update(real_3c, real=True)
    fid.update(fake_3c, real=False)
    fid_score = float(fid.compute())   # error occurs when batch == 1
    return fid_score

def LPIPS(real:np.ndarray, fake:np.ndarray, norm=True):
    """
        :real, fake para: np.ndarray, shape=(n, 1, h, w), should be a batch(n>1) of pictures
    """
    assert len(real.shape) == len(fake.shape) == 4
    
    real = real.astype(np.float32)
    fake = fake.astype(np.float32)
    
    if norm:
        for i in range(real.shape[0]):
            real[i] = np.array(norm_min_max(real[i], norm_type='self')[0]).astype(np.float32)
            fake[i] = np.array(norm_min_max(fake[i], norm_type='self')[0]).astype(np.float32)
    
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    real_3c = torch.tensor(np.concatenate((real, real, real), axis=1))
    fake_3c = torch.tensor(np.concatenate((fake, fake, fake), axis=1))
    print(real_3c.shape, real_3c.dtype)
    
    lpips_score = float(lpips(real_3c, fake_3c))
    return lpips_score

def perceptive_hash(mat1:np.ndarray, mat2:np.ndarray, norm=True):
    """
        :norm para: whether to normalize the input matrix to [0, 255]
    """
    assert mat1.shape == mat2.shape
    assert len(mat1.shape) == 3

    if norm:
        mat1 = np.array(norm_min_max(mat1, norm_type='self')[0]*255).astype(np.uint8)
        mat2 = np.array(norm_min_max(mat2, norm_type='self')[0]*255).astype(np.uint8)
    
    diff = 0
    for c in range(ch := mat1.shape[0]):
        m1 = resize(mat1[c], (32, 32), mode='reflect', anti_aliasing=True)
        m2 = resize(mat2[c], (32, 32), mode='reflect', anti_aliasing=True)

        m1 = Image.fromarray((m1 * 255).astype(np.uint8))
        m2 = Image.fromarray((m2 * 255).astype(np.uint8))
        
        hash1 = imagehash.phash(m1)
        hash2 = imagehash.phash(m2)
        
        diff += hash1 - hash2
        
    return diff / ch