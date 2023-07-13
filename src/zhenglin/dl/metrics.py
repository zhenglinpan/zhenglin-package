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

def SNR(original:np.ndarray, generated):
    snr = 0
    ch = original.shape[0]
    for c in range(ch):
        noise = original[c] - generated[c]
        snr += 10 * np.log10(np.sum(original**2) / np.sum(noise**2))
    return snr / ch

def PSNR(original:np.ndarray, generated:np.ndarray):
    original = np.array(norm_min_max(original, norm_type='self')[0]*255).astype(np.uint8)
    generated = np.array(norm_min_max(generated, norm_type='self')[0]*255).astype(np.uint8)
    if original.shape != generated.shape:
        raise Exception("Two inputs different in shapes.")
    psnr = 0
    ch = original.shape[0]
    for c in range(ch):
        ori = original[c]
        gen = generated[c]
        mse = np.mean((ori - gen) ** 2)
        if(mse == 0): 
            raise Exception("PSNR MSE is 0. This means two inputs are the same!")
        max_pixel = np.max(ori)
        psnr += 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr / ch

def SSIM(mat1:np.ndarray, mat2:np.ndarray):
    assert mat1.shape == mat2.shape
    assert len(mat1.shape) == 3
    
    mat1 = np.array(norm_min_max(mat1, norm_type='self')[0]*255).astype(np.uint8)
    mat2 = np.array(norm_min_max(mat2, norm_type='self')[0]*255).astype(np.uint8)
    
    ch = mat1.shape[0]
    if mat1.shape[0] != mat2.shape[0] or mat1.shape[1] != mat2.shape[1] or mat1.shape[2] != mat2.shape[2]:
        return 0
    ssim = 0
    for c in range(ch):
        ssim += structural_similarity(mat1[c], mat2[c])
    return ssim / ch

def FID(original:np.ndarray, generated:np.ndarray):
    original = np.array(norm_min_max(original, norm_type='self')[0]*255)
    generated = np.array(norm_min_max(generated, norm_type='self')[0]*255)

    score = 0
    ch = 2
    ori = original[:, None, ...]  # size[2, 1, 1024, 256]
    gen = generated[:, None, ...]

    ori = resize(ori, (2, 1, 299, 299)).astype(np.uint8)
    gen = resize(gen, (2, 1, 299, 299)).astype(np.uint8)

    ori_3c = torch.tensor(np.concatenate((ori, ori, ori), axis=1))  # size[2, 3, 1024, 256]
    gen_3c = torch.tensor(np.concatenate((gen, gen, gen), axis=1))
    
    fid = FrechetInceptionDistance(feature=64)  
    fid.update(ori_3c, real=True)
    fid.update(gen_3c, real=False)
    score += float(fid.compute())   # error occurs when batch == 1
    return score / ch

def LPIPS(mat1:np.ndarray, mat2:np.ndarray):
    mat1, _, _ = norm_min_max(mat1, norm_type='self')
    mat2, _, _ = norm_min_max(mat2, norm_type='self')
    mat1 = mat1 * 2 - 1
    mat2 = mat2 * 2 - 1
    
    score = 0
    ch = mat1.shape[0]
    lpips = LearnedPerceptualImagePatchSimilarity(net_type='vgg')
    for c in range(ch):
        m1 = mat1[c][None, :, :]
        m2 = mat2[c][None, :, :]
        
        m1_3c = torch.tensor(np.concatenate((m1, m1, m1), axis=0)[None, ...].astype(np.float32))
        m2_3c = torch.tensor(np.concatenate((m2, m2, m2), axis=0)[None, ...].astype(np.float32))
    
        score += float(lpips(m1_3c, m2_3c))
    return score / ch

def perceptive_hash(mat1:np.ndarray, mat2:np.ndarray):
    assert mat1.shape == mat2.shape
    assert len(mat1.shape) == 3
    
    mat1 = np.array(norm_min_max(mat1, norm_type='self')[0]*255).astype(np.uint8)
    mat2 = np.array(norm_min_max(mat2, norm_type='self')[0]*255).astype(np.uint8)
    
    ch = mat1.shape[0]
    diff = 0
    for c in range(ch):
        m1 = resize(mat1[c], (32, 32), mode='reflect', anti_aliasing=True)
        m2 = resize(mat2[c], (32, 32), mode='reflect', anti_aliasing=True)

        m1 = Image.fromarray((m1 * 255).astype(np.uint8))
        m2 = Image.fromarray((m2 * 255).astype(np.uint8))
        
        hash1 = imagehash.phash(m1)
        hash2 = imagehash.phash(m2)
        
        diff += hash1 - hash2
        
    return diff / ch