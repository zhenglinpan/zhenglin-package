"""
Implementation from: https://github.com/Beckschen/TransUNet/tree/main
Notations: https://aidenpan.notion.site/TranUNet-Structure-e3567c612b4e4779b6c607eb91aa2b99?pvs=4
Model parameters: 105,276,066
"""

import os
import numpy as np
from ..networks_.trans_unet.config import CONFIGS
from ..networks_.trans_unet.network import VisionTransformer

class TransUNet():
    def __new__(cls, size_input=224, n_skip=3, vit_name='R50-ViT-B_16', vit_patches_size=16):

        config = CONFIGS[vit_name]
        config.n_skip = n_skip
        
        # if use ResNet embedding setup
        if 'R50' in vit_name:
            config.patches.grid = (int(size_input / vit_patches_size), int(size_input / vit_patches_size))
        
        ### Build model
        net = VisionTransformer(config, img_size=size_input).cuda()
        
        ### Check pretrained weights
        if not os.path.exists(config.pretrained_path):
            cls._download_pretrained_weights(vit_name)
        
        ### Load pretrained weights
        try:
            net.load_from(weights=np.load(config.pretrained_path))
        except:
            print("Error loading pretrained weights.")
        
        return net
    
    def _download_pretrained_weights(vit_name):
        import urllib.request
        from tqdm import tqdm
        
        model_dir = '../models/vit_checkpoint/imagenet21k'
        os.makedirs(model_dir, exist_ok=True)
        url_dict = {"ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_16.npz",
                    "ViT-B_32": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-B_32.npz",
                    "ViT-L_16": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_16.npz",
                    "ViT-L_32": "https://storage.googleapis.com/vit_models/imagenet21k/ViT-L_32.npz",
                    "R50-ViT-B_16": "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-B_16.npz",
                    "R50-ViT-L_16": "https://storage.googleapis.com/vit_models/imagenet21k/R50+ViT-L_16.npz",}

        url = url_dict[vit_name]
        filename = os.path.join(model_dir, os.path.basename(url))
        if not os.path.exists(filename):
            print(f"Downloading {url} to {filename}")
            with tqdm(unit='B', unit_scale=True, unit_divisor=1024, miniters=1, desc=url.split('/')[-1]) as t:
                urllib.request.urlretrieve(url, filename, reporthook=lambda x, y, z: t.update(y))
        else:
            print(f"{filename} already exists.")
