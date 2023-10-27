"""
An Implementation from https://github.com/HuCaoFighting/Swin-Unet/tree/main
This network is modified and removed its segmentation head.
Model parameters: 27,168,036
"""

import os
import copy
import yaml
import subprocess
import ml_collections

import torch
import torch.nn as nn

from ..networks_.swin_unet.swin_transformer import SwinTransformer

class SwinUNet(nn.Module):
    def __init__(self, img_size=224, use_pretrain=True):
        super().__init__()
        config_ = yaml.load(open("../networks_/swin_unet/config.yaml", "r"), Loader=yaml.FullLoader)
        config = ml_collections.ConfigDict(config_)
        
        self.swin_unet = SwinTransformer(img_size=img_size,
                                patch_size=config.MODEL.SWIN.PATCH_SIZE,
                                in_chans=config.MODEL.SWIN.IN_CHANS,
                                embed_dim=config.MODEL.SWIN.EMBED_DIM,
                                depths=config.MODEL.SWIN.DEPTHS,
                                num_heads=config.MODEL.SWIN.NUM_HEADS,
                                window_size=config.MODEL.SWIN.WINDOW_SIZE,
                                mlp_ratio=config.MODEL.SWIN.MLP_RATIO,
                                qkv_bias=config.MODEL.SWIN.QKV_BIAS,
                                qk_scale=config.MODEL.SWIN.QK_SCALE,
                                drop_rate=config.MODEL.DROP_RATE,
                                drop_path_rate=config.MODEL.DROP_PATH_RATE,
                                ape=config.MODEL.SWIN.APE,
                                patch_norm=config.MODEL.SWIN.PATCH_NORM,
                                use_checkpoint=use_pretrain)

        if use_pretrain:
            model_path = config.MODEL.PRETRAIN_CKPT
            if not os.path.exists(model_path):
                # self.download_model()
                print("Predtrained model not found.")
                
            try:
                self.load_from(model_path)
            except:
                print("Predtrained model load failed.")

    def forward(self, x):
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
        x = self.swin_unet(x)
        
        return x

    def download_model(self):
        os.makedirs("../models", exist_ok=True)
        subprocess.run(["gdown https://drive.google.com/uc?id=1TyMf0_uvaxyacMmVzRfqvLLAWSOE2bJR -O ../models/swin_tiny_patch4_window7_224.pth"], shell=True)
    
    def load_from(self, model_path):
        device = torch.device(0)
        pretrained_dict = torch.load(model_path, map_location=device)
        
        if "model"  not in pretrained_dict:
            print("---start loading pretrained model by splitting---")
            pretrained_dict = {k[17:]:v for k,v in pretrained_dict.items()}
            for k in list(pretrained_dict.keys()):
                if "output" in k:
                    print("delete key:{}".format(k))
                    del pretrained_dict[k]
            msg = self.swin_unet.load_state_dict(pretrained_dict,strict=False)
            # print(msg)
            return
        
        pretrained_dict = pretrained_dict['model']
        print("---start loading pretrained model of swin encoder---")
        model_dict = self.swin_unet.state_dict()
        full_dict = copy.deepcopy(pretrained_dict)
        for k, v in pretrained_dict.items():
            if "layers." in k:
                current_layer_num = 3-int(k[7:8])
                current_k = "layers_up." + str(current_layer_num) + k[8:]
                full_dict.update({current_k:v})
        for k in list(full_dict.keys()):
            if k in model_dict:
                if full_dict[k].shape != model_dict[k].shape:
                    print("delete:{};shape pretrain:{};shape model:{}".format(k,v.shape,model_dict[k].shape))
                    del full_dict[k]