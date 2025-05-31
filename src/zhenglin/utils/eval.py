import torch
import torchvision.transforms as transforms
from torchvision.io import read_image
from torchmetrics.image import (
    LearnedPerceptualImagePatchSimilarity,
    StructuralSimilarityIndexMeasure,
    PeakSignalNoiseRatio,
    FrechetInceptionDistance
)
from torchmetrics.multimodal.clip_score import CLIPScore
import glob
import os
from PIL import Image
import numpy as np
from typing import List, Tuple, Dict, Union
import argparse
import cv2
import tempfile

from transformers import CLIPProcessor, CLIPModel
import torch.nn.functional as F

from abc import ABC, abstractmethod

try:
    from cdfvd import fvd
except ImportError:
    import subprocess
    import sys
    subprocess.check_call([sys.executable, "-m", "pip", "install", "cd-fvd"])
    from cdfvd import fvd


class Evaluator(ABC):
    def __init__(self, device="cuda"):
        self.device = device
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((512, 512)),
        ])
    
    @abstractmethod
    def eval(self, path_gt, path_pred) -> float:
        '''Implementation of evaluation method.'''
        pass

    def _load_images(self, dir_image: Union[str, List[str]]) -> List[torch.Tensor]:
        """Load and preprocess multiple images from a folder or list of paths."""
        if isinstance(dir_image, str):
            # If it's a directory, get all image files
            if os.path.isdir(dir_image):
                img_paths = glob.glob(os.path.join(dir_image, "*.png")) + \
                           glob.glob(os.path.join(dir_image, "*.jpg")) + \
                           glob.glob(os.path.join(dir_image, "*.jpeg"))
            else:
                # If it's a comma-separated string
                img_paths = dir_image.split(',')
        else:
            # If it's already a list
            img_paths = dir_image
        
        images = []
        for path in sorted(img_paths):
            try:
                img = Image.open(path).convert('RGB')
                img_tensor = self.transform(img).to(self.device)
                images.append(img_tensor)
            except Exception as e:
                print(f"Error loading image {path}: {e}")
        
        return images


class LPIPSEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.lpips = LearnedPerceptualImagePatchSimilarity(
            net_type='alex', normalize=True).to(device)

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        scores = []
        for gen_img, ref_img in zip(img_pred, img_gt):
            gen_normalized = gen_img * 0.999999  # LPIPS expects images in range [0, 1)
            ref_normalized = ref_img * 0.999999
            score = self.lpips(gen_normalized.unsqueeze(0), ref_normalized.unsqueeze(0))
            scores.append(score.item())
        
        return sum(scores) / len(scores) if scores else float('nan')


class SSIMEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.ssim = StructuralSimilarityIndexMeasure(data_range=1.0).to(device)

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        scores = []
        for gen_img, ref_img in zip(img_pred, img_gt):
            score = self.ssim(gen_img.unsqueeze(0), ref_img.unsqueeze(0))
            scores.append(score.item())
        
        return sum(scores) / len(scores) if scores else float('nan')


class PSNREvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.psnr = PeakSignalNoiseRatio().to(device)

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        scores = []
        for gen_img, ref_img in zip(img_pred, img_gt):
            score = self.psnr(gen_img.unsqueeze(0), ref_img.unsqueeze(0))
            scores.append(score.item())
        
        return sum(scores) / len(scores) if scores else float('nan')


class FIDEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.fid = FrechetInceptionDistance(feature=64).to(device)
        # Override transform for FID (requires 299x299 for Inception)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
        ])

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        self.fid.reset()
        
        # Add real images to FID
        self.fid.update((torch.stack(img_gt) * 255).type(torch.uint8), real=True)
        
        # Add generated images to FID
        self.fid.update((torch.stack(img_pred) * 255).type(torch.uint8), real=False)
        
        return self.fid.compute().item()


class FramewiseFIDEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.fid = FrechetInceptionDistance(feature=64).to(device)
        # Override transform for FID (requires 299x299 for Inception)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize((299, 299)),
        ])

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        fid_all = []
        for gen_img, ref_img in zip(img_pred, img_gt):
            self.fid.reset()
            
            self.fid.update((torch.stack([gen_img] * 2) * 255).type(torch.uint8), real=True)
            self.fid.update((torch.stack([ref_img] * 2) * 255).type(torch.uint8), real=False)
            
            fid_score = self.fid.compute().item()
            fid_all.append(fid_score)

        return np.mean(fid_all)


class CLIPScoreEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.clip_score = CLIPScore(model_name_or_path="openai/clip-vit-base-patch16").to(device)

    def eval(self, path_pred, text_prompt: str = None) -> float:
        if text_prompt is None:
            return float('nan')
            
        img_pred = self._load_images(path_pred)
        
        if not img_pred:
            return float('nan')
        
        scores = []
        for gen_img in img_pred:
            score = self.clip_score(gen_img.unsqueeze(0), text_prompt)
            scores.append(score.item())
        
        return sum(scores) / len(scores) if scores else float('nan')


class CLIPSimilarityEvaluator(Evaluator):
    def __init__(self, device="cuda"):
        super().__init__(device)
        self.clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16").to(device)
        self.clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch16")

    def eval(self, path_gt, path_pred) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        sims = []
        for gen_img, ref_img in zip(img_pred, img_gt):
            # Convert tensor to PIL image for processor
            gen_pil = transforms.ToPILImage()(gen_img.cpu())
            ref_pil = transforms.ToPILImage()(ref_img.cpu())

            inputs = self.clip_processor(images=[gen_pil, ref_pil], return_tensors="pt").to(self.device)

            with torch.no_grad():
                embeddings = self.clip_model.get_image_features(**inputs)

            # Normalize and compute cosine similarity
            emb_gen = F.normalize(embeddings[0], dim=0)
            emb_ref = F.normalize(embeddings[1], dim=0)
            similarity = F.cosine_similarity(emb_gen, emb_ref, dim=0).item()
            sims.append(similarity)

        return sum(sims) / len(sims) if sims else float('nan')


class FVDEvaluator(Evaluator):
    '''credit: https://github.com/JunyaoHu/common_metrics_on_video_quality/tree/main/fvd
    '''
    def __init__(self, device="cuda", method='videogpt'):
        super().__init__(device)
        self.method = method

    def eval(self, path_gt, path_pred, only_final: bool = True) -> float:
        img_gt = self._load_images(path_gt)
        img_pred = self._load_images(path_pred)
        
        if not img_gt or not img_pred:
            return float('nan')
        
        return self._calculate_fvd(img_pred, img_gt, only_final)

    def _calculate_fvd(self, generated_images: List[torch.Tensor], 
                      reference_images: List[torch.Tensor], 
                      only_final: bool = True) -> float:
        """
        Compute FVD using a selected method ('videogpt' or 'styleganv') from lists of frames (C, H, W).
        Assumes each list represents a single video.
        """
        from fvd.styleganv.fvd import get_fvd_feats as get_fvd_feats_styleganv, frechet_distance as fd_styleganv, load_i3d_pretrained as load_styleganv
        from fvd.videogpt.fvd import get_fvd_logits as get_fvd_feats_videogpt, frechet_distance as fd_videogpt, load_i3d_pretrained as load_videogpt
        from tqdm import tqdm

        def trans(x):
            # if grayscale, repeat channel
            if x.shape[-3] == 1:
                x = x.repeat(1, 1, 3, 1, 1)
            # permute BTCHW -> BCTHW
            return x.permute(0, 2, 1, 3, 4)

        # Prepare video tensors: [1, T, C, H, W]
        gen_video = torch.stack(generated_images).to(self.device)  # (T, C, H, W)
        ref_video = torch.stack(reference_images).to(self.device)

        # reshape ref_video to gen_video shape
        if ref_video.shape[2:] != gen_video.shape[2:]:
            gen_video = F.interpolate(gen_video, size=ref_video.shape[2:], mode='bilinear', align_corners=False)

        gen_video = gen_video.unsqueeze(0)  # (1, T, C, H, W)
        ref_video = ref_video.unsqueeze(0)

        B, T, C, H, W = gen_video.shape

        if T <= 7:
            print('WARNING: Video length shorter than 7! Duplicating longer...')
            gen_video = gen_video.repeat(1, int(np.ceil(8 / T)), 1, 1, 1)  # (1, T', C, H, W)
            ref_video = ref_video.repeat(1, int(np.ceil(8 / T)), 1, 1, 1)  # (1, T', C, H, W)
            # gen_video = torch.concat([gen_video, gen_video], dim=1)  # (1, 2T, C, H, W)
            # ref_video = torch.concat([ref_video, ref_video], dim=1)  # (1, 2T, C, H, W)

        assert gen_video.shape == ref_video.shape, "Generated and reference videos must have the same shape"

        # Select method
        if self.method == 'videogpt':
            load_i3d = load_videogpt
            get_feats = get_fvd_feats_videogpt
            frechet = fd_videogpt
        elif self.method == 'styleganv':
            load_i3d = load_styleganv
            get_feats = get_fvd_feats_styleganv
            frechet = fd_styleganv
        else:
            raise ValueError(f"Unsupported method: {self.method}")

        # Load model
        i3d = load_i3d(device=self.device)

        # Format input to BCTHW
        gen_video = trans(gen_video)
        ref_video = trans(ref_video)

        if only_final:
            feats1 = get_feats(gen_video, i3d=i3d, device=self.device)
            feats2 = get_feats(ref_video, i3d=i3d, device=self.device)
            fvd_value = frechet(feats1, feats2)
        else:
            fvd_list = []
            for clip_ts in tqdm(range(10, gen_video.shape[2] + 1)):
                feats1 = get_feats(gen_video[:, :, :clip_ts], i3d=i3d, device=self.device)
                feats2 = get_feats(ref_video[:, :, :clip_ts], i3d=i3d, device=self.device)
                fvd_list.append(frechet(feats1, feats2))
            fvd_value = float(np.mean(fvd_list))

        return float(fvd_value)

class UnifiedEvaluator:
    def __init__(self, 
                 eval_ssim=True, 
                 eval_psnr=True, 
                 eval_fid=True, 
                 eval_lpips=True,
                 eval_clip_sim=True, 
                 eval_clip_score=False,
                 eval_fvd=False, 
                 device="cuda"):
        
        self.eval_ssim = eval_ssim
        self.eval_psnr = eval_psnr
        self.eval_fid = eval_fid
        self.eval_lpips = eval_lpips
        self.eval_clip_sim = eval_clip_sim
        self.eval_clip_score = eval_clip_score
        self.eval_fvd = eval_fvd
        self.device = device

        self.eval_result = dict()

    def eval(self, path_gt, path_pred):
        """Evaluate all metrics based on the flags set during initialization."""
        if self.eval_ssim:
            print("Evaluating SSIM...")
            ssim_evaluator = SSIMEvaluator()
            self.eval_result['ssim'] = ssim_evaluator.eval(path_gt, path_pred)
        if self.eval_psnr:
            print("Evaluating PSNR...")
            psnr_evaluator = PSNREvaluator()
            self.eval_result['psnr'] = psnr_evaluator.eval(path_gt, path_pred)
        if self.eval_fid:
            print("Evaluating FID...")
            fid_evaluator = FIDEvaluator()
            self.eval_result['fid'] = fid_evaluator.eval(path_gt, path_pred)
        if self.eval_lpips:
            print("Evaluating LPIPS...")
            lpips_evaluator = LPIPSEvaluator()
            self.eval_result['lpips'] = lpips_evaluator.eval(path_gt, path_pred)
        if self.eval_clip_sim:
            print("Evaluating CLIP Similarity...")
            clip_sim_evaluator = CLIPSimilarityEvaluator()
            self.eval_result['clip_sim'] = clip_sim_evaluator.eval(path_gt, path_pred)
        if self.eval_clip_score:
            print("Evaluating CLIP Score...")
            clip_score_evaluator = CLIPScoreEvaluator()
            self.eval_result['clip_score'] = clip_score_evaluator.eval(path_pred, text_prompt="")
        if self.eval_fvd:
            print("Evaluating FVD...")
            fvd_evaluator = FVDEvaluator(method='videogpt')
            self.eval_result['fvd'] = fvd_evaluator.eval(path_gt, path_pred, only_final=True)
        
        return self.eval_result


# Example usage with the refactored classes
if __name__ == "__main__":
    dir_gt = r'C:\Users\Lenovo\Desktop\001-POT01_001_0036_37-PAINT-B\gt'
    dir_pred = r'C:\Users\Lenovo\Desktop\001-POT01_001_0036_37-PAINT-B\out'

    # fid_evaluator = FIDEvaluator()
    # fid_score = fid_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"FID Score: {fid_score}")
    
    # lpips_evaluator = LPIPSEvaluator()
    # lpips_score = lpips_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"LPIPS Score: {lpips_score}")
    
    # ssim_evaluator = SSIMEvaluator()
    # ssim_score = ssim_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"SSIM Score: {ssim_score}")
    
    # psnr_evaluator = PSNREvaluator()
    # psnr_score = psnr_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"PSNR Score: {psnr_score}")
    
    # clip_sim_evaluator = CLIPSimilarityEvaluator()
    # clip_sim_score = clip_sim_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"CLIP Similarity Score: {clip_sim_score}")
    
    # For CLIP Score with text prompt
    # clip_score_evaluator = CLIPScoreEvaluator()
    # clip_score = clip_score_evaluator.eval(path_pred=dir_pred, text_prompt="A beautiful landscape")
    # print(f"CLIP Score: {clip_score}")
    
    # # For FVD
    # fvd_evaluator = FVDEvaluator(method='videogpt')
    # fvd_score = fvd_evaluator.eval(path_gt=dir_gt, path_pred=dir_pred)
    # print(f"FVD Score: {fvd_score}")

    # Unified evaluation
    # evaluator = UnifiedEvaluator()
    # results = evaluator.eval(dir_gt, dir_pred)