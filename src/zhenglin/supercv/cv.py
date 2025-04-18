import cv2
import numpy as np
from PIL import Image
import torch

IMG_EXT = ('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.webp', 'tga')

class cv:
    def __init__(self, data):
        if isinstance(data, np.ndarray):
            self.data = data
        elif isinstance(data, Image.Image):
            self.data = np.array(data)
        elif isinstance(data, torch.Tensor):
            self.data = data.squeeze(0).permute(1, 2, 0).numpy()
        else:
            raise TypeError("Unsupported data type.")

    @staticmethod
    def imread(path_in, img_type='cv2', func=None, **kwargs):
        img = Image.open(path_in)
        if img_type == 'cv2':
            data = np.array(img)
        elif img_type == 'pil':
            data = img
        elif img_type == 'tensor':
            data = torch.from_numpy(np.array(img)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        else:
            raise ValueError(f"Unsupported img_type: {img_type}")

        if func is not None:
            data = func(data, **kwargs)
        return cv(data)

    @staticmethod
    def imwrite(obj, path_out, **kwargs):
        assert isinstance(obj, (cv, np.ndarray, Image.Image, torch.Tensor)), "Input must be cv object, numpy array, PIL image or torch tensor."
        if isinstance(obj, cv):
            obj = obj.data

        if isinstance(obj, np.ndarray):
            Image.fromarray(obj).save(path_out, **kwargs)
        elif isinstance(obj, Image.Image):
            obj.save(path_out, **kwargs)
        elif isinstance(obj, torch.Tensor):
            from torchvision.utils import save_image
            save_image(obj, path_out, **kwargs)
        return obj  # Return the same cv object

    @staticmethod
    def resize(obj, h, w, nearest=False):
        if not isinstance(obj, cv):
            raise TypeError("resize expects a cv object.")

        data = obj.data
        if isinstance(data, np.ndarray):
            interp = cv2.INTER_NEAREST if nearest else cv2.INTER_LINEAR
            resized = cv2.resize(data, dsize=(w, h), interpolation=interp)
        elif isinstance(data, Image.Image):
            interp = Image.NEAREST if nearest else Image.BILINEAR
            resized = data.resize((w, h), resample=interp)
        elif isinstance(data, torch.Tensor):
            interp = 'nearest' if nearest else 'bilinear'
            resized = torch.nn.functional.interpolate(data, size=(h, w), mode=interp, align_corners=False)
        else:
            raise TypeError("Unsupported data type for resize.")

        return cv(resized)

    def __call__(self, data):
        assert isinstance(data, (cv, np.ndarray, Image.Image, torch.Tensor)), "Unsupported type passed to cv instance."
        self.data = data
        return self

    @property
    def cv2(self):
        if isinstance(self.data, np.ndarray):
            return self.data
        elif isinstance(self.data, Image.Image):
            return np.array(self.data)
        elif isinstance(self.data, torch.Tensor):
            return self.data.squeeze(0).permute(1, 2, 0).numpy()

    @property
    def pil(self):
        if isinstance(self.data, np.ndarray):
            return Image.fromarray(self.data)
        elif isinstance(self.data, Image.Image):
            return self.data
        elif isinstance(self.data, torch.Tensor):
            return Image.fromarray(self.data.squeeze(0).permute(1, 2, 0).numpy())

    @property
    def tensor(self):
        if isinstance(self.data, np.ndarray):
            return torch.from_numpy(self.data).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif isinstance(self.data, Image.Image):
            return torch.from_numpy(np.array(self.data)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif isinstance(self.data, torch.Tensor):
            return self.data

    @property
    def shape(self):
        return self.data.shape

    @property
    def gray(self):
        if isinstance(self.data, np.ndarray):
            return cv2.cvtColor(self.data, cv2.COLOR_RGB2GRAY)
        elif isinstance(self.data, Image.Image):
            return self.data.convert("L")
        elif isinstance(self.data, torch.Tensor):
            return self.data.mean(dim=1, keepdim=True)

    def to(self, device):
        if isinstance(self.data, np.ndarray):
            self.data = torch.from_numpy(self.data).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        elif isinstance(self.data, Image.Image):
            self.data = torch.from_numpy(np.array(self.data)).permute(2, 0, 1).unsqueeze(0).float() / 255.0
        self.data = self.data.to(device)
        
        return self.data

if __name__=="__main__":
    import cv2

    # example usages
    img = cv.imread("img.png", 'pil')

    cv.imwrite(img.gray, "test.jpg")
    
    img = cv.resize(img, h=512, w=512, nearest=True)
    
    img = cv(cv2.resize(img.cv2, (512, 512), interpolation=cv2.INTER_NEAREST))

    