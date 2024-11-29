import cv2
import datasets
import numpy as np
import albumentations as A
from torch.utils.data import Dataset
from albumentations.pytorch import ToTensorV2

class MNIST(Dataset):
    def __init__(self):
        super().__init__()
        self.mnist = datasets.load_dataset("mnist", split="train")
        self.transform = A.Compose([A.HorizontalFlip(p=0.5), 
                                    ToTensorV2()], 
                                   additional_targets={"contour": "image"})

    def __len__(self):
        return len(self.mnist)

    def __getitem__(self, index):
        img = np.array(self.mnist[index]["image"])    # PIL image
        label = self.mnist[index]["label"]
        countour = cv2.Canny(img, 100, 200)
        
        transformed = self.transform(image=img, contour=countour)
        img = transformed["image"] / 255.
        countour = transformed["contour"] / 255.
        
        return {"image": img, "contour": countour, "label": label}
        

if __name__ == "__main__":
    from torch.utils.data import DataLoader
    from torchvision.utils import save_image
    
    dataset = MNIST()
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False, num_workers=4)
    
    for batch in dataloader:
        image = batch["image"]
        contour = batch["contour"]
        
        save_image(image, "image.png")
        save_image(contour, "contour.png")
        break