from torch.utils.data import Dataset
from torchvision.transforms import transforms

class MyDataset(Dataset):
    def __init__(self, trans=None, *args):
        self.trans = transforms.Compose(trans)
        self.data = []
        
    def __getitem__(self, index):
        data = self.trans(self.data[index]) # data: ndarray ([C, H, W])
    
    def __init__(self):
        return len(self.files)