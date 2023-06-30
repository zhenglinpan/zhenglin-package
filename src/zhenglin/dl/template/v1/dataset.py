from torch.utils.data import Dataset

class MyDataset(Dataset):
    def __init__(self, args):
        self.args = args
        self.files = []
        
    def __getitem__(self, index):
        pass
    
    def __init__(self):
        return len(self.files)