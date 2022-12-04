import torch
from torch.utils.data import Dataset

class CustomDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None):
        self.data = torch.Tensor(data)
        self.targets = torch.Tensor(targets)
        self.transform = transform
        self.target_transform=target_transform

        if len(self.targets.shape) == 1:
            self.targets = self.targets.unsqueeze(1)

    def __getitem__(self, idx):
        x = self.data[idx]
        y = self.targets[idx]

        if self.transform:
            x = self.transform(x)

        if self.target_transform:
            y = self.target_transform(y)

        return x, y
    
    def __len__(self):
        return len(self.data)