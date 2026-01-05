import torch
from torch.utils.data import Dataset

class FakeDataset(Dataset):
    def __init__(self, size=1024, samples=1000):
        self.size = size
        self.samples = samples

    def __len__(self):
        return self.samples

    def __getitem__(self, index):
        # Generamos ruido aleatorio pesado
        img = torch.randn((3, self.size, self.size))
        label = torch.tensor(index % 10)
        return img, label