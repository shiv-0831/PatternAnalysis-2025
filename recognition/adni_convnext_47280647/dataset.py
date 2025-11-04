from typing import Tuple
import torch
from torch.utils.data import Dataset, DataLoader, random_split

class RandomSliceDataset(Dataset):
    """Deterministic random 'images' for smoke testing the pipeline."""
    def __init__(self, n: int = 192, image_size: Tuple[int, int, int] = (1, 224, 224), num_classes: int = 2, seed: int = 42):
        super().__init__()
        g = torch.Generator().manual_seed(seed)
        self.x = torch.rand((n, *image_size), generator=g)
        self.y = torch.randint(low=0, high=num_classes, size=(n,), generator=g)

    def __len__(self):
        return self.x.shape[0]

    def __getitem__(self, i):
        return self.x[i], int(self.y[i])

def build_loaders(batch_size: int = 16, seed: int = 42):
    full = RandomSliceDataset(n=192, seed=seed)
    val_size = 64
    train_size = len(full) - val_size
    train_set, val_set = random_split(full, [train_size, val_size], generator=torch.Generator().manual_seed(seed))
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_set,   batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader
