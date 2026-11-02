import random
import numpy as np
import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Subset
from config import SEED, CIFAR_MEAN, CIFAR_STD

class Cutout:
    def __init__(self, n_holes=1, length=16):
        self.n_holes, self.length = n_holes, length

    def __call__(self, img):
        h, w = img.size(1), img.size(2)
        mask = np.ones((h, w), dtype=np.float32)
        for _ in range(self.n_holes):
            y, x = np.random.randint(h), np.random.randint(w)
            y1, y2 = np.clip(y-self.length//2, 0, h), np.clip(y+self.length//2, 0, h)
            x1, x2 = np.clip(x-self.length//2, 0, w), np.clip(x+self.length//2, 0, w)
            mask[y1:y2, x1:x2] = 0.0
        return img * torch.from_numpy(mask).expand_as(img)

def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def get_cifar10(batch_size):
    train_tf = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
        Cutout(n_holes=1, length=16),
    ])
    eval_tf = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(CIFAR_MEAN, CIFAR_STD),
    ])
    train_full = datasets.CIFAR10("./data", train=True, transform=train_tf, download=True)
    val_full = datasets.CIFAR10("./data", train=True, transform=eval_tf, download=True)
    test_data = datasets.CIFAR10("./data", train=False, transform=eval_tf, download=True)
    indices = torch.randperm(50000, generator=torch.Generator().manual_seed(SEED))
    train_data = Subset(train_full, indices[:45000].tolist())
    val_data = Subset(val_full, indices[45000:].tolist())
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False, num_workers=0)
    return train_loader, val_loader, test_loader
