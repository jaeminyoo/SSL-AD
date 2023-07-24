"""
https://github.com/sthalles/SimCLR/blob/master/data_aug/contrastive_learning_dataset.py
"""

from torch import nn
from torchvision.transforms import transforms


class SimCLR(nn.Module):
    def __init__(self, size, channels, s=1):
        super().__init__()
        color_jitter = transforms.ColorJitter(0.8 * s, 0.8 * s, 0.8 * s, 0.2 * s)
        if channels == 3:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.GaussianBlur(kernel_size=2 * (int(0.1 * size) // 2) + 1)
            ])
        else:
            self.transforms = transforms.Compose([
                transforms.RandomResizedCrop(size=size),
                transforms.RandomHorizontalFlip(),
                transforms.RandomApply([color_jitter], p=0.8),
                transforms.GaussianBlur(kernel_size=2 * (int(0.1 * size) // 2) + 1)
            ])

    def forward(self, img):
        return self.transforms(img)
