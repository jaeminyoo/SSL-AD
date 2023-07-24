import numpy as np
import torch
from torch import nn
from torchvision.transforms import transforms


def to_patch_position(img_h, img_w, patch_h, patch_w):
    patch_i = np.random.randint(img_h - patch_h)
    patch_j = np.random.randint(img_w - patch_w)
    return patch_i, patch_j


class CutPaste(nn.Module):
    def __init__(self, color_jitter=0.1):
        super().__init__()
        self.area_range = (0.02, 0.15)
        self.aspect_ranges = ((0.3, 1.), (1., 3.3))
        self.transform = transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter
        )

    def to_patch_size(self, img):
        img_area = img.shape[-2] * img.shape[-1]
        patch_area = np.random.uniform(*self.area_range) * img_area
        patch_aspect = np.random.choice([
            np.random.uniform(*self.aspect_ranges[0]),
            np.random.uniform(*self.aspect_ranges[1])
        ])
        patch_h = int(np.sqrt(patch_area / patch_aspect))
        patch_w = int(np.sqrt(patch_area * patch_aspect))
        return patch_h, patch_w

    def forward(self, img):
        assert isinstance(img, torch.Tensor)
        img_h, img_w = img.shape[-2:]
        h, w = self.to_patch_size(img)
        i, j = to_patch_position(img_h, img_w, h, w)
        patch = self.transform(img[..., i:i + h, j:j + w])

        i, j = to_patch_position(img_h, img_w, h, w)
        img = img.clone()
        img[..., i:i + h, j:j + w] = patch
        return img


class CutPasteScar(nn.Module):
    def __init__(self, color_jitter=0.1):
        super().__init__()
        self.height_range = (10, 25)
        self.width_range = (2, 16)
        self.rotate = transforms.RandomRotation(45, expand=True, fill=-1)
        self.transform = transforms.ColorJitter(
            brightness=color_jitter,
            contrast=color_jitter,
            saturation=color_jitter,
            hue=color_jitter
        )

    def to_patch_size(self):
        patch_h = np.random.randint(*self.height_range)
        patch_w = np.random.randint(*self.width_range)
        return patch_h, patch_w

    def forward(self, img):
        assert isinstance(img, torch.Tensor)
        img_h, img_w = img.shape[-2:]
        h, w = self.to_patch_size()
        i, j = to_patch_position(img_h, img_w, h, w)
        patch = self.rotate(self.transform(img[..., i:i + h, j:j + w]))
        patch_mask = patch >= 0
        h, w = patch.shape[-2:]
        i, j = to_patch_position(img_h, img_w, h, w)
        img = img.clone()
        img[..., i:i + h, j:j + w][patch_mask] = patch[patch_mask]
        return img
