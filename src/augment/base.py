import numpy as np
import torch
import torchvision.transforms.functional as functional
from torch import nn


class Identity(nn.Module):
    def __init__(self):
        super().__init__()

    @staticmethod
    def forward(x):
        return x


class OutlierExposure(nn.Module):
    def __init__(self, data, known_anomaly, obs_ratio=0.1):
        super().__init__()
        self.data = data
        candidates = torch.nonzero(data.targets == known_anomaly).view(-1)
        idx = torch.randperm(len(candidates))[:int(len(candidates) * obs_ratio)]
        self.candidates = candidates[idx]

    # noinspection PyUnusedLocal
    def forward(self, x):
        idx = np.random.randint(len(self.candidates))
        return self.data[self.candidates[idx]][0]


class UniformRotation(nn.Module):
    def __init__(self, angles=(-90, 0, 90, 180)):
        super().__init__()
        self.angles = angles

    def __call__(self, img):
        angle = self.angles[np.random.randint(len(self.angles))]
        return functional.rotate(img, angle)


class RandomInversion(nn.Module):
    def __init__(self, p=0.5):
        super().__init__()
        self.p = p

    def __call__(self, img):
        if np.random.rand() < self.p:
            img = functional.invert(img)
        return img


class RandomNoise(nn.Module):
    def __init__(self, level=0.1):
        super().__init__()
        self.std = level

    def __call__(self, img):
        return img + self.std * torch.randn_like(img)


class RandomMasking(nn.Module):
    def __init__(self, p=0.2, value=0):
        super().__init__()
        self.p = p
        self.value = value

    def __call__(self, img):
        img = img.clone()
        img[torch.rand_like(img) < self.p] = self.value
        return img
