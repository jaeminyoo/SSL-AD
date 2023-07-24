import numpy as np
from torch import nn
from torchvision import transforms
from torchvision.transforms import functional

from augment import UniformRotation


class GEOM(nn.Module):
    def __init__(self, size):
        super().__init__()
        self.flip = transforms.RandomHorizontalFlip()
        self.translation = 0.25 * size
        self.rotate = UniformRotation()

    def forward(self, img):
        img = self.flip(img)
        tx = np.random.choice([0, -self.translation, self.translation])
        ty = np.random.choice([0, -self.translation, self.translation])
        img = functional.affine(img, 0, [tx, ty], 1, 0, fill=0)
        return self.rotate(img)
