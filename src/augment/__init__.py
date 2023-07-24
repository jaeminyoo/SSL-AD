from torchvision import transforms

from augment.base import Identity, OutlierExposure, UniformRotation, \
    RandomMasking, RandomNoise, RandomInversion
from augment.cutpaste import CutPaste, CutPasteScar
from augment.geom import GEOM
from augment.simclr import SimCLR

FUNCTIONS = [
    'none',
    'mask',
    'noise',
    'rotate',
    'flip',
    'invert',
    'crop',
    'color',
    'blur',
    'geom',
    'simclr',
    'cutout',
    'cutpaste',
    'cp-scar'
]


def with_parameters(name):
    if name.startswith('cutout'):
        scale = float(name.split('-')[1])
        return transforms.RandomErasing(p=1.0, scale=(scale, scale),
                                        ratio=(1.0, 1.0))
    elif name.startswith('color'):
        scale = float(name.split('-')[1])
        return transforms.ColorJitter(brightness=0.8 * scale,
                                      contrast=0.8 * scale,
                                      saturation=0.8 * scale,
                                      hue=0.2 * scale)
    elif name.startswith('mask'):
        prob = float(name.split('-')[1])
        return RandomMasking(p=prob)
    else:
        raise ValueError(name)


def get_augment(name, size, channels=3, deterministic=False, **kwargs):
    if name is None or name.lower() == 'none':
        return Identity()
    elif name == 'mask':
        return RandomMasking(p=0.1)
    elif name == 'noise':
        return RandomNoise()
    elif name == 'rotate':
        return UniformRotation()
    elif name == 'crop':
        return transforms.RandomResizedCrop(size=size)
    elif name == 'color':  # From SimCLR
        return transforms.ColorJitter(0.8, 0.8, 0.8, 0.2)
    elif name == 'blur':  # From SimCLR
        return transforms.GaussianBlur(kernel_size=2 * (int(0.1 * size) // 2) + 1)
    elif name == 'geom':
        return GEOM(size)
    elif name == 'simclr':
        return SimCLR(size, channels)
    elif name == 'cutout':
        return transforms.RandomErasing(p=1.0)
    elif name == 'cutpaste':
        return CutPaste()
    elif name == 'cp-scar':
        return CutPasteScar()
    elif name == 'invert':  # For synthetic data
        return RandomInversion(p=1.0 if deterministic else 0.5)
    elif name == 'flip':  # For synthetic data
        return transforms.RandomVerticalFlip(p=1.0 if deterministic else 0.5)
    elif name.startswith('oe'):
        known_anomaly = int(name.split('-')[1])
        return OutlierExposure(kwargs['data'], known_anomaly)
    else:
        return with_parameters(name)


def get_category(name):
    if name in ['none']:
        return 0
    elif name in ['rotate', 'flip', 'geom', 'crop']:
        return 1
    elif name in ['mask', 'noise', 'blur']:
        return 2
    elif name in ['invert', 'color']:
        return 3
    elif name in ['cutout', 'cutpaste', 'cp-scar']:
        return 4
    elif name in ['simclr']:
        return 5
    else:
        raise ValueError(name)
