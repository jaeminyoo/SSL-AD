import os

import numpy as np
import torch
from PIL import Image
from torchvision import transforms
from torchvision.datasets import CIFAR10, SVHN, VisionDataset
from torchvision.io import read_image

import augment


def generate_anomalies(data, targets, classes, transform, normal_class,
                       transpose=False):
    aug_functions = [
        ('flip', transforms.RandomVerticalFlip(p=1.0)),
        ('cutout', transforms.RandomErasing(p=1.0)),
        ('invert', augment.RandomInversion(p=1.0)),
    ]
    images, labels = [], []
    for i in range(len(data)):
        if targets[i] == normal_class:
            if transpose:
                dat = np.transpose(data[i], (1, 2, 0))
            else:
                dat = data[i]
            img = Image.fromarray(dat)
            if transform is not None:
                img = transform(img)
            images.append(img)
            labels.append(normal_class)
            for j in range(len(aug_functions)):
                images.append(aug_functions[j][1](img))
                labels.append(len(classes) + j)

    data = torch.stack(images)
    targets = np.array(labels, dtype=np.int64)
    classes = classes + [e[0] for e in aug_functions]
    return data, targets, classes


class Synthetic(CIFAR10):
    def __init__(self, root, normal_class, augments=None, **kwargs):
        super().__init__(root, **kwargs)
        self.augments = augments
        self.targets = np.array(self.targets)

        if not self.train:
            self.data, self.targets, self.classes = \
                generate_anomalies(self.data, self.targets, self.classes,
                                   self.transform, normal_class)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.train:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

        aug_list = []
        if self.augments is not None:
            for aug in self.augments:
                aug_list.append(aug(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, aug_list, target


class SyntheticSVHN(SVHN):
    def __init__(self, root, normal_class, augments=None, **kwargs):
        self.train = kwargs['train']
        root = f'{root}/SVHN'
        split = 'train' if self.train else 'test'
        del kwargs['train']
        super().__init__(root, split, **kwargs)
        self.augments = augments
        self.targets = self.labels
        self.classes = [str(e) for e in range(10)]

        if not self.train:
            self.data, self.targets, self.classes = \
                generate_anomalies(self.data, self.targets, self.classes,
                                   self.transform, normal_class, transpose=True)

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        if self.train:
            img = Image.fromarray(np.transpose(img, (1, 2, 0)))
            if self.transform is not None:
                img = self.transform(img)

        aug_list = []
        if self.augments is not None:
            for aug in self.augments:
                aug_list.append(aug(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, aug_list, target


class SyntheticMVTec(VisionDataset):
    base_folder = 'mvtec_anomaly_detection'
    categories = [  # ordered by the number of training samples
        'hazelnut',
        'screw',
        'carpet',
        'pill',
        'grid',
        'wood',
        'leather',
        'zipper',
        'tile',
        'cable',
        'metal_nut',
        'capsule',
        'transistor',
        'bottle',
        'toothbrush',
    ]

    def __init__(self, root, normal_class=0, augments=None, train=True,
                 transform=None, target_transform=None, img_size=256):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.train = train
        self.augments = augments
        self.classes = list(self.categories)

        resize = transforms.Resize(img_size)
        split = 'train' if train else 'test'
        category = self.categories[normal_class]
        path = os.path.join(root, self.base_folder, category, split, 'good')

        self.data = []
        self.targets = []
        for file in os.listdir(path):
            image = read_image(os.path.join(path, file))
            if image.size(0) == 1:  # Convert to RGB
                image = image.expand(3, *image.size()[1:])
            if image.size(1) != img_size or image.size(2) != img_size:
                image = resize(image)
            self.data.append(image)
            self.targets.append(normal_class)

        self.data = np.stack(self.data).transpose((0, 2, 3, 1))
        self.targets = np.array(self.targets)

        if not train:
            self.data, self.targets, self.classes = \
                generate_anomalies(self.data, self.targets, self.categories,
                                   self.transform, normal_class)
        self.class_to_idx = {e: i for i, e in enumerate(self.classes)}

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
        if self.train:
            img = Image.fromarray(img)
            if self.transform is not None:
                img = self.transform(img)

        aug_list = []
        if self.augments is not None:
            for aug in self.augments:
                aug_list.append(aug(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, aug_list, target

    def __len__(self) -> int:
        return len(self.data)
