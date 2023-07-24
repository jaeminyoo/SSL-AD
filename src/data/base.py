import numpy as np
import torchvision
from PIL import Image


class MNIST(torchvision.datasets.MNIST):
    def __init__(self, root, augments=None, **kwargs):
        super().__init__(root, **kwargs)
        self.augments = augments

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        aug_list = []
        if self.augments is not None:
            for aug in self.augments:
                aug_list.append(aug(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, aug_list, target


class FashionMNIST(torchvision.datasets.FashionMNIST):
    def __init__(self, root, augments=None, **kwargs):
        super().__init__(root, **kwargs)
        self.augments = augments

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
        img = Image.fromarray(img.numpy(), mode='L')

        if self.transform is not None:
            img = self.transform(img)

        aug_list = []
        if self.augments is not None:
            for aug in self.augments:
                aug_list.append(aug(img))

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, aug_list, target


class SVHN(torchvision.datasets.SVHN):
    def __init__(self, root, augments=None, **kwargs):
        root = f'{root}/SVHN'
        split = 'train' if kwargs['train'] else 'test'
        del kwargs['train']
        super().__init__(root, split, **kwargs)
        self.augments = augments
        self.targets = self.labels
        self.classes = [str(e) for e in range(10)]

    def __getitem__(self, index):
        img, target = self.data[index], int(self.targets[index])
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


class CIFAR10(torchvision.datasets.CIFAR10):
    def __init__(self, root, augments=None, **kwargs):
        super().__init__(root, **kwargs)
        self.augments = augments
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
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


class CIFAR100(torchvision.datasets.CIFAR100):
    def __init__(self, root, augments=None, **kwargs):
        super().__init__(root, **kwargs)
        self.augments = augments
        self.targets = np.array(self.targets)

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]
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
