import os.path

import numpy as np
from PIL import Image
from torchvision.datasets import VisionDataset
from torchvision.io import read_image
from torchvision.transforms import transforms


class MVTecAD(VisionDataset):
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

    def __init__(self, root, normal_class=0, augments=None, anomaly='local',
                 train=True, transform=None, target_transform=None,
                 img_size=256):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        self.train = train
        self.category = self.categories[normal_class]
        self.anomaly = anomaly
        self.augments = augments
        self.normal_class = normal_class

        self.classes = self._get_class_names(self.category, anomaly)
        self.class_to_idx = {e: i for i, e in enumerate(self.classes)}

        self.data = []
        self.targets = []

        resize = transforms.Resize(img_size)
        for class_idx, path in zip(*self._get_directories()):
            for file in os.listdir(path):
                image = read_image(os.path.join(path, file))
                if image.size(0) == 1:  # Convert to RGB
                    image = image.expand(3, *image.size()[1:])
                if image.size(1) != img_size or image.size(2) != img_size:
                    image = resize(image)
                self.data.append(image)
                self.targets.append(class_idx)

        self.data = np.stack(self.data)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.targets = np.array(self.targets)

    def _get_class_names(self, category, anomaly):
        classes = list(self.categories)
        if anomaly == 'local':
            path = os.path.join(self.root, self.base_folder, category, 'test')
            anomalies = sorted(os.listdir(path))
            anomalies.remove('good')
            classes.extend(anomalies)
        return classes

    def _get_directories(self):
        def get_path(category, split, label):
            return os.path.join(self.root, self.base_folder, category,
                                split, label)

        if self.train:
            labels = [self.normal_class]
            paths = [get_path(self.category, 'train', 'good')]
        elif self.anomaly == 'local':
            labels = [self.normal_class]
            paths = [get_path(self.category, 'test', 'good')]
            for l in range(len(self.categories), len(self.classes)):
                labels.append(l)
                paths.append(get_path(self.category, 'test', self.classes[l]))
        elif self.anomaly == 'global':
            labels = list(range(len(self.categories)))
            paths = [get_path(c, 'test', 'good') for c in self.categories]
        else:
            raise ValueError()
        return labels, paths

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

    def __len__(self) -> int:
        return len(self.data)


class MVTecAll(MVTecAD):
    def __init__(self, root, augments=None, train=True, transform=None,
                 target_transform=None, img_size=256):
        super().__init__(root, transform=transform,
                         target_transform=target_transform)
        self.root = root
        self.train = train
        self.augments = augments
        self.img_size = img_size
        self.resize = transforms.Resize(img_size)

        self.data = []
        self.targets = []
        self.classes = ['good']
        if train:
            for category in self.categories:
                path = self._get_path(category, 'train', 'good')
                for file in os.listdir(path):
                    self.data.append(self._read_image(path, file))
                    self.targets.append(0)
        else:
            label_cnt = 1
            for category in self.categories:
                root = self._get_path(category, 'test')
                for curr_class in sorted(os.listdir(root)):
                    if curr_class == 'good':
                        label = 0
                    else:
                        label = label_cnt
                        label_cnt += 1
                        self.classes.append(f'{category}-{curr_class}')
                    path = self._get_path(category, 'test', curr_class)
                    for file in os.listdir(path):
                        self.data.append(self._read_image(path, file))
                        self.targets.append(label)

        self.data = np.stack(self.data)
        self.data = self.data.transpose((0, 2, 3, 1))
        self.targets = np.array(self.targets)

    def _read_image(self, path, file):
        image = read_image(os.path.join(path, file))
        if image.size(0) == 1:  # Convert to RGB
            image = image.expand(3, *image.size()[1:])
        if image.size(1) != self.img_size or image.size(2) != self.img_size:
            image = self.resize(image)
        return image

    def _get_path(self, category, split, label=None):
        out = os.path.join(self.root, self.base_folder, category, split)
        if label is not None:
            out = os.path.join(out, label)
        return out
