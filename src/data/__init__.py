from torchvision.transforms import transforms

from augment import get_augment
from data.base import MNIST, CIFAR10, CIFAR100, FashionMNIST, SVHN
from data.mvtec import MVTecAD, MVTecAll
from data.synthetic import Synthetic, SyntheticSVHN, SyntheticMVTec


def get_normal_classes(name):
    if name in ['mnist', 'fashion', 'svhn', 'cifar10', 'synthetic',
                'synthetic-svhn']:
        num_classes = 10
    elif name in ['mvtec', 'mvtec-global', 'synthetic-mvtec']:
        num_classes = 15
    elif name in ['mvtec-all']:
        num_classes = 1
    elif name in ['cifar100']:
        num_classes = 100
    elif '-' in name:
        return get_normal_classes(name[:name.rfind('-')])
    else:
        raise ValueError(name)
    return list(range(num_classes))


def get_image_size(name):
    if name in ['mnist', 'fashion']:
        channels = 1
    else:
        channels = 3

    if name in ['mvtec', 'mvtec-global', 'mvtec-all', 'synthetic-mvtec']:
        size = 256
    else:
        size = 32

    return size, channels


def get_real_data(name, root, augments, transform, download):
    if name == 'mnist':
        data_class = MNIST
    elif name == 'fashion':
        data_class = FashionMNIST
    elif name == 'svhn':
        data_class = SVHN
    elif name == 'cifar10':
        data_class = CIFAR10
    elif name == 'cifar100':
        data_class = CIFAR100
    else:
        raise ValueError(name)

    trn_data = data_class(root, augments, train=True, transform=transform,
                          download=download)
    test_data = data_class(root, augments, train=False, transform=transform,
                           download=download)
    return trn_data, test_data


def get_synthetic_data(name, root, augments, transform, normal_class):
    if name == 'synthetic':
        data_class = Synthetic
    elif name == 'synthetic-svhn':
        data_class = SyntheticSVHN
    elif name == 'synthetic-mvtec':
        data_class = SyntheticMVTec
    else:
        raise ValueError(name)

    trn_data = data_class(root, normal_class, augments, train=True,
                          transform=transform)
    test_data = data_class(root, normal_class, augments, train=False,
                           transform=transform)
    return trn_data, test_data


def get_mvtec_data(name, root, augments, transform, normal_class, img_size):
    if name == 'mvtec':
        anomaly = 'local'
    elif name == 'mvtec-global':
        anomaly = 'global'
    else:
        raise ValueError(name)
    trn_data = MVTecAD(root, normal_class, augments, anomaly, train=True,
                       transform=transform, img_size=img_size)
    test_data = MVTecAD(root, normal_class, augments, anomaly, train=False,
                        transform=transform, img_size=img_size)
    return trn_data, test_data


def get_oe_data(data, root, transform, download):
    if data == 'fashion':
        return FashionMNIST(
            root, train=True, transform=transform, download=download)
    else:
        raise ValueError(data)


def load_data(name, normal_class=0, augment=None, download=True,
              deterministic=False):
    root = f'/data/jaeminy/data'
    img_size, channels = get_image_size(name)
    transform = []
    if not name.startswith('mvtec'):
        transform.append(transforms.Resize(img_size))
    transform.append(transforms.ToTensor())
    transform = transforms.Compose(transform)

    if augment is None:
        aug_list = None
    else:
        aug_list = []
        if isinstance(augment, str):
            augment = [augment]
        for aug in augment:
            oe_data = None
            if aug.startswith('oe'):
                known_anomaly = int(aug.split('-')[1])
                if normal_class == known_anomaly:
                    aug = 'none'
                else:
                    oe_data = get_oe_data(name, root, transform, download)
            aug_list.append(
                get_augment(aug, img_size, channels, deterministic, data=oe_data)
            )

    if name in ['mnist', 'fashion', 'svhn', 'cifar10', 'cifar100']:
        trn_data, test_data = \
            get_real_data(name, root, aug_list, transform, download)
    elif name in ['synthetic', 'synthetic-svhn', 'synthetic-mvtec']:
        trn_data, test_data = \
            get_synthetic_data(name, root, aug_list, transform, normal_class)
    elif name in ['mvtec', 'mvtec-global']:
        trn_data, test_data = \
            get_mvtec_data(name, root, aug_list, transform, normal_class,
                           img_size)
    elif name in ['mvtec-all']:
        trn_data = MVTecAll(root, aug_list, train=True, transform=transform,
                            img_size=img_size)
        test_data = MVTecAll(root, aug_list, train=False, transform=transform,
                             img_size=img_size)
    else:
        raise ValueError(name)

    trn_data.data = trn_data.data[trn_data.targets == normal_class]
    trn_data.targets = trn_data.targets[trn_data.targets == normal_class]
    return trn_data, test_data
