import json
import os
from collections import defaultdict
from distutils.util import strtobool

import numpy as np
import torch

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def str2bool(x):
    return bool(strtobool(x))


def save_json(out, path):
    with open(path, 'w') as f:
        json.dump(out, f, indent=4, sort_keys=True)


def to_device(gpu):
    if torch.cuda.is_available():
        return torch.device(f'cuda:{gpu}')
    else:
        return torch.device('cpu')


def set_environment(num_threads, seed=None):
    torch.set_num_threads(num_threads)
    if seed is not None:
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)


def change_name(name):
    return name.replace(' ', '').replace('/', '_')


def to_dict(depth, base):
    if depth == 0:
        return base()
    elif depth > 0:
        return defaultdict(lambda: to_dict(depth - 1, base))
    else:
        raise ValueError(depth)
