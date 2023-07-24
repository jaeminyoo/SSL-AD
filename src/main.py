import argparse
import json
import os

import torch
from torch.utils.data import DataLoader

import data
import models
import utils
from trainer import Trainer


def count_parameters(model):
    out = 0
    for param in model.parameters():
        if param.requires_grad:
            out += param.numel()
    return out


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', type=str, default=f'{utils.ROOT}/out-tmp')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--verbose', type=utils.str2bool, default=True)
    parser.add_argument('--workers', type=int, default=12)
    parser.add_argument('--threads', type=int, default=16)

    # Experimental setup
    parser.add_argument('--data', type=str, default='cifar10')
    parser.add_argument('--normal-class', type=int, default=0)
    parser.add_argument('--augment', type=str, default=None)

    # Model parameters
    parser.add_argument('--model', type=str, default='conv')
    parser.add_argument('--hidden', type=int, default=256)
    parser.add_argument('--features', type=int, default=64)
    parser.add_argument('--decay', type=float, default=0)

    # Training parameters
    parser.add_argument('--epochs', type=int, default=256)
    parser.add_argument('--checkpoint', type=int, default=0)
    parser.add_argument('--batch-size', type=int, default=1024)
    return parser.parse_args()


def main():
    args = parse_args()
    utils.set_environment(args.threads, args.seed)
    os.makedirs(args.out, exist_ok=True)
    utils.save_json(vars(args), f'{args.out}/args.json')

    augments = []
    if args.augment is not None and args.augment.lower() != 'none':
        augments = [args.augment]
    trn_data, test_data = data.load_data(args.data, args.normal_class, augments)

    trn_loader = DataLoader(
        dataset=trn_data,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers
    )

    test_loader = DataLoader(
        dataset=test_data,
        batch_size=args.batch_size,
        num_workers=args.workers
    )

    sample = trn_data[0][0]
    if args.model == 'conv':
        if args.data in ['mvtec', 'mvtec-global', 'mvtec-all', 'synthetic-mvtec']:
            kernel_size = 5
            num_layers = 3
        else:
            kernel_size = 3
            num_layers = 4
        model = models.ConvAE(img_size=sample.size(1),
                              img_channels=sample.size(0),
                              hidden_size=args.hidden,
                              num_features=args.features,
                              kernel_size=kernel_size,
                              num_layers=num_layers)
    elif args.model == 'dense':
        model = models.DenseAE(img_size=sample.size(1),
                               img_channels=sample.size(0),
                               hidden_size=args.hidden)
    elif args.model == 'linear':
        model = models.LinearAE(img_size=sample.size(1),
                                img_channels=sample.size(0),
                                hidden_size=args.hidden)
    else:
        raise ValueError()

    if args.verbose:
        print(f'Parameters: {count_parameters(model)}')

    trainer = Trainer(args, model, test_data.classes)
    logs = trainer.fit(trn_loader, test_loader)

    utils.save_json(logs, f'{args.out}/log.json')
    torch.save(model.state_dict(), f'{args.out}/model.pth')

    test_out = trainer.evaluate_accuracy(test_loader, curve=True)
    utils.save_json(test_out, f'{args.out}/out.json')

    del test_out['all']['tpr']
    del test_out['all']['fpr']
    print(json.dumps(test_out['all'], indent=4, sort_keys=True))


if __name__ == '__main__':
    main()
