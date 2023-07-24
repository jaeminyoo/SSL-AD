import os
from typing import Iterable

import numpy as np
import torch
from sklearn import metrics
from torch import optim, nn
from tqdm import tqdm

import utils


def to_string(values):
    if not isinstance(values, Iterable):
        values = [values]
    return ', '.join(f'{e:.4f}' for e in values)


def to_metric_dict(labels, scores, curve=True):
    assert np.array_equal(np.unique(labels), [0, 1])

    auc = metrics.roc_auc_score(labels, scores)
    ap = metrics.average_precision_score(labels, scores)

    num_anomalies = sum(labels == 1)
    top_k = (-scores).argsort()[:num_anomalies]
    y_pred = np.zeros(len(scores))
    y_pred[top_k] = 1
    f1 = metrics.f1_score(labels, y_pred)
    ppv = metrics.precision_score(labels, y_pred)

    out = dict(auc=auc, ap=ap, f1=f1, ppv=ppv)
    if curve:
        fpr, tpr, _ = metrics.roc_curve(labels, scores)
        out['fpr'] = list(fpr)
        out['tpr'] = list(tpr)
    return out


def get_batch(loader):
    while True:
        for x in loader:
            yield x


def evaluate_classes(labels, scores, classes, normal_class, curve=True):
    out = {}
    for l in range(len(classes)):
        if l != normal_class and sum(labels == l) > 0:
            curr_y = np.concatenate([
                np.zeros(sum(labels == normal_class)),
                np.ones(sum(labels == l))
            ])
            curr_s = np.concatenate([
                scores[labels == normal_class],
                scores[labels == l]
            ])
            out[classes[l]] = to_metric_dict(curr_y, curr_s, curve)
    y_true = np.where(labels == normal_class, 0, 1)
    out['all'] = to_metric_dict(y_true, scores, curve)
    return out


class Trainer:
    def __init__(self, args, model, classes):
        self.out_path = args.out
        self.verbose = args.verbose
        self.normal_class = args.normal_class
        self.num_epochs = args.epochs
        self.checkpoint = args.checkpoint
        self.decay = args.decay
        self.device = utils.to_device(args.gpu)

        self.model = model.to(self.device)
        self.classes = classes

    def fit(self, trn_loader, test_loader):
        model = self.model
        device = self.device
        silent = not self.verbose
        cpt_path = f'{self.out_path}/checkpoint'
        log_path = f'{self.out_path}/log.json'

        optimizer = optim.Adam(model.parameters(), weight_decay=self.decay)
        scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, self.num_epochs)
        loss_func = nn.MSELoss()

        logs = []
        for epoch in range(self.num_epochs + 1):
            model.train()
            loss_list = []
            for ori_x, aug_xs, _ in tqdm(trn_loader, leave=False, disable=silent):
                losses = []
                ori_x = ori_x.to(device)

                if len(aug_xs) == 0:
                    _, ori_p = model(ori_x)
                    losses.append(loss_func(ori_p, ori_x))
                else:
                    aug_x = aug_xs[0].to(device)
                    _, aug_p = model(aug_x)
                    losses.append(loss_func(aug_p, ori_x))

                losses = torch.stack(losses)
                if epoch > 0:
                    optimizer.zero_grad()
                    losses.sum().backward()
                    optimizer.step()
                loss_list.append(losses.detach())

                if self.checkpoint > 0 and epoch % self.checkpoint == 0:
                    if 0 < epoch < self.num_epochs:
                        os.makedirs(cpt_path, exist_ok=True)
                        checkpoint = f'{cpt_path}/model-{epoch}.pth'
                        torch.save(model.state_dict(), checkpoint)
                        utils.save_json(logs, log_path)

            trn_loss = [e.item() for e in torch.stack(loss_list).mean(0)]
            if epoch > 0:
                scheduler.step()

            test_out = self.evaluate_accuracy(test_loader, curve=False)
            print(f'[epoch {epoch:3d}]'
                  f' [{to_string(trn_loss)}]'
                  f' [{to_string(test_out["all"]["auc"])}]')

            logs.append(dict(
                epoch=epoch,
                trn=trn_loss,
                test=test_out
            ))
        return logs

    @torch.no_grad()
    def evaluate_accuracy(self, test_loader, curve=True):
        model = self.model
        device = self.device
        classes = self.classes
        normal_class = self.normal_class

        model.eval()
        scores, labels = [], []
        for x, aug_xs, y in test_loader:
            x = x.to(device)
            _, pred = model(x)
            scores.append(((x - pred) ** 2).mean(dim=(1, 2, 3)).cpu().numpy())
            labels.append(y.numpy())
        scores = np.concatenate(scores)
        labels = np.concatenate(labels)
        return evaluate_classes(labels, scores, classes, normal_class, curve)
