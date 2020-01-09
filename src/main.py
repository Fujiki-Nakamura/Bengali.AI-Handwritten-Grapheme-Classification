import argparse
import datetime as dt
import os
from pathlib import Path
import random

import addict
import yaml
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import torch
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MyDataset as Dataset
import loss
import models
from score import F2Score
import utils


LOGDIR = Path('../logs')
input_d = Path('../inputs')
device = 'cuda:0'
RANDOM_SEED = 2020
random.seed(RANDOM_SEED)
os.environ['PYTHONHASHSEED'] = str(RANDOM_SEED)
np.random.seed(RANDOM_SEED)
torch.manual_seed(RANDOM_SEED)


def main(args):
    with open(args.config, 'r') as f:
        y = yaml.load(f, Loader=yaml.Loader)
    cfg = addict.Dict(y)
    cfg.general.config = args.config

    # log
    expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    cfg.general.logdir = str(LOGDIR/expid)
    if not os.path.exists(cfg.general.logdir):
        os.makedirs(cfg.general.logdir)
    os.chmod(cfg.general.logdir, 0o777)
    logger = utils.get_logger(os.path.join(cfg.general.logdir, 'main.log'))
    logger.info(f'Logging at {cfg.general.logdir}')
    logger.info(cfg)
    # model
    model = models.get_model(cfg=cfg)
    model = model.to(device)
    criterion = loss.get_loss_fn(cfg)
    optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)
    # data
    df_train = pd.read_csv(cfg.data.csv)
    kf = KFold(n_splits=cfg.training.n_splits, shuffle=True, random_state=cfg.general.random_state)  # noqa
    score_list = {'loss': [], 'F2': []}
    for fold_i, (train_idx, valid_idx) in enumerate(kf.split(df_train)):
        df_train_ = df_train.loc[train_idx].reset_index()
        df_valid_ = df_train.loc[valid_idx].reset_index()
        train_set = Dataset(df_label=df_train_)
        train_loader = DataLoader(
            train_set, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.training.n_worker)
        valid_set = Dataset(df_label=df_valid_)
        valid_loader = DataLoader(
            valid_set, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.training.n_worker)

        best = {'loss': 1e+9, 'F2': -1.}
        is_best = {'loss': False, 'score': False}
        for epoch_i in range(1, 1 + cfg.training.epochs):
            train = training(train_loader, model, criterion, optimizer)
            valid = training(valid_loader, model, criterion, optimizer, is_training=False)

            is_best['loss'] = valid['loss'] < best['loss']
            is_best['score'] = valid['F2'] > best['F2']
            if is_best['loss']:
                best['loss'] = valid['loss']
            if is_best['score']:
                best['F2'] = valid['F2']
            state_dict = {
                'epoch': epoch_i,
                'state_dict': model.state_dict(),
                'Loss/Valid': valid['loss'],
                'Acc/Valid': valid['acc'],
                'optimizer': optimizer.state_dict(),
            }
            utils.save_checkpoint(
                state_dict, is_best, Path(cfg.general.logdir)/f'fold_{fold_i}')

            log = f'[{expid}] Fold {fold_i+1} Epoch {epoch_i}/{cfg.training.epochs} '
            log += f'[Loss] {train["loss"]:.4f} Val {valid["loss"]:.4f} '
            log += f'[F2] {train["F2"]:.4f} Val {valid["F2"]:.4f} '
            log += f'Best {best["F2"]:.4f} '
            logger.info(log)
        score_list['loss'].append(best['loss'])
        score_list['F2'].append(best['F2'])
        if cfg.training.single_fold: break  # noqa

    log = f'[{expid}] '
    log += f'[Loss] {cfg.training.n_splits}-Fold/Mean {np.mean(score_list["loss"]):.4f} '
    log += f'[F2] {cfg.training.n_splits}-Fold/Mean {np.mean(score_list["F2"]):.4f} '
    logger.info(log)


def training(dataloader, model, criterion, optimizer, is_training=True):
    if is_training:
        model.train()
    else:
        model.eval()
    losses = utils.AverageMeter()
    acces = utils.AverageMeter()
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(is_training):
            bs = target.size(0)
            output = model(data)
            outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
            targets = torch.split(target, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
            loss_grapheme = criterion(outputs[:, 0], targets[:, 0])
            loss_vowel = criterion(outputs[:, 1], targets[:, 1])
            loss_consonant = criterion(outputs[:, 2], targets[:, 2])
            loss = loss_grapheme + loss_vowel + loss_consonant
            losses.update(loss.item(), bs)

            prob = torch.sigmoid(output)
            f2_score = F2Score(prob, target)
            f2_scores.update(f2_score.item(), bs)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    return {'loss': losses.avg, 'F2': f2_scores.avg}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='../configs/test.yaml')
    args = parser.parse_args()
    main(args)
