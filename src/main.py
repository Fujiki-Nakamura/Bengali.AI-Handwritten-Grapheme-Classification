import argparse
import datetime as dt
import os
from pathlib import Path
import random
import shutil

import addict
import yaml
import numpy as np
from sklearn.metrics import recall_score
from sklearn import model_selection
import torch
from torch.nn import functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from dataset import MyDataset as Dataset
import loss
import models
import utils


N_GRAPHEME = 168
N_VOWEL = 11
N_CONSONANT = 7


def main(args):
    with open(args.config, 'r') as f:
        y = yaml.load(f, Loader=yaml.Loader)
    cfg = addict.Dict(y)
    cfg.general.config = args.config

    # misc
    device = cfg.general.device
    random.seed(cfg.general.random_state)
    os.environ['PYTHONHASHSEED'] = str(cfg.general.random_state)
    np.random.seed(cfg.general.random_state)
    torch.manual_seed(cfg.general.random_state)

    # log
    # expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    expid = args.expid
    cfg.general.logdir = os.path.join(cfg.general.logdir, expid)
    if not os.path.exists(cfg.general.logdir):
        os.makedirs(cfg.general.logdir)
    os.chmod(cfg.general.logdir, 0o777)
    logger = utils.get_logger(os.path.join(cfg.general.logdir, 'main.log'))
    logger.info(f'Logging at {cfg.general.logdir}')
    logger.info(cfg)
    shutil.copyfile(str(args.config), cfg.general.logdir+'/config.yaml')
    # data
    X_train = np.load(cfg.data.X_train, allow_pickle=True)
    y_train = np.load(cfg.data.y_train, allow_pickle=True)
    logger.info('Loaded X_train, y_train')
    # CV
    kf = model_selection.__dict__[cfg.training.split](
        n_splits=cfg.training.n_splits, shuffle=True, random_state=cfg.general.random_state)  # noqa
    score_list = {'loss': [], 'score': []}
    for fold_i, (train_idx, valid_idx) in enumerate(
        kf.split(X=np.zeros(len(y_train)), y=y_train[:, 0])
    ):
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        X_valid_ = X_train[valid_idx]
        y_valid_ = y_train[valid_idx]
        train_set = Dataset(X_train_, y_train_, cfg, mode='train')
        valid_set = Dataset(X_valid_, y_valid_, cfg, mode='valid')
        train_loader = DataLoader(
            train_set, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.training.n_worker)
        valid_loader = DataLoader(
            valid_set, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.training.n_worker)

        # model
        model = models.get_model(cfg=cfg)
        model = model.to(device)
        criterion = loss.get_loss_fn(cfg)
        optimizer = optim.Adam(model.parameters(), lr=cfg.optimizer.lr)

        best = {'loss': 1e+9, 'score': -1.}
        is_best = {'loss': False, 'score': False}
        for epoch_i in range(1, 1 + cfg.training.epochs):
            train = training(train_loader, model, criterion, optimizer, config=cfg)
            valid = training(
                valid_loader, model, criterion, optimizer, is_training=False, config=cfg)

            is_best['loss'] = valid['loss'] < best['loss']
            is_best['score'] = valid['score'] > best['score']
            if is_best['loss']:
                best['loss'] = valid['loss']
            if is_best['score']:
                best['score'] = valid['score']
            state_dict = {
                'epoch': epoch_i,
                'state_dict': model.state_dict(),
                'loss/valid': valid['loss'],
                'score/valid': valid['score'],
                'optimizer': optimizer.state_dict(),
            }
            utils.save_checkpoint(
                state_dict, is_best, Path(cfg.general.logdir)/f'fold_{fold_i}')

            log = f'[{expid}] Fold {fold_i+1} Epoch {epoch_i}/{cfg.training.epochs} '
            log += f'[loss] {train["loss"]:.4f} Val {valid["loss"]:.4f} '
            log += f'[score] {train["score"]:.4f} Val {valid["score"]:.4f} '
            log += f'best {best["score"]:.4f} '
            logger.info(log)

        score_list['loss'].append(best['loss'])
        score_list['score'].append(best['score'])
        if cfg.training.single_fold: break  # noqa

    log = f'[{expid}] '
    log += f'[loss] {cfg.training.n_splits}-fold/mean {np.mean(score_list["loss"]):.4f} '
    log += f'[score] {cfg.training.n_splits}-fold/mean {np.mean(score_list["score"]):.4f} '  # noqa
    logger.info(log)


def training(dataloader, model, criterion, optimizer, config, is_training=True):
    device = config.general.device
    if is_training:
        model.train()
    else:
        model.eval()
    losses = utils.AverageMeter()
    pred = {'grapheme': [], 'vowel': [], 'consonant': []}
    true = {'grapheme': [], 'vowel': [], 'consonant': []}
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(is_training):
            bs = target.size(0)
            output = model(data)
            outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
            loss_grapheme = criterion(outputs[0], target[:, 0])
            loss_vowel = criterion(outputs[1], target[:, 1])
            loss_consonant = criterion(outputs[2], target[:, 2])
            loss = loss_grapheme + loss_vowel + loss_consonant
            losses.update(loss.item(), bs)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            pred['grapheme'].extend(
                F.softmax(outputs[0], dim=1).max(1)[1].detach().cpu().numpy().tolist())
            pred['vowel'].extend(
                F.softmax(outputs[1], dim=1).max(1)[1].detach().cpu().numpy().tolist())
            pred['consonant'].extend(
                F.softmax(outputs[2], dim=1).max(1)[1].detach().cpu().numpy().tolist())
            true['grapheme'].extend(target[:, 0].cpu().numpy().tolist())
            true['vowel'].extend(target[:, 1].cpu().numpy().tolist())
            true['consonant'].extend(target[:, 2].cpu().numpy().tolist())

        if config.training.single_iter: break  # noqa

    scores = []
    for component in ['grapheme', 'consonant', 'vowel']:
        scores.append(recall_score(true[component], pred[component], average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])

    return {'loss': losses.avg, 'score': final_score}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='default.yaml')
    parser.add_argument('--expid', type=str, default='default_expid')
    args = parser.parse_args()
    main(args)
