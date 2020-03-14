import argparse
import datetime as dt
import os
from pathlib import Path
import random
import shutil

import addict
import yaml
import numpy as np
from sklearn import model_selection
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from common import LOGDIR
from dataset import MyDataset as Dataset
import loss
import models
from trainer import training
import utils


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
    if cfg.general.expid == '':
        expid = dt.datetime.now().strftime('%Y%m%d%H%M%S')
        cfg.general.expid = expid
    else:
        expid = cfg.general.expid
    cfg.general.logdir = str(LOGDIR/expid)
    if not os.path.exists(cfg.general.logdir):
        os.makedirs(cfg.general.logdir)
    os.chmod(cfg.general.logdir, 0o777)
    logger = utils.get_logger(os.path.join(cfg.general.logdir, 'main.log'))
    logger.info(f'Logging at {cfg.general.logdir}')
    logger.info(cfg)
    shutil.copyfile(str(args.config), cfg.general.logdir+'/config.yaml')
    writer = SummaryWriter(cfg.general.logdir)

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
        if fold_i + 1 not in cfg.training.target_folds:
            continue
        X_train_ = X_train[train_idx]
        y_train_ = y_train[train_idx]
        X_valid_ = X_train[valid_idx]
        y_valid_ = y_train[valid_idx]
        if cfg.training.val90:
            assert cfg.training.n_splits == 5
            from sklearn.model_selection import train_test_split
            _X_train, X_valid_, _y_train, y_valid_ = train_test_split(
                X_valid_, y_valid_, test_size=0.5, random_state=cfg.general.random_state)
            X_train_ = np.concatenate([X_train_, _X_train], axis=0)
            y_train_ = np.concatenate([y_train_, _y_train], axis=0)
        train_set = Dataset(X_train_, y_train_, cfg, mode='train')
        valid_set = Dataset(X_valid_, y_valid_, cfg, mode='valid')
        if fold_i == 0:
            logger.info(train_set.transform)
            logger.info(valid_set.transform)
        train_loader = DataLoader(
            train_set, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.training.n_worker, pin_memory=True)
        valid_loader = DataLoader(
            valid_set, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.training.n_worker, pin_memory=True)

        # model
        model = models.get_model(cfg=cfg)
        model = model.to(device)
        criterion = loss.get_loss_fn(cfg)
        optimizer = utils.get_optimizer(model.parameters(), config=cfg)
        scheduler = utils.get_lr_scheduler(optimizer, config=cfg)

        start_epoch = 1
        best = {'loss': 1e+9, 'score': -1.}
        is_best = {'loss': False, 'score': False}

        # resume
        if cfg.model.resume:
            if os.path.isfile(cfg.model.resume):
                checkpoint = torch.load(cfg.model.resume)
                start_epoch = checkpoint['epoch'] + 1
                best['loss'] = checkpoint['loss/best']
                best['score'] = checkpoint['score/best']
                model.load_state_dict(checkpoint['state_dict'])
                if cfg.model.get('load_optimizer', True):
                    optimizer.load_state_dict(checkpoint['optimizer'])
                logger.info('Loaded checkpoint {} (epoch {})'.format(
                    cfg.model.resume, start_epoch - 1))
            else:
                raise IOError('No such file {}'.format(cfg.model.resume))

        for epoch_i in range(start_epoch, cfg.training.epochs + 1):
            if scheduler is not None:
                if cfg.training.lr_scheduler.name == 'MultiStepLR':
                    optimizer.zero_grad()
                    optimizer.step()
                    scheduler.step()
            for param_group in optimizer.param_groups:
                current_lr = param_group['lr']
            _ohem_loss = (cfg.training.ohem_loss and cfg.training.ohem_epoch < epoch_i)
            train = training(
                train_loader, model, criterion, optimizer, config=cfg,
                using_ohem_loss=_ohem_loss, lr=current_lr
            )
            valid = training(
                valid_loader, model, criterion, optimizer, is_training=False, config=cfg,
                lr=current_lr)

            if scheduler is not None and lr_scheduler.name != 'MultiStepLR':
                if cfg.training.lr_scheduler.name == 'ReduceLROnPlateau':
                    if scheduler.mode == 'min':
                        value = valid['loss']
                    elif scheduler.mode == 'max':
                        value = valid['score']
                    else:
                        raise NotImplementedError
                    scheduler.step(value)

            is_best['loss'] = valid['loss'] < best['loss']
            is_best['score'] = valid['score'] > best['score']
            if is_best['loss']:
                best['loss'] = valid['loss']
            if is_best['score']:
                best['score'] = valid['score']
            state_dict = {
                'epoch': epoch_i,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'loss/valid': valid['loss'],
                'score/valid': valid['score'],
                'loss/best': best['loss'],
                'score/best': best['score'],
            }
            utils.save_checkpoint(
                state_dict, is_best, epoch_i, valid['loss'], valid['score'],
                Path(cfg.general.logdir)/f'fold_{fold_i}',)

            # tensorboard
            writer.add_scalar('Loss/Train', train['loss'], epoch_i)
            writer.add_scalar('Loss/Valid', valid['loss'], epoch_i)
            writer.add_scalar('Loss/Best',  best['loss'], epoch_i)
            writer.add_scalar('Metrics/Train', train['score'], epoch_i)
            writer.add_scalar('Metrics/Valid', valid['score'], epoch_i)
            writer.add_scalar('Metrics/Best', best['score'], epoch_i)

            log = f'[{expid}] Fold {fold_i+1} Epoch {epoch_i}/{cfg.training.epochs} '
            log += f'[loss] {train["loss"]:.4f}/{valid["loss"]:.4f} '
            log += f'[score] {train["score"]:.4f}/{valid["score"]:.4f} '
            log += f'({best["score"]:.4f}) '
            log += f'lr {current_lr:.6f}'
            logger.info(log)

        score_list['loss'].append(best['loss'])
        score_list['score'].append(best['score'])
        if cfg.training.single_fold: break  # noqa

    log = f'[{expid}] '
    log += f'[loss] {cfg.training.n_splits}-fold/mean {np.mean(score_list["loss"]):.4f} '
    log += f'[score] {cfg.training.n_splits}-fold/mean {np.mean(score_list["score"]):.4f} '  # noqa
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, default='default.yaml')
    args = parser.parse_args()
    main(args)
