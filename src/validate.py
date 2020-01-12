import argparse
import os
from pathlib import Path
import random
import shutil

import addict
import yaml
import numpy as np
from tqdm import tqdm

from sklearn import metrics
from sklearn import model_selection
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from common import component_list, LOGDIR, N_GRAPHEME, N_VOWEL, N_CONSONANT
from dataset import MyDataset as Dataset
import models
import utils


best_model_name = 'bestLoss.pt'
device = 'cpu'


def main(args):
    # config
    config = LOGDIR/f'{args.expid}/config.yaml'
    with open(config, 'r') as f:
        y = yaml.load(f, Loader=yaml.Loader)
    cfg = addict.Dict(y)

    # misc
    random.seed(cfg.general.random_state)
    os.environ['PYTHONHASHSEED'] = str(cfg.general.random_state)
    np.random.seed(cfg.general.random_state)
    torch.manual_seed(cfg.general.random_state)

    # log
    expid = args.expid
    cfg.general.logdir = str(LOGDIR/expid)
    if not os.path.exists(cfg.general.logdir):
        os.makedirs(cfg.general.logdir)
    os.chmod(cfg.general.logdir, 0o777)
    logger = utils.get_logger(os.path.join(cfg.general.logdir, 'valid.log'))
    logger.info(f'Logging at {cfg.general.logdir}')
    logger.info(cfg)
    cfg.general.logdir = Path(cfg.general.logdir)
    shutil.copyfile(str(config), cfg.general.logdir/'config_val.yaml')

    # data
    X_train = np.load(cfg.data.X_train, allow_pickle=True)
    y_train = np.load(cfg.data.y_train, allow_pickle=True)
    logger.info('Loaded X_train, y_train')

    # CV
    kf = model_selection.__dict__[cfg.training.split](
        n_splits=cfg.training.n_splits, shuffle=True, random_state=cfg.general.random_state)  # noqa
    probas = {
        'grapheme': np.zeros((len(X_train), N_GRAPHEME)),
        'vowel': np.zeros((len(X_train), N_VOWEL)),
        'consonant': np.zeros((len(X_train), N_CONSONANT)),
    }
    for fold_i, (train_idx, valid_idx) in enumerate(kf.split(y_train[:, 0])):
        X_valid_ = X_train[valid_idx]
        y_valid_ = y_train[valid_idx]
        valid_set = Dataset(X_valid_, y_valid_, cfg, mode='valid')
        valid_loader = DataLoader(
            valid_set, batch_size=cfg.training.batch_size, shuffle=False,
            num_workers=cfg.training.n_worker)

        # model
        fold_d = cfg.general.logdir/f'fold_{fold_i}'
        checkpoint = fold_d/best_model_name
        checkpoint = torch.load(checkpoint, map_location=device)
        model = models.get_model(cfg=cfg)
        model.load_state_dict(checkpoint['state_dict'])
        model = model.to(device)
        model.eval()

        proba = {'grapheme': [], 'vowel': [], 'consonant': []}
        with torch.no_grad():
            pbar = tqdm(total=len(valid_loader), desc=f'Fold {fold_i+1}')
            for input_, target in valid_loader:
                input_ = input_.to(device)
                output = model(input_)
                outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
                for component_i, component in enumerate(component_list):
                    proba[component].extend(
                        F.softmax(outputs[component_i], dim=1).detach().cpu().numpy().tolist())
                pbar.update(1)
            pbar.close()

        for component in component_list:
            probas[component][valid_idx, :] = proba[component]

    pred = {}
    for component in component_list:
        pred[component] = probas[component].argmax(axis=1)

    acc = {}
    for component_i, component in enumerate(component_list):
        acc[component] = metrics.accuracy_score(y_train[:, component_i], pred[component])

    scores = []
    for component_i, component in enumerate(component_list):
        scores.append(metrics.recall_score(y_train[:, component_i], pred[component], average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])

    log = f'[{expid}] ({best_model_name}) '
    log += f'acc/grapheme {acc["grapheme"]:.4f} vowel {acc["vowel"]:.4f} consonant {acc["consonant"]:.4f} '  # noqa
    log += f'final_score {final_score:.4f}'
    logger.info(log)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expid', type=str)
    args = parser.parse_args()
    main(args)
