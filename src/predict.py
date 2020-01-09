import argparse
import datetime as dt
from pathlib import Path

import addict
import pandas as pd
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import DataLoader

from common import unique_label_list
from dataset import PlanetDataset
import models


TEST_CSV = '../inputs/orgs/sample_submission_v2.csv'
TEST_DATA_D = '../inputs/test-jpg/'
BATCH_SIZE = 1024
best_model_name = 'bestScore.pt'


def predict(args):
    args.logdir = Path(args.logdir)
    fpath = args.logdir/'main.log'
    with open(fpath, 'r') as f:
        cfg_d = eval(f.readlines()[1].split(' - ')[-1])
    cfg = addict.Dict(cfg_d)
    cfg.training.batch_size = BATCH_SIZE

    # data
    df_test = pd.read_csv(TEST_CSV)
    dataset = PlanetDataset(data_d=TEST_DATA_D, df_label=df_test, test=True)

    prob_list = []
    for fold_i in range(cfg.training.n_splits):
        dataloader = DataLoader(
            dataset, batch_size=cfg.training.batch_size, shuffle=True,
            num_workers=cfg.training.n_worker)
        # model
        fold_d = args.logdir/f'fold_{fold_i}'
        checkpoint = fold_d/best_model_name
        checkpoint = torch.load(checkpoint, map_location=cfg.device)
        model = models.get_model(cfg=cfg)
        model = model.to(cfg.general.device)
        model.load_state_dict(checkpoint['state_dict'])
        model.eval()

        batch_prob_list = []
        exp_id = args.logdir.as_posix().split('/')[-1]
        pbar = tqdm(total=len(dataloader.dataset), desc=f'[{exp_id}] Test/Fold {fold_i}')
        for input_ in dataloader:
            bs = input_.size(0)
            input_ = input_.to(cfg.general.device)
            with torch.no_grad():
                output = model(input_)
                output = torch.sigmoid(output)
                batch_prob_list.append(output.cpu().numpy())
            pbar.update(bs)
        pbar.close()
        prob = np.concatenate(batch_prob_list, axis=0)
        prob_list.append(prob)
    probs = np.stack(prob_list, axis=0)
    path = args.logdir/'probs.npy'
    np.save(path, probs)
    print(f'Saved at {path}')

    submit(path)


def submit(path):
    labels = np.asarray(unique_label_list)

    df = pd.read_csv('../inputs/orgs/sample_submission_v2.csv')
    probs = np.load(path)
    probs = probs.reshape(-1, len(df), 17)
    prob_mean = np.mean(probs, axis=0)
    pred = prob_mean.round()
    pred_bool = pred > 0

    tag_list = []
    for i in range(len(pred_bool)):
        tags = labels[pred_bool[i]]
        tag = ' '.join(tags)
        tag_list.append(tag)

    df['tags'] = tag_list
    datetime_id = dt.datetime.now().strftime('%Y%m%d%H%M%S')
    path = path.as_posix().replace('probs.npy', f'{datetime_id}.csv')
    df.to_csv(path, index=False)
    print(f'Saved at {path}')


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--logdir', type=str, default='')
    args = parser.parse_args()
    predict(args)


if __name__ == '__main__':
    main()
