from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


logs_d = Path('../logs')


def plot_learning_curve(expid, folds=[1, 2, 3, 4, 5], epochs=30):
    log_d = logs_d/expid
    fname = log_d/'main.log'
    with open(fname, 'r') as f:
        lines = f.readlines()

    n_lines = epochs
    for fold_i in range(5):
        if fold_i + 1 not in folds:
            continue
        start_line_i = 3 + fold_i * n_lines
        loss = {'train': [], 'valid': []}
        score = {'train': [], 'valid': []}
        for line_i in range(start_line_i, start_line_i + n_lines):
            train_loss, valid_loss = lines[line_i].split(
                expid)[-1].split('[loss]')[-1][:18].strip().split(' Val ')
            train_score, valid_score = lines[line_i].split(
                expid)[-1].split('[score]')[-1][:18].strip().split(' Val ')
            loss['train'].append(float(train_loss))
            loss['valid'].append(float(valid_loss))
            score['train'].append(float(train_score))
            score['valid'].append(float(valid_score))

        assert len(loss['train']) == len(loss['valid'])
        assert len(score['train']) == len(score['valid'])

        df_fold = pd.DataFrame({
            'train_loss': loss['train'],
            'valid_loss': loss['valid'],
            'train_score': score['train'],
            'valid_score': score['valid'],
        })

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(2*6, 4))
        df_fold.plot(ax=axes[0], y=['train_loss', 'valid_loss'],
                     title=f'{expid} / Fold {fold_i+1} / Loss')
        df_fold.plot(ax=axes[1], y=['train_score', 'valid_score'],
                     title=f'{expid} / Fold {fold_i+1} / Score')
        plt.show()


def plot_learning_curve_2(expid, folds=[1, 2, 3, 4, 5], epochs=30):
    log_d = logs_d/expid
    fname = log_d/'main.log'
    with open(fname, 'r') as f:
        lines = f.readlines()

    n_lines = epochs
    for fold_i in range(5):
        if fold_i + 1 not in folds:
            continue
        start_line_i = 3 + fold_i * n_lines
        loss = {'train': [], 'valid': []}
        score = {'train': [], 'valid': []}
        for line_i in range(start_line_i, start_line_i + n_lines):
            train_loss, valid_loss = lines[line_i].split(
                expid)[-1].split('[loss]')[-1][:15].strip().split('/')
            train_score, valid_score = lines[line_i].split(
                expid)[-1].split('[score]')[-1][:15].strip().split('/')
            loss['train'].append(float(train_loss))
            loss['valid'].append(float(valid_loss))
            score['train'].append(float(train_score))
            score['valid'].append(float(valid_score))

        assert len(loss['train']) == len(loss['valid'])
        assert len(score['train']) == len(score['valid'])

        df_fold = pd.DataFrame({
            'train_loss': loss['train'],
            'valid_loss': loss['valid'],
            'train_score': score['train'],
            'valid_score': score['valid'],
        })

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(2*6, 4))
        df_fold.plot(ax=axes[0], y=['train_loss', 'valid_loss'],
                     title=f'{expid} / Fold {fold_i+1} / Loss')
        df_fold.plot(ax=axes[1], y=['train_score', 'valid_score'],
                     title=f'{expid} / Fold {fold_i+1} / Score')
        plt.show()
