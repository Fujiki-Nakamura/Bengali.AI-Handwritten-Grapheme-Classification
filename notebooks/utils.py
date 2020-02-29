from pathlib import Path
import matplotlib.pyplot as plt
import pandas as pd


logs_d = Path('../logs')
logsd = Path('../logs')


def plot_learning_curve(expid, target_folds=[1, 2, 3, 4, 5]):
    logd = logsd/expid

    logfname = logd/'main.log'

    with open(logfname, 'r') as logf:
        lines = logf.readlines()

    results_d = {}
    for target_fold in target_folds:
        train = {'loss': [], 'score': []}
        valid = {'loss': [], 'score': []}
        results_d.update({
            f'fold_{target_fold}': {'train': train, 'valid': valid}})

    for line in lines:
        expid_fold_str = f'[{expid}] Fold '
        if expid_fold_str not in line:
            continue

        n_fold = int(line.split(expid_fold_str)[-1][:1])
        if n_fold not in target_folds:
            continue

        train_loss, valid_loss = line.split(
            expid)[-1].split('[loss]')[-1][:15].strip().split('/')
        train_score, valid_score = line.split(
            expid)[-1].split('[score]')[-1][:15].strip().split('/')
        results_d[f'fold_{n_fold}']['train']['loss'].append(float(train_loss))
        results_d[f'fold_{n_fold}']['valid']['loss'].append(float(valid_loss))
        results_d[f'fold_{n_fold}']['train']['score'].append(float(train_score))
        results_d[f'fold_{n_fold}']['valid']['score'].append(float(valid_score))

    for n_fold in target_folds:
        df_fold = pd.DataFrame({
            'train/loss': results_d[f'fold_{n_fold}']['train']['loss'],
            'valid/loss': results_d[f'fold_{n_fold}']['valid']['loss'],
            'train/score': results_d[f'fold_{n_fold}']['train']['score'],
            'valid/score': results_d[f'fold_{n_fold}']['valid']['score'],
        })

        fig, axes = plt.subplots(ncols=2, nrows=1, figsize=(2*6, 4))
        df_fold.plot(ax=axes[0], y=['train/loss', 'valid/loss'],
                     title=f'{expid} / Fold {n_fold} / Loss')
        df_fold.plot(ax=axes[1], y=['train/score', 'valid/score'],
                     title=f'{expid} / Fold {n_fold} / Score')
        plt.show()

    return results_d


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
