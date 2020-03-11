from pathlib import Path
import sys
from torch.utils.tensorboard import SummaryWriter

sys.path.append('../notebooks/')  # noqa
import utils


def main(log_d):
    writer = SummaryWriter(log_d)
    expid = log_d.parent.name
    fold_i = int(log_d.name.replace('fold_', ''))
    results_d = utils.plot_learning_curve(expid, target_folds=[fold_i+1, ])
    print(f'expid={expid} fold={fold_i+1}')

    n_epochs = len(results_d[f'fold_{fold_i+1}']['train']['loss'])
    for epoch_i in range(n_epochs):
        writer.add_scalar(
            'Loss/Train',
            results_d[f'fold_{fold_i+1}']['train']['loss'][epoch_i], epoch_i+1)
        writer.add_scalar(
            'Loss/Valid',
            results_d[f'fold_{fold_i+1}']['valid']['loss'][epoch_i], epoch_i+1)
        writer.add_scalar(
            'Metrics/Train',
            results_d[f'fold_{fold_i+1}']['train']['score'][epoch_i], epoch_i+1)
        writer.add_scalar(
            'Metrics/Valid',
            results_d[f'fold_{fold_i+1}']['valid']['score'][epoch_i], epoch_i+1)
    print(f'wrote to {log_d}')


if __name__ == '__main__':
    log_rootd = Path('../logs/')
    log_d = log_rootd/'20200216040313/fold_0/'
    main(log_d)
