import os
import shutil
import torch
import torch.optim as optim


def get_optimizer(params, config):
    name = config.optimizer.name
    kwargs = parse_arg_str(config.optimizer.args)
    optimizer = optim.__dict__[name](params, **kwargs)
    return optimizer


def get_lr_scheduler(optimizer, config):
    name = config.training.lr_scheduler.name
    kwargs = parse_arg_str(config.training.lr_scheduler.args)
    scheduler = optim.lr_scheduler.__dict__[name](optimizer, **kwargs)
    return scheduler


def parse_arg_str(arg_str):
    arg_dict = {}
    for arg in arg_str.split(','):
        arg = arg.strip()
        arg_dict[arg.split('=')[0]] = eval(arg.split('=')[1])
    return arg_dict


def save_checkpoint(state, is_best, logdir, filename='checkpoint.pt'):
    if not os.path.exists(logdir):
        os.makedirs(logdir)
    path = os.path.join(logdir, filename)
    torch.save(state, path)
    if is_best['loss']:
        shutil.copyfile(path, os.path.join(logdir, 'bestLoss.pt'))
    if is_best['score']:
        shutil.copyfile(path, os.path.join(logdir, 'bestScore.pt'))


def get_logger(log_file):
    from logging import getLogger, FileHandler, StreamHandler
    from logging import Formatter, DEBUG, INFO
    fh = FileHandler(log_file)
    fh.setLevel(INFO)
    sh = StreamHandler()
    sh.setLevel(DEBUG)
    for handler in [fh, sh]:
        formatter = Formatter('%(asctime)s - %(message)s')
        handler.setFormatter(formatter)
    logger = getLogger('log')
    logger.setLevel(DEBUG)
    logger.addHandler(fh)
    logger.addHandler(sh)
    return logger


class AverageMeter(object):
    """Computes and stores the average and current value
        adopted from https://github.com/pytorch/examples/blob/master/imagenet/main.py#L296
    """

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
