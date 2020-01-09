import torch.nn as nn


def get_loss_fn(cfg):
    _loss = cfg.training.loss
    losses = ['CrossEntropyLoss', 'BCEWithLogitsLoss', 'MultiLabelSoftMarginLoss']
    if _loss in losses:
        loss_fn = nn.__dict__.get(_loss)()
    else:
        raise NotImplementedError()
    return loss_fn
