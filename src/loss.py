import torch
import torch.nn as nn
import torch.nn.functional as F


def get_loss_fn(cfg):
    _loss = cfg.training.loss
    losses = ['CrossEntropyLoss', 'BCEWithLogitsLoss',
              'MultiLabelSoftMarginLoss']
    if _loss in losses:
        loss_fn = nn.__dict__.get(_loss)()
    elif _loss == 'OHEMLoss':
        loss_fn = ohem_loss
    else:
        raise NotImplementedError()
    return loss_fn


def ohem_loss(rate, cls_pred, cls_target):
    batch_size = cls_pred.size(0)
    ohem_cls_loss = F.cross_entropy(
        cls_pred, cls_target, reduction='none', ignore_index=-1)

    sorted_ohem_loss, idx = torch.sort(ohem_cls_loss, descending=True)
    keep_num = min(sorted_ohem_loss.size()[0], int(batch_size*rate))
    if keep_num < sorted_ohem_loss.size()[0]:
        keep_idx_cuda = idx[:keep_num]
        ohem_cls_loss = ohem_cls_loss[keep_idx_cuda]
    cls_loss = ohem_cls_loss.sum() / keep_num
    return cls_loss
