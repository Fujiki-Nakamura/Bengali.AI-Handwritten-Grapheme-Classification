from collections import OrderedDict
import numpy as np
from sklearn.metrics import recall_score
import torch
from torch.nn import functional as F
from tqdm import tqdm

from common import component_list, N_GRAPHEME, N_VOWEL, N_CONSONANT
from loss import ohem_loss
import utils


def training(
    dataloader, model, criterion, optimizer, config, is_training=True,
    using_ohem_loss=False, lr=None
):
    cfg = config
    device = config.general.device
    if is_training:
        model.train()
    else:
        model.eval()
    losses = utils.AverageMeter()
    pred = {'grapheme': [], 'vowel': [], 'consonant': []}
    true = {'grapheme': [], 'vowel': [], 'consonant': []}
    mode = 'train' if is_training else 'valid'
    _desc = f'[{config.general.expid}] {mode}'
    _desc = _desc + f' lr {lr:.4f}' if lr is not None else _desc
    _desc = _desc + f' OHEMLoss' if using_ohem_loss else _desc
    pbar = tqdm(total=len(dataloader), desc=_desc, position=0)
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(is_training):
            bs = target.size(0)

            aug_type = 'None  '
            # CutMix
            r = np.random.rand(1)
            r_mixup = np.random.rand(1)
            if is_training and config.cutmix.beta > 0 and r < config.cutmix.prob:
                aug_type = 'CutMix'
                lam = np.random.beta(config.cutmix.beta, config.cutmix.beta)
                rand_index = torch.randperm(data.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                bbx1, bby1, bbx2, bby2 = rand_bbox(data.size(), lam)
                data[:, :, bbx1:bbx2, bby1:bby2] = data[rand_index, :, bbx1:bbx2, bby1:bby2]  # noqa
                lam = 1 - ((bbx2 - bbx1) * (bby2 - bby1) / (data.size()[-1] * data.size()[-2]))  # noqa
                output = model(data)
                outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
                loss = 0.
                for i in range(len(component_list)):
                    coef = cfg.training.coef_list[i]
                    if using_ohem_loss:
                        _rate = cfg.training.ohem_rate
                        loss += coef * (
                            ohem_loss(_rate, outputs[i], target_a[:, i]) * lam +
                            ohem_loss(_rate, outputs[i], target_b[:, i]) * (1. - lam))
                    else:
                        loss += coef * (
                            criterion(outputs[i], target_a[:, i]) * lam +
                            criterion(outputs[i], target_b[:, i]) * (1. - lam))
            elif is_training and config.mixup.beta > 0 and r_mixup < config.mixup.prob:
                aug_type = 'Mixup'
                lam = np.random.beta(config.mixup.beta, config.mixup.beta)
                rand_index = torch.randperm(data.size()[0]).cuda()
                target_a = target
                target_b = target[rand_index]
                data = data * lam + data[rand_index] * (1 - lam)
                output = model(data)
                outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
                loss = 0.
                for i in range(len(component_list)):
                    coef = cfg.training.coef_list[i]
                    loss += coef * (
                            criterion(outputs[i], target_a[:, i]) * lam +
                            criterion(outputs[i], target_b[:, i]) * (1. - lam))
            else:
                output = model(data)
                outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
                loss = 0.
                if using_ohem_loss:
                    _rate = cfg.training.ohem_rate
                    for i in range(len(component_list)):
                        coef = cfg.training.coef_list[i]
                        loss += coef * ohem_loss(_rate, outputs[i], target[:, i])
                else:
                    # TODO: refactor
                    loss_grapheme = criterion(outputs[0], target[:, 0])
                    loss_vowel = criterion(outputs[1], target[:, 1])
                    loss_consonant = criterion(outputs[2], target[:, 2])
                    loss = 0.
                    loss += cfg.training.coef_list[0] * loss_grapheme
                    loss += cfg.training.coef_list[1] * loss_vowel
                    loss += cfg.training.coef_list[2] * loss_consonant
            losses.update(loss.item(), bs)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for component_i, component in enumerate(component_list):
                pred[component].extend(
                    F.softmax(outputs[component_i], dim=1).max(1)[1].detach().cpu().numpy().tolist())  # noqa
                true[component].extend(target[:, component_i].cpu().numpy().tolist())

        pbar.set_postfix(OrderedDict(aug=aug_type))
        pbar.update(1)
        if config.training.single_iter: break  # noqa
    pbar.close()

    scores = []
    for component in ['grapheme', 'consonant', 'vowel']:
        scores.append(recall_score(true[component], pred[component], average='macro'))
    final_score = np.average(scores, weights=[2, 1, 1])

    return {'loss': losses.avg, 'score': final_score}


def rand_bbox(size, lam):
    W = size[2]
    H = size[3]
    cut_rat = np.sqrt(1. - lam)
    cut_w = np.int(W * cut_rat)
    cut_h = np.int(H * cut_rat)

    # uniform
    cx = np.random.randint(W)
    cy = np.random.randint(H)

    bbx1 = np.clip(cx - cut_w // 2, 0, W)
    bby1 = np.clip(cy - cut_h // 2, 0, H)
    bbx2 = np.clip(cx + cut_w // 2, 0, W)
    bby2 = np.clip(cy + cut_h // 2, 0, H)

    return bbx1, bby1, bbx2, bby2
