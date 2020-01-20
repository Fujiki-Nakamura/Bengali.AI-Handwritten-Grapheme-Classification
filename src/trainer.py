import numpy as np
from sklearn.metrics import recall_score
import torch
from torch.nn import functional as F

from common import component_list, N_GRAPHEME, N_VOWEL, N_CONSONANT
import utils


def training(dataloader, model, criterion, optimizer, config, is_training=True):
    device = config.general.device
    if is_training:
        model.train()
    else:
        model.eval()
    losses = utils.AverageMeter()
    pred = {'grapheme': [], 'vowel': [], 'consonant': []}
    true = {'grapheme': [], 'vowel': [], 'consonant': []}
    for data, target in dataloader:
        data = data.to(device)
        target = target.to(device)
        with torch.set_grad_enabled(is_training):
            bs = target.size(0)

            # CutMix
            r = np.random.rand(1)
            if is_training and config.cutmix.beta > 0 and r < config.cutmix.prob:
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
                    loss += criterion(outputs[i], target_a[:, i]) * lam + criterion(
                            outputs[i], target_b[:, i]) * (1. - lam)
            else:
                output = model(data)
                outputs = torch.split(output, [N_GRAPHEME, N_VOWEL, N_CONSONANT], dim=1)
                loss_grapheme = criterion(outputs[0], target[:, 0])
                loss_vowel = criterion(outputs[1], target[:, 1])
                loss_consonant = criterion(outputs[2], target[:, 2])
                loss = loss_grapheme + loss_vowel + loss_consonant
            losses.update(loss.item(), bs)

            if is_training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            for component_i, component in enumerate(component_list):
                pred[component].extend(
                    F.softmax(outputs[component_i], dim=1).max(1)[1].detach().cpu().numpy().tolist())  # noqa
                true[component].extend(target[:, component_i].cpu().numpy().tolist())

        if config.training.single_iter: break  # noqa

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
