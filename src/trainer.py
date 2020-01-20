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
