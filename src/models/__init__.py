from torchvision import models
from .resnet import *


def get_model(cfg):
    if cfg.model.name.startswith('resnet'):
        model = eval(cfg.model.name)(
            input_dim=cfg.model.input_dim,
            num_classes=cfg.model.n_outputs,
            **cfg.model.config
        )
    else:
        model = models.__dict__[cfg.model.name](
            num_classes=cfg.model.n_outputs, **cfg.model.config)
    return model
