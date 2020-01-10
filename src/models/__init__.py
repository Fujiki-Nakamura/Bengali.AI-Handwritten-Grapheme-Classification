from torchvision import models
from .resnet import *
from .vgg import *


def get_model(cfg):
    model = eval(cfg.model.name)(
        input_dim=cfg.model.input_dim,
        num_classes=cfg.model.n_outputs,
        **cfg.model.config
    )
    return model
