from torchvision import models
import pretrainedmodels
from .senet import se_resnext50_32x4d_1
from .resnet import *
from .vgg import *


pretrainedmodels_model_name_list = [
    'se_resnet50',
    'se_resnext50_32x4d',
]
senet_model_name_list = [
    'se_resnext50_32x4d_1',
]


def get_model(cfg):
    if cfg.model.name in pretrainedmodels_model_name_list:
        from torch import nn
        block_expansion = 4
        model = pretrainedmodels.__dict__[cfg.model.name](
            num_classes=1000,
            pretrained=cfg.model.pretrained_type,
        )
        model.last_linear = nn.Linear(512 * block_expansion, cfg.model.n_outputs)
    elif cfg.model.name in senet_model_name_list:
        from torch import nn
        block_expansion = 4
        model = se_resnext50_32x4d_1(
            num_classes=1000,
            pretrained=cfg.model.pretrained_type,
        )
        model.last_linear = nn.Linear(512 * block_expansion, cfg.model.n_outputs)
    else:
        model = eval(cfg.model.name)(
            input_dim=cfg.model.input_dim,
            num_classes=cfg.model.n_outputs,
            **cfg.model.config
        )
    return model
