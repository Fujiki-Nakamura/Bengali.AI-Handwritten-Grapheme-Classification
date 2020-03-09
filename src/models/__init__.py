from torchvision import models
import pretrainedmodels
from .senet import se_resnext50_32x4d_1  # noqa
from .senet_dropblock import se_resnext50_32x4d_dropblock_1  # noqa
from .senet_dropblock_2 import se_resnext50_32x4d_dropblock_2  # noqa
from .senet_dropblock_3 import se_resnext50_32x4d_dropblock_3  # noqa
from .resnet import *
from .vgg import *
from .efficientnet_pytorch import EfficientNet


pretrainedmodels_model_name_list = [
    'se_resnet50',
    'se_resnext50_32x4d',
]
senet_model_name_list = [
    'se_resnext50_32x4d_1',
    'se_resnext50_32x4d_dropblock_1',
    'se_resnext50_32x4d_dropblock_2',
    'se_resnext50_32x4d_dropblock_3',
]


def get_model(cfg):
    if cfg.model.name.startswith('efficientnet'):
        model = EfficientNet.from_name(
            model_name=cfg.model.name,
            input_c=cfg.model.input_c, n_outputs=cfg.model.n_outputs)
    elif cfg.model.name in pretrainedmodels_model_name_list:
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
        model = eval(cfg.model.name)(
            num_classes=1000, pretrained=cfg.model.pretrained_type,
            dropout_p=cfg.model.dropout_p,
            strides=cfg.model.strides, adaptive_pool=cfg.model.adaptive_pool,
            input_c=cfg.model.get('input_c', 3),
            input_3x3=cfg.model.get('input_3x3', False),
        )
        model.last_linear = nn.Linear(512 * block_expansion, cfg.model.n_outputs)
    else:
        model = eval(cfg.model.name)(
            input_dim=cfg.model.input_dim,
            num_classes=cfg.model.n_outputs,
            **cfg.model.config
        )
    return model
