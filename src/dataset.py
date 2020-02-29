import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensor

import augment
from utils import parse_arg_str


augment_name_list = ['GridMask', ]


class MyDataset(Dataset):
    def __init__(self, data, label, config, mode='train'):
        assert mode in ['train', 'valid', 'test']
        self.data = np.expand_dims(data, axis=3)
        self.label = label
        self.is_training = (mode == 'train')
        self.is_testing = (mode == 'test')
        self.input_c = config.model.input_dim
        self.input_h = config.data.input_h
        self.input_w = config.data.input_w

        # transform
        transform_list = []
        transform_list.append(alb.Resize(height=self.input_h, width=self.input_w))
        if self.is_training:
            for aug in config.data.augmentation:
                name = aug.split('/')[0]
                arg_str = aug.split('/')[1]
                if name in augment_name_list:
                    transform_list.append(
                        augment.__dict__[name](**parse_arg_str(arg_str)))
                else:
                    transform_list.append(alb.__dict__[name](**parse_arg_str(arg_str)))
        transform_list.append(ToTensor())
        self.transform = alb.Compose(transform_list)

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # input
        input_ = self.data[idx]
        if self.input_c == 3:
            input_ = np.concatenate([input_, ]*3, axis=2)
        if self.transform is not None:
            input_ = self.transform(image=input_)['image']

        if self.is_testing:
            return input_

        # target
        target = self.label[idx]
        target = torch.from_numpy(np.asarray(target)).type(torch.LongTensor)

        return input_, target
