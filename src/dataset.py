import numpy as np
import torch
from torch.utils.data import Dataset
import albumentations as alb
from albumentations.pytorch import ToTensor


def parse_arg_str(arg_str):
    arg_dict = {}
    for arg in arg_str.split(', '):
        arg_dict[arg.split('=')[0]] = eval(arg.split('=')[1])
    return arg_dict


class MyDataset(Dataset):
    def __init__(self, data, label, config, mode='train'):
        self.data = np.expand_dims(data, axis=3)
        self.label = label
        self.is_training = (mode in ['train', 'valid'])
        self.input_c = config.model.input_dim
        self.input_h = config.data.input_h
        self.input_w = config.data.input_w

        # transform
        transform_list = []
        for aug in config.data.augmentation:
            name = aug.split('/')[0]
            arg_str = aug.split('/')[1]
            transform_list.append(alb.__dict__[name](**parse_arg_str(arg_str)))
        self.transform = alb.Compose([
            alb.Resize(self.input_h, self.input_w),
            ToTensor(),
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # input
        input_ = self.data[idx]
        if self.input_c == 3:
            input_ = np.concatenate([input_, ]*3, axis=2)
        if self.transform is not None:
            input_ = self.transform(image=input_)['image']

        if not self.is_training:
            return input_

        # target
        target = self.label[idx]
        target = torch.from_numpy(np.asarray(target)).type(torch.LongTensor)

        return input_, target
