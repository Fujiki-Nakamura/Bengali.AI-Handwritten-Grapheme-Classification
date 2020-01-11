import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data, label, config, mode='train'):
        self.data = np.expand_dims(data, axis=3)
        self.label = label
        self.is_training = (mode == 'train' or 'valid')
        # self.input_h = 224
        # self.input_w = 224
        self.transform = transforms.Compose([
            # transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.label)

    def __getitem__(self, idx):
        # input
        input_ = self.data[idx]
        if self.transform is not None:
            input_ = self.transform(input_)

        if not self.is_training:
            return input_

        # target
        target = self.label[idx]
        target = torch.from_numpy(np.asarray(target)).type(torch.LongTensor)

        return input_, target
