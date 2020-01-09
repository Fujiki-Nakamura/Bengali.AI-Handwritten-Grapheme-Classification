import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision import transforms


class MyDataset(Dataset):
    def __init__(self, data, label, train=False):
        self.data = data
        self.label = label
        self.train = train
        # self.input_h = 224
        # self.input_w = 224
        self.transform = transforms.Compose([
            # transforms.Resize((self.input_h, self.input_w)),
            transforms.ToTensor(),
        ])

    def __len__(self):
        return len(self.df_label)

    def __getitem__(self, idx):
        # input
        input_ = self.data[idx]
        if self.transform is not None:
            input_ = self.transform(input_)

        if self.test:
            return input_

        # target
        target = self.label[idx]
        target = torch.from_numpy(np.asarray(target).astype(np.float32))

        return input_, target
