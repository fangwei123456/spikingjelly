import torch
import numpy as np
import os
from torch.utils.data import dataset

class N_MNIST(dataset.Dataset):
    def __init__(self, root, train=True):
        super().__init__()
        if train:
            data_dict = np.load(os.path.join(root, 'train_dataset.npz'))
        else:
            data_dict = np.load(os.path.join(root, 'test_dataset.npz'))

        self.spikes = data_dict['data'].transpose(0, 2, 1, 3, 4)
        self.labels = data_dict['label']
    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, index):
        return torch.from_numpy(self.spikes[index]), self.labels[index].item()