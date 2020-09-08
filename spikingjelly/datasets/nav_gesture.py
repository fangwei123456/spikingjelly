import spikingjelly
import zipfile
import os
import threading
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import utils

# url md5
resource = {
    'walk':['https://www.neuromorphic-vision.com/public/downloads/navgesture/navgesture-walk.zip', None],
    'sit':['https://www.neuromorphic-vision.com/public/downloads/navgesture/navgesture-sit.zip', None]
}
class NavGesture(Dataset):
    @staticmethod
    def download(dataset_name: str, download_root: str, extract_root=None):
        utils.download_and_extract_archive(url=resource[dataset_name][0], download_root=download_root,
                                           extract_root=extract_root, md5=resource[dataset_name][1])

if __name__ == '__main__':
    NavGesture.download('walk', './NavGesture')