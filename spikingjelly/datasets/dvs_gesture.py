import spikingjelly
import zipfile
import os
import threading
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
# https://www.research.ibm.com/dvsgesture/
# https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794
resources = [

]
class DvsGesture(Dataset):
    @staticmethod
    def download(dir: str):
        pass

