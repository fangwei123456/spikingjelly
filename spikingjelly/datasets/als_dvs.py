import spikingjelly.datasets
import zipfile
import os
import threading
import tqdm
import numpy as np
import struct
from torchvision.datasets import utils
import time
import multiprocessing

labels_dict = {

}  # gesture_mapping.csv
# url md5
resource = ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0', None]

