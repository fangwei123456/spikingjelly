import spikingjelly.datasets
import zipfile
import os
import threading
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import utils

# https://www.garrickorchard.com/datasets/n-mnist
# https://www.dropbox.com/sh/tg2ljlbmtzygrag/AABrCc6FewNZSNsoObWJqY74a?dl=0
resource = {
    'Train': ['https://www.garrickorchard.com/datasets/n-mnist', '20959b8e626244a1b502305a9e6e2031'],
    'Test': ['https://www.garrickorchard.com/datasets/n-mnist', '69ca8762b2fe404d9b9bad1103e97832']
}


class NMNIST(spikingjelly.datasets.EventsFramesDatasetBase):

    @staticmethod
    def get_wh():
        return 34, 34

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: N-MNIST原始bin格式数据的文件名
        :return: 一个字典，键是{'t', 'x', 'y', 'p'}，值是np数组

        原始的N-MNIST提供的是bin格式数据，不能直接读取。本函数提供了一个读取的接口。
        本函数参考了 https://github.com/jackd/events-tfds 的代码。

        原始数据以二进制存储：

        Each example is a separate binary file consisting of a list of events. Each event occupies 40 bits as described below:
        bit 39 - 32: Xaddress (in pixels)
        bit 31 - 24: Yaddress (in pixels)
        bit 23: Polarity (0 for OFF, 1 for ON)
        bit 22 - 0: Timestamp (in microseconds)


        '''

        with open(file_name, 'rb') as bin_f:
            # `& 128` 是取一个8位二进制数的最高位
            # `& 127` 是取其除了最高位，也就是剩下的7位
            raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            x = raw_data[0::5]
            y = raw_data[1::5]
            rd_2__5 = raw_data[2::5]
            p = (rd_2__5 & 128) >> 7
            t = ((rd_2__5 & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        return {'t': t, 'x': x, 'y': y, 'p': p}

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        for key in resource.keys():
            file_name = os.path.join(download_root, key + '.zip')
            if os.path.exists(file_name):
                print('Train.zip already exists, check md5')
                if utils.check_md5(file_name, resource[key][1]):
                    print('md5 checked, extracting...')
                    utils.extract_archive(file_name, extract_root)
                else:
                    print(f'{file_name} corrupted.')
                    print(f'Please re-download {file_name} from {resource[key]} and save to {download_root} manually.')
                    raise NotImplementedError
            else:
                print(f'Please download from {resource[key]} and save to {download_root} manually.')
                raise NotImplementedError

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str,
                              normalization: str or None):
        width, height = NMNIST.get_wh()
        thread_list = []
        for key in resource.keys():
            source_dir = os.path.join(events_data_dir, key)
            target_dir = os.path.join(frames_data_dir, key)
            os.mkdir(target_dir)
            print(f'mkdir {target_dir}')
            print(f'convert {source_dir} to {target_dir}')
            for sub_dir in utils.list_dir(source_dir):
                source_sub_dir = os.path.join(source_dir, sub_dir)
                target_sub_dir = os.path.join(target_dir, sub_dir)
                os.mkdir(target_sub_dir)
                thread_list.append(spikingjelly.datasets.FunctionThread(
                    spikingjelly.datasets.convert_events_dir_to_frames_dir,
                    source_sub_dir, target_sub_dir, '.bin',
                    NMNIST.read_bin, height, width, frames_num, split_by, normalization))
                thread_list[-1].start()
                print(f'thread {thread_list.__len__() - 1} start')

            for i in range(thread_list.__len__()):
                thread_list[i].join()
                print(f'thread {i} finished')

    @staticmethod
    def get_events_item(file_name):
        return NMNIST.read_bin(file_name), int(os.path.dirname(file_name)[-1])

    @staticmethod
    def get_frames_item(file_name):
        return np.load(file_name), int(os.path.dirname(file_name)[-1])

    def __init__(self, root: str, train: bool, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        super().__init__()
        self.train = train
        events_root = os.path.join(root, 'events')
        if os.path.exists(events_root):
            print(f'{events_root} already exists')
        else:
            self.download_and_extract(root, events_root)

        self.use_frame = use_frame
        if use_frame:
            frames_root = os.path.join(root, 'frames')
            if os.path.exists(frames_root):
                print(f'{frames_root} already exists')
            else:
                os.mkdir(frames_root)
                print(f'mkdir {frames_root}')
                self.create_frames_dataset(events_root, frames_root, frames_num, split_by, normalization)
        self.data_dir = os.path.join(frames_root if use_frame else events_root, 'Train' if train else 'Test')

        self.file_name = []
        for sub_dir in utils.list_dir(self.data_dir, True):
            if self.use_frame:
                self.file_name.extend(utils.list_files(sub_dir, '.npy', True))
            else:
                self.file_name.extend(utils.list_files(sub_dir, '.bin', True))

    def __len__(self):
        return self.file_name.__len__()

    def __getitem__(self, index):
        if self.use_frame:
            return self.get_frames_item(self.file_name[index])
        else:
            return self.get_events_item(self.file_name[index])

