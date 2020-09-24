from .utils import (
    EventsFramesDatasetBase, 
    convert_events_dir_to_frames_dir,
    FunctionThread,
    normalize_frame,
)
import numpy as np
import os
from torchvision.datasets import utils
import torch
labels_dict = {
    'airplane': 0,
    'automobile': 1,
    'bird': 2,
    'cat': 3,
    'deer': 4,
    'dog': 5,
    'frog': 6,
    'horse': 7,
    'ship': 8,
    'truck': 9
}
# https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671
resource = {
    'airplane': ('https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
    'automobile': ('https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
    'bird': ('https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
    'cat': ('https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
    'deer': ('https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
    'dog': ('https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
    'frog': ('https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
    'horse': ('https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
    'ship': ('https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
    'truck': ('https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
}
# https://github.com/jackd/events-tfds/blob/master/events_tfds/data_io/aedat.py


EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event

def read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def load_raw_events(fp,
                    bytes_skip=0,
                    bytes_trim=0,
                    filter_dvs=False,
                    times_first=False):
    p = skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def parse_raw_address(addr,
                      x_mask=x_mask,
                      x_shift=x_shift,
                      y_mask=y_mask,
                      y_shift=y_shift,
                      polarity_mask=polarity_mask,
                      polarity_shift=polarity_shift):
    polarity = read_bits(addr, polarity_mask, polarity_shift).astype(np.bool)
    x = read_bits(addr, x_mask, x_shift)
    y = read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def load_events(
        fp,
        filter_dvs=False,
        # bytes_skip=0,
        # bytes_trim=0,
        # times_first=False,
        **kwargs):
    timestamp, addr = load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity



class CIFAR10DVS(EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 128, 128

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        for key in resource.keys():
            file_name = os.path.join(download_root, key + '.zip')
            if os.path.exists(file_name):
                if utils.check_md5(file_name, resource[key][1]):
                    print(f'extract {file_name} to {extract_root}')
                    utils.extract_archive(file_name, extract_root)
                else:
                    print(f'{file_name} corrupted, re-download...')
                    utils.download_and_extract_archive(resource[key][0], download_root, extract_root,
                                                       filename=key + '.zip',
                                                       md5=resource[key][1])
            else:
                utils.download_and_extract_archive(resource[key][0], download_root, extract_root, filename=key + '.zip',
                                                   md5=resource[key][1])


    @staticmethod
    def read_bin(file_name: str):
        with open(file_name, 'rb') as fp:
            t, x, y, p = load_events(fp,
                        x_mask=0xfE,
                        x_shift=1,
                        y_mask=0x7f00,
                        y_shift=8,
                        polarity_mask=1,
                        polarity_shift=None)
            return {'t': t, 'x': 127 - x, 'y': y, 'p': 1 - p.astype(int)}
        # 原作者的代码可能有一点问题，因此不是直接返回 t x y p

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str,
                              normalization: str or None):
        width, height = CIFAR10DVS.get_wh()
        thread_list = []
        for key in resource.keys():
            source_dir = os.path.join(events_data_dir, key)
            target_dir = os.path.join(frames_data_dir, key)
            os.mkdir(target_dir)
            print(f'mkdir {target_dir}')
            print(f'convert {source_dir} to {target_dir}')
            thread_list.append(FunctionThread(
                convert_events_dir_to_frames_dir,
                source_dir, target_dir, '.aedat',
                CIFAR10DVS.read_bin, height, width, frames_num, split_by, normalization, 1, True))
            thread_list[-1].start()
            print(f'thread {thread_list.__len__() - 1} start')

        for i in range(thread_list.__len__()):
            thread_list[i].join()
            print(f'thread {i} finished')

    @staticmethod
    def get_frames_item(file_name):
        return torch.from_numpy(np.load(file_name)['arr_0']).float(), labels_dict[file_name.split('_')[-2]]

    @staticmethod
    def get_events_item(file_name):
        return CIFAR10DVS.read_bin(file_name), labels_dict[file_name.split('_')[-2]]

    def __init__(self, root: str, train: bool, split_ratio=0.9, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        '''
        :param root: 保存数据集的根目录
        :type root: str
        :param train: 是否使用训练集
        :type train: bool
        :param split_ratio: 分割比例。每一类中前split_ratio的数据会被用作训练集，剩下的数据为测试集
        :type split_ratio: float
        :param use_frame: 是否将事件数据转换成帧数据
        :type use_frame: bool
        :param frames_num: 转换后数据的帧数
        :type frames_num: int
        :param split_by: 脉冲数据转换成帧数据的累计方式。``'time'`` 或 ``'number'``
        :type split_by: str
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；
                        为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                        为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                        为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :type normalization: str or None

        CIFAR10 DVS数据集，出自 `CIFAR10-DVS: An Event-Stream Dataset for Object Classification <https://www.frontiersin.org/articles/10.3389/fnins.2017.00309/full>`_，
        数据来源于DVS相机拍摄的显示器上的CIFAR10图片。原始数据的下载地址为 https://figshare.com/articles/dataset/CIFAR10-DVS_New/4724671。

        关于转换成帧数据的细节，参见 :func:`~spikingjelly.datasets.utils.integrate_events_to_frames`。
        '''
        super().__init__()
        self.train = train
        events_root = os.path.join(root, 'events')
        if os.path.exists(events_root):
            print(f'{events_root} already exists')
        else:
            self.download_and_extract(root, events_root)

        self.use_frame = use_frame
        if use_frame:
            self.normalization = normalization
            if normalization == 'frequency':
                dir_suffix = normalization
            else:
                dir_suffix = None
            frames_root = os.path.join(root, f'frames_num_{frames_num}_split_by_{split_by}_normalization_{dir_suffix}')
            if os.path.exists(frames_root):
                print(f'{frames_root} already exists')
            else:
                os.mkdir(frames_root)
                print(f'mkdir {frames_root}')
                self.create_frames_dataset(events_root, frames_root, frames_num, split_by, normalization)
        self.data_dir = frames_root if use_frame else events_root

        self.file_name = []
        if train:
            index = np.arange(0, int(split_ratio * 1000))
        else:
            index = np.arange(int(split_ratio * 1000), 1000)

        for class_name in labels_dict.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            for i in index:
                if self.use_frame:
                    self.file_name.append(os.path.join(class_dir, 'cifar10_' + class_name + '_' + str(i) + '.npz'))
                else:
                    self.file_name.append(os.path.join(class_dir, 'cifar10_' + class_name + '_' + str(i) + '.aedat'))

    def __len__(self):
        return self.file_name.__len__()


    def __getitem__(self, index):
        if self.use_frame:
            frames, labels = self.get_frames_item(self.file_name[index])
            if self.normalization is not None and self.normalization != 'frequency':
                frames = normalize_frame(frames, self.normalization)
            return frames, labels
        else:
            return self.get_events_item(self.file_name[index])
