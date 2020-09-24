from .utils import (
    EventsFramesDatasetBase, 
    convert_events_dir_to_frames_dir,
    FunctionThread,
    normalize_frame,
)
import os
import tqdm
import numpy as np
from torchvision.datasets import utils
import multiprocessing
import shutil
import scipy.io
import torch
labels_dict = {
'a': 0,
'b': 1,
'c': 2,
'd': 3,
'e': 4,
'f': 5,
'g': 6,
'h': 7,
'i': 8,
'k': 9,
'l': 10,
'm': 11,
'n': 12,
'o': 13,
'p': 14,
'q': 15,
'r': 16,
's': 17,
't': 18,
'u': 19,
'v': 20,
'w': 21,
'x': 22,
'y': 23
}  # gesture_mapping.csv
# url md5
resource = ['https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0',
            '8b46191acf6c1760ad3f2d2cb4380e24']


class ASLDVS(EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 240, 180

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        file_name = os.path.join(download_root, 'ICCV2019_DVS_dataset.zip')
        if os.path.exists(file_name):
            print('ICCV2019_DVS_dataset.zip already exists, check md5')

            if utils.check_md5(file_name, resource[1]):
                print('md5 checked, extracting...')
                temp_extract_root = os.path.join(download_root, 'temp_extract')
                os.mkdir(temp_extract_root)
                utils.extract_archive(file_name, temp_extract_root)
                for zip_file in tqdm.tqdm(utils.list_files(temp_extract_root, '.zip')):
                    utils.extract_archive(os.path.join(temp_extract_root, zip_file), extract_root)
                shutil.rmtree(temp_extract_root)
                return
            else:
                print(f'{file_name} corrupted.')

        print(f'Please download from {resource[0]} and save to {download_root} manually.')
        raise NotImplementedError

    @staticmethod
    def read_bin(file_name: str):
        events = scipy.io.loadmat(file_name)
        return {
            't': events['ts'].squeeze(),
            'x': events['x'].squeeze(),
            'y': events['y'].squeeze(),
            'p': events['pol'].squeeze()
        }

    @staticmethod
    def get_events_item(file_name):
        base_name = os.path.basename(file_name)
        return ASLDVS.read_bin(file_name), labels_dict[base_name[0]]

    @staticmethod
    def get_frames_item(file_name):
        base_name = os.path.basename(file_name)
        return torch.from_numpy(np.load(file_name)['arr_0']).float(), labels_dict[base_name[0]]

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        width, height = ASLDVS.get_wh()
        thread_list = []
        for source_dir in utils.list_dir(events_data_dir):
            abs_source_dir = os.path.join(events_data_dir, source_dir)
            abs_target_dir = os.path.join(frames_data_dir, source_dir)
            if not os.path.exists(abs_target_dir):
                os.mkdir(abs_target_dir)
                print(f'mkdir {abs_target_dir}')
            print(f'thread {thread_list.__len__()} convert events data in {abs_source_dir} to {abs_target_dir}')
            thread_list.append(
                FunctionThread(convert_events_dir_to_frames_dir,
                                                     abs_source_dir, abs_target_dir, '.mat', ASLDVS.read_bin,
                                                     height, width, frames_num, split_by, normalization, 1, True))
            # 文件数量太多，体积太大，因此采用压缩格式

        # 这个数据集的文件夹数量24，一次启动24个线程太多，因此一次只启动max_running_threads个线程
        max_running_threads = max(multiprocessing.cpu_count(), 8)
        for i in range(0, thread_list.__len__(), max_running_threads):
            for j in range(i, min(i + max_running_threads, thread_list.__len__())):
                thread_list[j].start()
                print(f'thread {j} start')
            for j in range(i, min(i + max_running_threads, thread_list.__len__())):
                print('thread', j, 'join')
                thread_list[j].join()
                print('thread', j, 'finished')

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

        ASL-DVS数据集，出自 `Graph-Based Object Classification for Neuromorphic Vision Sensing <https://arxiv.org/abs/1908.06648>`_，
        包含24个英文字母（从A到Y，排除J）的美国手语，American Sign Language (ASL)。更多信息参见 https://github.com/PIX2NVS/NVS2Graph，
        原始数据的下载地址为 https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0。

        关于转换成帧数据的细节，参见 :func:`~spikingjelly.datasets.utils.integrate_events_to_frames`。
        '''
        super().__init__()
        self.train = train
        events_root = os.path.join(root, 'events')
        if os.path.exists(events_root):
            # 如果root目录下存在events_root目录
            print(f'events data root {events_root} already exists.')
        else:
            self.download_and_extract(root, events_root)
        self.file_name = []  # 保存数据文件的路径
        self.use_frame = use_frame
        self.data_dir = None

        if use_frame:
            self.normalization = normalization
            if normalization == 'frequency':
                dir_suffix = normalization
            else:
                dir_suffix = None
            frames_root = os.path.join(root,
                                       f'frames_num_{frames_num}_split_by_{split_by}_normalization_{dir_suffix}')
            if os.path.exists(frames_root):
                # 如果root目录下存在frames_root目录，且frames_root下有10个子文件夹，则认为数据集文件存在
                print(f'frames data root {frames_root} already exists.')
            else:
                os.mkdir(frames_root)
                self.create_frames_dataset(events_root, frames_root, frames_num, split_by, normalization)
            self.data_dir = frames_root
        else:
            self.data_dir = events_root

        if train:
            # + 1 是因为文件的下标从1开始
            index = np.arange(0, int(split_ratio * 4200)) + 1
        else:
            index = np.arange(int(split_ratio * 4200), 4200) + 1

        for class_name in labels_dict.keys():
            class_dir = os.path.join(self.data_dir, class_name)
            for i in index:
                self.file_name.append(os.path.join(class_dir, class_name + '_' + str(i).zfill(4)))

    def __len__(self):
        return self.file_name.__len__()

    def __getitem__(self, index):
        if self.use_frame:
            frames, labels = self.get_frames_item(self.file_name[index] + '.npz')
            if self.normalization is not None and self.normalization != 'frequency':
                frames = normalize_frame(frames, self.normalization)
            return frames, labels
        else:
            return self.get_events_item(self.file_name[index] + '.mat')


