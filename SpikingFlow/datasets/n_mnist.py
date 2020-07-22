import SpikingFlow
import zipfile
import os
import threading
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset

class NMNIST(Dataset):

    @ staticmethod
    def download_zip(zip_dir):
        '''
        :param zip_dir: 保存下载的N-MNIST训练集和测试集zip压缩包的文件夹
        :return: None

        .. warning::
            代码尚未完成，请勿使用。

        运行此函数，会将N-MNIST训练集和测试集zip压缩包下载到 ``zip_dir``，下载好\
        的文件夹是如下形式：

        .. code-block:: bash

            zip_dir/
            |-- Train.zip
            |-- Test.zip
        '''
        raise NotImplementedError

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: N-MNIST原始bin格式数据的文件名
        :return: 一个字典，键是{'t', 'x', 'y', 'p'}，值是np数组
        
        原始的N-MNIST提供的是bin格式数据，不能直接读取。本函数提供了一个读取的接口。
        '''

        with open(file_name, 'rb') as bin_f:
            raw_data = np.uint32(np.fromfile(bin_f, dtype=np.uint8))
            # y = raw_data[1::5]
            # x = raw_data[0::5]
            x = raw_data[1::5]
            y = raw_data[0::5]
            p = (raw_data[2::5] & 128) >> 7  # bit 7
            t = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])
        return {'t': t, 'x': x, 'y': y, 'p': p}

    @staticmethod
    def create_frames_dataset(events_data_dir, frames_data_dir, frames_num=10, split_by='time', normalization=None):
        '''
        :param events_data_dir: 保存N-MNIST原始数据集bin文件的文件夹，例如解压后的 ``Train`` 文件夹
        :param frames_data_dir: 保存frames数据的文件夹
        :param frames_num: 转换后数据的帧数
        :param split_by: ``'time'`` 或 ``'number'``。为 ``'time'`` 表示将events数据在时间上分段，例如events记录的 ``t`` 介于
                        [0, 105]且 ``frames_num=10``，则转化得到的10帧分别为 ``t`` 属于[0, 10), [10,20), ..., [90, 105)的
                        脉冲的累加；
                        为 ``'number'`` 表示将events数据在数量上分段，例如events一共有105个且 ``frames_num=10``，则转化得到
                        的10帧分别是第[0, 10), [10,20), ..., [90, 105)个脉冲的累加
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                            为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                            为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :return: None

        将N-MNIST的events数据进行分段，每段累加得到一帧，转换成 ``frames_num`` 帧数。

        ``events_data_dir`` 文件夹应该包含解压后的N-MNIST数据集，会具有如下的格式：

        .. code-block:: bash

            events_data_dir/
            |-- 0
            |   |-- 0.bin
            |   |-- ...
            |-- 1
            |-- ..


        本函数会在 ``frames_data_dir`` 文件夹下生成与 ``events_data_dir`` 相同的目录结构，每个子文件夹内也是npz的数据，以键为 ``'arr_0'`` 的字典（numpy默认）保存数据。
        '''
        def cvt_data_in_dir(source_dir, target_dir, show_bar):
            print('processing', source_dir)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                print('mkdir', target_dir)
            if show_bar:
                for file_name in tqdm.tqdm(os.listdir(source_dir)):
                    events = NMNIST.read_bin(os.path.join(source_dir, file_name))
                    # events: {'t', 'x', 'y', 'p'}
                    frames = SpikingFlow.datasets.integrate_events_to_frames(events=events, weight=34, height=34,
                                                                               frames_num=frames_num, split_by=split_by, normalization=normalization)
                    # os.path.splitext是为了去掉'.bin'，防止保存的文件命为'*.bin.npz'
                    np.savez_compressed(os.path.join(target_dir, os.path.splitext(file_name)[0]), frames)
            else:
                for file_name in os.listdir(source_dir):
                    events = NMNIST.read_bin(os.path.join(source_dir, file_name))
                    # events: {'t', 'x', 'y', 'p'}
                    frames = SpikingFlow.datasets.integrate_events_to_frames(events=events, weight=34, height=34,
                                                                               frames_num=frames_num, split_by=split_by, normalization=normalization)
                    np.savez_compressed(os.path.join(target_dir, os.path.splitext(file_name)[0]), frames)

        thread_list = []

        sub_dir_list = os.listdir(events_data_dir)
        for i in range(sub_dir_list.__len__()):
            sub_dir = sub_dir_list[i]
            events_sub_dir = os.path.join(events_data_dir, sub_dir)
            frames_sub_dir = os.path.join(frames_data_dir, sub_dir)
            if i == sub_dir_list.__len__() - 1:
                show_bar = True
            else:
                show_bar = False
            thread_list.append(SpikingFlow.datasets.FunctionThread(f=cvt_data_in_dir, source_dir=events_sub_dir,
                                                                   target_dir=frames_sub_dir, show_bar=show_bar))
            print('start thread', thread_list.__len__())
            thread_list[-1].start()

        for i in range(thread_list.__len__()):
            thread_list[i].join()
            print('thread', i, 'finished')

    @staticmethod
    def install_dataset(zip_dir, frames_data_dir, frames_num=10, split_by='time', normalization=None):
        '''
        :param zip_dir: 下载N-MNIST数据集的 ``Train.zip`` 和 ``Test.zip`` 保存到哪个文件夹。如果这个文件夹内已经存在这2个文件，则不会下载
        :param frames_data_dir: 保存frames数据的文件夹，会将训练集和测试集分别保存到 ``Train`` 和 ``Test`` 文件夹内
        :param frames_num: 转换后数据的帧数
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                            为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                            为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :return: None

        一步完成对N-MNIST数据集的下载，解压，转换为frames数据。
        '''

        # 数据集不存在则下载
        if not os.path.exists(os.path.join(zip_dir, 'Train.zip')) or not os.path.exists(os.path.join(zip_dir, 'Test.zip')):
            print('download N-MNIST')
            NMNIST.download_zip(zip_dir)

        # 解压数据集
        ext_dir = os.path.join(zip_dir, 'extract')
        print('unzip Train.zip and Test.zip to', ext_dir)
        SpikingFlow.datasets.extract_zip_in_dir(zip_dir, ext_dir)

        if not os.path.exists(frames_data_dir):
            os.mkdir(frames_data_dir)
            print('mkdir', frames_data_dir)
        # 转换events为frames
        train_events_dir = os.path.join(ext_dir, 'Train/Train')
        train_frames_dir = os.path.join(frames_data_dir, 'Train')
        if not os.path.exists(train_frames_dir):
            os.mkdir(train_frames_dir)
            print('mkdir', train_frames_dir)
        test_events_dir = os.path.join(ext_dir, 'Test/Test')
        test_frames_dir = os.path.join(frames_data_dir, 'Test')
        if not os.path.exists(test_frames_dir):
            os.mkdir(test_frames_dir)
            print('mkdir', test_frames_dir)


        print('convert train dataset')
        NMNIST.create_frames_dataset(train_events_dir, train_frames_dir, frames_num=frames_num, split_by=split_by, normalization=normalization)
        print('convert test dataset')
        NMNIST.create_frames_dataset(test_events_dir, test_frames_dir, frames_num=frames_num, split_by=split_by, normalization=normalization)

        print('create frames data in', train_frames_dir, test_frames_dir)
        print('frames_num', frames_num, 'split_by', split_by, 'normalization', normalization)



    def __init__(self, frames_data_dir: str, train=True):
        '''
        :param frames_data_dir: 保存frame格式的N MNIST数据集的文件夹，其中包含训练集和测试集， ``Train`` 和 ``Test`` 文件夹
        :param train: 训练还是测试
        :param split_ratio: 训练集占数据集的比例。对于每一类，会抽取前 ``split_ratio`` 的数据作为训练集，而剩下 ``split_ratio`` 的数据集作为测试集

        N-MNIST数据集由以下论文发布：

        Orchard G, Jayawant A, Cohen G, et al. Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades[J]. Frontiers in Neuroscience, 2015: 437-437.

        为了构造好这样的数据集文件夹，建议遵循如下顺序：

        1.原始的N-MNIST数据集 ``zip_dir``。可以手动下载，也可以调用静态方法 ``NMNIST.download_zip``；

        2.分别解压训练集和测试集到 ``events_data_dir`` 目录。可以手动解压，也可以调用静态方法 ``SpikingFlow.datasets.extract_zip_in_dir``；

        3.将events数据转换成frames数据。调用 ``NMNIST.create_frames_dataset``。

        由于DVS数据集体积庞大，在生成需要的frames格式的数据后，可以考虑删除之前下载的原始数据。
        '''
        super().__init__()

        self.file_path = []
        self.label = []
        if train:
            self.root = os.path.join(frames_data_dir, 'Train')
        else:
            self.root = os.path.join(frames_data_dir, 'Test')

        for class_name in os.listdir(self.root):
            sub_dir = os.path.join(self.root, class_name)
            for file_name in os.listdir(sub_dir):
                self.file_path.append(os.path.join(sub_dir, file_name))
                self.label.append(int(class_name))

    def __len__(self):
        return self.file_path.__len__()

    def __getitem__(self, index):
        frame = torch.from_numpy(np.load(self.file_path[index])['arr_0']).float()
        # shape=[frames_num, 2, weight, height]
        return frame, self.label[index]

