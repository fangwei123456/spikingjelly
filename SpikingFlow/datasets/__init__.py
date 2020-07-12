from torch.utils.data import Dataset
import tqdm
import os
import numpy as np
import torch
import threading


class CVTThread(threading.Thread):
    def __init__(self, show_bar, sub_dir, events_data_dir, frames_data_dir, weight, height, frames_num, normalization):
        super().__init__()
        self.show_bar = show_bar
        self.events_sub_dir = os.path.join(events_data_dir, sub_dir)
        self.frames_sub_dir = os.path.join(frames_data_dir, sub_dir)
        self.weight = weight
        self.height = height
        self.frames_num = frames_num
        self.normalization = normalization
        self.cvted_num = 0

    def cvt(self, file_name):
        events = np.load(os.path.join(self.events_sub_dir, file_name))
        # events: {'t', 'x', 'y', 'p'}
        frames = np.zeros(shape=[self.frames_num, 2, self.weight, self.height])
        dt = events['t'].shape[0] // self.frames_num
        for i in range(self.frames_num):
            # 将events在t维度分割成frames_num段，每段的范围是[i * dt, (i + 1) * dt)，每段累加得到一帧
            index_l = i * dt
            index_r = index_l + dt
            frames[i, events['p'][index_l:index_r], events['x'][index_l:index_r], events['y'][index_l:index_r]] \
                += events['t'][index_l:index_r]
            if self.normalization == 'frequency':
                frames[i] /= dt  # 表示脉冲发放的频率
            elif self.normalization == 'max':
                frames[i] /= frames[i].max()
            elif self.normalization == 'norm':
                frames[i] = (frames[i] - frames[i].mean()) / np.sqrt((frames[i].var() + 1e-5))
            elif self.normalization is None:
                continue
            else:
                raise NotImplementedError
        np.savez_compressed(os.path.join(self.frames_sub_dir, file_name), frames)
        self.cvted_num += 1

    def run(self):

        print('processing', self.events_sub_dir)
        if not os.path.exists(self.frames_sub_dir):
            os.mkdir(self.frames_sub_dir)
            print('mkdir', self.frames_sub_dir)
        if self.show_bar:
            for file_name in tqdm.tqdm(os.listdir(self.events_sub_dir)):
                self.cvt(file_name)
        else:
            for file_name in os.listdir(self.events_sub_dir):
                self.cvt(file_name)



def convert_events_to_frames(events_data_dir, frames_data_dir, weight, height, frames_num=10, normalization=None):
    '''
    :param events_data_dir: 保存events数据的文件夹
    :param frames_data_dir: 保存frames数据的文件夹
    :param weight: 脉冲数据的宽度，例如对于DVS CIFAR10是128
    :param height: 脉冲数据的高度，例如对于DVS CIFAR10是128
    :param frames_num: 转换后数据的帧数
    :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                        为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                        为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
    :return: None

    将DVS的events数据，在t维度分割成 ``frames_num`` 段，每段的范围是 ``[i * dt, (i + 1) * dt)`` ，每段累加得到一帧，转换成 ``frames_num`` 帧数。

    ``events_data_dir`` 文件夹中应该包含多个子文件夹，每个子文件夹内是npz的数据，以键为 ``'t','x','y','p'`` 的字典保存数据，例如：

    .. code-block:: bash

        dvs_cifar10_npz/
        |-- airplane
        |   |-- 0.npz
        |   |-- ...
        |-- automobile
        |-- bird
        |-- cat
        |-- deer
        |-- dog
        |-- frog
        |-- horse
        |-- ship
        `-- truck

    本函数会在 ``frames_data_dir`` 文件夹下生成与 ``events_data_dir`` 相同的子文件夹，每个子文件夹内也是npz的数据，以键为 ``'arr_0'`` 的字典（numpy默认）保存数据。
    '''

    data_num = 0
    thread_list = []
    sub_dir_list = os.listdir(events_data_dir)
    for i in range(sub_dir_list.__len__()):
        sub_dir = sub_dir_list[i]
        if i == sub_dir_list.__len__() - 1:
            show_bar = True
        else:
            show_bar = False
        thread_list.append(CVTThread(show_bar, sub_dir, events_data_dir, frames_data_dir, weight, height,
                                     frames_num, normalization))
        print('start thread', thread_list.__len__())
        thread_list[i].start()

    for td in thread_list:
        td.join()
    for td in thread_list:
        data_num += td.cvted_num
    print('converted data num = ', data_num)


class SubDirDataset(Dataset):
    def __init__(self, root, train=True, split_ratio=0.9):
        '''
        :param root: 保存数据集的文件夹
        :param train: 训练还是测试
        :param split_ratio: 训练集占数据集的比例，对于每一类的数据，按文件夹内部数据文件的命名排序，取前 ``split_ratio`` 的数据作为训练集，
                            其余数据作为测试集

        适用于包含多个子文件夹，每个子文件夹名称为类别名，子文件夹内部是npz格式的数据的数据集基类。文件结构类似如下所示：

        .. code-block:: bash

            dvs_cifar10_npz/
            |-- airplane
            |   |-- 0.npz
            |   |-- ...
            |-- automobile
            |-- bird
            |-- cat
            |-- deer
            |-- dog
            |-- frog
            |-- horse
            |-- ship
            `-- truck
        '''

        self.root = root
        self.label_name = os.listdir(self.root)
        self.file_path = []
        self.label = []

        for i in range(self.label_name.__len__()):
            sub_dir_path = os.path.join(self.root, self.label_name[i])
            file_names = os.listdir(sub_dir_path)
            split_boundary = int(file_names.__len__() * split_ratio)
            if train:
                for j in range(0, split_boundary):
                    self.file_path.append(os.path.join(sub_dir_path, file_names[j]))
                    self.label.append(i)
            else:
                for j in range(split_boundary, file_names.__len__()):
                    self.file_path.append(os.path.join(sub_dir_path, file_names[j]))
                    self.label.append(i)

    def __len__(self):
        return self.file_path.__len__()

    def __getitem__(self, index):
        frame = torch.from_numpy(np.load(self.file_path[index])['arr_0']).float()
        return frame, self.label[index]

class N_MNIST(Dataset):
    def __init__(self, root, train=True):
        '''
        临时使用，随时删除
        '''
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