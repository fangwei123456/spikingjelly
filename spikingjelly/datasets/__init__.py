from torch.utils.data import Dataset
import tqdm
import os
import numpy as np
import torch
import threading
import zipfile
from torchvision.datasets import utils
class FunctionThread(threading.Thread):
    def __init__(self, f, *args, **kwargs):
        super().__init__()
        self.f = f
        self.args = args
        self.kwargs = kwargs
    def run(self):
        self.f(*self.args, **self.kwargs)

def integrate_events_to_frames(events, height, width, frames_num=10, split_by='time', normalization=None):
    '''
    :param events: 键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
    :param height: 脉冲数据的高度，例如对于CIFAR10-DVS是128
    :param width: 脉冲数据的宽度，例如对于CIFAR10-DVS是128
    :param frames_num: 转换后数据的帧数
    :param split_by: 脉冲数据转换成帧数据的累计方式。``'time'`` 或 ``'number'``。为 ``'time'`` 表示将events数据在时间上分段，例如events记录的 ``t`` 介于
                        [0, 105]且 ``frames_num=10``，则转化得到的10帧分别为 ``t`` 属于[0, 10), [10,20), ..., [90, 105)的
                        脉冲的累加；
                        为 ``'number'`` 表示将events数据在数量上分段，例如events一共有105个且 ``frames_num=10``，则转化得到
                        的10帧分别是第[0, 10), [10,20), ..., [90, 105)个脉冲的累加
    :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；
                        为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                        为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                        为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
    :return: 转化后的frames数据，是一个 ``shape = [frames_num, 2, height, width]`` 的np数组

    '''
    frames = np.zeros(shape=[frames_num, 2, height, width])
    eps = 1e-5  # 涉及到除法的地方，被除数加上eps，防止出现除以0
    if split_by == 'time':
        # 按照脉冲的发生时刻进行等分
        events['t'] -= events['t'][0]
        dt = events['t'][-1] // frames_num
        t_interval = np.arange(dt, frames_num * dt, dt)  # frames_num - 1个元素
        # 在时间上，可以将event划分为frames_num段，分别是
        # [0, t_interval[0]), [t_interval[0], t_interval[1]), ...,
        # [t_interval[frames_num - 3], t_interval[frames_num - 2]), [t_interval[frames_num - 2], events['t'][-1])
        # 找出t_interval对应的索引
        index_list = np.searchsorted(events['t'], t_interval).tolist()
        index_l = 0
        for i in range(frames_num):
            if i == frames_num - 1:
                index_r = events['t'].shape[0]
            else:
                index_r = index_list[i]
            frames[i, events['p'][index_l:index_r], events['y'][index_l:index_r], events['x'][index_l:index_r]] \
                += events['t'][index_l:index_r]
            index_l = index_r
            if normalization == 'frequency':
                frames[i] /= dt  # 表示脉冲发放的频率
            elif normalization == 'max':
                frames[i] /= max(frames[i].max(), eps)
            elif normalization == 'norm':
                frames[i] = (frames[i] - frames[i].mean()) / np.sqrt(max(frames[i].var(), eps))
            elif normalization is None:
                continue
            else:
                raise NotImplementedError
        return frames
    elif split_by == 'number':
        # 按照脉冲数量进行等分
        dt = events['t'].shape[0] // frames_num
        for i in range(frames_num):
            # 将events在t维度分割成frames_num段
            index_l = i * dt
            if i == frames_num - 1:
                index_r = events['t'].shape[0]
            else:
                index_r = index_l + dt
            frames[i, events['p'][index_l:index_r], events['y'][index_l:index_r], events['x'][index_l:index_r]] \
                += events['t'][index_l:index_r]
            if normalization == 'frequency':
                frames[i] /= dt  # 表示脉冲发放的频率
            elif normalization == 'max':
                frames[i] /= max(frames[i].max(), eps)

            elif normalization == 'norm':
                frames[i] = (frames[i] - frames[i].mean()) / np.sqrt(max(frames[i].var(), eps))
            elif normalization is None:
                continue
            else:
                raise NotImplementedError
        return frames
    else:
        raise NotImplementedError

def convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, suffix, read_function, height, width,
                                              frames_num=10, split_by='time', normalization=None, thread_num=1, compress=False):
    # 遍历events_data_dir目录下的所有脉冲数据文件，在frames_data_dir目录下生成帧数据文件
    def cvt_fun(events_file_list):
        for events_file in events_file_list:
            frames = integrate_events_to_frames(read_function(events_file), height, width, frames_num, split_by,
                                                normalization)
            if compress:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npz')
                np.savez_compressed(frames_file, frames)
            else:
                frames_file = os.path.join(frames_data_dir,
                                           os.path.basename(events_file)[0: -suffix.__len__()] + '.npy')
                np.save(frames_file, frames)
    events_file_list = utils.list_files(events_data_dir, suffix, True)
    if thread_num == 1:
        cvt_fun(events_file_list)
    else:
        # 多线程加速
        thread_list = []
        block = events_file_list.__len__() // thread_num
        for i in range(thread_num - 1):
            thread_list.append(FunctionThread(cvt_fun, events_file_list[i * block: (i + 1) * block]))
            thread_list[-1].start()
            print(f'thread {i} start, processing files index: {i * block} : {(i + 1) * block}.')
        thread_list.append(FunctionThread(cvt_fun, events_file_list[(thread_num - 1) * block:]))
        thread_list[-1].start()
        print(f'thread {thread_num} start, processing files index: {(thread_num - 1) * block} : {events_file_list.__len__()}.')
        for i in range(thread_num):
            thread_list[i].join()
            print(f'thread {i} finished.')

def extract_zip_in_dir(source_dir, target_dir):
    '''
    :param source_dir: 保存有zip文件的文件夹
    :param target_dir: 保存zip解压后数据的文件夹
    :return: None

    将 ``source_dir`` 目录下的所有*.zip文件，解压到 ``target_dir`` 目录下的对应文件夹内
    '''

    for file_name in os.listdir(source_dir):
        if file_name[-3:] == 'zip':
            with zipfile.ZipFile(os.path.join(source_dir, file_name), 'r') as zip_file:
                zip_file.extractall(os.path.join(target_dir, file_name[:-4]))

class EventsFramesDatasetBase(Dataset):
    @staticmethod
    def get_wh():
        '''
        :return: (width, height)
            width: int
                events或frames图像的宽度
            height: int
                events或frames图像的高度
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: 脉冲数据的文件名
        :type file_name: str
        :return: events
            键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
        :rtype: dict
        '''
        raise NotImplementedError

    @staticmethod
    def get_events_item(file_name):
        '''
        :param file_name: 脉冲数据的文件名
        :type file_name: str
        :return: (events, label)
            events: dict
                键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
            label: int
                数据的标签
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def get_frames_item(file_name):
        '''
        :param file_name: 帧数据的文件名
        :type file_name: str
        :return: (frames, label)
            frames: np.ndarray
                ``shape = [frames_num, 2, height, width]`` 的np数组
            label: int
                数据的标签
        :rtype: tuple
        '''
        raise NotImplementedError

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        '''
        :param download_root: 保存下载文件的文件夹
        :type download_root: str
        :param extract_root: 保存解压后文件的文件夹
        :type extract_root: str

        下载数据集到 ``download_root``，然后解压到 ``extract_root``。
        '''
        raise NotImplementedError

    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        '''
        :param events_data_dir: 保存脉冲数据的文件夹，文件夹的文件全部是脉冲数据
        :type events_data_dir: str
        :param frames_data_dir: 保存帧数据的文件夹
        :type frames_data_dir: str
        :param frames_num: 转换后数据的帧数
        :type frames_num: int
        :param split_by: 脉冲数据转换成帧数据的累计方式
        :type split_by: str
        :param normalization: 归一化方法
        :type normalization: str or None

        将 ``events_data_dir`` 文件夹下的脉冲数据全部转换成帧数据，并保存在 ``frames_data_dir``。
        转换参数的详细含义，参见 ``integrate_events_to_frames`` 函数。
        '''
        raise NotImplementedError