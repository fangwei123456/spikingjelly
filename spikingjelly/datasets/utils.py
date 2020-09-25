from torch.utils.data import Dataset
import os
import numpy as np
import threading
import zipfile
from torchvision.datasets import utils
import torch

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
    * :ref:`API in English <integrate_events_to_frames.__init__-en>`

    .. _integrate_events_to_frames.__init__-cn:

    :param events: 键是{'t', 'x', 'y', 'p'}，值是np数组的的字典
    :param height: 脉冲数据的高度，例如对于CIFAR10-DVS是128
    :param width: 脉冲数据的宽度，例如对于CIFAR10-DVS是128
    :param frames_num: 转换后数据的帧数
    :param split_by: 脉冲数据转换成帧数据的累计方式，允许的取值为 ``'number', 'time'``
    :param normalization: 归一化方法，允许的取值为 ``None, 'frequency', 'max', 'norm', 'sum'``
    :return: 转化后的frames数据，是一个 ``shape = [frames_num, 2, height, width]`` 的np数组

    记脉冲数据为 :math:`E_{i} = (t_{i}, x_{i}, y_{i}, p_{i}), i=0,1,...,N-1`，转换为帧数据 :math:`F(j, p, x, y), j=0,1,...,M-1`。

    若划分方式 ``split_by`` 为 ``'time'``，则

    .. math::

        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases} \\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    若划分方式 ``split_by`` 为 ``'number'``，则

    .. math::

        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}\\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    其中 :math:`\\mathcal{I}` 为示性函数，当且仅当 :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})` 时为1，否则为0。

    若 ``normalization`` 为 ``'frequency'``，

        若 ``split_by`` 为 ``time`` 则

            .. math::
                F_{norm}(j, p, x, y) = \\begin{cases} \\frac{F(j, p, x, y)}{\\Delta T}, & j < M - 1
                \\cr \\frac{F(j, p, x, y)}{\\Delta T + (t_{N-1} - t_{0}) \\bmod M}, & j = M - 1 \\end{cases}

        若 ``split_by`` 为 ``number`` 则

            .. math::
                F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{t_{j_{r}} - t_{j_{l}}}


    若 ``normalization`` 为 ``'max'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\mathrm{max} F(j, p)}

    若 ``normalization`` 为 ``'norm'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y) - \\mathrm{E}(F(j, p))}{\\sqrt{\\mathrm{Var}(F(j, p))}}

    若 ``normalization`` 为 ``'sum'`` 则

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\sum_{a, b} F(j, p, a, b)}

    * :ref:`中文API <integrate_events_to_frames.__init__-cn>`

    .. _integrate_events_to_frames.__init__-en:

    :param events: a dict with keys are {'t', 'x', 'y', 'p'} and values are numpy arrays
    :param height: the height of events data, e.g., 128 for CIFAR10-DVS
    :param width: the width of events data, e.g., 128 for CIFAR10-DVS
    :param frames_num: frames number
    :param split_by: how to split the events, can be ``'number', 'time'``
    :param normalization: how to normalize frames, can be ``None, 'frequency', 'max', 'norm', 'sum'``
    :return: the frames data with ``shape = [frames_num, 2, height, width]``

    The events data are denoted by :math:`E_{i} = (t_{i}, x_{i}, y_{i}, p_{i}), i=0,1,...,N-1`, and the converted frames
    data are denoted by :math:`F(j, p, x, y), j=0,1,...,M-1`.

    If ``split_by`` is ``'time'``, then

    .. math::

        \\Delta T & = [\\frac{t_{N-1} - t_{0}}{M}] \\\\
        j_{l} & = \\mathop{\\arg\\min}\\limits_{k} \\{t_{k} | t_{k} \\geq t_{0} + \\Delta T \\cdot j\\} \\\\
        j_{r} & = \\begin{cases} \\mathop{\\arg\\max}\\limits_{k} \\{t_{k} | t_{k} < t_{0} + \\Delta T \\cdot (j + 1)\\} + 1, & j <  M - 1 \\cr N, & j = M - 1 \\end{cases} \\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    If ``split_by`` is ``'number'``, then

    .. math::

        j_{l} & = [\\frac{N}{M}] \\cdot j \\\\
        j_{r} & = \\begin{cases} [\\frac{N}{M}] \\cdot (j + 1), & j <  M - 1 \\cr N, & j = M - 1 \\end{cases}\\\\
        F(j, p, x, y) & = \\sum_{i = j_{l}}^{j_{r} - 1} \\mathcal{I_{p, x, y}(p_{i}, x_{i}, y_{i})}

    where :math:`\\mathcal{I}` is the characteristic function，if and only if :math:`(p, x, y) = (p_{i}, x_{i}, y_{i})`,
    this function is identically 1 else 0.

    If ``normalization`` is ``'frequency'``,

        if ``split_by`` is ``time``,

            .. math::
                F_{norm}(j, p, x, y) = \\begin{cases} \\frac{F(j, p, x, y)}{\\Delta T}, & j < M - 1
                \\cr \\frac{F(j, p, x, y)}{\\Delta T + (t_{N-1} - t_{0}) \\bmod M}, & j = M - 1 \\end{cases}

        if ``split_by`` is ``number``,

            .. math::
                F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{t_{j_{r}} - t_{j_{l}}}

    If ``normalization`` is ``'max'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\mathrm{max} F(j, p)}

    If ``normalization`` is ``'norm'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y) - \\mathrm{E}(F(j, p))}{\\sqrt{\\mathrm{Var}(F(j, p))}}

    If ``normalization`` is ``'sum'``, then

    .. math::
        F_{norm}(j, p, x, y) = \\frac{F(j, p, x, y)}{\\sum_{a, b} F(j, p, a, b)}
    '''
    frames = np.zeros(shape=[frames_num, 2, height * width])

    # 创建j_{l}和j_{r}
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    if split_by == 'time':
        events['t'] -= events['t'][0]  # 时间从0开始
        assert events['t'][-1] > frames_num
        dt = events['t'][-1] // frames_num  # 每一段的持续时间
        idx = np.arange(events['t'].size)
        for i in range(frames_num):
            t_l = dt * i
            t_r = t_l + dt
            mask = np.logical_and(events['t'] >= t_l, events['t'] < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1 if i < frames_num - 1 else events['t'].size

    elif split_by == 'number':
        di = events['t'].size // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di if i < frames_num - 1 else events['t'].size
    else:
        raise NotImplementedError

    # 开始累计脉冲
    # 累计脉冲需要用bitcount而不能直接相加，原因可参考下面的示例代码，以及
    # https://stackoverflow.com/questions/15973827/handling-of-duplicate-indices-in-numpy-assignments
    # height = 3
    # width = 3
    # frames = np.zeros(shape=[2, height, width])
    # events = {
    #     'x': np.asarray([1, 2, 1, 1]),
    #     'y': np.asarray([1, 1, 1, 2]),
    #     'p': np.asarray([0, 1, 0, 1])
    # }
    #
    # frames[0, events['y'], events['x']] += (1 - events['p'])
    # frames[1, events['y'], events['x']] += events['p']
    # print('wrong accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # for i in range(events['p'].__len__()):
    #     frames[events['p'][i], events['y'][i], events['x'][i]] += 1
    # print('correct accumulation\n', frames)
    #
    # frames = np.zeros(shape=[2, height, width])
    # frames = frames.reshape(2, -1)
    #
    # mask = [events['p'] == 0]
    # mask.append(np.logical_not(mask[0]))
    # for i in range(2):
    #     position = events['y'][mask[i]] * height + events['x'][mask[i]]
    #     events_number_per_pos = np.bincount(position)
    #     idx = np.arange(events_number_per_pos.size)
    #     frames[i][idx] += events_number_per_pos
    # frames = frames.reshape(2, height, width)
    # print('correct accumulation by bincount\n', frames)

    for i in range(frames_num):
        x = events['x'][j_l[i]:j_r[i]]
        y = events['y'][j_l[i]:j_r[i]]
        p = events['p'][j_l[i]:j_r[i]]
        mask = []
        mask.append(p == 0)
        mask.append(np.logical_not(mask[0]))
        for j in range(2):
            position = y[mask[j]] * height + x[mask[j]]
            events_number_per_pos = np.bincount(position)
            frames[i][j][np.arange(events_number_per_pos.size)] += events_number_per_pos

        if normalization == 'frequency':
            if split_by == 'time':
                if i < frames_num - 1:
                    frames[i] /= dt
                else:
                    frames[i] /= (dt + events['t'][-1] % frames_num)
            elif split_by == 'number':
                    frames[i] /= (events['t'][j_r[i]] - events['t'][j_l[i]])  # 表示脉冲发放的频率

            else:
                raise NotImplementedError

        # 其他的normalization方法，在数据集类读取数据的时候进行通过调用normalize_frame(frames: np.ndarray, normalization: str)
        # 函数操作，而不是在转换数据的时候进行
    return frames.reshape((frames_num, 2, height, width))

def normalize_frame(frames: np.ndarray or torch.Tensor, normalization: str):
    eps = 1e-5  # 涉及到除法的地方，被除数加上eps，防止出现除以0
    for i in range(frames.shape[0]):
        if normalization == 'max':
            frames[i][0] /= max(frames[i][0].max(), eps)
            frames[i][1] /= max(frames[i][1].max(), eps)

        elif normalization == 'norm':
            frames[i][0] = (frames[i][0] - frames[i][0].mean()) / np.sqrt(max(frames[i][0].var(), eps))
            frames[i][1] = (frames[i][1] - frames[i][1].mean()) / np.sqrt(max(frames[i][1].var(), eps))

        elif normalization == 'sum':
            frames[i][0] /= max(frames[i][0].sum(), eps)
            frames[i][1] /= max(frames[i][1].sum(), eps)

        else:
            raise NotImplementedError
    return frames

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
