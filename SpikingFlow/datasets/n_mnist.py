import SpikingFlow
import zipfile
import os
import threading
import tqdm
import numpy as np

t_bit_mask = int('1' * 23, base=2)
class CVTThread(SpikingFlow.datasets.CVTThread):
    def cvt(self, file_name):
        t_list, x_list, y_list, p_list = np.asarray(NMNIST.read_bin(os.path.join(self.events_sub_dir, file_name)))

        events = {}
        events['t'] = np.asarray(t_list)
        events['x'] = np.asarray(x_list)
        events['y'] = np.asarray(y_list)
        events['p'] = np.asarray(p_list)
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

class NMNIST(SpikingFlow.datasets.SubDirDataset):

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
            |-- train.zip
            |-- test.zip
        '''
        raise NotImplementedError

    @ staticmethod
    def unzip(zip_dir, events_data_dir):
        '''
        :param zip_dir: 保存下载的N-MNIST训练集和测试集zip压缩包的文件夹
        :param events_data_dir: 保存数据集的文件夹，运行 ``download_zip(zip_dir)`` 下载的N-MNIST训练集和测试集zip压缩包，会被逐个解压到 ``events_data_dir`` 目录
        :return: None

        ``events_data_dir`` 文件夹在数据集全部解压后，会具有如下的格式：

        .. code-block:: bash

            events_data_dir/
            |-- 0
            |   |-- 0.bin
            |   |-- ...
            |-- 1
            |-- 2
            |-- 3
            |-- 4
            |-- 5
            |-- 6
            |-- 7
            |-- 8
            `-- 9
        '''
        for i in range(10):
            class_name = str(i)
            zip_file_name = os.path.join(zip_dir, class_name + '.zip')
            un_dir = os.path.join(events_data_dir, class_name)
            print('unzip', zip_file_name, 'to', un_dir)
            with zipfile.ZipFile(zip_file_name, 'r') as zip_file:
                zip_file.extractall(un_dir)
            print('extra file number', os.listdir(un_dir).__len__())

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: N-MNIST原始bin格式数据的文件名
        :return: 4个list，分别是t, x, y, p
        
        原始的N-MNIST提供的是bin格式数据，不能直接读取。本函数提供了一个读取的接口。
        '''
        t_list = []
        x_list = []
        y_list = []
        p_list = []
        with open(file_name, 'rb') as bin_f:
            while True:
                bin_x = bin_f.read(1)
                if bin_x.__len__() == 0:
                    break
                x = int.from_bytes(bin_x, byteorder='big')
                y = int.from_bytes(bin_f.read(1), byteorder='big')
                bits = int.from_bytes(bin_f.read(3), byteorder='big')
                p = bits >> 23
                t = bits & t_bit_mask
                t_list.append(t)
                x_list.append(x)
                y_list.append(y)
                p_list.append(p)
        return t_list, x_list, y_list, p_list

    @staticmethod
    def convert_events_to_frames(events_data_dir, frames_data_dir, frames_num=10, normalization=None):
        '''
        :param events_data_dir: 保存N-MNIST原始数据集bin文件的文件夹
        :param frames_data_dir: 保存frames数据的文件夹
        :param frames_num: 转换后数据的帧数
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                            为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                            为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :return: None

        将N-MNIST的events数据，在t维度分割成 ``frames_num`` 段，每段的范围是 ``[i * dt, (i + 1) * dt)`` ，每段累加得到一帧，转换成 ``frames_num`` 帧数。

        ``events_data_dir`` 文件夹应该包含解压后的N-MNIST数据集，会具有如下的格式：

        .. code-block:: bash

            events_data_dir/
            |-- 0
            |   |-- 0.bin
            |   |-- ...
            |-- 1
            |-- 2
            |-- 3
            |-- 4
            |-- 5
            |-- 6
            |-- 7
            |-- 8
            `-- 9

        本函数会在 ``frames_data_dir`` 文件夹下生成与 ``events_data_dir`` 相同的子文件夹，每个子文件夹内也是npz的数据，以键为 ``'arr_0'`` 的字典（numpy默认）保存数据。
        '''
        weight = 28
        height = 28
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

    def __init__(self, frames_data_dir: str, train=True, split_ratio=0.9):
        '''
        :param frames_data_dir: 保存frame格式的N MNIST数据集的文件夹
        :param train: 训练还是测试
        :param split_ratio: 训练集占数据集的比例。对于每一类，会抽取前 ``split_ratio`` 的数据作为训练集，而剩下 ``split_ratio`` 的数据集作为测试集

        N-MNIST数据集由以下论文发布：

        Orchard G, Jayawant A, Cohen G, et al. Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades[J]. Frontiers in Neuroscience, 2015: 437-437.

        需要保证``root`` 文件夹具有如下的格式：

        .. code-block:: bash

            frames_data_dir/
            |-- 0
            |   |-- 0.npz
            |   |-- ...
            |-- 1
            |-- 2
            |-- 3
            |-- 4
            |-- 5
            |-- 6
            |-- 7
            |-- 8
            `-- 9

        其中的npz文件，以键为't','x','y','p'的字典保存数据。

        为了构造好这样的数据集文件夹，建议遵循如下顺序：

        1.原始的N-MNIST数据集 ``zip_dir``。可以手动下载，也可以调用静态方法 ``NMNIST.download_zip(zip_dir)``；

        2.分别解压训练集和测试集到 ``events_data_dir`` 目录。可以手动解压，也可以调用静态方法 ``NMNIST.unzip(zip_dir, events_data_dir)``；

        3.将events数据转换成frames数据。调用 ``NMNIST.convert_events_to_frames(events_data_dir, frames_data_dir, frames_num=10, normalization=None)``。

        由于DVS数据集体积庞大，在生成需要的frames格式的数据后，可以考虑删除之前下载的原始数据。
        '''

        super().__init__(frames_data_dir, train, split_ratio)
