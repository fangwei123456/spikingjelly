import SpikingFlow
import zipfile
import os
import tqdm
import numpy as np
cifar10_class = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


class CIFAR10DVS(SpikingFlow.datasets.SubDirDataset):

    @ staticmethod
    def download_zip(zip_dir):
        '''
        :param zip_dir: 保存SpikingFlow提供的CIFAR10-DVS对应的10个zip文件的文件夹
        :return: None

        .. warning::
            代码尚未完成，请勿使用。

        原始的CIFAR10-DVS数据集位于 https://figshare.com/articles/CIFAR10-DVS_New/4724671。原始的CIFAR10-DVS使用jAER格式，\
        需要首先使用MATLAB转换成mat格式才能使用，较为繁琐。SpikingFlow的开发者将原始数据集转化为numpy数组并存为npz文件，\
        每一类的数据都重新压缩并重新上传到了figshare。运行此函数，会将SpikingFlow提供的10个zip文件下载到 ``zip_dir``，下载好\
        的文件夹是如下形式：

        .. code-block:: bash

            zip_dir/
            |-- airplane.zip
            |-- automobile.zip
            |-- bird.zip
            |-- cat.zip
            |-- deer.zip
            |-- dog.zip
            |-- frog.zip
            |-- horse.zip
            |-- ship.zip
            `-- truck.zip

        '''
        raise NotImplementedError

    @staticmethod
    def create_frames_dataset(events_data_dir, frames_data_dir, frames_num=10, split_by='time', normalization=None):
        '''
        :param events_data_dir: 保存events数据的文件夹
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

        def cvt_data_in_dir(source_dir, target_dir, show_bar):
            print('processing', source_dir)
            if not os.path.exists(target_dir):
                os.mkdir(target_dir)
                print('mkdir', target_dir)
            if show_bar:
                for file_name in tqdm.tqdm(os.listdir(source_dir)):
                    events = np.load(os.path.join(source_dir, file_name))
                    # events: {'t', 'x', 'y', 'p'}
                    frames = SpikingFlow.datasets.integrate_events_to_frames(events=events, weight=128, height=128,
                                                                               frames_num=frames_num, split_by=split_by,
                                                                             normalization=normalization)
                    np.savez_compressed(os.path.join(target_dir, file_name), frames)
            else:
                for file_name in os.listdir(source_dir):
                    events = np.load(os.path.join(source_dir, file_name))
                    # events: {'t', 'x', 'y', 'p'}
                    frames = SpikingFlow.datasets.integrate_events_to_frames(events=events, weight=128, height=128,
                                                                               frames_num=frames_num, split_by=split_by,
                                                                             normalization=normalization)
                    np.savez_compressed(os.path.join(target_dir, file_name), frames)


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

    def __init__(self, frames_data_dir: str, train=True, split_ratio=0.9):
        '''
        :param frames_data_dir: 保存frame格式的CIFAR10-DVS数据集的文件夹
        :param train: 训练还是测试
        :param split_ratio: 训练集占数据集的比例。对于每一类，会抽取前 ``split_ratio`` 的数据作为训练集，而剩下 ``split_ratio`` 的数据集作为测试集

        CIFAR10-DVS数据集由以下论文发布：

        Li H, Liu H, Ji X, Li G and Shi L (2017) CIFAR10-DVS: An Event-Stream Dataset for Object Classification. Front. Neurosci. 11:309. doi: 10.3389/fnins.2017.00309

        需要保证``root`` 文件夹具有如下的格式：

        .. code-block:: bash

            frames_data_dir/
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

        其中的npz文件，以键为't','x','y','p'的字典保存数据。

        为了构造好这样的数据集文件夹，建议遵循如下顺序：

        1.下载SpikingFlow提供的10个zip文件下载到 ``zip_dir``。可以手动下载，也可以调用静态方法 ``DVSCIFAR10.download_zip``；

        2.解压这10个文件夹到 ``events_data_dir`` 目录。可以手动解压，也可以调用静态方法 ``SpikingFlow.datasets.extract_zip_in_dir``；

        3.将events数据转换成frames数据。调用 ``CIFAR10DVS.create_frames_dataset``。

        由于DVS数据集体积庞大，在生成需要的frames格式的数据后，可以考虑删除之前下载的原始数据。
        '''

        super().__init__(frames_data_dir, train, split_ratio)







