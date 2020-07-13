import SpikingFlow
import zipfile
import os

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

    @ staticmethod
    def unzip(zip_dir, events_data_dir):
        '''
        :param zip_dir: 保存SpikingFlow提供的CIFAR10-DVS对应的10个zip文件的文件夹
        :param events_data_dir: 保存数据集的文件夹，运行 ``download_zip(zip_dir)`` 下载的10个zip文件，会被逐个解压到 ``events_data_dir`` 目录
        :return: None

        ``events_data_dir`` 文件夹在数据集全部解压后，会具有如下的格式：

        .. code-block:: bash

            events_data_dir/
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
        for class_name in cifar10_class:
            zip_file_name = os.path.join(zip_dir, class_name + '.zip')
            un_dir = os.path.join(events_data_dir, class_name)
            print('unzip', zip_file_name, 'to', un_dir)
            with zipfile.ZipFile(zip_file_name, 'r') as zip_file:
                zip_file.extractall(un_dir)
            print('extra file number', os.listdir(un_dir).__len__())

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

        1.下载SpikingFlow提供的10个zip文件下载到 ``zip_dir``。可以手动下载，也可以调用静态方法 ``DVSCIFAR10.download_zip(zip_dir)``；

        2.解压这10个文件夹到 ``events_data_dir`` 目录。可以手动解压，也可以调用静态方法 ``DVSCIFAR10.unzip(zip_dir, events_data_dir)``；

        3.将events数据转换成frames数据。调用 ``SpikingFlow.datasets.convert_events_to_frames(events_data_dir, frames_data_dir, weight=128, height=128, frames_num=10, normalization=None)``。

        由于DVS数据集体积庞大，在生成需要的frames格式的数据后，可以考虑删除之前下载的原始数据。
        '''

        super().__init__(frames_data_dir, train, split_ratio)







