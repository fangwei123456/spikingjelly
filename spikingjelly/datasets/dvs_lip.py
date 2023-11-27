from typing import Callable, Dict,  Optional, Tuple
import spikingjelly.datasets as sjds
import scipy.io
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import shutil
from .. import configure
from ..datasets import np_savez

class DVSLip(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The DVS-Lip dataset, which is proposed by `Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading <https://openaccess.thecvf.com/content/CVPR2022/html/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.html>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.
        """
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)

    @staticmethod
    def resource_url_md5() -> list:
        return [
            ('DVS-Lip.zip', 'https://sites.google.com/view/event-based-lipreading', '2dcb959255122d4cdeb6094ca282494b')
        ]


    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        zip_file = os.path.join(download_root, 'DVS-Lip.zip')
        print(f'Extract [{zip_file}] to [{extract_root}].')
        extract_archive(zip_file, extract_root)


    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):

        for train_test in ('train', 'test'):
            source_train_test_dir = os.path.join(extract_root, 'DVS-Lip', train_test)
            target_train_test_dir = os.path.join(events_np_root, train_test)
            os.mkdir(target_train_test_dir)
            for class_name in os.listdir(source_train_test_dir):
                source_class_dir = os.path.join(source_train_test_dir, class_name)
                target_class_dir = os.path.join(target_train_test_dir, class_name)
                os.mkdir(target_class_dir)
                for fname in os.listdir(source_class_dir):
                    source_file = os.path.join(source_class_dir, fname)
                    target_file = os.path.join(target_class_dir, fname)
                    # 由于原始数据已经是npy了，故只创建个软连接
                    os.symlink(source_file, target_file)




