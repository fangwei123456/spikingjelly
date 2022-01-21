from typing import Callable, Dict, Optional, Tuple
from .. import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from .. import configure
from ..datasets import np_savez

class NMNIST(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
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
        The N-MNIST dataset, which is proposed by `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.
        """
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://www.garrickorchard.com/datasets/n-mnist'
        return [
            ('Train.zip', url, '20959b8e626244a1b502305a9e6e2031'),
            ('Test.zip', url, '69ca8762b2fe404d9b9bad1103e97832')
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
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                tpe.submit(extract_archive, zip_file, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''

        return sjds.load_ATIS_bin(file_name)

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 34, 34

    @staticmethod
    def read_bin_save_to_np(bin_file: str, np_file: str):
        events = NMNIST.load_origin_data(bin_file)
        np_savez(np_file,
                 t=events['t'],
                 x=events['x'],
                 y=events['y'],
                 p=events['p']
                 )
        print(f'Save [{bin_file}] to [{np_file}].')


    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
            # too many threads will make the disk overload
            for train_test_dir in ['Train', 'Test']:
                source_dir = os.path.join(extract_root, train_test_dir)
                target_dir = os.path.join(events_np_root, train_test_dir.lower())
                os.mkdir(target_dir)
                print(f'Mkdir [{target_dir}].')
                for class_name in os.listdir(source_dir):
                    bin_dir = os.path.join(source_dir, class_name)
                    np_dir = os.path.join(target_dir, class_name)
                    os.mkdir(np_dir)
                    print(f'Mkdir [{np_dir}].')
                    for bin_file in os.listdir(bin_dir):
                        source_file = os.path.join(bin_dir, bin_file)
                        target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
                        print(f'Start to convert [{source_file}] to [{target_file}].')
                        tpe.submit(NMNIST.read_bin_save_to_np, source_file,
                                   target_file)


        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
