from typing import Any, Callable, cast, Dict, List, Optional, Tuple
import numpy as np
import spikingjelly.datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import shutil

class ASLDVS(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        '''
        :param root: root path of the dataset
        :type root: str
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``.

        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`cal_fixed_frames_number_segment_index` for
            more details.

        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.

        '''
        super().__init__(root, None, data_type, frames_number, split_by, duration, transform,
                         target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0'
        return [
            ('ICCV2019_DVS_dataset.zip', url, '8b46191acf6c1760ad3f2d2cb4380e24')
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
        temp_ext_dir = os.path.join(download_root, 'temp_ext')
        os.mkdir(temp_ext_dir)
        print(f'Mkdir [{temp_ext_dir}].')
        extract_archive(os.path.join(download_root, 'ICCV2019_DVS_dataset.zip'), temp_ext_dir)
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            for zip_file in os.listdir(temp_ext_dir):
                if os.path.splitext(zip_file)[1] == '.zip':
                    zip_file = os.path.join(temp_ext_dir, zip_file)
                    print(f'Extract [{zip_file}] to [{extract_root}].')
                    tpe.submit(extract_archive, zip_file, extract_root)

        shutil.rmtree(temp_ext_dir)
        print(f'Rmtree [{temp_ext_dir}].')

    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ['t', 'x', 'y', 'p'] and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''

        return sjds.load_matlab_mat(file_name)

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 180, 240

    @staticmethod
    def read_mat_save_to_np(mat_file: str, np_file: str):
        events = ASLDVS.load_origin_data(mat_file)
        np.savez(np_file,
                 t=events['t'],
                 x=events['x'],
                 y=events['y'],
                 p=events['p']
                 )
        print(f'Save [{mat_file}] to [{np_file}].')


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
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 64)) as tpe:
            for class_name in os.listdir(extract_root):
                mat_dir = os.path.join(extract_root, class_name)
                np_dir = os.path.join(events_np_root, class_name)
                os.mkdir(np_dir)
                print(f'Mkdir [{np_dir}].')
                for bin_file in os.listdir(mat_dir):
                    source_file = os.path.join(mat_dir, bin_file)
                    target_file = os.path.join(np_dir, os.path.splitext(bin_file)[0] + '.npz')
                    print(f'Start to convert [{source_file}] to [{target_file}].')
                    tpe.submit(ASLDVS.read_mat_save_to_np, source_file,
                               target_file)


        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
