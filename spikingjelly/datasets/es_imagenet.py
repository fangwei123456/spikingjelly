from typing import Callable, Dict, Optional, Tuple
import numpy as np
from .. import datasets as sjds
import os
import rarfile
import time


def load_events(fname: str):
    events = np.load(fname)
    e_pos = events['pos']
    e_neg = events['neg']
    e_pos = np.hstack((e_pos, np.ones((e_pos.shape[0], 1))))
    e_neg = np.hstack((e_neg, np.zeros((e_neg.shape[0], 1))))
    events = np.vstack((e_pos, e_neg))  # shape = [N, 4], N * (x, y, t, p)
    idx = np.argsort(events[:, 2])
    events = events[idx]
    return {
        'x': events[:, 1],
        'y': events[:, 0],
        't': events[:, 2],
        'p': events[:, 3]
    }


class ESImageNet(sjds.NeuromorphicDatasetFolder):
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
        The ES-ImageNet dataset, which is proposed by `ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks <https://www.frontiersin.org/articles/10.3389/fnins.2021.726582/full>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.
        """
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)

        if data_type == 'event':
            self.loader = load_events

    @staticmethod
    def load_events_np(fname: str):
        return load_events(fname)

    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        urls = [
            ('ES-imagenet-0.18.part01.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part01.rar&dl=1',
             '900bdd57b5641f7d81cd4620283fef76'),
            ('ES-imagenet-0.18.part02.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part02.rar&dl=1',
             '5982532009e863a8f4e18e793314c54b'),
            ('ES-imagenet-0.18.part03.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part03.rar&dl=1',
             '8f408c1f5a1d4604e48d0d062a8289a0'),
            ('ES-imagenet-0.18.part04.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part04.rar&dl=1',
             '5c5b5cf0a55954eb639964e3da510097'),
            ('ES-imagenet-0.18.part05.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part05.rar&dl=1',
             '51feb661b4c9fa87860b63e76b914673'),
            ('ES-imagenet-0.18.part06.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part06.rar&dl=1',
             'fcd007a2b17b7c13f338734c53f6db31'),
            ('ES-imagenet-0.18.part07.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part07.rar&dl=1',
             'd3e74b96d9c5df15714bbc3abcd329fc'),
            ('ES-imagenet-0.18.part08.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part08.rar&dl=1',
             '65b9cf7fa63e18d2e7d92ff45a42a5e5'),
            ('ES-imagenet-0.18.part09.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part09.rar&dl=1',
             '241c9a37a83ff9efd305fe46d012211e'),
            ('ES-imagenet-0.18.part10.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part10.rar&dl=1',
             'ceee96971008e30d0cdc34086c49fd75'),
            ('ES-imagenet-0.18.part11.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part11.rar&dl=1',
             '4fbfefbe6e48758fbb72427c81f119cf'),
            ('ES-imagenet-0.18.part12.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part12.rar&dl=1',
             'c8cc163be4e5f6451201dccbded4ec24'),
            ('ES-imagenet-0.18.part13.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part13.rar&dl=1',
             '08c9dff32f6b42c49ef7cd78e37c728e'),
            ('ES-imagenet-0.18.part14.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part14.rar&dl=1',
             '43aa157dc5bd5fcea81315a46e0322cf'),
            ('ES-imagenet-0.18.part15.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part15.rar&dl=1',
             '480a69b050f465ef01efcc44ae29f7df'),
            ('ES-imagenet-0.18.part16.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part16.rar&dl=1',
             '11abd24d92b93e7f85acd63abd4a18ab'),
            ('ES-imagenet-0.18.part17.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part17.rar&dl=1',
             '3891486a6862c63a325c5f16cd01fdd1'),
            ('ES-imagenet-0.18.part18.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part18.rar&dl=1',
             'cf8bb0525b514f411bca9d7c2d681f7c'),
            ('ES-imagenet-0.18.part19.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part19.rar&dl=1',
             '3766bc35572ccacc03f0f293c571d0ae'),
            ('ES-imagenet-0.18.part20.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part20.rar&dl=1',
             'bf73a5e338644122220e41da7b5630e6'),
            ('ES-imagenet-0.18.part21.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part21.rar&dl=1',
             '564de4a2609cbb0bb67ffa1bc51f2487'),
            ('ES-imagenet-0.18.part22.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part22.rar&dl=1',
             '60a9e52db1acadfccc9a9809073f0b04'),
            ('ES-imagenet-0.18.part23.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part23.rar&dl=1',
             '373b5484826d40d7ec35f0e1605cb6ea'),
            ('ES-imagenet-0.18.part24.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part24.rar&dl=1',
             'a50612e889b20f99cc7b2725dfd72e9e'),
            ('ES-imagenet-0.18.part25.rar',
             'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part25.rar&dl=1',
             '0802ccdeb0cff29237faf55164524101')
        ]

        return urls


    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return True

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
        rar_file = os.path.join(download_root, 'ES-imagenet-0.18.part01.rar')
        print(f'Extract [{rar_file}] to [{extract_root}].')
        rar_file = rarfile.RarFile(rar_file)
        rar_file.extractall(extract_root)
        rar_file.close()



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
        train_dir = os.path.join(events_np_root, 'train')
        os.mkdir(train_dir)
        print(f'Mkdir [{train_dir}].')
        sjds.create_same_directory_structure(os.path.join(extract_root, 'ES-imagenet-0.18/train'), train_dir)
        for class_dir in os.listdir(os.path.join(extract_root, 'ES-imagenet-0.18/train')):
            source_dir = os.path.join(extract_root, 'ES-imagenet-0.18/train', class_dir)
            target_dir = os.path.join(train_dir, class_dir)
            print(f'Create soft links from [{source_dir}] to [{target_dir}].')
            for class_sample in os.listdir(source_dir):
                os.symlink(os.path.join(source_dir, class_sample),
                           os.path.join(target_dir, class_sample))




        val_label = np.loadtxt(os.path.join(extract_root, 'ES-imagenet-0.18/vallabel.txt'), delimiter=' ', usecols=(1, ), dtype=int)
        val_fname = np.loadtxt(os.path.join(extract_root, 'ES-imagenet-0.18/vallabel.txt'), delimiter=' ', usecols=(0, ), dtype=str)
        source_dir = os.path.join(extract_root, 'ES-imagenet-0.18/val')
        target_dir = os.path.join(events_np_root, 'test')
        os.mkdir(target_dir)
        print(f'Mkdir [{target_dir}].')
        sjds.create_same_directory_structure(train_dir, target_dir)

        for i in range(val_fname.__len__()):
            os.symlink(os.path.join(source_dir, val_fname[i]), os.path.join(target_dir, f'class{val_label[i]}/{val_fname[i]}'))

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'Note that files in [{events_np_root}] are soft links whose source files are in [{extract_root}]. If you want to use events, do not delete [{extract_root}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 256, 256