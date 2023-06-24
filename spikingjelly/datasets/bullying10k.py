import multiprocessing
import os
import shutil
import time
from concurrent.futures import ThreadPoolExecutor
from typing import Callable, List, Optional, Tuple

import numpy as np
from torchvision.datasets import utils

from .. import configure
from .. import datasets as sjds
from ..datasets import np_savez

CATEGORY_LABEL = {
    "handshake": 0,
    "slapping": 1,
    "punching": 2,
    "walking": 3,
    "fingerguess": 4,
    "strangling": 5,
    "greeting": 6,
    "pushing": 7,
    "hairgrabs": 8,
    "kicking": 9,
}

class Bullying10kClassification(sjds.NeuromorphicDatasetFolder):

    def __init__(
        self, 
        root: str, 
        train: Optional[bool] = None,
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The Bullying10K dataset for action recognition (classification), which 
        is proposed by `Bullying10K: A Neuromorphic Dataset towards 
        Privacy-Preserving Bullying Recognition <https://arxiv.org/abs/2306.11546>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.
        """
        if train is None:
            raise ValueError(
                "The argument `train` must be specified as a boolean value."
            )
        super().__init__(
            root, train, data_type, frames_number, split_by, duration, 
            custom_integrate_function, custom_integrated_frames_dir_name, 
            transform, target_transform
        )
        

    @staticmethod
    def resource_url_md5() -> List[Tuple[str, str, str]]:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        return [
            ("handshake.zip", "https://figshare.com/ndownloader/files/41268834", "681d70f499e736a1e805305284ddc425"),
            ("slapping.zip", "https://figshare.com/ndownloader/files/41247021", "84b41d6805958f9f62f425223916ffc2"),
            ("punching.zip", "https://figshare.com/ndownloader/files/41263314", "40954f480ab210099d448b7b88fc4719"),
            ("walking.zip", "https://figshare.com/ndownloader/files/41247024", "56e4cac9c0814ce701c3b2292c15b6a9"),
            ("fingerguess.zip", "https://figshare.com/ndownloader/files/41253057", "f83114e5b4f0ea57cac86fb080c7e4d7"),
            ("strangling.zip", "https://figshare.com/ndownloader/files/41261904", "8185ecd6f3147e9b609d22f06270aa86"),
            ("greeting.zip", "https://figshare.com/ndownloader/files/41268792", "4a763fad728b04c8356db8544f1121fe"),
            ("pushing.zip", "https://figshare.com/ndownloader/files/41268951", "7986c74ade7149a98672120a89b13ba8"),
            ("hairgrabs.zip", "https://figshare.com/ndownloader/files/41277855", "a9cf690ed0a3305da4a4b8e110f64db1"),
            ("kicking.zip", "https://figshare.com/ndownloader/files/41278008", "6c3218f977de4ac29c84a10b17779c33"),
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return True

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str) -> None:
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract downloaded files.
        '''
        with ThreadPoolExecutor(
            max_workers=min(multiprocessing.cpu_count(), 10)
        ) as tpe:
            sub_threads = []
            for file_name in os.listdir(download_root):
                if not file_name.endswith(".zip"):
                    # move the json files to extract_root directly
                    src_file = os.path.join(download_root, file_name)
                    dst_file = os.path.join(extract_root, file_name)
                    shutil.copy(src_file, dst_file)
                else:
                    zip_file = os.path.join(download_root, file_name)
                    print(f'Extract [{zip_file}] to [{extract_root}].')
                    sub_threads.append(tpe.submit(
                        utils.extract_archive, zip_file, extract_root
                    ))

            for sub_thread in sub_threads:
                if sub_thread.exception():
                    print(sub_thread.exception())
                    exit(-1)

    @staticmethod
    def convert_npy_to_npz(src_path: str, dst_dir: str, label: int):
        """
        :param src_path: the path of a npy file
        :type src_path: str
        :param dst_dir: the path of the directory to contain the converted npz files
        :type dst_dir: str
        :param label: the label of the sample, ranging from 0 to 9
        :type label: int
        :return: None
        
        This function defines how to convert a single npy file to npz format and save converted file in ``dst_dir``.
        """
        original_data = np.load(src_path, allow_pickle=True)
        original_data = [y for x in original_data for y in x]
        # original_data: [(t, x, y, p, ...), ...]
        # npz data: {"t": t, "x": x, "y": y, "p": p, "label": label}
        t = np.array([d[0] for d in original_data])
        x = np.array([d[1] for d in original_data])
        y = np.array([d[2] for d in original_data])
        p = np.array([d[3] for d in original_data])
        fname = os.path.split(src_path)[-1].split(".")[0]
        target_file_path = os.path.join(
            dst_dir, str(label), f'{fname}.npz'
        )
        np_savez(
            target_file_path, t=t, x=x, y=y, p=p, label=label
        )
        print(f"[{target_file_path}] saved.")

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str) -> None:
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root: str
        :return: None

        This function defines how to convert the extracted npy data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        train_dir= os.path.join(events_np_root, "train")
        val_dir = os.path.join(events_np_root, "val")
        os.mkdir(train_dir)
        os.mkdir(val_dir)
        print(f"Mkdir [{train_dir}] and [{val_dir}].")
        for label in range(10):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(val_dir, str(label)))
        print(
            f"Mkdir {os.listdir(train_dir)} in [{train_dir}] "
            f"and {os.listdir(val_dir)} in [{val_dir}]."
        )

        all_files_labels = []
        categories = list(filter(
            lambda x: (not x.endswith(".json")) and (not x.startswith(".")), 
            os.listdir(extract_root)
        ))
        for c in categories:
            cp = os.path.join(extract_root, c)
            for dir_path, _, dir_file_names in os.walk(cp):
                for dfn in dir_file_names:
                    all_files_labels.append(
                        (os.path.join(dir_path, dfn), CATEGORY_LABEL[c])
                    )
        num_files = len(all_files_labels)
        all_files_labels = np.array(all_files_labels)
        print(f"Found {num_files} files in total.")

        # the same way to split training / validation sets as the original work:
        # https://github.com/Brain-Cog-Lab/Bullying10K/blob/main/Bullying10k.py
        val_loc = np.zeros(num_files, dtype=bool)
        val_loc[range(0, num_files, 5)] = 1
        train_files_labels = all_files_labels[~val_loc]
        val_files_labels = all_files_labels[val_loc]
        print(
            f"Training set: {len(train_files_labels)} files. "
            f"Validation set: {len(val_files_labels)} files."
        )

        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(
            multiprocessing.cpu_count(), 
            configure.max_threads_number_for_datasets_preprocess
        )) as tpe:
            sub_threads = []
            print(
                f"Start the ThreadPoolExecutor with max workers"
                f" = [{tpe._max_workers}]."
            )
            for fp, label in train_files_labels:
                sub_threads.append(tpe.submit(
                    Bullying10kClassification.convert_npy_to_npz,
                    fp, train_dir, label
                ))
            for fp, label in val_files_labels:
                sub_threads.append(tpe.submit(
                    Bullying10kClassification.convert_npy_to_npz,
                    fp, val_dir, label
                ))
        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(
            f"All npy files have been converted into npz files "
            f"and into [{train_dir, val_dir}]."
        )

        # remove the extracted files, since they're too large
        print(f"Remove the directory [{extract_root}].")
        shutil.rmtree(extract_root)

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 260, 346

