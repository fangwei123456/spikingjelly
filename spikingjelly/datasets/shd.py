from typing import Callable, Dict, Optional, Tuple

import h5py
import numpy as np
from torch.utils.data import Dataset
from torchvision.datasets import utils
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import math
import bisect
from .. import configure
from ..datasets import np_savez

def cal_fixed_frames_number_segment_index_shd(events_t: np.ndarray, split_by: str, frames_num: int) -> tuple:
    j_l = np.zeros(shape=[frames_num], dtype=int)
    j_r = np.zeros(shape=[frames_num], dtype=int)
    N = events_t.size

    if split_by == 'number':
        di = N // frames_num
        for i in range(frames_num):
            j_l[i] = i * di
            j_r[i] = j_l[i] + di
        j_r[-1] = N

    elif split_by == 'time':
        dt = (events_t[-1] - events_t[0]) / frames_num
        idx = np.arange(N)
        for i in range(frames_num):
            t_l = dt * i + events_t[0]
            t_r = t_l + dt
            mask = np.logical_and(events_t >= t_l, events_t < t_r)
            idx_masked = idx[mask]
            j_l[i] = idx_masked[0]
            j_r[i] = idx_masked[-1] + 1

        j_r[-1] = N
    else:
        raise NotImplementedError

    return j_l, j_r



def integrate_events_segment_to_frame_shd(x: np.ndarray, W: int, j_l: int = 0, j_r: int = -1) -> np.ndarray:

    frame = np.zeros(shape=[W])
    x = x[j_l: j_r].astype(int)  # avoid overflow

    position = x
    events_number_per_pos = np.bincount(position)
    frame[np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame

def integrate_events_by_fixed_frames_number_shd(events: Dict, split_by: str, frames_num: int, W: int) -> np.ndarray:
    t, x = (events[key] for key in ('t', 'x'))
    j_l, j_r = cal_fixed_frames_number_segment_index_shd(t, split_by, frames_num)
    frames = np.zeros([frames_num, W])
    for i in range(frames_num):
        frames[i] = integrate_events_segment_to_frame_shd(x, W, j_l[i], j_r[i])
    return frames

def integrate_events_file_to_frames_file_by_fixed_frames_number_shd(h5_file: h5py.File, i: int, output_dir: str, split_by: str, frames_num: int, W: int, print_save: bool = False) -> None:
    events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
    label = h5_file['labels'][i]
    fname = os.path.join(output_dir, str(label), str(i))
    np_savez(fname, frames=integrate_events_by_fixed_frames_number_shd(events, split_by, frames_num, W))
    if print_save:
        print(f'Frames [{fname}] saved.')

def integrate_events_by_fixed_duration_shd(events: Dict, duration: int, W: int) -> np.ndarray:

    x = events['x']
    t = 1000*events['t']
    t = t - t[0]
    
    N = t.size

    frames_num = int(math.ceil(t[-1] / duration))
    frames = np.zeros([frames_num, W])
    frame_index = t // duration
    left = 0

    for i in range(frames_num - 1):
        right = np.searchsorted(frame_index, i + 1, side='left')
        frames[i] = integrate_events_segment_to_frame_shd(x, W, left, right)
        left = right

    frames[-1] = integrate_events_segment_to_frame_shd(x, W, left, N)
    return frames

def integrate_events_file_to_frames_file_by_fixed_duration_shd(h5_file: h5py.File, i: int, output_dir: str, duration: int, W: int, print_save: bool = False) -> None:
    events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
    label = h5_file['labels'][i]
    fname = os.path.join(output_dir, str(label), str(i))

    frames = integrate_events_by_fixed_duration_shd(events, duration, W)

    np_savez(fname, frames=frames)
    if print_save:
        print(f'Frames [{fname}] saved.')
    return frames.shape[0]

def custom_integrate_function_example(h5_file: h5py.File, i: int, output_dir: str, W: int):
    events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
    label = h5_file['labels'][i]
    frames = np.zeros([2, W])
    index_split = np.random.randint(low=0, high=events['t'].__len__())
    frames[0] = integrate_events_segment_to_frame_shd(events['x'], W, 0, index_split)
    frames[1] = integrate_events_segment_to_frame_shd(events['x'], W, index_split, events['t'].__len__())
    fname = os.path.join(output_dir, str(label), str(i))
    np_savez(fname, frames=frames)




class SpikingHeidelbergDigits(Dataset):
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
        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__()
        self.root = root
        self.train = train
        self.data_type = data_type
        self.frames_number = frames_number
        self.split_by = split_by
        self.duration = duration
        self.custom_integrate_function = custom_integrate_function
        self.custom_integrated_frames_dir_name = custom_integrated_frames_dir_name
        self.transform = transform
        self.target_transform = target_transform

        download_root = os.path.join(root, 'download')
        extract_root = os.path.join(root, 'extract')

        if not os.path.exists(extract_root):

            if os.path.exists(download_root):
                print(f'The [{download_root}] directory for saving downloaded files already exists, check files...')
                # check files
                resource_list = self.resource_url_md5()
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')

                        if os.path.exists(fpath):
                            # If file is corrupted, we will remove it.
                            os.remove(fpath)
                            print(f'Remove [{fpath}]')

                        if self.downloadable():
                            # If file does not exist, we will download it.
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            raise NotImplementedError(
                                f'This dataset can not be downloaded by SpikingJelly, please download [{file_name}] from [{url}] manually and put files at {download_root}.')

            else:
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                resource_list = self.resource_url_md5()
                if self.downloadable():
                    # download and extract file
                    for i in range(resource_list.__len__()):
                        file_name, url, md5 = resource_list[i]
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(f'This dataset can not be downloaded by SpikingJelly, '
                                              f'please download files manually and put files at [{download_root}]. '
                                              f'The resources file_name, url, and md5 are: \n{resource_list}')

            os.mkdir(extract_root)
            print(f'Mkdir [{extract_root}].')
            self.extract_downloaded_files(download_root, extract_root)

        else:
            print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                  f'SpikingJelly will not check the data integrity of extracted files.\n'
                  f'If extracted files are not integrated, please delete [{extract_root}] manually, '
                  f'then SpikingJelly will re-extract files from [{download_root}].')
            # shutil.rmtree(extract_root)
            # print(f'Delete [{extract_root}].')

        if self.data_type == 'event':
            if self.train:
                self.h5_file = h5py.File(os.path.join(extract_root, 'shd_train.h5'))
            else:
                self.h5_file = h5py.File(os.path.join(extract_root, 'shd_test.h5'))
            self.length = self.h5_file['labels'].__len__()

            return

        elif self.data_type == 'frame':

            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by == 'time' or split_by == 'number'
                frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')

                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')






                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        sub_threads = []

                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        h5_file = h5py.File(os.path.join(extract_root, 'shd_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number_shd, h5_file, i,
                                       frames_np_train_root, self.split_by, frames_number, self.get_W(), True))


                        h5_file = h5py.File(os.path.join(extract_root, 'shd_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number_shd, h5_file, i,
                                       frames_np_test_root, self.split_by, frames_number, self.get_W(), True))



                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)



                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                self.frames_np_root = frames_np_root
            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')


                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        sub_threads = []

                        h5_file = h5py.File(os.path.join(extract_root, 'shd_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration_shd, h5_file, i,
                                       frames_np_train_root, self.duration, self.get_W(), True))

                        h5_file = h5py.File(os.path.join(extract_root, 'shd_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration_shd, h5_file, i,
                                       frames_np_test_root, self.duration, self.get_W(), True))

                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
                self.frames_np_root = frames_np_root
            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        sub_threads = []

                        h5_file = h5py.File(os.path.join(extract_root, 'shd_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(custom_integrate_function, h5_file, i,
                                       frames_np_train_root, self.get_W()))

                        h5_file = h5py.File(os.path.join(extract_root, 'shd_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(custom_integrate_function, h5_file, i,
                                       frames_np_test_root, self.get_W()))

                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)


                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                self.frames_np_root = frames_np_root
            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')

            self.frames_path = []
            self.frames_label = []
            if self.train:
                sub_dir = 'train'
            else:
                sub_dir = 'test'

            for i in range(self.classes_number()):
                for fname in os.listdir(os.path.join(self.frames_np_root, sub_dir, str(i))):
                    self.frames_path.append(
                        os.path.join(self.frames_np_root, sub_dir, str(i), fname)
                    )
                    self.frames_label.append(i)

            self.length = self.frames_label.__len__()

        else:
                raise NotImplementedError(self.data_type)


    def classes_number(self):
        return 20

    def __len__(self):
        return self.length

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            if self.transform is not None:
                frames = self.transform(frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return frames, label




    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        return [
            ('shd_train.h5.zip', 'https://zenkelab.org/datasets/shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
            ('shd_test.h5.zip', 'https://zenkelab.org/datasets/shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f'),
        ]

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
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            sub_threads = []
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                sub_threads.append(tpe.submit(extract_archive, zip_file, extract_root))

            for sub_thread in sub_threads:
                if sub_thread.exception():
                    print(sub_thread.exception())
                    exit(-1)

    @staticmethod
    def get_W():
        return 700




class SpikingSpeechCommands(Dataset):
    def __init__(
            self,
            root: str,
            split: str = 'train',
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
        The Spiking Speech Commands (SSC) dataset, which is proposed by `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.

        .. admonition:: Note
            :class: note

            Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``. Thus, this dataset is not inherited from :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` directly. But their procedures are similar.

        :class:`spikingjelly.datasets.shd.custom_integrate_function_example` is an example of ``custom_integrate_function``, which is similar to the cunstom function for DVS Gesture in the ``Neuromorphic Datasets Processing`` tutorial.
        """
        super().__init__()
        self.root = root
        self.split = split
        self.data_type = data_type
        self.frames_number = frames_number
        self.split_by = split_by
        self.duration = duration
        self.custom_integrate_function = custom_integrate_function
        self.custom_integrated_frames_dir_name = custom_integrated_frames_dir_name
        self.transform = transform
        self.target_transform = target_transform

        download_root = os.path.join(root, 'download')
        extract_root = os.path.join(root, 'extract')

        if not os.path.exists(extract_root):

            if os.path.exists(download_root):
                print(f'The [{download_root}] directory for saving downloaded files already exists, check files...')
                # check files
                resource_list = self.resource_url_md5()
                for i in range(resource_list.__len__()):
                    file_name, url, md5 = resource_list[i]
                    fpath = os.path.join(download_root, file_name)
                    if not utils.check_integrity(fpath=fpath, md5=md5):
                        print(f'The file [{fpath}] does not exist or is corrupted.')

                        if os.path.exists(fpath):
                            # If file is corrupted, we will remove it.
                            os.remove(fpath)
                            print(f'Remove [{fpath}]')

                        if self.downloadable():
                            # If file does not exist, we will download it.
                            print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                            utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                        else:
                            raise NotImplementedError(
                                f'This dataset can not be downloaded by SpikingJelly, please download [{file_name}] from [{url}] manually and put files at {download_root}.')

            else:
                os.mkdir(download_root)
                print(f'Mkdir [{download_root}] to save downloaded files.')
                resource_list = self.resource_url_md5()
                if self.downloadable():
                    # download and extract file
                    for i in range(resource_list.__len__()):
                        file_name, url, md5 = resource_list[i]
                        print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                        utils.download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(f'This dataset can not be downloaded by SpikingJelly, '
                                              f'please download files manually and put files at [{download_root}]. '
                                              f'The resources file_name, url, and md5 are: \n{resource_list}')

            os.mkdir(extract_root)
            print(f'Mkdir [{extract_root}].')
            self.extract_downloaded_files(download_root, extract_root)

        else:
            print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                  f'SpikingJelly will not check the data integrity of extracted files.\n'
                  f'If extracted files are not integrated, please delete [{extract_root}] manually, '
                  f'then SpikingJelly will re-extract files from [{download_root}].')
            # shutil.rmtree(extract_root)
            # print(f'Delete [{extract_root}].')

        if self.data_type == 'event':
            if self.split == 'train':
                self.h5_file = h5py.File(os.path.join(extract_root, 'ssc_train.h5'))
            elif self.split == 'valid':
                self.h5_file = h5py.File(os.path.join(extract_root, 'ssc_valid.h5'))
            else:
                self.h5_file = h5py.File(os.path.join(extract_root, 'ssc_test.h5'))
            self.length = self.h5_file['labels'].__len__()

            return

        elif self.data_type == 'frame':

            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by == 'time' or split_by == 'number'
                frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')

                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_valid_root = os.path.join(frames_np_root, 'valid')
                    os.mkdir(frames_np_valid_root)
                    print(f'Mkdir [{frames_np_valid_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_valid_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_valid_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')



                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        sub_threads = []
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number_shd, h5_file,
                                       i, frames_np_train_root, self.split_by, frames_number, self.get_W(), True))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_valid.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(f'Start to integrate [{i}]-th valid sample to frames and save to [{frames_np_valid_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number_shd, h5_file, i,
                                       frames_np_test_root, self.split_by, frames_number, self.get_W(), True))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_frames_number_shd, h5_file, i,
                                       frames_np_test_root, self.split_by, frames_number, self.get_W(), True))


                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)




                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                self.frames_np_root = frames_np_root
            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_valid_root = os.path.join(frames_np_root, 'valid')
                    os.mkdir(frames_np_valid_root)
                    print(f'Mkdir [{frames_np_valid_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_valid_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_valid_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')


                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        sub_threads = []
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration_shd, h5_file, i,
                                       frames_np_train_root, self.duration, self.get_W(), True))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_valid.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th valid sample to frames and save to [{frames_np_valid_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration_shd, h5_file, i,
                                       frames_np_valid_root, self.duration, self.get_W(), True))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(integrate_events_file_to_frames_file_by_fixed_duration_shd, h5_file, i,
                                       frames_np_test_root, self.duration, self.get_W(), True))


                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
                self.frames_np_root = frames_np_root
            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    frames_np_train_root = os.path.join(frames_np_root, 'train')
                    os.mkdir(frames_np_train_root)
                    print(f'Mkdir [{frames_np_train_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_train_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_train_root, str(i))}].')

                    frames_np_valid_root = os.path.join(frames_np_root, 'valid')
                    os.mkdir(frames_np_valid_root)
                    print(f'Mkdir [{frames_np_valid_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_valid_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_valid_root, str(i))}].')

                    frames_np_test_root = os.path.join(frames_np_root, 'test')
                    os.mkdir(frames_np_test_root)
                    print(f'Mkdir [{frames_np_test_root}].')
                    for i in range(self.classes_number()):
                        os.mkdir(os.path.join(frames_np_test_root, str(i)))
                        print(f'Mkdir [{os.path.join(frames_np_test_root, str(i))}].')

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        sub_threads = []

                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_train.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th train sample to frames and save to [{frames_np_train_root}].')
                            sub_threads.append(tpe.submit(custom_integrate_function, h5_file, i,
                                       frames_np_train_root, self.get_W()))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_valid.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th valid sample to frames and save to [{frames_np_valid_root}].')
                            sub_threads.append(tpe.submit(custom_integrate_function, h5_file, i,
                                       frames_np_valid_root, self.get_W()))


                        h5_file = h5py.File(os.path.join(extract_root, 'ssc_test.h5'))
                        for i in range(h5_file['labels'].__len__()):
                            print(
                                f'Start to integrate [{i}]-th test sample to frames and save to [{frames_np_test_root}].')
                            sub_threads.append(tpe.submit(custom_integrate_function, h5_file, i,
                                       frames_np_test_root, self.get_W()))

                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)



                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                self.frames_np_root = frames_np_root
            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')

            self.frames_path = []
            self.frames_label = []
            if self.split == 'train':
                sub_dir = 'train'
            elif self.split == 'valid':
                sub_dir = 'valid'
            else:
                sub_dir = 'test'

            for i in range(self.classes_number()):
                for fname in os.listdir(os.path.join(self.frames_np_root, sub_dir, str(i))):
                    self.frames_path.append(
                        os.path.join(self.frames_np_root, sub_dir, str(i), fname)
                    )
                    self.frames_label.append(i)

            self.length = self.frames_label.__len__()

        else:
                raise NotImplementedError(self.data_type)

    def classes_number(self):
        return 35
    
    def __len__(self):
        return self.length

    def __getitem__(self, i: int):
        if self.data_type == 'event':
            events = {'t': self.h5_file['spikes']['times'][i], 'x': self.h5_file['spikes']['units'][i]}
            label = self.h5_file['labels'][i]
            if self.transform is not None:
                events = self.transform(events)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return events, label

        elif self.data_type == 'frame':
            frames = np.load(self.frames_path[i], allow_pickle=True)['frames'].astype(np.float32)
            label = self.frames_label[i]

            if self.transform is not None:
                frames = self.transform(frames)
            if self.target_transform is not None:
                label = self.target_transform(label)

            return frames, label




    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        return [
            ('ssc_train.h5.zip', 'https://zenkelab.org/datasets/ssc_train.h5.zip', 'd102be95e7144fcc0553d1f45ba94170'),
            ('ssc_valid.h5.zip', 'https://zenkelab.org/datasets/ssc_valid.h5.zip', 'b4eee3516a4a90dd0c71a6ac23a8ae43'),
            ('ssc_test.h5.zip', 'https://zenkelab.org/datasets/ssc_test.h5.zip', 'a35ff1e9cffdd02a20eb850c17c37748'),
        ]

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
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            sub_threads = []
            for zip_file in os.listdir(download_root):
                zip_file = os.path.join(download_root, zip_file)
                print(f'Extract [{zip_file}] to [{extract_root}].')
                sub_threads.append(tpe.submit(extract_archive, zip_file, extract_root))

            for sub_thread in sub_threads:
                if sub_thread.exception():
                    print(sub_thread.exception())
                    exit(-1)

    @staticmethod
    def get_W():
        return 700