import time
import os
import abc
from typing import Callable, Optional, Tuple

import numpy as np
from torchvision.datasets import utils
from torchvision.datasets import DatasetFolder
from concurrent.futures import ThreadPoolExecutor

from .. import configure
from . import utils


class NeuromorphicDatasetFolder(DatasetFolder):
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
        '''
        * **English**

        The base class for neuromorphic dataset.
        Users can define a new dataset by inheriting this class and implementing
        all abstract methods.
        Users can refer to :class:`spikingjelly.datasets.dvs128_gesture.DVS128Gesture`.

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.
        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :func:`cal_fixed_frames_number_segment_index <spikingjelly.datasets.utils.cal_fixed_frames_number_segment_index>` for
            more details.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, and ``duration`` is not ``None``
            events will be integrated to frames with fixed time duration.
        If ``data_type == 'frame'``, ``frames_number`` is ``None``, ``duration`` is ``None``, and ``custom_integrate_function`` is not ``None``:
            events will be integrated by the user-defined function and saved to the ``custom_integrated_frames_dir_name`` directory in ``root`` directory.
            Here is an example from SpikingJelly's tutorials:

            .. code-block:: python

                from spikingjelly.datasets.dvs128_gesture import DVS128Gesture
                from typing import Dict
                import numpy as np
                import spikingjelly.datasets as sjds
                from spikingjelly.datasets.utils import play_frame

                def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
                    index_split = np.random.randint(low=0, high=events['t'].__len__())
                    frames = np.zeros([2, 2, H, W])
                    t, x, y, p = (events[key] for key in ('t', 'x', 'y', 'p'))
                    frames[0] = sjds.utils.integrate_events_segment_to_frame(x, y, p, H, W, 0, index_split)
                    frames[1] = sjds.utils.integrate_events_segment_to_frame(x, y, p, H, W, index_split, events['t'].__len__())
                    return frames

                root_dir = 'D:/datasets/DVS128Gesture'
                train_set = DVS128Gesture(
                    root_dir, train=True, data_type='frame',
                    custom_integrate_function=integrate_events_to_2_frames_randomly
                )
                frame, label = train_set[500]
                play_frame(frame)

        :param root: root path of the dataset
        :type root: str

        :param train: whether use the train set. Set ``True`` or ``False`` for
            those datasets provide train/test division, e.g., DVS128 Gesture.
            If the dataset does not provide train/test division, e.g., CIFAR10-DVS,
            please set ``None`` and use :func:`split_to_train_test_set <spikingjelly.datasets.utils.split_to_train_test_set>`
            function to get train/test set
        :type train: bool

        :param data_type: `event` or `frame`
        :type data_type: str

        :param frames_number: the number of integrated frames
        :type frames_number: int

        :param split_by: `time` or `number`
        :type split_by: str

        :param duration: the time duration of each frame
        :type duration: int

        :param custom_integrate_function: a user-defined function whose inputs
            are ``events, H, W``. ``events`` is a dict whose keys are
            ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
            ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, H=128 and W=128 for the DVS128 Gesture dataset.
            The user should define how to integrate events to frames, and return the frames.
        :type custom_integrate_function: Callable

        :param custom_integrated_frames_dir_name: The name of directory for
            saving frames integrating by ``custom_integrate_function``.
            If ``custom_integrated_frames_dir_name`` is ``None``, it will be set to ``custom_integrate_function.__name__``
        :type custom_integrated_frames_dir_name: Optional[str]

        :param transform: a function/transform that takes in a sample and
            returns a transformed version. E.g, ``transforms.RandomCrop`` for images.
        :type transform: Callable

        :param target_transform: a function/transform that takes in the target
            and transforms it.
        :type target_transform: Callable
        '''

        events_np_root = os.path.join(root, 'events_np')

        if not os.path.exists(events_np_root):

            download_root = os.path.join(root, 'download')

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

            # We have downloaded files and checked files. Now, let us extract the files
            extract_root = os.path.join(root, 'extract')
            if os.path.exists(extract_root):
                print(f'The directory [{extract_root}] for saving extracted files already exists.\n'
                      f'SpikingJelly will not check the data integrity of extracted files.\n'
                      f'If extracted files are not integrated, please delete [{extract_root}] manually, '
                      f'then SpikingJelly will re-extract files from [{download_root}].')
                # shutil.rmtree(extract_root)
                # print(f'Delete [{extract_root}].')
            else:
                os.mkdir(extract_root)
                print(f'Mkdir [{extract_root}].')
                self.extract_downloaded_files(download_root, extract_root)

            # Now let us convert the origin binary files to npz files
            os.mkdir(events_np_root)
            print(f'Mkdir [{events_np_root}].')
            print(f'Start to convert the origin data from [{extract_root}] to [{events_np_root}] in np.ndarray format.')
            self.create_events_np_files(extract_root, events_np_root)

        H, W = self.get_H_W()

        if data_type == 'event':
            _root = events_np_root
            _loader = np.load
            _transform = transform
            _target_transform = target_transform

        elif data_type == 'frame':
            if frames_number is not None:
                assert frames_number > 0 and isinstance(frames_number, int)
                assert split_by == 'time' or split_by == 'number'
                frames_np_root = os.path.join(root, f'frames_number_{frames_number}_split_by_{split_by}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')

                    # create the same directory structure
                    utils.create_same_directory_structure(events_np_root, frames_np_root)

                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        sub_threads = []
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    sub_threads.append(
                                        tpe.submit(
                                            utils.integrate_events_file_to_frames_file_by_fixed_frames_number, 
                                            self.load_events_np, events_np_file, output_dir, split_by, frames_number, H, W, True
                                        )
                                    )
                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = utils.load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif duration is not None:
                assert duration > 0 and isinstance(duration, int)
                frames_np_root = os.path.join(root, f'duration_{duration}')
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')

                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    utils.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        sub_threads = []
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    sub_threads.append(tpe.submit(utils.integrate_events_file_to_frames_file_by_fixed_duration, self.load_events_np, events_np_file, output_dir, duration, H, W, True))
                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)

                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = utils.load_npz_frames
                _transform = transform
                _target_transform = target_transform

            elif custom_integrate_function is not None:
                if custom_integrated_frames_dir_name is None:
                    custom_integrated_frames_dir_name = custom_integrate_function.__name__

                frames_np_root = os.path.join(root, custom_integrated_frames_dir_name)
                if os.path.exists(frames_np_root):
                    print(f'The directory [{frames_np_root}] already exists.')
                else:
                    os.mkdir(frames_np_root)
                    print(f'Mkdir [{frames_np_root}].')
                    # create the same directory structure
                    utils.create_same_directory_structure(events_np_root, frames_np_root)
                    # use multi-thread to accelerate
                    t_ckp = time.time()
                    with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                        print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
                        sub_threads = []
                        for e_root, e_dirs, e_files in os.walk(events_np_root):
                            if e_files.__len__() > 0:
                                output_dir = os.path.join(frames_np_root, os.path.relpath(e_root, events_np_root))
                                for e_file in e_files:
                                    events_np_file = os.path.join(e_root, e_file)
                                    print(
                                        f'Start to integrate [{events_np_file}] to frames and save to [{output_dir}].')
                                    sub_threads.append(tpe.submit(utils.save_frames_to_npz_and_print, os.path.join(output_dir, os.path.basename(events_np_file)), custom_integrate_function(np.load(events_np_file), H, W)))

                        for sub_thread in sub_threads:
                            if sub_thread.exception():
                                print(sub_thread.exception())
                                exit(-1)
                    print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

                _root = frames_np_root
                _loader = utils.load_npz_frames
                _transform = transform
                _target_transform = target_transform


            else:
                raise ValueError('At least one of "frames_number", "duration" and "custom_integrate_function" should not be None.')

        if train is not None:
            if train:
                _root = os.path.join(_root, 'train')
            else:
                _root = os.path.join(_root, 'test')
        else:
            _root = self.set_root_when_train_is_none(_root)

        super().__init__(root=_root, loader=_loader, extensions=('.npz', '.npy'), transform=_transform,
                         target_transform=_target_transform)

    def set_root_when_train_is_none(self, _root: str):
        return _root

    @staticmethod
    @abc.abstractmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None
        This function defines how to extract download files.
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None
        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        pass

    @staticmethod
    @abc.abstractmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        pass

    @staticmethod
    def load_events_np(fname: str):
        '''
        :param fname: file name
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        This function defines how to load a sample from `events_np`. In most cases, this function is `np.load`.
        But for some datasets, e.g., ES-ImageNet, it can be different.
        '''
        return np.load(fname)