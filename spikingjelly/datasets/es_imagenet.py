from typing import Callable, Dict, Optional, Tuple
from .. import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from .. import configure
from ..datasets import np_savez

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
        '''
        :param root: root path of the dataset
        :type root: str
        :param train: whether use the train set
        :type train: bool
        :param data_type: `event` or `frame`
        :type data_type: str
        :param frames_number: the integrated frame number
        :type frames_number: int
        :param split_by: `time` or `number`
        :type split_by: str
        :param duration: the time duration of each frame
        :type duration: int
        :param custom_integrate_function: a user-defined function that inputs are ``events, H, W``.
            ``events`` is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
            ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, H=128 and W=128 for the DVS128 Gesture dataset.
            The user should define how to integrate events to frames, and return frames.
        :type custom_integrate_function: Callable
        :param custom_integrated_frames_dir_name: The name of directory for saving frames integrating by ``custom_integrate_function``.
            If ``custom_integrated_frames_dir_name`` is ``None``, it will be set to ``custom_integrate_function.__name__``
        :type custom_integrated_frames_dir_name: str or None
        :param transform: a function/transform that takes in
            a sample and returns a transformed version.
            E.g, ``transforms.RandomCrop`` for images.
        :type transform: callable
        :param target_transform: a function/transform that takes
            in the target and transforms it.
        :type target_transform: callable

        If ``data_type == 'event'``
            the sample in this dataset is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.

        If ``data_type == 'frame'`` and ``frames_number`` is not ``None``
            events will be integrated to frames with fixed frames number. ``split_by`` will define how to split events.
            See :class:`spikingjelly.datasets.cal_fixed_frames_number_segment_index` for
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
                def integrate_events_to_2_frames_randomly(events: Dict, H: int, W: int):
                    index_split = np.random.randint(low=0, high=events['t'].__len__())
                    frames = np.zeros([2, 2, H, W])
                    frames[0] = sjds.integrate_events_segment_to_frame(events, H, W, 0, index_split)
                    frames[1] = sjds.integrate_events_segment_to_frame(events, H, W, index_split, events['t'].__len__())
                    return frames

                root_dir = 'D:/datasets/DVS128Gesture'
                train_set = DVS128Gesture(root_dir, train=True, data_type='frame', custom_integrate_function=integrate_events_to_2_frames_randomly)

                from spikingjelly.datasets import play_frame
                frame, label = train_set[500]
                play_frame(frame)

        '''
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        urls = [
            ('Validation Set.zip', 'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FValidation%20Set.zip&dl=1', None)
        ]
        for i in range(1, 26):
            idx = str(i).zfill(2)
            urls.append(
                (f'	ES-imagenet-0.18.part{idx}.rar', f'https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part{idx}.rar&dl=1', None)
            )


    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return True
