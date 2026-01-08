from typing import Callable, Optional, Tuple
import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import math

import h5py
import numpy as np
from torchvision.datasets.utils import extract_archive

from .. import configure
from . import utils
from .base import NeuromorphicDatasetFolder
from .base import NeuromorphicDatasetBuilder
from .base import NeuromorphicDatasetConfig


__all__ = [
    "SHD_N_CLASSES", "SpikingHeidelbergDigits",
    "SSC_N_CLASSES", "SpikingSpeechCommands",
]

SHD_N_CLASSES = 20
SSC_N_CLASSES = 35

def _cal_fixed_frames_number_segment_index(
    events_t: np.ndarray, split_by: str, frames_num: int
) -> tuple:
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
        dt = (events_t[-1] - events_t[0]) / frames_num # different from utils.cal_fixed_frames_number_segment_index
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


def _integrate_events_segment_to_frame(
    x: np.ndarray, W: int, j_l: int = 0, j_r: int = -1
) -> np.ndarray:
    frame = np.zeros(shape=[W])
    x = x[j_l: j_r].astype(int)  # avoid overflow

    position = x
    events_number_per_pos = np.bincount(position)
    frame[np.arange(events_number_per_pos.size)] += events_number_per_pos
    return frame


def _integrate_events_by_fixed_frames_number(
    events: dict, split_by: str, frames_num: int, W: int
) -> np.ndarray:
    t, x = (events[key] for key in ('t', 'x'))
    j_l, j_r = _cal_fixed_frames_number_segment_index(t, split_by, frames_num)
    frames = np.zeros([frames_num, W])
    for i in range(frames_num):
        frames[i] = _integrate_events_segment_to_frame(x, W, j_l[i], j_r[i])
    return frames


def _integrate_events_file_to_frames_file_by_fixed_frames_number(
    h5_file: h5py.File, i: int, output_dir: str, split_by: str, frames_num: int,
    W: int, print_save: bool = False
) -> None:
    events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
    label = h5_file['labels'][i]
    fname = os.path.join(output_dir, str(label), str(i))
    utils.np_savez(fname, frames=_integrate_events_by_fixed_frames_number(events, split_by, frames_num, W))
    if print_save:
        print(f'Frames [{fname}] saved.')


def _integrate_events_by_fixed_duration(
    events: dict, duration: int, W: int
) -> np.ndarray:
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
        frames[i] = _integrate_events_segment_to_frame(x, W, left, right)
        left = right

    frames[-1] = _integrate_events_segment_to_frame(x, W, left, N)
    return frames


def _integrate_events_file_to_frames_file_by_fixed_duration(
    h5_file: h5py.File, i: int, output_dir: str, duration: int, W: int, 
    print_save: bool = False
) -> None:
    events = {'t': h5_file['spikes']['times'][i], 'x': h5_file['spikes']['units'][i]}
    label = h5_file['labels'][i]
    fname = os.path.join(output_dir, str(label), str(i))

    frames = _integrate_events_by_fixed_duration(events, duration, W)

    utils.np_savez(fname, frames=frames)
    if print_save:
        print(f'Frames [{fname}] saved.')
    return frames.shape[0]


class NullBuilder(NeuromorphicDatasetBuilder):

    def build_impl(self) -> None:
        pass

    def build(self) -> Tuple[Path, Callable]:
        return self.processed_root, self.get_loader()

    @property
    def processed_root(self) -> Path:
        return self.raw_root

    def get_loader(self) -> Callable:
        return lambda x: x


class SHDFrameFixedNumberBuilder(NeuromorphicDatasetBuilder):

    def __init__(
        self, cfg: NeuromorphicDatasetConfig, raw_root: Path, W: int,
        dataset_name: str = "shd", splits: Tuple[str] = ('train', 'test'),
        n_classes: int = SHD_N_CLASSES
    ):
        super().__init__(cfg, raw_root)
        self.W = W
        self.dataset_name = dataset_name
        self.splits = splits
        self.n_classes = n_classes

    def build_impl(self):
        for split in self.splits:
            processed_root = self.processed_root / split
            processed_root.mkdir()
            print(f'Mkdir [{processed_root}]')
            for i in range(self.n_classes):
                processed_class_root = processed_root / str(i)
                processed_class_root.mkdir()
                print(f'Mkdir [{processed_class_root}]')

            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                futures = []
                print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                h5_file = h5py.File(self.raw_root / f'{self.dataset_name}_{split}.h5')
                for i in range(len(h5_file['labels'])):
                    print(
                        f'Start to integrate [{i}]-th {split} sample to frames and '
                        f'save to [{processed_root}].'
                    )
                    futures.append(tpe.submit(
                        _integrate_events_file_to_frames_file_by_fixed_frames_number,
                        h5_file, i, processed_root, self.cfg.split_by,
                        self.cfg.frames_number, self.W, True
                    ))

                for future in futures:
                    future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        return self.cfg.root / f'frames_number_{self.cfg.frames_number}_split_by_{self.cfg.split_by}'

    def get_loader(self):
        return utils.load_npz_frames


class SHDFrameFixedDurationBuilder(NeuromorphicDatasetBuilder):

    def __init__(
        self, cfg: NeuromorphicDatasetConfig, raw_root: Path, W: int,
        dataset_name: str = "shd", splits: Tuple[str] = ('train', 'test'),
        n_classes: int = SHD_N_CLASSES
    ):
        super().__init__(cfg, raw_root)
        self.W = W
        self.dataset_name = dataset_name
        self.splits = splits
        self.n_classes = n_classes

    def build_impl(self):
        for split in self.splits:
            processed_root = self.processed_root / split
            processed_root.mkdir()
            print(f'Mkdir [{processed_root}]')
            for i in range(self.n_classes):
                processed_class_root = processed_root / str(i)
                processed_class_root.mkdir()
                print(f'Mkdir [{processed_class_root}]')

            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                futures = []
                print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                h5_file = h5py.File(self.raw_root / f'{self.dataset_name}_{split}.h5')
                for i in range(len(h5_file['labels'])):
                    print(
                        f'Start to integrate [{i}]-th {split} sample to frames and '
                        f'save to [{processed_root}].'
                    )
                    futures.append(tpe.submit(
                        _integrate_events_file_to_frames_file_by_fixed_duration,
                        h5_file, i, processed_root, self.cfg.duration,
                        self.W, True
                    ))

                for future in futures:
                    future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        return self.cfg.root / f'duration_{self.cfg.duration}'

    def get_loader(self):
        return utils.load_npz_frames


class SHDFrameCustomIntegrateBuilder(NeuromorphicDatasetBuilder):

    def __init__(
        self, cfg: NeuromorphicDatasetConfig, raw_root: Path, W: int,
        dataset_name: str = "shd", splits: Tuple[str] = ('train', 'test'),
        n_classes: int = SHD_N_CLASSES
    ):
        super().__init__(cfg, raw_root)
        self.W = W
        self.dataset_name = dataset_name
        self.splits = splits
        self.n_classes = n_classes

    def build_impl(self):
        for split in self.splits:
            processed_root = self.processed_root / split
            processed_root.mkdir()
            print(f'Mkdir [{processed_root}]')
            for i in range(self.n_classes):
                processed_class_root = processed_root / str(i)
                processed_class_root.mkdir()
                print(f'Mkdir [{processed_class_root}]')

            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
                futures = []
                print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')

                h5_file = h5py.File(self.raw_root / f'{self.dataset_name}_{split}.h5')
                for i in range(len(h5_file['labels'])):
                    print(
                        f'Start to integrate [{i}]-th {split} sample to frames and '
                        f'save to [{processed_root}].'
                    )
                    futures.append(tpe.submit(
                        self.cfg.custom_integrate_function, h5_file, i,
                        processed_root, self.W
                    ))

                for future in futures:
                    future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        name = self.cfg.custom_integrated_frames_dir_name
        if name is None:
            name = self.cfg.custom_integrate_function.__name__
        return self.cfg.root / name

    def get_loader(self):
        return utils.load_npz_frames


class SpikingHeidelbergDigits(NeuromorphicDatasetFolder):

    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        * **English**

        The Spiking Heidelberg Digits (SHD) dataset, which is proposed by
        `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.

        .. note::

            Unlike other datasets in SpikingJelly, SHD is a neuromorphic audio dataset.

            #. Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``.
            #. The raw dataset replicates the extracted dataset (by symbolic links). The raw dataset consists of two ``.h5`` files instead of a series of ``.npz`` files.
            #. When ``data_type == "event"``, the data loading procedure of ``DatasetFolder`` will be bypassed. Instead, data will be loaded in ``Dataset`` style.
        """
        if train is None:
            raise ValueError("`train` must be `True` or `False`")

        self.cfg = NeuromorphicDatasetConfig(
            root=Path(root),
            train=train,
            data_type=data_type,
            frames_number=frames_number,
            split_by=split_by,
            duration=duration,
            custom_integrate_function=custom_integrate_function,
            custom_integrated_frames_dir_name=custom_integrated_frames_dir_name,
            transform=transform,
            target_transform=target_transform
        )

        self.prepare_raw_dataset()
        builder = self.get_dataset_builder()
        self.processed_root, loader = builder.build()

        split_root = self.processed_root / ('train' if self.cfg.train else 'test')

        if data_type == 'event': # init as Dataset
            self.transform = transform
            self.target_transform = target_transform
        else: # init as DatasetFolder
            super(NeuromorphicDatasetFolder, self).__init__(
                root=split_root,
                loader=loader,
                extensions=self.get_extensions(),
                transform=self.cfg.transform,
                target_transform=self.cfg.target_transform
            )

    @property
    def raw_root(self) -> Path:
        """
        ``root / "events_h5"``
        """
        return self.cfg.root / "events_h5"

    def get_dataset_builder(self):
        if self.cfg.data_type == 'event':
            # prepare for manual __getitem__
            h5_file = self.raw_root / (
                "shd_train.h5" if self.cfg.train else "shd_test.h5"
            )
            self.h5_file = h5py.File(h5_file)
            self.length = len(self.h5_file['labels'])
            return NullBuilder(self.cfg, self.raw_root)

        _, W = self.get_H_W()
        if self.cfg.frames_number is not None:
            return SHDFrameFixedNumberBuilder(self.cfg, self.raw_root, W)
        elif self.cfg.duration is not None:
            return SHDFrameFixedDurationBuilder(self.cfg, self.raw_root, W)
        elif self.cfg.custom_integrate_function is not None:
            return SHDFrameCustomIntegrateBuilder(self.cfg, self.raw_root, W)
        else:
            # not reachable
            raise NotImplementedError(
                f'Please specify the frames number or duration or '
                f'custom integrate function.'
            )

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(None, 700)`` (i.e., 700 channels)
        """
        return None, 700

    @classmethod
    def resource_url_md5(cls) -> list:
        return [
            ('shd_train.h5.zip', 'https://zenkelab.org/datasets/shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
            ('shd_test.h5.zip', 'https://zenkelab.org/datasets/shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f'),
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``True``
        """
        return True

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            futures = []
            for zip_file in download_root.iterdir():
                print(f'Extract [{zip_file}] to [{extract_root}].')
                futures.append(tpe.submit(extract_archive, zip_file, extract_root))

            for future in futures:
                future.result()

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        for f in extract_root.iterdir():
            target = raw_root / f.name
            if target.exists():
                continue
            target.symlink_to(f)

    def __len__(self):
        if self.cfg.data_type == "event":
            return self.length
        return super().__len__()

    def __getitem__(self, index):
        if self.cfg.data_type != "event":
            return super().__getitem__(index)

        events = {
            't': self.h5_file['spikes']['times'][index],
            'x': self.h5_file['spikes']['units'][index]
        }
        label = self.h5_file['labels'][index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return events, label


class SpikingSpeechCommands(NeuromorphicDatasetFolder):

    def __init__(
        self,
        root: str,
        split: str = "train", # 'train' | 'valid' | 'test'
        data_type: str = 'event',
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        * **English**

        The Spiking Speech Commands (SSC) dataset, which is proposed by
        `The Heidelberg Spiking Data Sets for the Systematic Evaluation of Spiking Neural Networks <https://doi.org/10.1109/TNNLS.2020.3044364>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.

        .. note::

            Unlike other datasets in SpikingJelly, SSC is a neuromorphic audio dataset.

            #. Events in this dataset are in the format of ``(x, t)`` rather than ``(x, y, t, p)``.
            #. The raw dataset replicates the extracted dataset (by symbolic links). The raw dataset consists of three ``.h5`` files instead of a series of ``.npz`` files.
            #. When ``data_type == "event"``, the data loading procedure of ``DatasetFolder`` will be bypassed. Instead, data will be loaded in ``Dataset`` style.
        """
        self.splits = ("train", "valid", "test")
        if split not in self.splits:
            raise ValueError(f'Invalid split: {split}; valid splits are {self.splits}')
        self.split = split

        self.cfg = NeuromorphicDatasetConfig(
            root=Path(root),
            train=None,
            data_type=data_type,
            frames_number=frames_number,
            split_by=split_by,
            duration=duration,
            custom_integrate_function=custom_integrate_function,
            custom_integrated_frames_dir_name=custom_integrated_frames_dir_name,
            transform=transform,
            target_transform=target_transform
        )

        self.prepare_raw_dataset()
        builder = self.get_dataset_builder()
        self.processed_root, loader = builder.build()

        split_root = self.get_root_when_train_is_none(self.processed_root)

        if data_type == 'event': # init as Dataset
            self.transform = transform
            self.target_transform = target_transform
        else: # init as DatasetFolder
            super(NeuromorphicDatasetFolder, self).__init__(
                root=split_root,
                loader=loader,
                extensions=self.get_extensions(),
                transform=self.cfg.transform,
                target_transform=self.cfg.target_transform
            )

    def get_root_when_train_is_none(self, _root: Path):
        return _root / self.split

    @property
    def raw_root(self) -> Path:
        """
        ``root / "events_h5"``
        """
        return self.cfg.root / "events_h5"

    def get_dataset_builder(self):
        if self.cfg.data_type == 'event':
            # prepare for manual __getitem__
            h5_file = self.raw_root / f"ssc_{self.split}.h5"
            self.h5_file = h5py.File(h5_file)
            self.length = len(self.h5_file['labels'])
            return NullBuilder(self.cfg, self.raw_root)

        _, W = self.get_H_W()
        if self.cfg.frames_number is not None:
            return SHDFrameFixedNumberBuilder(
                self.cfg, self.raw_root, W, dataset_name="ssc",
                splits=self.splits, n_classes=SSC_N_CLASSES
            )
        elif self.cfg.duration is not None:
            return SHDFrameFixedDurationBuilder(
                self.cfg, self.raw_root, W, dataset_name="ssc",
                splits=self.splits, n_classes=SSC_N_CLASSES
            )
        elif self.cfg.custom_integrate_function is not None:
            return SHDFrameCustomIntegrateBuilder(
                self.cfg, self.raw_root, W, dataset_name="ssc",
                splits=self.splits, n_classes=SSC_N_CLASSES
            )
        else:
            # not reachable
            raise NotImplementedError(
                f'Please specify the frames number or duration or '
                f'custom integrate function.'
            )

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(None, 700)`` (i.e., 700 channels)
        """
        return None, 700

    @classmethod
    def resource_url_md5(cls) -> list:
        return [
            ('shd_train.h5.zip', 'https://zenkelab.org/datasets/shd_train.h5.zip', 'f3252aeb598ac776c1b526422d90eecb'),
            ('shd_test.h5.zip', 'https://zenkelab.org/datasets/shd_test.h5.zip', '1503a5064faa34311c398fb0a1ed0a6f'),
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``True``
        """
        return True

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            futures = []
            for zip_file in download_root.iterdir():
                print(f'Extract [{zip_file}] to [{extract_root}].')
                futures.append(tpe.submit(extract_archive, zip_file, extract_root))

            for future in futures:
                future.result()

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        for f in extract_root.iterdir():
            target = raw_root / f.name
            if target.exists():
                continue
            target.symlink_to(f)

    def __len__(self):
        if self.cfg.data_type == "event":
            return self.length
        return super().__len__()

    def __getitem__(self, index):
        if self.cfg.data_type != "event":
            return super().__getitem__(index)

        events = {
            't': self.h5_file['spikes']['times'][index],
            'x': self.h5_file['spikes']['units'][index]
        }
        label = self.h5_file['labels'][index]
        if self.transform is not None:
            events = self.transform(events)
        if self.target_transform is not None:
            label = self.target_transform(label)
        return events, label
