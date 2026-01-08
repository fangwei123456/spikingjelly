import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from typing import Callable, Optional, Tuple, Union

import numpy as np
from torchvision.datasets.utils import extract_archive

from .. import configure
from . import utils
from .base import NeuromorphicDatasetFolder, NeuromorphicDatasetConfig
# https://github.com/jackd/events-tfds/blob/master/events_tfds/data_io/aedat.py


__all__ = [
    "CIFAR10DVS_CLASS_NAMES",
    "CIFAR10DVS",
    "CIFAR10DVSTEBNSplit",
]

CIFAR10DVS_CLASS_NAMES = (
    'airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
)

EVT_DVS = 0  # DVS event type
EVT_APS = 1  # APS event


def _read_bits(arr, mask=None, shift=None):
    if mask is not None:
        arr = arr & mask
    if shift is not None:
        arr = arr >> shift
    return arr


y_mask = 0x7FC00000
y_shift = 22

x_mask = 0x003FF000
x_shift = 12

polarity_mask = 0x800
polarity_shift = 11

valid_mask = 0x80000000
valid_shift = 31


def _skip_header(fp):
    p = 0
    lt = fp.readline()
    ltd = lt.decode().strip()
    while ltd and ltd[0] == "#":
        p += len(lt)
        lt = fp.readline()
        try:
            ltd = lt.decode().strip()
        except UnicodeDecodeError:
            break
    return p


def _load_raw_events(
    fp, bytes_skip=0, bytes_trim=0, filter_dvs=False, times_first=False
):
    p = _skip_header(fp)
    fp.seek(p + bytes_skip)
    data = fp.read()
    if bytes_trim > 0:
        data = data[:-bytes_trim]
    data = np.fromstring(data, dtype='>u4')
    if len(data) % 2 != 0:
        print(data[:20:2])
        print('---')
        print(data[1:21:2])
        raise ValueError('odd number of data elements')
    raw_addr = data[::2]
    timestamp = data[1::2]
    if times_first:
        timestamp, raw_addr = raw_addr, timestamp
    if filter_dvs:
        valid = _read_bits(raw_addr, valid_mask, valid_shift) == EVT_DVS
        timestamp = timestamp[valid]
        raw_addr = raw_addr[valid]
    return timestamp, raw_addr


def _parse_raw_address(
    addr, x_mask=x_mask, x_shift=x_shift, y_mask=y_mask, y_shift=y_shift,
    polarity_mask=polarity_mask, polarity_shift=polarity_shift
):
    polarity = _read_bits(addr, polarity_mask, polarity_shift).astype(np.bool_)
    x = _read_bits(addr, x_mask, x_shift)
    y = _read_bits(addr, y_mask, y_shift)
    return x, y, polarity


def _load_events(
        fp,
        filter_dvs=False,
        # bytes_skip=0,
        # bytes_trim=0,
        # times_first=False,
        **kwargs):
    timestamp, addr = _load_raw_events(
        fp,
        filter_dvs=filter_dvs,
        #   bytes_skip=bytes_skip,
        #   bytes_trim=bytes_trim,
        #   times_first=times_first
    )
    x, y, polarity = _parse_raw_address(addr, **kwargs)
    return timestamp, x, y, polarity


def _load_origin_data(file_name: Union[str, Path]) -> dict:
    with open(file_name, 'rb') as fp:
        t, x, y, p = _load_events(
            fp, x_mask=0xfE, x_shift=1, y_mask=0x7f00, y_shift=8, 
            polarity_mask=1, polarity_shift=None
        )
        return {'t': t, 'x': 127 - y, 'y': 127 - x, 'p': 1 - p.astype(int)}


def _read_aedat_save_to_np(bin_file: Union[str, Path], np_file: Union[str, Path]):
    events = _load_origin_data(bin_file)
    utils.np_savez(
        np_file, t=events['t'], x=events['x'], y=events['y'], p=events['p']
    )
    print(f'Save [{bin_file}] to [{np_file}].')


class CIFAR10DVS(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        * **English**

        The CIFAR10-DVS dataset, which is proposed by
        `CIFAR10-DVS: An Event-Stream Dataset for Object Classification <https://internal-journal.frontiersin.org/articles/10.3389/fnins.2017.00309/full>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.
        """
        super().__init__(
            root, None, data_type, frames_number, split_by, duration,
            custom_integrate_function, custom_integrated_frames_dir_name,
            transform, target_transform
        )

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(128, 128)``
        """
        return 128, 128

    @classmethod
    def resource_url_md5(cls) -> list:
        return [
            ('airplane.zip', 'https://ndownloader.figshare.com/files/7712788', '0afd5c4bf9ae06af762a77b180354fdd'),
            ('automobile.zip', 'https://ndownloader.figshare.com/files/7712791', '8438dfeba3bc970c94962d995b1b9bdd'),
            ('bird.zip', 'https://ndownloader.figshare.com/files/7712794', 'a9c207c91c55b9dc2002dc21c684d785'),
            ('cat.zip', 'https://ndownloader.figshare.com/files/7712812', '52c63c677c2b15fa5146a8daf4d56687'),
            ('deer.zip', 'https://ndownloader.figshare.com/files/7712815', 'b6bf21f6c04d21ba4e23fc3e36c8a4a3'),
            ('dog.zip', 'https://ndownloader.figshare.com/files/7712818', 'f379ebdf6703d16e0a690782e62639c3'),
            ('frog.zip', 'https://ndownloader.figshare.com/files/7712842', 'cad6ed91214b1c7388a5f6ee56d08803'),
            ('horse.zip', 'https://ndownloader.figshare.com/files/7712851', 'e7cbbf77bec584ffbf913f00e682782a'),
            ('ship.zip', 'https://ndownloader.figshare.com/files/7712836', '41c7bd7d6b251be82557c6cce9a7d5c9'),
            ('truck.zip', 'https://ndownloader.figshare.com/files/7712839', '89f3922fd147d9aeff89e76a2b0b70a7')
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``True``
        """
        return True

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 10)) as tpe:
            futures = []
            for zip_file in download_root.iterdir():
                print(f'Extract [{zip_file}] to [{extract_root}].')
                futures.append(tpe.submit(
                    extract_archive, zip_file, extract_root
                ))

            for future in futures:
                future.result()

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
            futures = []
            for class_name in os.listdir(extract_root):
                aedat_dir = extract_root / class_name
                np_dir = raw_root / class_name
                np_dir.mkdir()
                print(f'Mkdir [{np_dir}].')
                for bin_file in os.listdir(aedat_dir):
                    source_file = aedat_dir / bin_file
                    target_file = np_dir / (os.path.splitext(bin_file)[0] + '.npz')
                    print(f'Start to convert [{source_file}] to [{target_file}].')
                    futures.append(tpe.submit(
                        _read_aedat_save_to_np, source_file, target_file
                    ))
            for future in futures:
                future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')


def _move_data(root: Union[str, Path]):
    root = Path(root)

    for cn in CIFAR10DVS_CLASS_NAMES:
        source = root / cn

        target = root / "test" / cn
        if not target.exists():
            target.mkdir(parents=True)
            print(f"mkdir [{target}]")
            for i in range(100):
                source_file = source / f"cifar10_{cn}_{i}.npz"
                target_file = target / f"cifar10_{cn}_{i}.npz"
                target_file.symlink_to(source_file)
                print(f"symlink: [{target_file}] -> [{source_file}]")

        target = root / "train" / cn
        if not target.exists():
            target.mkdir(parents=True)
            print(f"mkdir [{target}]")
            for i in range(100, 1000):
                source_file = source / f"cifar10_{cn}_{i}.npz"
                target_file = target / f"cifar10_{cn}_{i}.npz"
                target_file.symlink_to(source_file)
                print(f"symlink: [{target_file}] -> [{source_file}]")


class CIFAR10DVSTEBNSplit(CIFAR10DVS):

    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        * **English**

        The CIFAR10-DVS dataset, which is proposed by
        `CIFAR10-DVS: An Event-Stream Dataset for Object Classification <https://internal-journal.frontiersin.org/articles/10.3389/fnins.2017.00309/full>`_.

        The original CIFAR10-DVS dataset does not provide train and test split.
        In `Temporal Effective Batch Normalization in Spiking Neural Networks <https://proceedings.neurips.cc/paper_files/paper/2022/hash/de2ad3ed44ee4e675b3be42aa0b615d0-Abstract-Conference.html>`_ ,
        the authors use sample 0-99 in each class as the test set, and the 100-999 as the train set.
        This split is widely used by later works. This class implements this split.

        .. note::

            The validation accuracy on this split is typically much higher than
            that on a random split. Be careful when making comparisons!

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`
        for more details about params information.
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
        if not split_root.exists():
            print(
                f"We have the unsplit processed dataset at [{self.processed_root}]. "
                f"_move_data() is called to split the dataset following TEBN's approach."
            )
            _move_data(self.processed_root)
        print("CIFAR10-DVS has been split after TEBN's approach.")

        super(NeuromorphicDatasetFolder, self).__init__(
            root=split_root,
            loader=loader,
            extensions=self.get_extensions(),
            transform=self.cfg.transform,
            target_transform=self.cfg.target_transform
        )