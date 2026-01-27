from typing import Callable, Optional, Tuple, Union
import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

from torchvision.datasets.utils import extract_archive

from .. import configure
from . import utils
from .base import NeuromorphicDatasetFolder


__all__ = ["NMNIST"]


def _read_bin_save_to_np(bin_file: Union[str, Path], np_file: Union[str, Path]):
    events = utils.load_ATIS_bin(bin_file)
    utils.np_savez(np_file, t=events["t"], x=events["x"], y=events["y"], p=events["p"])
    print(f"Save [{bin_file}] to [{np_file}].")


class NMNIST(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: Optional[str] = "event",
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        **API Language:**
        :ref:`中文 <NMNIST.__init__-cn>` | :ref:`English <NMNIST.__init__-en>`

        ----

        .. _NMNIST.__init__-cn:

        * **中文**

        N-MNIST 数据集，由 `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_ 提出。

        有关参数的更多详细信息，请参考 :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        ----

        .. _NMNIST.__init__-en:

        * **English**

        The N-MNIST dataset, which is proposed by
        `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.
        """
        if train is None:
            raise ValueError("`train` must be `True` or `False`")
        super().__init__(
            root,
            train,
            data_type,
            frames_number,
            split_by,
            duration,
            custom_integrate_function,
            custom_integrated_frames_dir_name,
            transform,
            target_transform,
        )

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(34, 34)``
        """
        return 34, 34

    @classmethod
    def resource_url_md5(cls) -> list:
        url = "https://www.garrickorchard.com/datasets/n-mnist"
        return [
            ("Train.zip", url, "20959b8e626244a1b502305a9e6e2031"),
            ("Test.zip", url, "69ca8762b2fe404d9b9bad1103e97832"),
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``False``
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            futures = []
            for zip_file in os.listdir(download_root):
                zip_file = download_root / zip_file
                print(f"Extract [{zip_file}] to [{extract_root}].")
                futures.append(tpe.submit(extract_archive, zip_file, extract_root))

            for future in futures:
                future.result()

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        t_ckp = time.time()
        with ThreadPoolExecutor(
            max_workers=min(
                multiprocessing.cpu_count(),
                configure.max_threads_number_for_datasets_preprocess,
            )
        ) as tpe:
            futures = []
            for train_test_dir in ["Train", "Test"]:
                source_dir = extract_root / train_test_dir
                target_dir = raw_root / train_test_dir.lower()
                target_dir.mkdir()
                print(f"Mkdir [{target_dir}].")
                for class_name in os.listdir(source_dir):
                    bin_dir = source_dir / class_name
                    np_dir = target_dir / class_name
                    np_dir.mkdir()
                    print(f"Mkdir [{np_dir}].")
                    for bin_file in os.listdir(bin_dir):
                        source_file = bin_dir / bin_file
                        target_file = np_dir / (os.path.splitext(bin_file)[0] + ".npz")
                        print(f"Start to convert [{source_file}] to [{target_file}].")
                        futures.append(
                            tpe.submit(_read_bin_save_to_np, source_file, target_file)
                        )

            for future in futures:
                future.result()

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")
