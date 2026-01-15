from typing import Callable, Union, Optional, Tuple
import os
from pathlib import Path
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time

from torchvision.datasets.utils import extract_archive

from .. import configure
from . import utils
from .base import NeuromorphicDatasetFolder


__all__ = ["NCaltech101"]


def _read_bin_save_to_np(bin_file: Union[str, Path], np_file: Union[str, Path]):
    events = utils.load_ATIS_bin(bin_file)
    utils.np_savez(np_file, t=events["t"], x=events["x"], y=events["y"], p=events["p"])
    print(f"Save [{bin_file}] to [{np_file}].")


class NCaltech101(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        data_type: str = "event",
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ) -> None:
        """
        * **English**

        The N-Caltech101 dataset, which is proposed by
        `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.
        """
        super().__init__(
            root,
            None,
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
        :return: ``(180, 240)``
        """
        return 180, 240

    @classmethod
    def resource_url_md5(cls) -> list:
        url = "https://www.garrickorchard.com/datasets/n-caltech101"
        return [
            ("Caltech101.zip", url, "66201824eabb0239c7ab992480b50ba3"),
            ("Caltech101_annotations.zip", url, "25e64cea645291e368db1e70f214988e"),
            (
                "ReadMe(Caltech101)-SINAPSE-G.txt",
                url,
                "d464b81684e0af3b5773555eb1d5b95c",
            ),
            ("ReadMe(Caltech101).txt", url, "33632a7a5c46074c70509f960d0dd5e5"),
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``False``
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        zip_file = download_root / "Caltech101.zip"
        print(f"Extract [{zip_file}] to [{extract_root}].")
        extract_archive(zip_file, extract_root)

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        t_ckp = time.time()
        extract_root = extract_root / "Caltech101"
        with ThreadPoolExecutor(
            max_workers=min(
                multiprocessing.cpu_count(),
                configure.max_threads_number_for_datasets_preprocess,
            )
        ) as tpe:
            futures = []
            for class_name in os.listdir(extract_root):
                bin_dir = extract_root / class_name
                np_dir = raw_root / class_name
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
