import os
from pathlib import Path
from typing import Callable, Optional, Tuple, Union
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
import shutil

import scipy.io
from torchvision.datasets.utils import extract_archive

from .. import configure
from .base import NeuromorphicDatasetFolder
from . import utils


__all__ = ["ASLDVS"]


def _read_mat_save_to_np(mat_file: Union[str, Path], np_file: Union[str, Path]):
    mat_file, np_file = str(mat_file), str(np_file)
    events = scipy.io.loadmat(mat_file)
    events = {
        "t": events["ts"].squeeze(),
        "x": 239 - events["x"].squeeze(),
        "y": 179 - events["y"].squeeze(),
        "p": events["pol"].squeeze(),
    }
    utils.np_savez(np_file, t=events["t"], x=events["x"], y=events["y"], p=events["p"])
    print(f"Save [{mat_file}] to [{np_file}].")


class ASLDVS(NeuromorphicDatasetFolder):
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
    ):
        """
        * **English**

        The ASL-DVS dataset, which is proposed by `Graph-based Object Classification for Neuromorphic Vision Sensing <https://openaccess.thecvf.com/content_ICCV_2019/html/Bi_Graph-Based_Object_Classification_for_Neuromorphic_Vision_Sensing_ICCV_2019_paper.html>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`
        for more details about params information.

        .. note::

            ASLDVS's Dropbox link is expired. Users can download this dataset
            from the OpenI mirror manually by the following commands:

            .. code:: shell

                pip install openi
                openi dataset download OpenI/ASLDVS --local_dir ./ASLDVS --max_workers 10

            Then you can extract ``ASLDVS.zip`` and get ``ICCV2019_DVS_dataset.zip`` .
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
        print(
            "The ICCV2019_DVS_dataset.zip is packed by dropbox. We find that the"
            "MD5 of this zip file can change. So, MD5 check will not be used for"
            "ASL-DVS dataset."
        )
        print(
            "Update: The Dropbox link is expired now. You can download this dataset"
            "from the OpenI mirror manually by the following commands:\n"
            "----------\n"
            "pip install openi\n"
            "openi dataset download OpenI/ASLDVS --local_dir ./ASLDVS --max_workers 10\n"
            "----------\n"
            'Then you can extract "ASLDVS.zip" and get "ICCV2019_DVS_dataset.zip".'
        )
        url = (
            "https://www.dropbox.com/sh/ibq0jsicatn7l6r/AACNrNELV56rs1YInMWUs9CAa?dl=0"
        )
        return [("ICCV2019_DVS_dataset.zip", url, None)]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``False``
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        temp_ext_dir = download_root / "temp_ext"
        temp_ext_dir.mkdir()
        print(f"Mkdir [{temp_ext_dir}].")
        extract_archive(download_root / "ICCV2019_DVS_dataset.zip", temp_ext_dir)

        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            futures = []
            for zip_file in temp_ext_dir.iterdir():
                if zip_file.suffix == ".zip":
                    print(f"Extract [{zip_file}] to [{extract_root}].")
                    futures.append(tpe.submit(extract_archive, zip_file, extract_root))
            for future in futures:
                future.result()

        shutil.rmtree(temp_ext_dir)
        print(f"Rmtree [{temp_ext_dir}].")

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
            for class_name in os.listdir(extract_root):
                mat_dir = extract_root / class_name
                np_dir = raw_root / class_name
                np_dir.mkdir()
                print(f"Mkdir [{np_dir}].")
                for bin_file in os.listdir(mat_dir):
                    source_file = mat_dir / bin_file
                    target_file = np_dir / (os.path.splitext(bin_file)[0] + ".npz")
                    print(f"Start to convert [{source_file}] to [{target_file}].")
                    futures.append(
                        tpe.submit(_read_mat_save_to_np, source_file, target_file)
                    )
            for future in futures:
                future.result()

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")
