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
        **API Language:**
        :ref:`中文 <NCaltech101.__init__-cn>` | :ref:`English <NCaltech101.__init__-en>`

        ----

        .. _NCaltech101.__init__-cn:

        * **中文**

        N-Caltech101 数据集，由 `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_ 提出。

        有关参数的更多详细信息，请参考 :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        ----

        .. _NCaltech101.__init__-en:

        * **English**

        The N-Caltech101 dataset, which is proposed by
        `Converting Static Image Datasets to Spiking Neuromorphic Datasets Using Saccades <https://www.frontiersin.org/articles/10.3389/fnins.2015.00437/full>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        :param root: 数据集的根路径
        :type root: Union[str, Path]
        :param data_type: ``\"event\"`` 或 ``\"frame\"``
        :type data_type: str
        :param frames_number: 积分帧的数量
        :type frames_number: Optional[int]
        :param split_by: ``\"time\"`` 或 ``\"number\"``
        :type split_by: Optional[str]
        :param duration: 每帧的时间时长
        :type duration: Optional[int]
        :param custom_integrate_function: 用户自定义积分函数
        :type custom_integrate_function: Optional[Callable]
        :param custom_integrated_frames_dir_name: 自定义积分帧目录名
        :type custom_integrated_frames_dir_name: Optional[str]
        :param transform: 数据变换
        :type transform: Optional[Callable]
        :param target_transform: 标签变换
        :type target_transform: Optional[Callable]

        :param root: Root directory of the dataset
        :type root: Union[str, Path]
        :param data_type: ``\"event\"`` or ``\"frame\"``
        :type data_type: str
        :param frames_number: Number of frames to integrate
        :type frames_number: Optional[int]
        :param split_by: ``\"time\"`` or ``\"number\"``
        :type split_by: Optional[str]
        :param duration: Time duration per frame
        :type duration: Optional[int]
        :param custom_integrate_function: User-defined integrate function
        :type custom_integrate_function: Optional[Callable]
        :param custom_integrated_frames_dir_name: Custom frames directory name
        :type custom_integrated_frames_dir_name: Optional[str]
        :param transform: Transform function
        :type transform: Optional[Callable]
        :param target_transform: Target transform function
        :type target_transform: Optional[Callable]

        :return: None
        :rtype: None
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
        r"""
        **API Language:**
        :ref:`中文 <n_caltech101.get_H_W-cn>` | :ref:`English <n_caltech101.get_H_W-en>`

        ----

        .. _n_caltech101.get_H_W-cn:

        * **中文**

        :return: ``(180, 240)``
        :rtype: Tuple

        ----

        .. _n_caltech101.get_H_W-en:

        * **English**

        :return: ``(180, 240)``
        :rtype: Tuple
        """
        return 180, 240

    @classmethod
    def resource_url_md5(cls) -> list:
        r"""
        **API Language:**
        :ref:`中文 <n_caltech101.resource_url_md5-cn>` | :ref:`English <n_caltech101.resource_url_md5-en>`

        ----

        .. _n_caltech101.resource_url_md5-cn:

        * **中文**

        :return: N-Caltech101 数据集的下载链接与 MD5 校验值列表
        :rtype: list

        ----

        .. _n_caltech101.resource_url_md5-en:

        * **English**

        :return: List of download URLs and MD5 checksums for the N-Caltech101 dataset
        :rtype: list
        """
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
        r"""
        **API Language:**
        :ref:`中文 <n_caltech101.downloadable-cn>` | :ref:`English <n_caltech101.downloadable-en>`

        ----

        .. _n_caltech101.downloadable-cn:

        * **中文**

        由于数据集版权限制，N-Caltech101 不提供自动下载，用户需手动下载。

        :return: ``False``
        :rtype: bool

        ----

        .. _n_caltech101.downloadable-en:

        * **English**

        The N-Caltech101 dataset does not provide automatic download due to copyright restrictions. Users need to download it manually.

        :return: ``False``
        :rtype: bool
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        r"""
        **API Language:**
        :ref:`中文 <n_caltech101.extract_downloaded_files-cn>` | :ref:`English <n_caltech101.extract_downloaded_files-en>`

        ----

        .. _n_caltech101.extract_downloaded_files-cn:

        * **中文**

        从 ``download_root`` 中的 ``Caltech101.zip`` 提取文件到 ``extract_root``。

        :param download_root: 下载文件所在目录
        :type download_root: Path
        :param extract_root: 提取目标目录
        :type extract_root: Path
        :return: None
        :rtype: None

        ----

        .. _n_caltech101.extract_downloaded_files-en:

        * **English**

        Extract ``Caltech101.zip`` from ``download_root`` into ``extract_root``.

        :param download_root: Directory containing the downloaded files
        :type download_root: Path
        :param extract_root: Directory to extract into
        :type extract_root: Path
        :return: None
        :rtype: None
        """
        zip_file = download_root / "Caltech101.zip"
        print(f"Extract [{zip_file}] to [{extract_root}].")
        extract_archive(zip_file, extract_root)

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        r"""
        **API Language:**
        :ref:`中文 <n_caltech101.create_raw_from_extracted-cn>` | :ref:`English <n_caltech101.create_raw_from_extracted-en>`

        ----

        .. _n_caltech101.create_raw_from_extracted-cn:

        * **中文**

        将提取后的 ATIS 二进制文件转换为 ``.npz`` 格式并保存到 ``raw_root``。
        每个类别目录下的 ``.bin`` 文件会被并行转换为 ``.npz`` 文件。

        :param extract_root: 包含已提取文件的目录
        :type extract_root: Path
        :param raw_root: 保存原始数据的目录
        :type raw_root: Path
        :return: None
        :rtype: None

        ----

        .. _n_caltech101.create_raw_from_extracted-en:

        * **English**

        Convert extracted ATIS binary files to ``.npz`` format and save them to ``raw_root``.
        Each ``.bin`` file under the class directories is converted to ``.npz`` in parallel.

        :param extract_root: Directory containing the extracted files
        :type extract_root: Path
        :param raw_root: Directory to save the raw dataset
        :type raw_root: Path
        :return: None
        :rtype: None
        """
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
