import os
from pathlib import Path
from typing import Callable, Optional, Tuple

from torchvision.datasets.utils import extract_archive

from .base import NeuromorphicDatasetFolder


__all__ = ["DVSLip"]


class DVSLip(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
        data_type: str = "event",
        frames_number: Optional[int] = None,
        split_by: Optional[str] = None,
        duration: Optional[int] = None,
        custom_integrate_function: Optional[Callable] = None,
        custom_integrated_frames_dir_name: Optional[str] = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        """
        **API Language:**
        :ref:`中文 <DVSLip.__init__-cn>` | :ref:`English <DVSLip.__init__-en>`

        ----

        .. _DVSLip.__init__-cn:

        * **中文**

        DVS-Lip 数据集，由 `Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading <https://openaccess.thecvf.com/content/CVPR2022/html/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.html>`_ 提出。

        有关参数的更多详细信息，请参考 :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        ----

        .. _DVSLip.__init__-en:

        * **English**

        The DVS-Lip dataset, which is proposed by `Multi-Grained Spatio-Temporal Features Perceived Network for Event-Based Lip-Reading <https://openaccess.thecvf.com/content/CVPR2022/html/Tan_Multi-Grained_Spatio-Temporal_Features_Perceived_Network_for_Event-Based_Lip-Reading_CVPR_2022_paper.html>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`
            for more details about params information.

            :param root: 数据集的根路径
            :type root: Union[str, Path]
            :param train: 是否使用训练集
            :type train: Optional[bool]
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
            :param train: Whether to use training set or test set
            :type train: Optional[bool]
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
        """
        if train is None:
            raise ValueError(
                "The argument `train` must be specified as a boolean value."
            )
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
        r"""
        **API Language:**
        :ref:`中文 <DVSLip.get_H_W-cn>` | :ref:`English <DVSLip.get_H_W-en>`

        ----

        .. _DVSLip.get_H_W-cn:

        * **中文**

        :return: ``(128, 128)``
        :rtype: Tuple[int, int]

        ----

        .. _DVSLip.get_H_W-en:

        * **English**

        :return: ``(128, 128)``
        :rtype: Tuple[int, int]
        """
        return 128, 128

    @classmethod
    def resource_url_md5(cls) -> list:
        return [
            (
                "DVS-Lip.zip",
                "https://sites.google.com/view/event-based-lipreading",
                "2dcb959255122d4cdeb6094ca282494b",
            )
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``False``
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        zip_file = download_root / "DVS-Lip.zip"
        print(f"Extract [{zip_file}] to [{extract_root}].")
        extract_archive(zip_file, extract_root)

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        for split in ("train", "test"):
            source_split_dir = extract_root / "DVS-Lip" / split
            target_split_dir = raw_root / split
            target_split_dir.mkdir()
            for class_name in os.listdir(source_split_dir):
                source_class_dir = source_split_dir / class_name
                target_class_dir = target_split_dir / class_name
                target_class_dir.mkdir()
                for fname in os.listdir(source_class_dir):
                    source_file = source_class_dir / fname
                    target_file = target_class_dir / fname
                    target_file.symlink_to(source_file)
        # The data in source_file is a structured array
        # whose dtype is [('t', '<i4'), ('x', 'i1'), ('y', 'i1'), ('p', 'i1')].
        # Although its form is like [(t0, x0, y0, p0), ...],
        # we can access all "t"s by `source_file['t']` directly!!!
        # This is exactly the same as what we do for npz files, where four arrays
        # `t`, `x`, `y`, `p` are separately saved.
        # In other words, .npy data with structured array is compatible with .npz
        # data with multiple arrays!!!
