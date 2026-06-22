import multiprocessing
import shutil
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable, Optional, Tuple

from torchvision.datasets.utils import extract_archive

from .base import NeuromorphicDatasetFolder

__all__ = ["HARDVS"]


class HARDVS(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        train_test_val: str = None,
        data_type: str = "event",
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <HARDVS.__init__-cn>` | :ref:`English <HARDVS.__init__-en>`

        ----

        .. _HARDVS.__init__-cn:

        * **中文**

        HARDVS 数据集，由 `HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors. <https://arxiv.org/pdf/2211.09648>`_ 提出。

        有关参数的更多详细信息，请参考 :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        ----

        .. _HARDVS.__init__-en:

        * **English**

        The HARDVS dataset, which is proposed by
        `HARDVS: Revisiting Human Activity Recognition with Dynamic Vision Sensors. <https://arxiv.org/pdf/2211.09648>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.

        :param root: 数据集的根路径
        :type root: Union[str, Path]
        :param train_test_val: ``"train"``、``"val"`` 或 ``"test"``
        :type train_test_val: str
        :param data_type: ``"event"`` 或 ``"frame"``
        :type data_type: str
        :param frames_number: 积分帧的数量
        :type frames_number: Optional[int]
        :param split_by: ``"time"`` 或 ``"number"``
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
        :param train_test_val: ``"train"``, ``"val"``, or ``"test"``
        :type train_test_val: str
        :param data_type: ``"event"`` or ``"frame"``
        :type data_type: str
        :param frames_number: Number of frames to integrate
        :type frames_number: Optional[int]
        :param split_by: ``"time"`` or ``"number"``
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
        self.train_test_val = train_test_val
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

    def get_root_when_train_is_none(self, _root: Path):
        return _root / self.train_test_val

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(260, 346)``
        """
        return 260, 346

    @classmethod
    def resource_url_md5(cls) -> list:
        url = "https://github.com/Event-AHU/HARDVS"
        return [
            ("MINI_HARDVS_files.zip", url, "9c4cc0d9ba043faa17f6f1a9e9aff982"),
            ("test_label.txt", url, "5b664af5843f9b476a9c22626f7f5a59"),
            ("train_label.txt", url, "0d642b6e6871034f151b2649a89d8d3c"),
            ("val_label.txt", url, "cd2cebcba80e4552102bbacf2b5df812"),
        ]

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
        extract_archive(download_root / "MINI_HARDVS_files.zip", temp_ext_dir)

        with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), 2)) as tpe:
            futures = []
            for i in range(1, 301):
                s = str(i).zfill(3)
                zip_file = temp_ext_dir / "MINI_HARDVS_files" / f"action_{s}.zip"
                target_dir = extract_root / f"action_{s}"
                print(f"Extract [{zip_file}] to [{target_dir}].")
                futures.append(tpe.submit(extract_archive, zip_file, target_dir))

            for future in futures:
                future.result()

        shutil.rmtree(temp_ext_dir)
        print(f"Rmtree [{temp_ext_dir}].")

        shutil.copy(download_root / "train_label.txt", extract_root / "train_label.txt")
        shutil.copy(download_root / "val_label.txt", extract_root / "val_label.txt")
        shutil.copy(download_root / "test_label.txt", extract_root / "test_label.txt")
        print(
            f"Copy [{download_root / 'train_label.txt'}], "
            f"[{download_root / 'val_label.txt'}], "
            f"[{download_root / 'test_label.txt'}] to [{extract_root}]."
        )

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        for prefix in ("train", "val", "test"):
            target_dir = raw_root / prefix
            target_dir.mkdir()
            print(f"Mkdir {target_dir}.")
            for i in range(1, 301):
                class_dir = target_dir / f"action_{str(i).zfill(3)}"
                class_dir.mkdir()
                print(f"Mkdir {class_dir}.")

            with open(extract_root / f"{prefix}_label.txt") as txt_file:
                for line in txt_file:
                    if len(line) <= 1:
                        continue
                    # e.g., "action_001/dvSave-2021_10_15_19_18_02"
                    class_name, sample_name = line.split(" ")[0].split("/")
                    source_file = extract_root / class_name / f"{sample_name}.npz"
                    target_file = target_dir / class_name / f"{sample_name}.npz"
                    target_file.symlink_to(source_file)
