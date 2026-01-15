import os
from pathlib import Path
import time
from typing import Callable, Optional, Tuple, Union
from concurrent.futures import ThreadPoolExecutor

import numpy as np

from . import utils
from .base import NeuromorphicDatasetFolder
from .base import NeuromorphicDatasetBuilder
from .base import NeuromorphicDatasetConfig
from .. import configure


__all__ = ["ESImageNet"]


def _load_events(fname: Union[str, Path]):
    events = np.load(fname)
    e_pos = events["pos"]
    e_neg = events["neg"]
    e_pos = np.hstack((e_pos, np.ones((e_pos.shape[0], 1))))
    e_neg = np.hstack((e_neg, np.zeros((e_neg.shape[0], 1))))
    events = np.vstack((e_pos, e_neg))  # shape = [N, 4], N * (x, y, t, p)
    idx = np.argsort(events[:, 2])
    events = events[idx]
    return {"x": events[:, 1], "y": events[:, 0], "t": events[:, 2], "p": events[:, 3]}


class ESImageNetEventBuilder(NeuromorphicDatasetBuilder):
    def build_impl(self) -> None:
        pass

    def build(self) -> Tuple[Path, Callable]:
        return self.processed_root, self.get_loader()

    @property
    def processed_root(self) -> Path:
        return self.raw_root

    def get_loader(self) -> Callable:
        return _load_events


class ESImageNetFrameFixedNumberBuilder(NeuromorphicDatasetBuilder):
    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)

        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(
            max_workers=configure.max_threads_number_for_datasets_preprocess
        ) as tpe:
            futures = []
            print(f"Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].")
            for e_root, e_dirs, e_files in os.walk(self.raw_root):
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file = e_root / e_file
                    print(
                        f"Start to integrate [{events_np_file}] to frames "
                        f"and save to [{output_dir}]."
                    )
                    futures.append(
                        tpe.submit(
                            utils.integrate_events_file_to_frames_file_by_fixed_frames_number,
                            _load_events,
                            events_np_file,
                            output_dir,
                            self.cfg.split_by,
                            self.cfg.frames_number,
                            self.H,
                            self.W,
                            True,
                        )
                    )
            for future in futures:
                future.result()

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")

    @property
    def processed_root(self) -> Path:
        return (
            self.cfg.root
            / f"frames_number_{self.cfg.frames_number}_split_by_{self.cfg.split_by}"
        )

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class ESImageNetFrameFixedDurationBuilder(NeuromorphicDatasetBuilder):
    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)

        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(
            max_workers=configure.max_threads_number_for_datasets_preprocess
        ) as tpe:
            futures = []
            print(f"Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].")
            for e_root, e_dirs, e_files in os.walk(self.raw_root):
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file = e_root / e_file
                    print(
                        f"Start to integrate [{events_np_file}] to frames "
                        f"and save to [{output_dir}]."
                    )
                    futures.append(
                        tpe.submit(
                            utils.integrate_events_file_to_frames_file_by_fixed_duration,
                            _load_events,
                            events_np_file,
                            output_dir,
                            self.cfg.duration,
                            self.H,
                            self.W,
                            True,
                        )
                    )
            for future in futures:
                future.result()

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")

    @property
    def processed_root(self) -> Path:
        return self.cfg.root / f"duration_{self.cfg.duration}"

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class ESImageNetFrameCustomIntegrateBuilder(NeuromorphicDatasetBuilder):
    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)
        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(
            max_workers=configure.max_threads_number_for_datasets_preprocess
        ) as tpe:
            futures = []
            print(f"Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].")
            for e_root, e_dirs, e_files in os.walk(self.raw_root):
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file: Path = e_root / e_file
                    print(
                        f"Start to integrate [{events_np_file}] to frames "
                        f"and save to [{output_dir}]."
                    )
                    futures.append(
                        tpe.submit(
                            utils.save_frames_to_npz_and_print,
                            output_dir / events_np_file.name,
                            self.cfg.custom_integrate_function(
                                _load_events(events_np_file), self.H, self.W
                            ),
                        )
                    )

            for future in futures:
                future.result()

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")

    @property
    def processed_root(self) -> Path:
        custom_dir_name = self.cfg.custom_integrated_frames_dir_name
        if custom_dir_name is None:
            custom_dir_name = self.cfg.custom_integrate_function.__name__
        return self.cfg.root / custom_dir_name

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class ESImageNet(NeuromorphicDatasetFolder):
    def __init__(
        self,
        root: str,
        train: bool = True,
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

        The ES-ImageNet dataset, which is proposed by `ES-ImageNet: A Million Event-Stream Classification Dataset for Spiking Neural Networks <https://www.frontiersin.org/articles/10.3389/fnins.2021.726582/full>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.
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

    def get_dataset_builder(self):
        if self.cfg.data_type == "event":
            return ESImageNetEventBuilder(self.cfg, self.raw_root)

        H, W = self.get_H_W()
        if self.cfg.frames_number is not None:
            return ESImageNetFrameFixedNumberBuilder(self.cfg, self.raw_root, H, W)
        elif self.cfg.duration is not None:
            return ESImageNetFrameFixedDurationBuilder(self.cfg, self.raw_root, H, W)
        elif self.cfg.custom_integrate_function is not None:
            return ESImageNetFrameCustomIntegrateBuilder(self.cfg, self.raw_root, H, W)
        else:
            # not reachable
            raise NotImplementedError(
                f"Please specify the frames number or duration or "
                f"custom integrate function."
            )

    @classmethod
    def get_H_W(cls) -> Tuple:
        """
        :return: ``(256, 256)``
        """
        return 256, 256

    @classmethod
    def resource_url_md5(cls) -> list:
        urls = [
            (
                "ES-imagenet-0.18.part01.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part01.rar&dl=1",
                "900bdd57b5641f7d81cd4620283fef76",
            ),
            (
                "ES-imagenet-0.18.part02.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part02.rar&dl=1",
                "5982532009e863a8f4e18e793314c54b",
            ),
            (
                "ES-imagenet-0.18.part03.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part03.rar&dl=1",
                "8f408c1f5a1d4604e48d0d062a8289a0",
            ),
            (
                "ES-imagenet-0.18.part04.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part04.rar&dl=1",
                "5c5b5cf0a55954eb639964e3da510097",
            ),
            (
                "ES-imagenet-0.18.part05.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part05.rar&dl=1",
                "51feb661b4c9fa87860b63e76b914673",
            ),
            (
                "ES-imagenet-0.18.part06.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part06.rar&dl=1",
                "fcd007a2b17b7c13f338734c53f6db31",
            ),
            (
                "ES-imagenet-0.18.part07.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part07.rar&dl=1",
                "d3e74b96d9c5df15714bbc3abcd329fc",
            ),
            (
                "ES-imagenet-0.18.part08.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part08.rar&dl=1",
                "65b9cf7fa63e18d2e7d92ff45a42a5e5",
            ),
            (
                "ES-imagenet-0.18.part09.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part09.rar&dl=1",
                "241c9a37a83ff9efd305fe46d012211e",
            ),
            (
                "ES-imagenet-0.18.part10.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part10.rar&dl=1",
                "ceee96971008e30d0cdc34086c49fd75",
            ),
            (
                "ES-imagenet-0.18.part11.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part11.rar&dl=1",
                "4fbfefbe6e48758fbb72427c81f119cf",
            ),
            (
                "ES-imagenet-0.18.part12.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part12.rar&dl=1",
                "c8cc163be4e5f6451201dccbded4ec24",
            ),
            (
                "ES-imagenet-0.18.part13.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part13.rar&dl=1",
                "08c9dff32f6b42c49ef7cd78e37c728e",
            ),
            (
                "ES-imagenet-0.18.part14.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part14.rar&dl=1",
                "43aa157dc5bd5fcea81315a46e0322cf",
            ),
            (
                "ES-imagenet-0.18.part15.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part15.rar&dl=1",
                "480a69b050f465ef01efcc44ae29f7df",
            ),
            (
                "ES-imagenet-0.18.part16.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part16.rar&dl=1",
                "11abd24d92b93e7f85acd63abd4a18ab",
            ),
            (
                "ES-imagenet-0.18.part17.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part17.rar&dl=1",
                "3891486a6862c63a325c5f16cd01fdd1",
            ),
            (
                "ES-imagenet-0.18.part18.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part18.rar&dl=1",
                "cf8bb0525b514f411bca9d7c2d681f7c",
            ),
            (
                "ES-imagenet-0.18.part19.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part19.rar&dl=1",
                "3766bc35572ccacc03f0f293c571d0ae",
            ),
            (
                "ES-imagenet-0.18.part20.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part20.rar&dl=1",
                "bf73a5e338644122220e41da7b5630e6",
            ),
            (
                "ES-imagenet-0.18.part21.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part21.rar&dl=1",
                "564de4a2609cbb0bb67ffa1bc51f2487",
            ),
            (
                "ES-imagenet-0.18.part22.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part22.rar&dl=1",
                "60a9e52db1acadfccc9a9809073f0b04",
            ),
            (
                "ES-imagenet-0.18.part23.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part23.rar&dl=1",
                "373b5484826d40d7ec35f0e1605cb6ea",
            ),
            (
                "ES-imagenet-0.18.part24.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part24.rar&dl=1",
                "a50612e889b20f99cc7b2725dfd72e9e",
            ),
            (
                "ES-imagenet-0.18.part25.rar",
                "https://cloud.tsinghua.edu.cn/d/94873ab4ec2a4eb497b3/files/?p=%2FES-imagenet-0.18.part25.rar&dl=1",
                "0802ccdeb0cff29237faf55164524101",
            ),
        ]
        return urls

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``True``
        """
        return True

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        """
        .. warning::

            Require ``rarfile`` package.
        """
        import rarfile

        rar_file = download_root / "ES-imagenet-0.18.part01.rar"
        print(f"Extract [{rar_file}] to [{extract_root}].")
        rar_file = rarfile.RarFile(rar_file)
        rar_file.extractall(extract_root)
        rar_file.close()

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        t_ckp = time.time()
        train_dir = raw_root / "train"
        source_train_dir = extract_root / "ES-imagenet-0.18" / "train"
        train_dir.mkdir()
        print(f"Mkdir [{train_dir}].")
        utils.create_same_directory_structure(source_train_dir, train_dir)
        for class_dir in os.listdir(source_train_dir):
            source_dir = source_train_dir / class_dir
            target_dir = train_dir / class_dir
            print(f"Create soft links from [{source_dir}] to [{target_dir}].")
            for sample in os.listdir(source_dir):
                source_file = source_dir / sample
                target_file = target_dir / sample
                target_file.symlink_to(source_file)

        val_txt_path = extract_root / "ES-imagenet-0.18" / "vallabel.txt"
        val_label = np.loadtxt(val_txt_path, delimiter=" ", usecols=(1,), dtype=int)
        val_fname = np.loadtxt(val_txt_path, delimiter=" ", usecols=(0,), dtype=str)
        source_dir = extract_root / "ES-imagenet-0.18" / "val"
        target_dir = raw_root / "test"
        target_dir.mkdir()
        print(f"Mkdir [{target_dir}].")
        utils.create_same_directory_structure(source_train_dir, target_dir)

        for fname, label in zip(val_fname, val_label):
            source_file = source_dir / fname
            target_file = target_dir / f"class{label}" / f"{fname}"
            target_file.symlink_to(source_file)

        print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")
        print(
            f"Note that files in [{raw_root}] are soft links whose source files "
            f"are in [{extract_root}]. If you want to use events, do not "
            f"delete [{extract_root}]."
        )
