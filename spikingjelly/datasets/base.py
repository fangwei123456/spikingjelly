import time
import os
from pathlib import Path
import abc
from typing import Callable, Optional, Tuple, Union
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor

import numpy as np
from torchvision.datasets.utils import check_integrity, download_url
from torchvision.datasets import DatasetFolder

from .. import configure
from . import utils


__all__ = [
    "NeuromorphicDatasetFolder",
    "NeuromorphicDatasetBuilder",
    "EventBuilder",
    "FrameFixedNumberBuilder",
    "FrameFixedDurationBuilder",
    "FrameCustomIntegrateBuilder",
    "NeuromorphicDatasetConfig",
]


@dataclass(frozen=True)
class NeuromorphicDatasetConfig:
    """
    * **English**

    Configuration container for neuromorphic datasets.

    This dataclass encapsulates all user-specified options that control how a
    neuromorphic dataset is prepared, processed, and loaded. It is **immutable**,
    and is **validated upon initialization**.
    """

    root: Path
    train: Optional[bool]
    data_type: str = 'event'  # 'event' | 'frame'
    frames_number: Optional[int] = None
    split_by: Optional[str] = None  # 'time' | 'number'
    duration: Optional[int] = None
    custom_integrate_function: Optional[Callable] = None
    custom_integrated_frames_dir_name: Optional[str] = None
    transform: Optional[Callable] = None
    target_transform: Optional[Callable] = None

    def __post_init__(self):
        if self.data_type not in ('event', 'frame'):
            raise ValueError(
                f'data_type must be "event" or "frame", but got {self.data_type}'
            )

        if self.data_type == 'event':
            return

        # data_type == "frame"
        cnt = sum([
            self.frames_number is not None,
            self.duration is not None,
            self.custom_integrate_function is not None
        ])
        if cnt != 1:
            raise ValueError(
                'When data_type="frame", one and only one of '
                '"frames_number", "duration", or "custom_integrate_function" should be set.'
            )

        if self.frames_number is not None:
            if not isinstance(self.frames_number, int) or self.frames_number <= 0:
                raise ValueError('frames_number must be a positive integer')
            if self.split_by not in ('time', 'number'):
                raise ValueError('split_by must be "time" or "number"')

        if self.duration is not None:
            if not isinstance(self.duration, int) or self.duration <= 0:
                raise ValueError('duration must be a positive integer')


class NeuromorphicDatasetBuilder(abc.ABC):

    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path):
        r'''
        * **English**

        Abstract base class for neuromorphic dataset builders.

        A dataset builder defines how raw event data are converted into a
        processed dataset that can be loaded by :class:`DatasetFolder <torchvision.datasets.DatasetFolder>`.
        Each builder encapsulates one concrete preprocessing strategy (e.g., event data, fixed-frame
        integration, fixed-duration integration).

        The builder is responsible for:

        * Determining the directory where the processed dataset is saved.
        * Creating processed files if they do not already exist.
        * Providing a loader function for :class:`torchvision.datasets.DatasetFolder`.

        Subclasses should implement the abstract methods :meth:`build_impl`,
        :meth:`get_loader` and property :attr:`processed_root`.

        :param cfg: dataset configuration.
        :type cfg: NeuromorphicDatasetConfig

        :param raw_root: root directory of the raw dataset. The builder will read
            data from this directory.
        :type raw_root: pathlib.Path
        '''
        self.cfg = cfg
        self.raw_root = raw_root

    @property
    @abc.abstractmethod
    def processed_root(self) -> Path:
        r'''
        * **English**

        Root directory of the processed dataset.

        This directory stores the output of the preprocessing step defined by
        the builder.
        '''
        return self.cfg.root / "processed"

    def build(self) -> Tuple[Path, Callable]:
        r'''
        * **English**

        Build the processed dataset if necessary.

        If the processed dataset directory already exists, this method skips
        preprocessing. Otherwise, it invokes :meth:`build_impl` to generate
        processed files.

        :return: a tuple ``(processed_root, loader)``. ``processed_root`` is
            defined by property :attr:`processed_root` . ``loader`` is a
            function that loads individual samples.
        :rtype: Tuple[pathlib.Path, Callable]
        '''
        if self.processed_root.exists():
            print(f'The directory [{self.processed_root}] already exists.')
        else:
            self.processed_root.mkdir()
            print(f'Mkdir [{self.processed_root}].')
            self.build_impl()
        return self.processed_root, self.get_loader()

    @abc.abstractmethod
    def build_impl(self) -> None:
        r'''
        * **English**

        Implement dataset-specific preprocessing logic.

        This method defines how raw data are transformed into processed
        dataset files and saved under :attr:`processed_root`.

        Subclasses must implement this method.
        '''
        pass

    @abc.abstractmethod
    def get_loader(self) -> Callable:
        r'''
        * **English**

        Return a loader function for processed dataset files.

        The returned callable should load a single processed file and return
        the corresponding sample. It will be passed to :class:`DatasetFolder <torchvision.datasets.DatasetFolder>` .
        '''
        pass


class EventBuilder(NeuromorphicDatasetBuilder):

    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path):
        r'''
        * **English**

        Dataset builder for raw event data.

        This builder performs no preprocessing and directly uses the raw dataset as
        the processed dataset. Each sample is loaded directly by ``np.load`` as a raw event
        file (e.g., ``.npz``) without frame integration.

        Typically, this builder is used when ``data_type == "event"``.
        '''
        super().__init__(cfg, raw_root)

    def build_impl(self) -> None:
        pass

    def build(self) -> Tuple[Path, Callable]:
        return self.processed_root, self.get_loader()

    @property
    def processed_root(self) -> Path:
        return self.raw_root

    def get_loader(self) -> Callable:
        return np.load


class FrameFixedNumberBuilder(NeuromorphicDatasetBuilder):

    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        r'''
        * **English**

        Dataset builder for fixed-frame-number integration.

        This builder converts raw event data into a fixed number of frames per
        sample. Events are split according to the specified strategy
        (by time or by event count) and integrated into frames.

        It is used when ``data_type == "frame"`` and ``frames_number`` is specified.

        :param H: height of the output frames.
        :type H: int

        :param W: width of the output frames.
        :type W: int

        Other arguments are the same as those in :class:`NeuromorphicDatasetBuilder`.
        '''
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)

        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
            futures = []
            print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
            for e_root, e_dirs, e_files in os.walk(self.raw_root): 
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file = e_root / e_file
                    print(
                        f'Start to integrate [{events_np_file}] to frames '
                        f'and save to [{output_dir}].'
                    )
                    futures.append(tpe.submit(
                        utils.integrate_events_file_to_frames_file_by_fixed_frames_number,
                        np.load,
                        events_np_file,
                        output_dir,
                        self.cfg.split_by,
                        self.cfg.frames_number,
                        self.H,
                        self.W,
                        True
                    ))
            for future in futures:
                future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        return self.cfg.root / f"frames_number_{self.cfg.frames_number}_split_by_{self.cfg.split_by}"

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class FrameFixedDurationBuilder(NeuromorphicDatasetBuilder):

    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        r'''
        * **English**

        Dataset builder for fixed-duration integration.

        This builder converts raw event data into frame sequences where each frame
        corresponds to a fixed time duration. Different samples may have different lengths.

        It is used when ``data_type == "frame"`` and ``duration`` is specified.

        :param H: height of the output frames.
        :type H: int

        :param W: width of the output frames.
        :type W: int

        Other arguments are the same as those in :class:`NeuromorphicDatasetBuilder`.
        '''
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)

        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
            futures = []
            print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
            for e_root, e_dirs, e_files in os.walk(self.raw_root):
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file = e_root / e_file
                    print(
                        f'Start to integrate [{events_np_file}] to frames '
                        f'and save to [{output_dir}].'
                    )
                    futures.append(tpe.submit(
                        utils.integrate_events_file_to_frames_file_by_fixed_duration,
                        np.load,
                        events_np_file,
                        output_dir,
                        self.cfg.duration,
                        self.H,
                        self.W,
                        True
                    ))
            for future in futures:
                future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        return self.cfg.root / f"duration_{self.cfg.duration}"

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class FrameCustomIntegrateBuilder(NeuromorphicDatasetBuilder):

    def __init__(self, cfg: NeuromorphicDatasetConfig, raw_root: Path, H: int, W: int):
        r'''
        * **English**

        Dataset builder for custom event-to-frame integration.

        This builder applies a user-defined integration function to convert raw
        event data into frame sequences. The resulting frames are saved on disk
        under a user-specified directory. Refer to :doc:`../tutorials/en/neuromorphic_datasets`
        for the way to define a custom integration function.

        It is used when ``data_type == "frame"`` and ``custom_integrate_function`` is specified.

        :param H: height of the output frames.
        :type H: int

        :param W: width of the output frames.
        :type W: int

        Other arguments are the same as those in :class:`NeuromorphicDatasetBuilder`.
        '''
        super().__init__(cfg, raw_root)
        self.H, self.W = H, W

    def build_impl(self) -> None:
        # create the same directory structure
        utils.create_same_directory_structure(self.raw_root, self.processed_root)
        # use multi-thread to accelerate
        t_ckp = time.time()
        with ThreadPoolExecutor(max_workers=configure.max_threads_number_for_datasets_preprocess) as tpe:
            futures = []
            print(f'Start ThreadPoolExecutor with max workers = [{tpe._max_workers}].')
            for e_root, e_dirs, e_files in os.walk(self.raw_root):
                #! Path.walk is not available until Python 3.12
                e_root = Path(e_root)
                if len(e_files) <= 0:
                    continue
                output_dir = self.processed_root / e_root.relative_to(self.raw_root)
                for e_file in e_files:
                    events_np_file: Path = e_root / e_file
                    print(
                        f'Start to integrate [{events_np_file}] to frames '
                        f'and save to [{output_dir}].'
                    )
                    futures.append(tpe.submit(
                        utils.save_frames_to_npz_and_print,
                        output_dir / events_np_file.name,
                        self.cfg.custom_integrate_function(
                            np.load(events_np_file), self.H, self.W
                        )
                    ))

            for future in futures:
                future.result()

        print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')

    @property
    def processed_root(self) -> Path:
        custom_dir_name = self.cfg.custom_integrated_frames_dir_name
        if custom_dir_name is None:
            custom_dir_name = self.cfg.custom_integrate_function.__name__
        return self.cfg.root / custom_dir_name

    def get_loader(self) -> Callable:
        return utils.load_npz_frames


class NeuromorphicDatasetFolder(DatasetFolder):

    def __init__(
        self,
        root: Union[str, Path],
        train: bool = None,
        data_type: str = 'event',
        frames_number: int = None,
        split_by: str = None,
        duration: int = None,
        custom_integrate_function: Callable = None,
        custom_integrated_frames_dir_name: str = None,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
    ):
        r'''
        * **English**

        The base class for SpikingJelly's neuromorphic datasets. Users can define
        a new dataset by inheriting this class and implementing all abstract methods.
        Users can refer to :class:`DVS128Gesture <spikingjelly.datasets.dvs128_gesture.DVS128Gesture>`.

        Users can control data formats by setting arguments:

        If ``data_type == 'event'`` :
            Each sample in this dataset is a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.
        If ``data_type == 'frame'`` :
            If ``frames_number`` is not ``None`` :
                Events will be integrated to frames with fixed frames number. ``split_by`` defines how to split the events.
                See :func:`cal_fixed_frames_number_segment_index <spikingjelly.datasets.utils.cal_fixed_frames_number_segment_index>` for
                more details.
            Else if ``duration`` is not ``None`` :
                Events will be integrated to frames with fixed time duration for each frame.
                The lengths of the resulting sequences are different from one another.
            Else if ``custom_integrate_function`` is not ``None`` :
                Events will be integrated by the user-defined function and saved to
                the ``custom_integrated_frames_dir_name`` directory in ``root`` directory.
                See :doc:`../tutorials/en/neuromorphic_datasets` for more details.

        Dataset preparation process consists of the following steps:

        #. Arguments check.
            This is done by :class:`NeuromorphicDatasetConfig`.
        #. Prepare the *raw dataset*.
            #. Dataset files are downloaded to ``root/download`` (if supported) and verified.
            #. Downloaded files are extracted to ``root/extract``
            #. Extracted data are converted into a unified raw event format (e.g., ``.npz``) and saved to :attr:`raw_root`.
        #. Convert the raw dataset to the *processed dataset*.
            The raw event data are converted into the final dataset format
            according to ``data_type`` and related parameters. This process is
            done by :class:`NeuromorphicDatasetBuilder`.
            Processed dataset is saved to a auto-generated directory :attr:`processed_root`.
        #. Load the processed dataset.
            By inheriting :class:`DatasetFolder <torchvision.datasets.DatasetFolder>` and using its ``__getitem__()``.

        :param root: root path of the dataset
        :type root: Union[str, Path]

        :param train: whether use the train set. Set to ``True`` or ``False`` for
            those datasets provide train/test division, e.g., DVS128 Gesture.
            If the dataset does not provide train/test division, e.g., CIFAR10-DVS,
            please set to ``None`` and use :func:`split_to_train_test_set <spikingjelly.datasets.utils.split_to_train_test_set>`
            function to get train/test set
        :type train: bool

        :param data_type: ``"event"`` or ``"frame"``
        :type data_type: str

        :param frames_number: the number of integrated frames
        :type frames_number: int

        :param split_by: ``"time"`` or ``"number"``
        :type split_by: str

        :param duration: the time duration of each frame, whose unit is the same
            as the time unit of the specific dataset
        :type duration: int

        :param custom_integrate_function: a user-defined function whose inputs
            are ``events, H, W``. ``events`` is a dict whose keys are
            ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``.
            ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, ``H=128`` and ``W=128`` for the DVS128 Gesture dataset.
            The integrated frame sequence (``np.ndarray``) should be returned.
        :type custom_integrate_function: Callable

        :param custom_integrated_frames_dir_name: The name of directory for
            saving the frames integrating by ``custom_integrate_function``.
            If ``None``, it will be set to ``custom_integrate_function.__name__``
        :type custom_integrated_frames_dir_name: Optional[str]

        :param transform: a function/transform that takes in a sample and
            returns a transformed version. E.g, ``transforms.RandomCrop`` for images.
        :type transform: Callable

        :param target_transform: a function/transform that takes in the target
            and transforms it.
        :type target_transform: Callable
        '''
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

        if self.cfg.train is not None:
            split_root = self.processed_root / ('train' if self.cfg.train else 'test')
        else:
            split_root = self.get_root_when_train_is_none(self.processed_root)

        super().__init__(
            root=split_root,
            loader=loader,
            extensions=self.get_extensions(),
            transform=self.cfg.transform,
            target_transform=self.cfg.target_transform
        )

    @property
    def raw_root(self) -> Path:
        r'''
        * **English**

        Root directory of the raw dataset.

        **Raw dataset** serves as an intermediate and unified representation of
        the original dataset. Processed dataset is generated based on the raw dataset.

        :return: default to ``root/events_np``
        '''
        return self.cfg.root / "events_np"

    @classmethod
    def _check_downloaded_files(cls, download_root: Path):
        resource_list = cls.resource_url_md5()
        for file_name, url, md5 in resource_list:
            fpath = download_root / file_name
            if not check_integrity(fpath=fpath, md5=md5):
                print(f'The file [{fpath}] does not exist or is corrupted.')
                if fpath.exists():
                    fpath.unlink()
                    print(f'Remove [{fpath}]')

                if cls.downloadable():
                    print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                    download_url(url=url, root=download_root, filename=file_name, md5=md5)
                else:
                    raise NotImplementedError(
                        f'This dataset can not be downloaded by SpikingJelly, '
                        f'please download [{file_name}] from [{url}] manually '
                        f'and put files at {download_root}.'
                    )

    @classmethod
    def _download_all_files(cls, download_root: Path):
        resource_list = cls.resource_url_md5()
        if cls.downloadable():
            for file_name, url, md5 in resource_list:
                print(f'Download [{file_name}] from [{url}] to [{download_root}]')
                download_url(url=url, root=download_root, filename=file_name, md5=md5)
        else:
            raise NotImplementedError(
                f'This dataset can not be downloaded by SpikingJelly, '
                f'please download files manually and put files at [{download_root}]. '
                f'The resources file_name, url, and md5 are: \n{resource_list}'
            )

    def prepare_raw_dataset(self):
        r'''
        * **English**

        Prepare the **raw dataset**.

        This method ensures that the raw dataset exists under :attr:`raw_root`.
        If not, it performs the following steps sequentially:

        1. Download dataset files to ``root/download`` (if supported) or verify existing downloads.
        2. Extract downloaded files into ``root/extract`` by calling :meth:`extract_downloaded_files`.
        3. Convert extracted data into raw dataset by calling :meth:`create_raw_from_extracted`, and save the raw dataset to :attr:`raw_root`.
        '''
        if self.raw_root.exists():
            return

        # download
        download_root = self.cfg.root / "download"
        if download_root.exists():
            print(
                f'The [{download_root}] directory for saving downloaded files '
                f'already exists, check files...'
            )
            self._check_downloaded_files(download_root)
        else:
            download_root.mkdir()
            print(f'Mkdir [{download_root}] to save downloaded files.')
            self._download_all_files(download_root)

        # extract
        extract_root = self.cfg.root / "extract"
        if extract_root.exists():
            print(
                f'The directory [{extract_root}] for saving extracted files already exists.\n'
                f'SpikingJelly will not check the data integrity of extracted files.\n'
                f'If extracted files are corrupted, please delete [{extract_root}] manually.'
            )
        else:
            extract_root.mkdir()
            print(f'Mkdir [{extract_root}].')
            self.extract_downloaded_files(download_root, extract_root)

        # generate raw dataset in self.raw_root
        self.raw_root.mkdir(exist_ok=True) # raw_root might be equal to extract_root
        print(f'Mkdir [{self.raw_root}].')
        print(
            f'Start to convert the extracted dataset from [{extract_root}] to '
            f'raw dataset in [{self.raw_root}].'
        )
        self.create_raw_from_extracted(extract_root, self.raw_root)

    def get_dataset_builder(self):
        r'''
        * **English**

        Create a dataset builder according to the configuration.

        The builder defines **how raw dataset are converted into the final
        processed dataset**. The specific builder is selected based on
        ``data_type`` and related parameters.

        :return: A dataset builder instance.
        :rtype: :class:`NeuromorphicDatasetBuilder`
        '''
        if self.cfg.data_type == "event":
            return EventBuilder(self.cfg, self.raw_root)

        H, W = self.get_H_W()
        if self.cfg.frames_number is not None:
            return FrameFixedNumberBuilder(self.cfg, self.raw_root, H, W)
        elif self.cfg.duration is not None:
            return FrameFixedDurationBuilder(self.cfg, self.raw_root, H, W)
        elif self.cfg.custom_integrate_function is not None:
            return FrameCustomIntegrateBuilder(self.cfg, self.raw_root, H, W)
        else:
            # not reachable
            raise NotImplementedError(
                f'Please specify the frames number or duration or '
                f'custom integrate function.'
            )

    def get_root_when_train_is_none(self, _root: Path) -> Path:
        r'''
        * **English**

        Determine the directory of the processed dataset when ``train`` is ``None``.

        This method is used for datasets that do not provide a predefined
        train/test split. Subclasses may override this method to implement
        custom directory layouts.

        :param _root: root directory of the processed dataset.
        :type _root: pathlib.Path

        :return: directory of the processed dataset used by :class:`DatasetFolder <torchvision.datasets.DatasetFolder>`.
        :rtype: pathlib.Path
        '''
        return _root

    @classmethod
    def get_extensions(cls) -> Tuple[str]:
        r'''
        * **English**

        Return valid file extensions for processed dataset samples.

        These extensions are passed to :class:`DatasetFolder <torchvision.datasets.DatasetFolder>`
        to identify valid data files.

        :return: tuple of supported file extensions.
        :rtype: Tuple[str]
        '''
        return (".npy", ".npz")

    @classmethod
    @abc.abstractmethod
    def get_H_W(cls) -> Tuple[int]:
        '''
        * **English**

        :return: a tuple ``(H, W)``, where ``H`` is the height of the data and ``W`` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: Tuple[int]
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def resource_url_md5(cls) -> list:
        '''
        * **English**

        :return: a list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def downloadable(cls) -> bool:
        '''
        * **English**

        :return: whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        '''
        * **English**

        This function defines how to extract downloaded files.

        :param download_root: root directory path which saves downloaded dataset files
        :type download_root: pathlib.Path

        :param extract_root: root directory path which saves extracted files from downloaded files
        :type extract_root: pathlib.Path

        :return: None
        '''
        pass

    @classmethod
    @abc.abstractmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        '''
        * **English**

        This function defines how to convert the extracted dataset in
        ``extract_root`` to raw dataset and save the converted files to
        ``raw_root``.

        :param extract_root: root directory where extracted files are saved
        :type extract_root: pathlib.Path

        :param raw_root: root directory where raw dataset files are saved
        :type raw_root: pathlib.Path

        :return: None
        '''
        pass
