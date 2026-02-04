import time
from typing import Callable, Optional, Tuple
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import os
from pathlib import Path

import numpy as np
from torchvision.datasets.utils import extract_archive

from .. import configure
from . import utils
from .base import NeuromorphicDatasetFolder


__all__ = ["DVS128Gesture"]


def _split_aedat_files_to_np(
    fname: str, aedat_file: Path, csv_file: Path, output_dir: Path
):
    events = utils.load_aedat_v3(aedat_file)
    print(f"Start to split [{aedat_file}] to samples.")
    # read csv file and get time stamp and label of each sample
    # then split the origin data to samples
    csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=",", skiprows=1)

    # Note that there are some files that many samples have the same label, e.g., user26_fluorescent_labels.csv
    label_file_num = [0] * 11

    # There are some wrong time stamp in this dataset, e.g., in user22_led_labels.csv, ``endTime_usec`` of the class 9 is
    # larger than ``startTime_usec`` of the class 10. So, the following codes, which are used in old version of SpikingJelly,
    # are replaced by new codes.

    for i in range(csv_data.shape[0]):
        # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        label = csv_data[i][0] - 1
        t_start = csv_data[i][1]
        t_end = csv_data[i][2]
        mask = np.logical_and(events["t"] >= t_start, events["t"] < t_end)
        file_name = output_dir / str(label) / f"{fname}_{label_file_num[label]}.npz"
        utils.np_savez(
            file_name,
            t=events["t"][mask],
            x=events["x"][mask],
            y=events["y"][mask],
            p=events["p"][mask],
        )
        print(f"[{file_name}] saved.")
        label_file_num[label] += 1


class DVS128Gesture(NeuromorphicDatasetFolder):
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
        **API Language:**
        :ref:`中文 <DVS128Gesture.__init__-cn>` | :ref:`English <DVS128Gesture.__init__-en>`

        ----

        .. _DVS128Gesture.__init__-cn:

        * **中文**

        DVS128 Gesture 数据集，由 `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_ 提出。

        有关参数的更多详细信息，请参考 :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>`

        .. admonition:: 注意
            :class: note

            在 SpikingJelly 中，有 1176 个训练样本和 288 个测试样本。总样本数为 1464 个。

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                data_dir = "D:/datasets/DVS128Gesture"
                train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True)
                test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False)
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1176, test samples = 288
                # total samples = 1464


            虽然原始论文指出，`the DvsGesture dataset comprises 1342 instances of a set of 11 hand and arm
            gestures`。差异可能是由于不同的预处理方法导致的。

            `snnTorch <https://snntorch.readthedocs.io/>`_ 与 SpikingJelly 的数字相同：

            .. code-block:: python

                from snntorch.spikevision import spikedata

                train_set = spikedata.DVSGesture(
                    "D:/datasets/DVS128Gesture/temp2",
                    train=True,
                    num_steps=500,
                    dt=1000,
                )
                test_set = spikedata.DVSGesture(
                    "D:/datasets/DVS128Gesture/temp2",
                    train=False,
                    num_steps=1800,
                    dt=1000,
                )
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1176, test samples = 288
                # total samples = 1464


            但 `tonic <https://tonic.readthedocs.io/>`_ 的数字不同，接近 `1342`：

            .. code-block:: python

                import tonic

                train_set = tonic.datasets.DVSGesture(
                    save_to="D:/datasets/DVS128Gesture/temp", train=True
                )
                test_set = tonic.datasets.DVSGesture(
                    save_to="D:/datasets/DVS128Gesture/temp", train=False
                )
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1077, test samples = 264
                # total samples = 1341


            下面说明 SpikingJelly 如何获得 1176 个训练样本和 288 个测试样本。

            原始数据集通过 ``trials_to_train.txt`` 和 ``trials_to_test.txt`` 划分为训练集和测试集。

            .. code-block:: shell

                trials_to_train.txt:

                    user01_fluorescent.aedat
                    user01_fluorescent_led.aedat
                    ...
                    user23_lab.aedat
                    user23_led.aedat

                trials_to_test.txt:

                    user24_fluorescent.aedat
                    user24_fluorescent_led.aedat
                    ...
                    user29_led.aedat
                    user29_natural.aedat

            SpikingJelly 将读取 txt 文件并获取 aedat 文件名，如 ``user01_fluorescent.aedat``。对应的
            标签文件名将被视为 ``user01_fluorescent_labels.csv``。

            .. code-block:: shell

                user01_fluorescent_labels.csv:

                    class	startTime_usec	endTime_usec
                    1	80048239	85092709
                    2	89431170	95231007
                    3	95938861	103200075
                    4	114845417	123499505
                    5	124344363	131742581
                    6	133660637	141880879
                    7	142360393	149138239
                    8	150717639	157362334
                    8	157773346	164029864
                    9	165057394	171518239
                    10	172843790	179442817
                    11	180675853	187389051

            然后 SpikingJelly 将根据 csv 文件中的时间范围和类别将 aedat 切分为样本。在这个示例中，
            第一个样本 ``user01_fluorescent_0.npz`` 是从原始事件 ``user01_fluorescent.aedat`` 中切分的，
            带有 ``80048239 <= t < 85092709`` 和 ``label=0``。``user01_fluorescent_0.npz`` 将保存在 ``root/events_np/train/0``。

        ----

        .. _DVS128Gesture.__init__-en:

        * **English**

        The DVS128 Gesture dataset, which is proposed by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_.

        Refer to :class:`NeuromorphicDatasetFolder <spikingjelly.datasets.base.NeuromorphicDatasetFolder>` for more details about params information.

        .. admonition:: Note
            :class: note

            In SpikingJelly, there are 1176 train samples and 288 test samples. The total samples number is 1464.

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                data_dir = "D:/datasets/DVS128Gesture"
                train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True)
                test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False)
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1176, test samples = 288
                # total samples = 1464


            While from the origin paper, `the DvsGesture dataset comprises 1342 instances of a set of 11 hand and arm
            gestures`. The difference may be caused by different pre-processing methods.

            `snnTorch <https://snntorch.readthedocs.io/>`_ have the same numbers with SpikingJelly:

            .. code-block:: python

                from snntorch.spikevision import spikedata

                train_set = spikedata.DVSGesture(
                    "D:/datasets/DVS128Gesture/temp2",
                    train=True,
                    num_steps=500,
                    dt=1000,
                )
                test_set = spikedata.DVSGesture(
                    "D:/datasets/DVS128Gesture/temp2",
                    train=False,
                    num_steps=1800,
                    dt=1000,
                )
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1176, test samples = 288
                # total samples = 1464


            But `tonic <https://tonic.readthedocs.io/>`_ has different numbers, which are close to `1342`:

            .. code-block:: python

                import tonic

                train_set = tonic.datasets.DVSGesture(
                    save_to="D:/datasets/DVS128Gesture/temp", train=True
                )
                test_set = tonic.datasets.DVSGesture(
                    save_to="D:/datasets/DVS128Gesture/temp", train=False
                )
                print(
                    f"train samples = {train_set.__len__()}, test samples = {test_set.__len__()}"
                )
                print(f"total samples = {train_set.__len__() + test_set.__len__()}")

                # train samples = 1077, test samples = 264
                # total samples = 1341


            Here we show how 1176 train samples and 288 test samples are got in SpikingJelly.

            The origin dataset is split to train and test set by ``trials_to_train.txt`` and ``trials_to_test.txt``.

            .. code-block:: shell

                trials_to_train.txt:

                    user01_fluorescent.aedat
                    user01_fluorescent_led.aedat
                    ...
                    user23_lab.aedat
                    user23_led.aedat

                trials_to_test.txt:

                    user24_fluorescent.aedat
                    user24_fluorescent_led.aedat
                    ...
                    user29_led.aedat
                    user29_natural.aedat

            SpikingJelly will read the txt file and get the aedat file name like ``user01_fluorescent.aedat``. The corresponding
            label file name will be regarded as ``user01_fluorescent_labels.csv``.

            .. code-block:: shell

                user01_fluorescent_labels.csv:

                    class	startTime_usec	endTime_usec
                    1	80048239	85092709
                    2	89431170	95231007
                    3	95938861	103200075
                    4	114845417	123499505
                    5	124344363	131742581
                    6	133660637	141880879
                    7	142360393	149138239
                    8	150717639	157362334
                    8	157773346	164029864
                    9	165057394	171518239
                    10	172843790	179442817
                    11	180675853	187389051

            Then SpikingJelly will split the aedat to samples by the time range and class in the csv file. In this sample,
            the first sample ``user01_fluorescent_0.npz`` is sliced from the origin events ``user01_fluorescent.aedat`` with
            ``80048239 <= t < 85092709`` and ``label=0``. ``user01_fluorescent_0.npz`` will be saved in ``root/events_np/train/0``.
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
        :return: ``(128, 128)``
        """
        return 128, 128

    @classmethod
    def resource_url_md5(cls) -> list:
        url = "https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794"
        return [
            ("DvsGesture.tar.gz", url, "8a5c71fb11e24e5ca5b11866ca6c00a1"),
            ("gesture_mapping.csv", url, "109b2ae64a0e1f3ef535b18ad7367fd1"),
            ("LICENSE.txt", url, "065e10099753156f18f51941e6e44b66"),
            ("README.txt", url, "a0663d3b1d8307c329a43d949ee32d19"),
        ]

    @classmethod
    def downloadable(cls) -> bool:
        """
        :return: ``False``
        """
        return False

    @classmethod
    def extract_downloaded_files(cls, download_root: Path, extract_root: Path):
        fpath = download_root / "DvsGesture.tar.gz"
        print(f"Extract [{fpath}] to [{extract_root}].")
        extract_archive(fpath, extract_root)

    @classmethod
    def create_raw_from_extracted(cls, extract_root: Path, raw_root: Path):
        aedat_dir = extract_root / "DvsGesture"
        train_dir = raw_root / "train"
        test_dir = raw_root / "test"
        train_dir.mkdir()
        test_dir.mkdir()
        print(f"Mkdir [{train_dir, test_dir}.")
        for label in range(11):
            (train_dir / str(label)).mkdir()
            (test_dir / str(label)).mkdir()
        print(
            f"Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}]."
        )

        with (
            open(aedat_dir / "trials_to_train.txt") as trials_to_train_txt,
            open(aedat_dir / "trials_to_test.txt") as trials_to_test_txt,
        ):
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(
                max_workers=min(
                    multiprocessing.cpu_count(),
                    configure.max_threads_number_for_datasets_preprocess,
                )
            ) as tpe:
                futures = []
                print(
                    f"Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}]."
                )

                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if len(fname) > 0:
                        aedat_file = aedat_dir / fname
                        fname = os.path.splitext(fname)[0]
                        futures.append(
                            tpe.submit(
                                _split_aedat_files_to_np,
                                fname,
                                aedat_file,
                                aedat_dir / f"{fname}_labels.csv",
                                train_dir,
                            )
                        )

                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if len(fname) > 0:
                        aedat_file = aedat_dir / fname
                        fname = os.path.splitext(fname)[0]
                        futures.append(
                            tpe.submit(
                                _split_aedat_files_to_np,
                                fname,
                                aedat_file,
                                aedat_dir / f"{fname}_labels.csv",
                                test_dir,
                            )
                        )

                for future in futures:
                    future.result()

            print(f"Used time = [{round(time.time() - t_ckp, 2)}s].")

        print(
            f"All aedat files have been split to samples and saved into [{train_dir, test_dir}]."
        )
