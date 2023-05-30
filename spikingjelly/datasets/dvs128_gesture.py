from typing import Callable, Dict, Optional, Tuple
import numpy as np
from .. import datasets as sjds
from torchvision.datasets.utils import extract_archive
import os
import multiprocessing
from concurrent.futures import ThreadPoolExecutor
import time
from .. import configure
from ..datasets import np_savez

class DVS128Gesture(sjds.NeuromorphicDatasetFolder):
    def __init__(
            self,
            root: str,
            train: bool = None,
            data_type: str = 'event',
            frames_number: int = None,
            split_by: str = None,
            duration: int = None,
            custom_integrate_function: Callable = None,
            custom_integrated_frames_dir_name: str = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
    ) -> None:
        """
        The DVS128 Gesture dataset, which is proposed by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/html/Amir_A_Low_Power_CVPR_2017_paper.html>`_.

        Refer to :class:`spikingjelly.datasets.NeuromorphicDatasetFolder` for more details about params information.


        .. admonition:: Note
            :class: note

            In SpikingJelly, there are 1176 train samples and 288 test samples. The total samples number is 1464.

            .. code-block:: python

                from spikingjelly.datasets import dvs128_gesture

                data_dir = 'D:/datasets/DVS128Gesture'
                train_set = dvs128_gesture.DVS128Gesture(data_dir, train=True)
                test_set = dvs128_gesture.DVS128Gesture(data_dir, train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            While from the origin paper, `the DvsGesture dataset comprises 1342 instances of a set of 11 hand and arm \
            gestures`. The difference may be caused by different pre-processing methods.

            `snnTorch <https://snntorch.readthedocs.io/>`_ have the same numbers with SpikingJelly:

            .. code-block:: python

                from snntorch.spikevision import spikedata

                train_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=True, num_steps=500, dt=1000)
                test_set = spikedata.DVSGesture("D:/datasets/DVS128Gesture/temp2", train=False, num_steps=1800, dt=1000)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

                # train samples = 1176, test samples = 288
                # total samples = 1464


            But `tonic <https://tonic.readthedocs.io/>`_ has different numbers, which are close to `1342`:

            .. code-block:: python

                import tonic

                train_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=True)
                test_set = tonic.datasets.DVSGesture(save_to='D:/datasets/DVS128Gesture/temp', train=False)
                print(f'train samples = {train_set.__len__()}, test samples = {test_set.__len__()}')
                print(f'total samples = {train_set.__len__() + test_set.__len__()}')

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

            SpikingJelly will read the txt file and get the aedat file name like ``user01_fluorescent.aedat``. The corresponding \
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




            Then SpikingJelly will split the aedat to samples by the time range and class in the csv file. In this sample, \
            the first sample ``user01_fluorescent_0.npz`` is sliced from the origin events ``user01_fluorescent.aedat`` with \
            ``80048239 <= t < 85092709`` and ``label=0``. ``user01_fluorescent_0.npz`` will be saved in ``root/events_np/train/0``.





        """
        assert train is not None
        super().__init__(root, train, data_type, frames_number, split_by, duration, custom_integrate_function, custom_integrated_frames_dir_name, transform, target_transform)
    @staticmethod
    def resource_url_md5() -> list:
        '''
        :return: A list ``url`` that ``url[i]`` is a tuple, which contains the i-th file's name, download link, and MD5
        :rtype: list
        '''
        url = 'https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794'
        return [
            ('DvsGesture.tar.gz', url, '8a5c71fb11e24e5ca5b11866ca6c00a1'),
            ('gesture_mapping.csv', url, '109b2ae64a0e1f3ef535b18ad7367fd1'),
            ('LICENSE.txt', url, '065e10099753156f18f51941e6e44b66'),
            ('README.txt', url, 'a0663d3b1d8307c329a43d949ee32d19')
        ]

    @staticmethod
    def downloadable() -> bool:
        '''
        :return: Whether the dataset can be directly downloaded by python codes. If not, the user have to download it manually
        :rtype: bool
        '''
        return False

    @staticmethod
    def extract_downloaded_files(download_root: str, extract_root: str):
        '''
        :param download_root: Root directory path which saves downloaded dataset files
        :type download_root: str
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :return: None

        This function defines how to extract download files.
        '''
        fpath = os.path.join(download_root, 'DvsGesture.tar.gz')
        print(f'Extract [{fpath}] to [{extract_root}].')
        extract_archive(fpath, extract_root)


    @staticmethod
    def load_origin_data(file_name: str) -> Dict:
        '''
        :param file_name: path of the events file
        :type file_name: str
        :return: a dict whose keys are ``['t', 'x', 'y', 'p']`` and values are ``numpy.ndarray``
        :rtype: Dict

        This function defines how to read the origin binary data.
        '''
        return sjds.load_aedat_v3(file_name)

    @staticmethod
    def split_aedat_files_to_np(fname: str, aedat_file: str, csv_file: str, output_dir: str):
        events = DVS128Gesture.load_origin_data(aedat_file)
        print(f'Start to split [{aedat_file}] to samples.')
        # read csv file and get time stamp and label of each sample
        # then split the origin data to samples
        csv_data = np.loadtxt(csv_file, dtype=np.uint32, delimiter=',', skiprows=1)

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
            mask = np.logical_and(events['t'] >= t_start, events['t'] < t_end)
            file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
            np_savez(file_name,
                     t=events['t'][mask],
                     x=events['x'][mask],
                     y=events['y'][mask],
                     p=events['p'][mask]
                     )
            print(f'[{file_name}] saved.')
            label_file_num[label] += 1

        # old codes:

        # index = 0
        # index_l = 0
        # index_r = 0
        # for i in range(csv_data.shape[0]):
        #     # the label of DVS128 Gesture is 1, 2, ..., 11. We set 0 as the first label, rather than 1
        #     label = csv_data[i][0] - 1
        #     t_start = csv_data[i][1]
        #     t_end = csv_data[i][2]
        #
        #     while True:
        #         t = events['t'][index]
        #         if t < t_start:
        #             index += 1
        #         else:
        #             index_l = index
        #             break
        #     while True:
        #         t = events['t'][index]
        #         if t < t_end:
        #             index += 1
        #         else:
        #             index_r = index
        #             break
        #
        #     file_name = os.path.join(output_dir, str(label), f'{fname}_{label_file_num[label]}.npz')
        #     np.savez(file_name,
        #         t=events['t'][index_l:index_r],
        #         x=events['x'][index_l:index_r],
        #         y=events['y'][index_l:index_r],
        #         p=events['p'][index_l:index_r]
        #     )
        #     print(f'[{file_name}] saved.')
        #     label_file_num[label] += 1

    @staticmethod
    def create_events_np_files(extract_root: str, events_np_root: str):
        '''
        :param extract_root: Root directory path which saves extracted files from downloaded files
        :type extract_root: str
        :param events_np_root: Root directory path which saves events files in the ``npz`` format
        :type events_np_root:
        :return: None

        This function defines how to convert the origin binary data in ``extract_root`` to ``npz`` format and save converted files in ``events_np_root``.
        '''
        aedat_dir = os.path.join(extract_root, 'DvsGesture')
        train_dir = os.path.join(events_np_root, 'train')
        test_dir = os.path.join(events_np_root, 'test')
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        print(f'Mkdir [{train_dir, test_dir}.')
        for label in range(11):
            os.mkdir(os.path.join(train_dir, str(label)))
            os.mkdir(os.path.join(test_dir, str(label)))
        print(f'Mkdir {os.listdir(train_dir)} in [{train_dir}] and {os.listdir(test_dir)} in [{test_dir}].')

        with open(os.path.join(aedat_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            # use multi-thread to accelerate
            t_ckp = time.time()
            with ThreadPoolExecutor(max_workers=min(multiprocessing.cpu_count(), configure.max_threads_number_for_datasets_preprocess)) as tpe:
                sub_threads = []
                print(f'Start the ThreadPoolExecutor with max workers = [{tpe._max_workers}].')


                for fname in trials_to_train_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        sub_threads.append(tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file, os.path.join(aedat_dir, fname + '_labels.csv'), train_dir))


                for fname in trials_to_test_txt.readlines():
                    fname = fname.strip()
                    if fname.__len__() > 0:
                        aedat_file = os.path.join(aedat_dir, fname)
                        fname = os.path.splitext(fname)[0]
                        sub_threads.append(tpe.submit(DVS128Gesture.split_aedat_files_to_np, fname, aedat_file,
                                   os.path.join(aedat_dir, fname + '_labels.csv'), test_dir))


                for sub_thread in sub_threads:
                    if sub_thread.exception():
                        print(sub_thread.exception())
                        exit(-1)

            print(f'Used time = [{round(time.time() - t_ckp, 2)}s].')
        print(f'All aedat files have been split to samples and saved into [{train_dir, test_dir}].')

    @staticmethod
    def get_H_W() -> Tuple:
        '''
        :return: A tuple ``(H, W)``, where ``H`` is the height of the data and ``W` is the weight of the data.
            For example, this function returns ``(128, 128)`` for the DVS128 Gesture dataset.
        :rtype: tuple
        '''
        return 128, 128