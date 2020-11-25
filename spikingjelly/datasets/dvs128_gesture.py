from .utils import (
    EventsFramesDatasetBase, 
    convert_events_dir_to_frames_dir,
    FunctionThread,
    normalize_frame,
)
import os
import tqdm
import numpy as np
import struct
from torchvision.datasets import utils
import time
import multiprocessing
import torch
# https://www.research.ibm.com/dvsgesture/
# https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794

labels_dict = {
'hand_clapping': 1,  # 注意不是从0开始
'right_hand_wave': 2,
'left_hand_wave': 3,
'right_arm_clockwise': 4,
'right_arm_counter_clockwise': 5,
'left_arm_clockwise': 6,
'left_arm_counter_clockwise': 7,
'arm_roll': 8,
'air_drums': 9,
'air_guitar': 10,
'other_gestures': 11
}  # gesture_mapping.csv
# url md5
resource = ['https://ibm.ent.box.com/s/3hiq58ww1pbbjrinh367ykfdf60xsfm8/folder/50167556794', '8a5c71fb11e24e5ca5b11866ca6c00a1']

class DVS128Gesture(EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 128, 128

    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        file_name = os.path.join(download_root, 'DvsGesture.tar.gz')
        if os.path.exists(file_name):
            print('DvsGesture.tar.gz already exists, check md5')
            if utils.check_md5(file_name, resource[1]):
                print('md5 checked, extracting...')
                utils.extract_archive(file_name, extract_root)
                return
            else:
                print(f'{file_name} corrupted.')


        print(f'Please download from {resource[0]} and save to {download_root} manually.')
        raise NotImplementedError


    @staticmethod
    def read_bin(file_name: str):
        # https://gitlab.com/inivation/dv/dv-python/
        with open(file_name, 'rb') as bin_f:
            # skip ascii header
            line = bin_f.readline()
            while line.startswith(b'#'):
                if line == b'#!END-HEADER\r\n':
                    break
                else:
                    line = bin_f.readline()

            txyp = {
                't': [],
                'x': [],
                'y': [],
                'p': []
            }
            while True:
                header = bin_f.read(28)
                if not header or len(header) == 0:
                    break

                # read header
                e_type = struct.unpack('H', header[0:2])[0]
                e_source = struct.unpack('H', header[2:4])[0]
                e_size = struct.unpack('I', header[4:8])[0]
                e_offset = struct.unpack('I', header[8:12])[0]
                e_tsoverflow = struct.unpack('I', header[12:16])[0]
                e_capacity = struct.unpack('I', header[16:20])[0]
                e_number = struct.unpack('I', header[20:24])[0]
                e_valid = struct.unpack('I', header[24:28])[0]

                data_length = e_capacity * e_size
                data = bin_f.read(data_length)
                counter = 0

                if e_type == 1:
                    while data[counter:counter + e_size]:
                        aer_data = struct.unpack('I', data[counter:counter + 4])[0]
                        timestamp = struct.unpack('I', data[counter + 4:counter + 8])[0] | e_tsoverflow << 31
                        x = (aer_data >> 17) & 0x00007FFF
                        y = (aer_data >> 2) & 0x00007FFF
                        pol = (aer_data >> 1) & 0x00000001
                        counter = counter + e_size
                        txyp['x'].append(x)
                        txyp['y'].append(y)
                        txyp['t'].append(timestamp)
                        txyp['p'].append(pol)
                else:
                    # non-polarity event packet, not implemented
                    pass
            txyp['x'] = np.asarray(txyp['x'])
            txyp['y'] = np.asarray(txyp['y'])
            txyp['t'] = np.asarray(txyp['t'])
            txyp['p'] = np.asarray(txyp['p'])
            return txyp


    @staticmethod
    def convert_aedat_dir_to_npy_dir(aedat_data_dir: str, events_npy_train_root: str, events_npy_test_root: str):
        def cvt_files_fun(aedat_file_list, output_dir):
            for aedat_file in aedat_file_list:
                base_name = aedat_file[0: -6]
                events = DVS128Gesture.read_bin(os.path.join(aedat_data_dir, aedat_file))
                # 读取csv文件，获取各段的label，保存对应的数据和label
                events_csv = np.loadtxt(os.path.join(aedat_data_dir, base_name + '_labels.csv'),
                                        dtype=np.uint32, delimiter=',', skiprows=1)
                index = 0
                index_l = 0
                index_r = 0
                for i in range(events_csv.shape[0]):
                    label = events_csv[i][0]
                    t_start = events_csv[i][1]
                    t_end = events_csv[i][2]

                    while True:
                        t = events['t'][index]
                        if t < t_start:
                            index += 1
                        else:
                            index_l = index  # 左闭
                            break
                    while True:
                        t = events['t'][index]
                        if t < t_end:
                            index += 1
                        else:
                            index_r = index  # 右开
                            break
                    # [index_l, index_r)
                    j = 0
                    while True:
                        file_name = os.path.join(output_dir, f'{base_name}_{label}_{j}.npy')
                        # 由于不同线程执行的base_name一定不相同，因此这里不会出现多线程之间的数据复用造成的错误
                        if os.path.exists(file_name):  # 防止同一个aedat里存在多个相同label的数据段
                            j += 1
                        else:
                            np.save(file=file_name, arr={
                                't': events['t'][index_l:index_r],
                                'x': events['x'][index_l:index_r],
                                'y': events['y'][index_l:index_r],
                                'p': events['p'][index_l:index_r]
                            })
                            break

        with open(os.path.join(aedat_data_dir, 'trials_to_train.txt')) as trials_to_train_txt, open(
                os.path.join(aedat_data_dir, 'trials_to_test.txt')) as trials_to_test_txt:
            train_list = []
            for fname in trials_to_train_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:
                    train_list.append(fname)
            test_list = []
            for fname in trials_to_test_txt.readlines():
                fname = fname.strip()
                if fname.__len__() > 0:
                    test_list.append(fname)


        # 将aedat_data_dir目录下的.aedat文件读取并转换成np保存的字典，保存在npy_data_dir目录
        print('convert events data from aedat to numpy format.')
        # 速度很慢，并行化

        npy_data_num = train_list.__len__() + test_list.__len__()
        thread_num = max(multiprocessing.cpu_count(), 2)
        block = train_list.__len__() // (thread_num - 1)  # 训练集分成thread_num - 1个子任务
        thread_list = []
        for i in range(thread_num - 1):

            thread_list.append(FunctionThread(cvt_files_fun, train_list[i * block: (i + 1) * block], events_npy_train_root))
            print(f'thread {i} start')
            thread_list[-1].start()

        # 测试集再单独作为一个线程
        thread_list.append(FunctionThread(cvt_files_fun, test_list, events_npy_test_root))
        print(f'thread {thread_num - 1} start')
        thread_list[-1].start()

        with tqdm.tqdm(total=npy_data_num) as pbar:
            while True:
                working_thread = []
                finished_thread = []
                for i in range(thread_list.__len__()):
                    if thread_list[i].is_alive():
                        working_thread.append(i)
                    else:
                        finished_thread.append(i)
                pbar.update(utils.list_files(events_npy_train_root, '.npy').__len__() + utils.list_files(events_npy_test_root, '.npy').__len__())
                print('wroking thread:', working_thread)
                print('finished thread:', finished_thread)
                if finished_thread.__len__() == thread_list.__len__():
                    return
                else:
                    time.sleep(10)





    @staticmethod
    def create_frames_dataset(events_data_dir: str, frames_data_dir: str, frames_num: int, split_by: str, normalization: str or None):
        width, height = DVS128Gesture.get_wh()
        def read_fun(file_name):
            return np.load(file_name, allow_pickle=True).item()
        convert_events_dir_to_frames_dir(events_data_dir, frames_data_dir, '.npy',
                                                               read_fun, height, width, frames_num, split_by,
                                                               normalization, thread_num=4)

    @staticmethod
    def get_events_item(file_name):
        return np.load(file_name, allow_pickle=True).item(), int(os.path.basename(file_name).split('_')[-2]) - 1

    @staticmethod
    def get_frames_item(file_name):
        return torch.from_numpy(np.load(file_name)).float(), int(os.path.basename(file_name).split('_')[-2]) - 1

    def __init__(self, root: str, train: bool, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        '''
        :param root: 保存数据集的根目录。其中应该至少包含 `DvsGesture.tar.gz` 和 `gesture_mapping.csv`
        :type root: str
        :param train: 是否使用训练集
        :type train: bool
        :param use_frame: 是否将事件数据转换成帧数据
        :type use_frame: bool
        :param frames_num: 转换后数据的帧数
        :type frames_num: int
        :param split_by: 脉冲数据转换成帧数据的累计方式。``'time'`` 或 ``'number'``
        :type split_by: str
        :param normalization: 归一化方法，为 ``None`` 表示不进行归一化；
                        为 ``'frequency'`` 则每一帧的数据除以每一帧的累加的原始数据数量；
                        为 ``'max'`` 则每一帧的数据除以每一帧中数据的最大值；
                        为 ``norm`` 则每一帧的数据减去每一帧中的均值，然后除以标准差
        :type normalization: str or None

        DVS128 Gesture数据集，出自 `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf>`_，
        数据来源于DVS相机拍摄的手势。原始数据的原始下载地址参见 https://www.research.ibm.com/dvsgesture/。

        关于转换成帧数据的细节，参见 :func:`~spikingjelly.datasets.utils.integrate_events_to_frames`。

        :param root: root directory of dataset, which should contain `DvsGesture.tar.gz` and `gesture_mapping.csv`
        :type root: str
        :param train: whether use the train dataset. If `False`, use the test dataset
        :type train: bool
        :param use_frame: whether use the frames data. If `False`, use the events data
        :type use_frame: bool
        :param frames_num: the number of frames
        :type frames_num: int
        :param split_by: how to split the events, can be ``'number', 'time'``
        :type split_by: str
        :param normalization: how to normalize frames, can be ``None, 'frequency', 'max', 'norm', 'sum'``
        :type normalization: str or None

        DVS128 Gesture dataset, which is provided by `A Low Power, Fully Event-Based Gesture Recognition System <https://openaccess.thecvf.com/content_cvpr_2017/papers/Amir_A_Low_Power_CVPR_2017_paper.pdf>`, contains the gesture
        recorded by a DVS128 camera. The origin dataset can be downloaded from https://www.research.ibm.com/dvsgesture/.

        For more details about converting events to frames, see :func:`~spikingjelly.datasets.utils.integrate_events_to_frames`。
        '''
        super().__init__()
        events_npy_root = os.path.join(root, 'events_npy')
        events_npy_train_root = os.path.join(events_npy_root, 'train')
        events_npy_test_root = os.path.join(events_npy_root, 'test')
        if os.path.exists(events_npy_train_root) and os.path.exists(events_npy_test_root):
            print(f'npy format events data root {events_npy_train_root}, {events_npy_test_root} already exists')
        else:

            extracted_root = os.path.join(root, 'extracted')
            if os.path.exists(extracted_root):
                print(f'extracted root {extracted_root} already exists.')
            else:
                self.download_and_extract(root, extracted_root)
            if not os.path.exists(events_npy_root):
                os.mkdir(events_npy_root)
                print(f'mkdir {events_npy_root}')
            os.mkdir(events_npy_train_root)
            print(f'mkdir {events_npy_train_root}')
            os.mkdir(events_npy_test_root)
            print(f'mkdir {events_npy_test_root}')
            print('read events data from *.aedat and save to *.npy...')
            self.convert_aedat_dir_to_npy_dir(os.path.join(extracted_root, 'DvsGesture'), events_npy_train_root, events_npy_test_root)


        self.file_name = []  # 保存数据文件的路径
        self.use_frame = use_frame
        self.data_dir = None
        if use_frame:
            self.normalization = normalization
            if normalization == 'frequency':
                dir_suffix = normalization
            else:
                dir_suffix = None
            frames_root = os.path.join(root, f'frames_num_{frames_num}_split_by_{split_by}_normalization_{dir_suffix}')
            frames_train_root = os.path.join(frames_root, 'train')
            frames_test_root = os.path.join(frames_root, 'test')
            if os.path.exists(frames_root):
                # 如果root目录下存在frames_root目录，则认为数据集文件存在
                print(f'frames data root {frames_root} already exists.')
            else:
                os.mkdir(frames_root)
                os.mkdir(frames_train_root)
                os.mkdir(frames_test_root)
                print(f'mkdir {frames_root}, {frames_train_root}, {frames_test_root}.')
                print('creating frames data..')
                self.create_frames_dataset(events_npy_train_root, frames_train_root, frames_num, split_by, normalization)
                self.create_frames_dataset(events_npy_test_root, frames_test_root, frames_num, split_by, normalization)
            if train:
                self.data_dir = frames_train_root
            else:
                self.data_dir = frames_test_root


            self.file_name = utils.list_files(self.data_dir, '.npy', True)

        else:
            if train:
                self.data_dir = events_npy_train_root
            else:
                self.data_dir = events_npy_test_root
            self.file_name = utils.list_files(self.data_dir, '.npy', True)


    def __len__(self):
        return self.file_name.__len__()
    def __getitem__(self, index):
        if self.use_frame:
            frames, labels = self.get_frames_item(self.file_name[index])
            if self.normalization is not None and self.normalization != 'frequency':
                frames = normalize_frame(frames, self.normalization)
            return frames, labels
        else:
            return self.get_events_item(self.file_name[index])

