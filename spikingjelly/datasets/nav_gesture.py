import spikingjelly
import zipfile
import os
import threading
import tqdm
import numpy as np
import torch
from torch.utils.data import Dataset
from torchvision.datasets import utils
import shutil
import loris

# url md5
resource = {
    'walk': ['https://www.neuromorphic-vision.com/public/downloads/navgesture/navgesture-walk.zip',
             '5d305266f13005401959e819abe206f0']
}
labels = {
    'do': 0,
    'up': 1,
    'le': 2,
    'ri': 3,
    'se': 4,
    'ho': 5
}


class NavGesture(spikingjelly.datasets.EventsFramesDatasetBase):
    @staticmethod
    def get_wh():
        return 304, 240

    @staticmethod
    def read_bin(file_name: str):
        '''
        :param file_name: NavGesture原始bin格式数据的文件名
        :return: 一个字典，键是{'t', 'x', 'y', 'p'}，值是np数组

        原始的NavGesture提供的是bin格式数据，不能直接读取。本函数提供了一个读取的接口。
        原始数据以二进制存储：

        Events are encoded in binary format on 64 bits (8 bytes):
        32 bits for timestamp
        9 bits for x address
        8 bits for y address
        2 bits for polarity
        13 bits padding
        '''
        txyp = loris.read_file(file_name)['events']
        # txyp.p是bool类型，转换成int
        return {'t': txyp.t, 'x': txyp.x, 'y': txyp.y, 'p': txyp.p.astype(int)}

    @staticmethod
    def get_label(file_name):
        # 6 gestures: left, right, up, down, home, select.
        # 10 subjects, holding the phone in one hand (selfie mode) while walking indoor and outdoor. It contains 339 clips.
        # No train/test split, scores should be reported using average score with one-versus-all cross-validation.
        # Files are named userID_classID_userclipID.dat and allow to identify the user and gesture class. For example, "user09_do_11.dat" is a "Down Swipe" gesture from user09. classID can be:
        # do: down swipe ; up: up swipe ; le: left swipe ; ri: right swipe ; se: select ; ho: home
        base_name = os.path.basename(file_name)
        return labels[base_name.split('_')[1]]

    @staticmethod
    def get_events_item(file_name):
        events = NavGesture.read_bin(file_name)
        return events, NavGesture.get_label(file_name)
    
    @staticmethod
    def get_frames_item(file_name):
        frames = np.load(file_name)
        return frames, NavGesture.get_label(file_name)
    @staticmethod
    def download_and_extract(download_root: str, extract_root: str):
        dataset_name = 'walk'
        file_name = os.path.basename(resource[dataset_name][0])
        temp_extract_root = os.path.join(extract_root, 'temp_extract')
        utils.download_and_extract_archive(url=resource[dataset_name][0], download_root=download_root,
                                           extract_root=temp_extract_root,
                                           filename=file_name, md5=resource[dataset_name][1])
        # 解压后仍然是zip 要继续解压
        for zip_file in utils.list_files(root=temp_extract_root, suffix='.zip', prefix=True):
            print(f'extract {zip_file} to {extract_root}')
            utils.extract_archive(zip_file, extract_root)
        shutil.rmtree(temp_extract_root)

    @staticmethod
    def create_frames_dataset(events_data_dir, frames_data_dir, frames_num=10, split_by='time', normalization=None):
        width, height = NavGesture.get_wh()
        thread_list = []
        for source_dir in utils.list_dir(events_data_dir):
            abs_source_dir = os.path.join(events_data_dir, source_dir)
            abs_target_dir = os.path.join(frames_data_dir, source_dir)
            if not os.path.exists(abs_target_dir):
                os.mkdir(abs_target_dir)
                print(f'mkdir {abs_target_dir}')
            print(f'thread {thread_list.__len__()} convert events data in {abs_source_dir} to {abs_target_dir}')
            thread_list.append(spikingjelly.datasets.FunctionThread(spikingjelly.datasets.convert_events_dir_to_frames_dir,
                abs_source_dir, abs_target_dir, '.dat', NavGesture.read_bin, height, width, frames_num, split_by, normalization))
            thread_list[-1].start()
        for i in range(thread_list.__len__()):
            thread_list[i].join()
            print('thread', i, 'finished')

    def __init__(self, root: str, use_frame=True, frames_num=10, split_by='number', normalization='max'):
        events_root = os.path.join(root, 'events')
        if os.path.exists(events_root) and os.listdir(events_root).__len__() == 9:
            # 如果root目录下存在events_root目录，且events_root下有10个子文件夹，则认为数据集文件存在
            print(f'events data root {events_root} already exists.')
        else:
           self.download_and_extract(root, events_root)
        self.file_name = []  # 保存数据文件的路径
        self.use_frame = use_frame
        self.data_dir = None
        if use_frame:
            frames_root = os.path.join(root, f'frames_num_{frames_num}_split_by_{split_by}_normalization_{normalization}')
            if os.path.exists(frames_root) and os.listdir(frames_root).__len__() == 9:
                # 如果root目录下存在frames_root目录，且frames_root下有10个子文件夹，则认为数据集文件存在
                print(f'frames data root {frames_root} already exists.')
            else:
                os.mkdir(frames_root)
                NavGesture.create_frames_dataset(events_root, frames_root, frames_num, split_by, normalization)
            for sub_dir in utils.list_dir(frames_root, True):
                    self.file_name.extend(utils.list_files(sub_dir, '.npy', True))
            self.data_dir = frames_root
            self.get_item_fun = NavGesture.get_frames_item

        else:
            for sub_dir in utils.list_dir(events_root, True):
                    self.file_name.extend(utils.list_files(sub_dir, '.dat', True))
            self.data_dir = events_root
            self.get_item_fun = NavGesture.get_events_item

    def __len__(self):
        return self.file_name.__len__()
    
    def __getitem__(self, index):
        return self.get_item_fun(self.file_name[index])


