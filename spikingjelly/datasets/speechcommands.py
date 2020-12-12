import os
from typing import Callable, Tuple, Dict, Optional

import torch
import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files
)
from torchvision import transforms
from torchvision.datasets.utils import verify_str_arg
import numpy as np

FOLDER_IN_ARCHIVE = "SpeechCommands"
URL = "speech_commands_v0.02"
HASH_DIVIDER = "_nohash_"
EXCEPT_FOLDER = "_background_noise_"
_CHECKSUMS = {
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz":
    "3cd23799cb2bbdec517f1cc028f8d43c",
    "https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.02.tar.gz":
    "6b74f3901214cb2c2934e98196829835",
}
VAL_RECORD = "validation_list.txt"
TEST_RECORD = "testing_list.txt"
TRAIN_RECORD = "training_list.txt"


def load_speechcommands_item(relpath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    filepath = os.path.join(path, relpath)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    def __init__(self,
                 label_dict: Dict,
                 root: str,
                 transform: Optional[Callable] = None,
                 url: Optional[str] = URL,
                 split: Optional[str] = "train",
                 folder_in_archive: Optional[str] = FOLDER_IN_ARCHIVE,
                 download: Optional[bool] = False) -> None:
        '''
        :param label_dict: 标签与类别的对应字典
        :type label_dict: Dict
        :param root: 数据集的根目录
        :type root: str
        :param transform: A function/transform that takes in a raw audio
        :type transform: Callable, optional
        :param url: 数据集版本，默认为v0.02
        :type url: str, optional
        :param split: 数据集划分，可以是 ``"train", "test", "val"``，默认为 ``"train"``
        :type split: str, optional
        :param folder_in_archive: 解压后的目录名称，默认为 ``"SpeechCommands"``
        :type folder_in_archive: str, optional
        :param download: 是否下载数据，默认为False
        :type download: bool, optional

        SpeechCommands语音数据集，出自 `Speech Commands: A Dataset for Limited-Vocabulary Speech Recognition <https://arxiv.org/abs/1804.03209>`_，根据给出的测试集与验证集列表进行了划分，包含v0.01与v0.02两个版本。

        数据集包含三大类单词的音频：

        #. 指令单词，共10个，"Yes", "No", "Up", "Down", "Left", "Right", "On", "Off", "Stop", "Go". 对于v0.02，还额外增加了5个："Forward", "Backward", "Follow", "Learn", "Visual".

        #. 0~9的数字，共10个："One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine".

        #. 非关键词，可以视为干扰词，共10个："Bed", "Bird", "Cat", "Dog", "Happy", "House", "Marvin", "Sheila", "Tree", "Wow".

        v0.01版本包含共计30类，64,727个音频片段，v0.02版本包含共计35类，105,829个音频片段。更详细的介绍参见前述论文，以及数据集的README。

        代码实现基于torchaudio并扩充了功能，同时也参考了 `原论文的实现 <https://github.com/romainzimmer/s2net/blob/b073f755e70966ef133bbcd4a8f0343354f5edcd/data.py>`_。
        '''

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        self.label_dict = label_dict
        self.transform = transform
        
        if url in [
            "speech_commands_v0.01",
            "speech_commands_v0.02",
        ]:
            base_url = "https://storage.googleapis.com/download.tensorflow.org/data/"
            ext_archive = ".tar.gz"

            url = os.path.join(base_url, url + ext_archive)

        basename = os.path.basename(url)
        archive = os.path.join(root, basename)

        basename = basename.rsplit(".", 2)[0]
        folder_in_archive = os.path.join(folder_in_archive, basename)

        self._path = os.path.join(root, folder_in_archive)

        if download:
            if not os.path.isdir(self._path):
                if not os.path.isfile(archive):
                    checksum = _CHECKSUMS.get(url, None)
                    download_url(url, root, hash_value=checksum, hash_type="md5")
                extract_archive(archive, self._path)
        elif not os.path.isdir(self._path):
            raise FileNotFoundError("Audio data not found. Please specify \"download=True\" and try again.")


        if self.split == "train":
            record = os.path.join(self._path, TRAIN_RECORD)
            if os.path.exists(record):
                with open(record, 'r') as f:
                    self._walker = list([line.rstrip('\n') for line in f])
            else:
                print("No training list, generating...")
                walker = walk_files(self._path, suffix=".wav", prefix=True)
                walker = filter(lambda w: HASH_DIVIDER in w and EXCEPT_FOLDER not in w, walker)
                walker = map(lambda w: os.path.relpath(w, self._path), walker)

                walker = set(walker)

                val_record = os.path.join(self._path, VAL_RECORD)
                with open(val_record, 'r') as f:
                    val_walker = set([line.rstrip('\n') for line in f])

                test_record = os.path.join(self._path, TEST_RECORD)
                with open(test_record, 'r') as f:
                    test_walker = set([line.rstrip('\n') for line in f])

                walker = walker - val_walker - test_walker
                self._walker = list(walker)

                with open(record, 'w') as f:
                    f.write('\n'.join(self._walker))

                print("Training list generated!")

            labels = [self.label_dict.get(os.path.split(relpath)[0]) for relpath in self._walker]
            label_weights = 1. / np.unique(labels, return_counts=True)[1]
            label_weights /= np.sum(label_weights)
            self.weights = torch.DoubleTensor([label_weights[label] for label in labels])

        else:
            if self.split == "val":
                record = os.path.join(self._path, VAL_RECORD)
            else:
                record = os.path.join(self._path, TEST_RECORD)
            with open(record, 'r') as f:
                self._walker = list([line.rstrip('\n') for line in f])

    def __getitem__(self, n: int) -> Tuple[Tensor, int]:
        fileid = self._walker[n]
        waveform, sample_rate, label, speaker_id, utterance_number = load_speechcommands_item(fileid, self._path)
        m = waveform.abs().max()
        if m > 0:
            waveform /= m
        if self.transform is not None:
            waveform = self.transform(waveform)

        label = self.label_dict.get(label)
        return waveform, label

    def __len__(self) -> int:
        return len(self._walker)