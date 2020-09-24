import os
from typing import Tuple

import torchaudio
from torch.utils.data import Dataset
from torch import Tensor
from torchaudio.datasets.utils import (
    download_url,
    extract_archive,
    walk_files
)
from torchvision.datasets.utils import (
    verify_str_arg,
    list_dir
)

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


def load_speechcommands_item(filepath: str, path: str) -> Tuple[Tensor, int, str, str, int]:
    relpath = os.path.relpath(filepath, path)
    label, filename = os.path.split(relpath)
    speaker, _ = os.path.splitext(filename)

    speaker_id, utterance_number = speaker.split(HASH_DIVIDER)
    utterance_number = int(utterance_number)

    # Load audio
    waveform, sample_rate = torchaudio.load(filepath)
    return waveform, sample_rate, label, speaker_id, utterance_number


class SPEECHCOMMANDS(Dataset):
    """
    Create a Dataset for Speech Commands. Each item is a tuple of the form:
    waveform, sample_rate, label, speaker_id, utterance_number
    """

    def __init__(self,
                 root: str,
                 url: str = URL,
                 split: str = "train",
                 folder_in_archive: str = FOLDER_IN_ARCHIVE,
                 download: bool = False) -> None:

        self.split = verify_str_arg(split, "split", ("train", "val", "test"))
        
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

                val_record = os.path.join(self._path, VAL_RECORD)
                with open(val_record, 'r') as f:
                    val_walker= list([line.rstrip('\n') for line in f])

                test_record = os.path.join(self._path, TEST_RECORD)
                with open(test_record, 'r') as f:
                    test_walker = list([line.rstrip('\n') for line in f])

                walker = filter(lambda w: w not in val_walker and w not in test_walker, walker)
                self._walker = list(walker)

                with open(record, 'w') as f:
                    f.write('\n'.join(self._walker))

                print("Training list generated!")

        else:
            if self.split == "val":
                record = os.path.join(self._path, VAL_RECORD)
            else:
                record = os.path.join(self._path, TEST_RECORD)
            with open(record, 'r') as f:
                self._walker = list([line.rstrip('\n') for line in f])

    def __getitem__(self, n: int) -> Tuple[Tensor, int, str, str, int]:
        fileid = self._walker[n]
        return load_speechcommands_item(fileid, self._path)

    def __len__(self) -> int:
        return len(self._walker)