"""
.. codeauthor:: Yanqi Chen <chyq@pku.edu.cn>, Ismail Khalfaoui Hassani <ismail.khalfaoui-hassani@univ-tlse3.fr>

A reproduction of the paper `Technical report: supervised training of convolutional spiking neural networks with PyTorch <https://arxiv.org/pdf/1911.10124.pdf>`_\ .

This code reproduces an audio recognition task using convolutional SNN. It provides comparable performance to ANN.

..  note::

    To prevent too much dependency like `librosa <https://librosa.org/doc/latest/index.html>`_, we implement MelScale ourselves. We provide two kinds of DCT types: Slaney & HTK. Slaney style is used in the original paper and will be applied by default.

Confusion matrix of TEST set after training (50 epochs):

+------------------------+--------------------------------------------------------------------------------------------------+
| Count                  | Prediction                                                                                       |
|                        +-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|                        | "Yes" | "Stop" | "No" | "Right" | "Up" | "Left" | "On" | "Down" | "Off" | "Go" | Other | Silence |
+--------------+---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
| Ground Truth | "Yes"   | 234   | 0      | 2    | 0       | 0    | 3      | 0    | 0      | 0     | 1    | 16    | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Stop"  | 0     | 233    | 0    | 1       | 5    | 0      | 0    | 0      | 0     | 1    | 9     | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "No"    | 0     | 1      | 223  | 1       | 0    | 1      | 0    | 5      | 0     | 9    | 12    | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Right" | 0     | 0      | 0    | 234     | 0    | 0      | 0    | 0      | 0     | 0    | 24    | 1       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Up"    | 0     | 4      | 0    | 0       | 249  | 0      | 0    | 0      | 8     | 0    | 11    | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Left"  | 3     | 1      | 2    | 3       | 1    | 250    | 0    | 0      | 1     | 0    | 6     | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "On"    | 0     | 3      | 0    | 0       | 0    | 0      | 231  | 0      | 2     | 1    | 9     | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Down"  | 0     | 0      | 7    | 0       | 0    | 1      | 2    | 230    | 0     | 4    | 8     | 1       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Off"   | 0     | 0      | 2    | 1       | 4    | 2      | 6    | 0      | 237   | 1    | 9     | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | "Go"    | 0     | 2      | 5    | 0       | 0    | 2      | 0    | 1      | 5     | 220  | 16    | 0       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | Other   | 6     | 21     | 12   | 25      | 22   | 19     | 25   | 14     | 11    | 40   | 4072  | 1       |
|              +---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
|              | Silence | 0     | 0      | 0    | 0       | 0    | 0      | 0    | 0      | 0     | 0    | 0     | 260     |
+--------------+---------+-------+--------+------+---------+------+--------+------+--------+-------+------+-------+---------+
"""

import torch
from torch import Tensor, nn
from torch.optim import Adam
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms
from torchaudio.transforms import Spectrogram
from spikingjelly.clock_driven import neuron, surrogate

from spikingjelly.datasets.speechcommands import SPEECHCOMMANDS
from spikingjelly.clock_driven.functional import reset_net
from scipy.signal import savgol_filter

from sklearn.metrics import confusion_matrix

import numpy as np

import math
import time
import argparse
from typing import Optional
from tqdm import tqdm

label_dict = {'yes': 0, 'stop': 1, 'no': 2, 'right': 3, 'up': 4, 'left': 5, 'on': 6, 'down': 7, 'off': 8, 'go': 9, 'bed': 10, 'three': 10, 'one': 10, 'four': 10, 'two': 10, 'five': 10, 'cat': 10, 'dog': 10, 'eight': 10, 'bird': 10, 'happy': 10, 'sheila': 10, 'zero': 10, 'wow': 10, 'marvin': 10, 'house': 10, 'six': 10, 'seven': 10, 'tree': 10, 'nine': 10, '_silence_': 11}
label_cnt = len(set(label_dict.values()))
n_mels = 40
f_max = 4000
f_min = 20
delta_order = 0
size = 16000
try:
    import cupy
    backend = 'cupy'
except ModuleNotFoundError:
    backend = 'torch'
    print('Cupy is not intalled. Using torch backend for neurons.')

def mel_to_hz(mels, dct_type):
    if dct_type == 'htk':
        return 700.0 * (10 ** (mels / 2595.0) - 1.0)

    # Fill in the linear scale
    f_min = 0.0
    f_sp = 200.0 / 3
    freqs = f_min + f_sp * mels

    # And now the nonlinear scale
    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(mels) and mels.ndim:
        # If we have vector data, vectorize
        log_t = mels >= min_log_mel
        freqs[log_t] = min_log_hz * \
            torch.exp(logstep * (mels[log_t] - min_log_mel))
    elif mels >= min_log_mel:
        # If we have scalar data, check directly
        freqs = min_log_hz * math.exp(logstep * (mels - min_log_mel))

    return freqs


def hz_to_mel(frequencies, dct_type):
    if dct_type == 'htk':
        if torch.is_tensor(frequencies) and frequencies.ndim:
            return 2595.0 * torch.log10(1.0 + frequencies / 700.0)
        return 2595.0 * math.log10(1.0 + frequencies / 700.0)

    # Fill in the linear part
    f_min = 0.0
    f_sp = 200.0 / 3

    mels = (frequencies - f_min) / f_sp

    # Fill in the log-scale part

    min_log_hz = 1000.0  # beginning of log region (Hz)
    min_log_mel = (min_log_hz - f_min) / f_sp  # same (Mels)
    logstep = math.log(6.4) / 27.0  # step size for log region

    if torch.is_tensor(frequencies) and frequencies.ndim:
        # If we have array data, vectorize
        log_t = frequencies >= min_log_hz
        mels[log_t] = min_log_mel + \
            torch.log(frequencies[log_t] / min_log_hz) / logstep
    elif frequencies >= min_log_hz:
        # If we have scalar data, heck directly
        mels = min_log_mel + math.log(frequencies / min_log_hz) / logstep

    return mels


def create_fb_matrix(
        n_freqs: int,
        f_min: float,
        f_max: float,
        n_mels: int,
        sample_rate: int,
        dct_type: Optional[str] = 'slaney') -> Tensor:

    if dct_type != "htk" and dct_type != "slaney":
        raise ValueError("DCT type must be either 'htk' or 'slaney'")

    # freq bins
    # Equivalent filterbank construction by Librosa
    all_freqs = torch.linspace(0, sample_rate // 2, n_freqs)

    # calculate mel freq bins
    # hertz to mel(f)
    m_min = hz_to_mel(f_min, dct_type)
    m_max = hz_to_mel(f_max, dct_type)
    m_pts = torch.linspace(m_min, m_max, n_mels + 2)
    # mel to hertz(mel)
    f_pts = mel_to_hz(m_pts, dct_type)
    # calculate the difference between each mel point and each stft freq point in hertz
    f_diff = f_pts[1:] - f_pts[:-1]  # (n_mels + 1)
    # (n_freqs, n_mels + 2)
    slopes = f_pts.unsqueeze(0) - all_freqs.unsqueeze(1)
    # create overlapping triangles
    zero = torch.zeros(1)
    down_slopes = (-1.0 * slopes[:, :-2]) / f_diff[:-1]  # (n_freqs, n_mels)
    up_slopes = slopes[:, 2:] / f_diff[1:]  # (n_freqs, n_mels)
    fb = torch.max(zero, torch.min(down_slopes, up_slopes))

    if dct_type == "slaney":
        # Slaney-style mel is scaled to be approx constant energy per channel
        enorm = 2.0 / (f_pts[2:n_mels + 2] - f_pts[:n_mels])
        fb *= enorm.unsqueeze(0)

    return fb


class MelScaleDelta(nn.Module):
    __constants__ = ['n_mels', 'sample_rate', 'f_min', 'f_max']

    def __init__(self,
                 order,
                 n_mels: int = 128,
                 sample_rate: int = 16000,
                 f_min: float = 0.,
                 f_max: Optional[float] = None,
                 n_stft: Optional[int] = None,
                 dct_type: Optional[str] = 'slaney') -> None:
        super(MelScaleDelta, self).__init__()
        self.order = order
        self.n_mels = n_mels
        self.sample_rate = sample_rate
        self.f_max = f_max if f_max is not None else float(sample_rate // 2)
        self.f_min = f_min
        self.dct_type = dct_type

        assert f_min <= self.f_max, 'Require f_min: {} < f_max: {}'.format(
            f_min, self.f_max)

        fb = torch.empty(0) if n_stft is None else create_fb_matrix(
            n_stft, self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
        self.register_buffer('fb', fb)

    def forward(self, specgram: Tensor) -> Tensor:
        # pack batch
        shape = specgram.size()
        specgram = specgram.reshape(-1, shape[-2], shape[-1])

        if self.fb.numel() == 0:
            tmp_fb = create_fb_matrix(specgram.size(
                1), self.f_min, self.f_max, self.n_mels, self.sample_rate, self.dct_type)
            # Attributes cannot be reassigned outside __init__ so workaround
            self.fb.resize_(tmp_fb.size())
            self.fb.copy_(tmp_fb)

        # (channel, frequency, time).transpose(...) dot (frequency, n_mels)
        # -> (channel, time, n_mels).transpose(...)
        mel_specgram = torch.matmul(
            specgram.transpose(1, 2), self.fb).transpose(1, 2)

        # unpack batch
        mel_specgram = mel_specgram.reshape(
            shape[:-2] + mel_specgram.shape[-2:]).squeeze()

        M = torch.max(torch.abs(mel_specgram))
        if M > 0:
            feat = torch.log1p(mel_specgram/M)
        else:
            feat = mel_specgram

        feat_list = [feat.numpy().T]
        for k in range(1, self.order + 1):
            feat_list.append(savgol_filter(
                feat.numpy(), 9, deriv=k, axis=-1, mode='interp', polyorder=k).T)

        return torch.as_tensor(np.expand_dims(np.stack(feat_list), axis=0))


class Pad(object):

    def __init__(self, size):
        self.size = size

    def __call__(self, wav):
        wav_size = wav.shape[-1]
        pad_size = (self.size - wav_size) // 2
        padded_wav = torch.nn.functional.pad(
            wav, (pad_size, self.size-wav_size-pad_size), mode='constant', value=0)
        return padded_wav


class Rescale(object):

    def __call__(self, input):
        std = torch.std(input, axis=2, keepdims=True, unbiased=False) # Numpy std is calculated via the Numpy's biased estimator. https://github.com/romainzimmer/s2net/blob/82c38bf80b55d16d12d0243440e34e52d237a2df/data.py#L201 
        std.masked_fill_(std == 0, 1)

        return input / std

def collate_fn(data):

    X_batch = torch.cat([d[0] for d in data])
    std = X_batch.std(axis=(0, 2), keepdim=True, unbiased=False)
    X_batch.div_(std)

    y_batch = torch.tensor([d[1] for d in data])

    return X_batch, y_batch

#### Network ####
class LIFWrapper(nn.Module):
    def __init__(self, module, flatten=False):
        super().__init__()
        self.module = module
        self.flatten = flatten
    
    def forward(self, x_seq: torch.Tensor):
        '''
        :param x_seq: shape=[batch size, channel, T, n_mel]
        :type x_seq: torch.Tensor
        :return: y_seq, shape=[batch size, channel, T, n_mel]
        :rtype: torch.Tensor
        '''
        # Input: [batch size, channel, T, n_mel]
        y_seq = self.module(x_seq.transpose(0, 2)) # [T, channel, batch size, n_mel]
        if self.flatten:
            y_seq = y_seq.permute(2, 0, 1, 3) # [batch size, T, channel, n_mel]
            shape = y_seq.shape[:2]
            return y_seq.reshape(shape + (-1,)) # [batch size, T, channel * n_mel]
        else:
            return y_seq.transpose(0, 2) # [batch size, channel, T, n_mel]

class Net(nn.Module):
    def __init__(self):
        super().__init__()

        self.train_times = 0
        self.epochs = 0
        self.max_test_acccuracy = 0

        # batch size * delta_order+1 * T * n_mel
        self.conv = nn.Sequential(
            # 101 * 40
            nn.Conv2d(in_channels=delta_order+1, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(2, 1), bias=False),
            LIFWrapper(neuron.MultiStepLIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend)),

            # 102 * 40
            nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(6, 3), dilation=(4, 3), bias=False),
            LIFWrapper(neuron.MultiStepLIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend)),

            # 102 * 40
                nn.Conv2d(in_channels=64, out_channels=64,
                      kernel_size=(4, 3), stride=1, padding=(24, 9), dilation=(16, 9), bias=False),
            LIFWrapper(neuron.MultiStepLIFNode(tau=10.0 / 7, surrogate_function=surrogate.Sigmoid(alpha=10.), backend=backend), flatten=True),
        )
        # [batch size, T, channel * n_mel]
        self.fc = nn.Linear(64 * 40, label_cnt)

    def forward(self, x):
        x = self.fc(self.conv(x)) # [batch size, T, #Class]
        return x.mean(dim=1) # [batch size, #Class]


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-b', '--batch-size', type=int, default=64)
    parser.add_argument('-sr', '--sample-rate', type=int, default=16000)
    parser.add_argument('-lr', '--learning-rate', type=float, default=1e-2)
    parser.add_argument('-dir', '--dataset-dir', type=str)
    parser.add_argument('-e', '--epoch', type=int, default=50)
    parser.add_argument('-d', '--device', type=str, default='cuda:0')
    args = parser.parse_args()

    sr = args.sample_rate
    n_fft = int(30e-3*sr) # 48
    hop_length = int(10e-3*sr) # 16
    dataset_dir = args.dataset_dir
    batch_size = args.batch_size
    lr = args.learning_rate
    epoch = args.epoch
    device = args.device

    pad = Pad(size)
    spec = Spectrogram(n_fft=n_fft, hop_length=hop_length)
    melscale = MelScaleDelta(order=delta_order, n_mels=n_mels,
                             sample_rate=sr, f_min=f_min, f_max=f_max, dct_type='slaney')
    rescale = Rescale()

    transform = torchvision.transforms.Compose([pad,
                                                spec,
                                                melscale,
                                                rescale])

    print(label_cnt)

    train_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=2300, url="speech_commands_v0.01", split="train", transform=transform, download=True)
    train_sampler = torch.utils.data.WeightedRandomSampler(
        train_dataset.weights, len(train_dataset.weights))
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, num_workers=16,
                                  sampler=train_sampler, collate_fn=collate_fn)

    test_dataset = SPEECHCOMMANDS(
        label_dict, dataset_dir, silence_cnt=260, url="speech_commands_v0.01", split="test", transform=transform, download=True)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, num_workers=16, collate_fn=collate_fn, shuffle=False,
                                 drop_last=False)

    net = Net().to(device)

    optimizer = Adam(net.parameters(), lr=lr)
    gamma = 0.85
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma, last_epoch=-1)
    warmup_epochs = 1
    print(net)

    writer = SummaryWriter('./logs/')

    criterion = nn.CrossEntropyLoss().to(device)

    for e in range(epoch):
        net.train()
        print(f'Epoch {net.epochs}')

        time_start = time.time()
        ##### TRAIN #####
        for audios, labels in tqdm(train_dataloader):
            audios = audios.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            optimizer.zero_grad()

            out_spikes_counter_frequency = net(audios)

            loss = criterion(out_spikes_counter_frequency, labels)
            loss.backward()

            # nn.utils.clip_grad_value_(net.parameters(), 5)

            optimizer.step()

            reset_net(net)

            # Rate-based output decoding
            correct_rate = (out_spikes_counter_frequency.argmax(
                dim=1) == labels).float().mean().item()

            net.train_times += 1

        if e >= warmup_epochs:
            lr_scheduler.step()

        net.eval()

        writer.add_scalar('Train Loss', loss.item(), global_step=net.epochs)

        ##### TEST #####
        with torch.no_grad():
            test_sum = 0
            correct_sum = 0
            pred = []
            label = []
            for audios, labels in tqdm(test_dataloader):
                audios = audios.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)

                out_spikes_counter = net(audios)

                preds = out_spikes_counter.argmax(dim=1)

                correct_sum += (preds == labels).float().sum().item()

                pred.append(preds)
                label.append(labels)

                test_sum += labels.numel()
                reset_net(net)

            pred = torch.cat(pred).cpu().numpy()
            label = torch.cat(label).cpu().numpy()

            # Confusion matrix
            cmatrix = confusion_matrix(label, pred)

            print("Confusion Matrix:")
            print(cmatrix)

            # plt.clf()
            # fig = plt.figure()
            # plt.imshow(cmatrix)
            # writer.add_figure('Confusion Matrix', figure=fig,
            #                   global_step=net.epochs)

            test_accuracy = correct_sum / test_sum
            writer.add_scalar('Test Acc.', test_accuracy, global_step=net.epochs)

        net.epochs += 1
        time_end = time.time()
        print(f'Test Acc: {test_accuracy} Loss: {loss} Elapse: {time_end - time_start:.2f}s')
