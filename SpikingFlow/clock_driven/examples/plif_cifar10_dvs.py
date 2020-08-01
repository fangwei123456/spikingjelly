import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from SpikingFlow.clock_driven import functional, layer, neuron
from torch.utils.tensorboard import SummaryWriter
import os
import readline
from SpikingFlow.datasets.cifar10_dvs import CIFAR10DVS
'''
dvs_cifar10_frames_10_split_by_number_normalization_max 0.679
dvs_cifar10_frames_20_split_by_number_normalization_max 0.692
'''

class Net(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        super().__init__()

        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0

        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 64 * 64

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 32 * 32

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 16 * 16

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            nn.MaxPool2d(2, 2),  # 8 * 8

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5),
            nn.Linear(128 * 8 * 8, 128 * 4 * 4, bias=False),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            layer.Dropout(0.5),
            nn.Linear(128 * 4 * 4, 128, bias=False),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(128, 10, bias=False),
            neuron.PLIFNode(decay=True, v_threshold=v_threshold, v_reset=v_reset)
        )

    def forward(self, x):
        return self.fc(self.conv(x))

def main():
    '''
    * :ref:`API in English <plif_cifar10_dvs.main-en>`

    .. _plif_cifar10_dvs.main-cn:

    :return: None

    `Leaky integrate-and-fire spiking neuron with learnable membrane time parameter <https://arxiv.org/abs/2007.05785>`_ 中
    分类CIFAR10-DVS数据集的训练代码。在测试集上最高正确率为 ``69.2%``。原始超参数为：

    .. code-block:: python

        batch_size = 16
        learning_rate = 1e-4
        T = 20

    数据集是对原始CIFAR10-DVS使用 ``SpikingFlow.datasets.CIFAR10DVS`` 中的函数转换而来的帧数据，超参数为：

    .. code-block:: python

        frames_num=20
        split_by='number'
        normalization='max'

    * :ref:`中文API <plif_cifar10_dvs.main-cn>`

    .. _plif_cifar10_dvs.main-en:

    :return: None

    The network for classifying CIFAR10-DVS proposed in `Leaky integrate-and-fire spiking neuron with learnable membrane time parameter <https://arxiv.org/abs/2007.05785>`_.
    The max accuracy on test dataset is ``69.2%``. The origin hyper-parameters are:

    .. code-block:: python

        batch_size = 16
        learning_rate = 1e-4
        T = 20

    The frames dataset is converted from the origin CIFAR10-DVS dataset by functions in ``SpikingFlow.datasets.CIFAR10DVS``. The
    hyper-parameters are:

    .. code-block:: python

        frames_num=20
        split_by='number'
        normalization='max'
    '''
    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存N-MNIST数据集的位置，例如“./”\n input root directory for saving N-MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')
    # 初始化数据加载器


    train_data_loader = torch.utils.data.DataLoader(
        dataset=CIFAR10DVS(dataset_dir, train=True, split_ratio=0.9),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=CIFAR10DVS(dataset_dir, train=False, split_ratio=0.9),
        batch_size=batch_size * 2,
        shuffle=False,
        drop_last=False)


    writer = SummaryWriter(log_dir)



    # 初始化网络
    if os.path.exists(log_dir + '/net.pkl'):
        net = torch.load(log_dir + '/net.pkl', map_location=device)
        print(net.train_times, net.max_test_accuracy)
    else:
        net = Net().to(device)
    print(net)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)


    for net.epoch in range(net.epoch, 999999999):
        net.train()
        for img, label in train_data_loader:
            img = img.to(device).permute(1, 0, 2, 3, 4)  # [10, N, 2, 34, 34]
            label = label.to(device)
            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(img[t])
                else:
                    out_spikes_counter += net(img[t])

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            if net.train_times % 256 == 0:
                accuracy = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
                writer.add_scalar('train_accuracy', accuracy, net.train_times)
                writer.add_scalar('train_loss', loss.item(), net.train_times)

            if net.train_times % 1024 == 0:
                print(device, dataset_dir, batch_size, learning_rate, T, log_dir, net.max_test_accuracy)
                print(sys.argv, 'train_times', net.train_times, 'train_accuracy', accuracy)
            net.train_times += 1
        net.eval()

        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device).permute(1, 0, 2, 3, 4)  # [10, N, 2, 34, 34]
                label = label.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(img[t])
                    else:
                        out_spikes_counter += net(img[t])

                correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, net.epoch)
            print('test_accuracy', test_accuracy)
            if net.max_test_accuracy < test_accuracy:
                print('save model with test_accuracy = ', test_accuracy)
                net.max_test_accuracy = test_accuracy
                torch.save(net, log_dir + '/net.pkl')




if __name__ == '__main__':
    main()




