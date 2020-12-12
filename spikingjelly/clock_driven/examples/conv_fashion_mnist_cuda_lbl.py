import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import functional, surrogate, layer
from spikingjelly.cext import neuron as cext_neuron
from torch.utils.tensorboard import SummaryWriter
import sys
import time
if sys.platform != 'win32':
    import readline
import numpy as np
torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

class Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            cext_neuron.MultiStepIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
            layer.SeqToANNContainer(
                    nn.MaxPool2d(2, 2),  # 14 * 14
                    nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(128),
            ),
            cext_neuron.MultiStepIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
        )
        self.fc = nn.Sequential(
            layer.SeqToANNContainer(
                    nn.MaxPool2d(2, 2),  # 7 * 7
                    nn.Flatten(),
            ),
            layer.MultiStepDropout(0.5),
            layer.SeqToANNContainer(nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False)),
            cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
            layer.MultiStepDropout(0.5),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
            layer.SeqToANNContainer(nn.Linear(128, 10, bias=False)),
            cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0)
        )


    def forward(self, x):
        x_seq = self.static_conv(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
        # [N, C, H, W] -> [1, N, C, H, W] -> [T, N, C, H, W]

        out_spikes_counter = self.fc(self.conv(x_seq)).sum(0)
        return out_spikes_counter / self.T
def main():
    '''
    * :ref:`API in English <conv_fashion_mnist_cuda_lbl.main-en>`

    .. _conv_fashion_mnist_cuda_lbl.main-cn:

    :return: None

    :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist` 的逐层传播版本。

    训练100个epoch，训练batch和测试集上的正确率如下：

    .. image:: ./_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.*
        :width: 100%

    * :ref:`中文API <conv_fashion_mnist_cuda_lbl.main-cn>`

    .. _conv_fashion_mnist_cuda_lbl.main-en:

    The layer-by-layer version of :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist`.

    After 100 epochs, the accuracy on train batch and test dataset is as followed:

    .. image:: ./_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.*
        :width: 100%
    '''
    device = input('输入运行的GPU，例如“cuda:0”\n input GPU index, e.g., "cuda:0": ')
    if device == 'cpu':
        print("conv_fashion_mnist_cuda_lbl only supports GPU.")
        exit()
    dataset_dir = input('输入保存Fashion MNIST数据集的位置，例如“./”\n input root directory for saving Fashion MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“8”\n input simulating steps, e.g., "8": '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“2.0”\n input membrane time constant, tau, for LIF neurons, e.g., "2.0": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')
    # device = 'cuda:0'
    # dataset_dir = './'
    # batch_size = 128
    # learning_rate = 1e-3
    # T = 8
    # tau = 2.0
    # train_epoch = 100
    # log_dir = './logs2'


    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # 初始化网络
    net = Net(tau=tau, T=T).to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    train_times = 0
    max_test_accuracy = 0
    for epoch in range(train_epoch):
        net.train()
        t_start = time.perf_counter()
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            out_spikes_counter_frequency = net(img)

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的MSE
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            accuracy = (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().mean().item()
            if train_times % 256 == 0:
                writer.add_scalar('train_accuracy', accuracy, train_times)
            train_times += 1
        t_train = time.perf_counter() - t_start
        net.eval()
        t_start = time.perf_counter()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                out_spikes_counter_frequency = net(img)

                correct_sum += (out_spikes_counter_frequency.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = correct_sum / test_sum
            t_test = time.perf_counter() - t_start
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            if max_test_accuracy < test_accuracy:
                max_test_accuracy = test_accuracy
                print('saving net...')
                torch.save(net, log_dir + '/net_max_acc.pt')
                print('saved')

        print(
            'epoch={}, t_train={}, t_test={}, device={}, dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                epoch, t_train, t_test, device, dataset_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy,
                train_times))


if __name__ == '__main__':
    main()




