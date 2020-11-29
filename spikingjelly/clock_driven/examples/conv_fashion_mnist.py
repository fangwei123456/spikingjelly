import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import readline

class Net(nn.Module):
    def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.7),
            nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.7),
            nn.Linear(128 * 3 * 3, 128, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128, 10, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T
def main():
    '''
    * :ref:`API in English <conv_fashion_mnist.main-en>`

    .. _conv_fashion_mnist.main-cn:

    :return: None

    使用卷积-全连接的网络结构，进行Fashion MNIST识别。这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。会将训练过
    程中测试集正确率最高的网络保存在 ``tensorboard`` 日志文件的同级目录下。这个目录的位置，是在运行 ``main()``
    函数时由用户输入的。

    训练100个epoch，训练batch和测试集上的正确率如下：

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%

    * :ref:`中文API <conv_fashion_mnist.main-cn>`

    .. _conv_fashion_mnist.main-en:

    The network with Conv-FC structure for classifying Fashion MNIST. This function initials the network, starts training
    and shows accuracy on test dataset. The net with the max accuracy on test dataset will be saved in
    the root directory for saving ``tensorboard`` logs, which is inputted by user when running the ``main()``  function.

    After 100 epochs, the accuracy on train batch and test dataset is as followed:

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%
    '''
    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存Fashion MNIST数据集的位置，例如“./”\n input root directory for saving Fashion MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“8”\n input simulating steps, e.g., "8": '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“2.0”\n input membrane time constant, tau, for LIF neurons, e.g., "2.0": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')


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
        net.eval()
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
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            if max_test_accuracy < test_accuracy:
                max_test_accuracy = test_accuracy
                print('saving net...')
                torch.save(net, log_dir + '/net_max_acc.pt')
                print('saved')

        print(
            'device={}, dataset_dir={}, batch_size={}, learning_rate={}, T={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                device, dataset_dir, batch_size, learning_rate, T, log_dir, max_test_accuracy, train_times
            ))

if __name__ == '__main__':
    main()




