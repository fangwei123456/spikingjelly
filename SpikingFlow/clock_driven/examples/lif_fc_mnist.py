import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from SpikingFlow.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
import readline

class Net(nn.Module):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        # 网络结构，简单的双层全连接网络，每一层之后都是LIF神经元
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 14 * 14, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(14 * 14, 10, bias=False),
            neuron.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        )

    def forward(self, x):
        return self.fc(x)
def main():
    '''
    :return: None

    使用全连接-LIF-全连接-LIF的网络结构，进行MNIST识别。运行示例：

    .. code-block:: python

        >>> import SpikingFlow.clock_driven.examples.lif_fc_mnist as lif_fc_mnist
        >>> lif_fc_mnist.main()
        输入运行的设备，例如“cpu”或“cuda:0”
         input device, e.g., "cpu" or "cuda:0": cuda:15
        输入保存MNIST数据集的位置，例如“./”
         input root directory for saving MNIST dataset, e.g., "./": ./mnist
        输入batch_size，例如“64”
         input batch_size, e.g., "64": 128
        输入学习率，例如“1e-3”
         input learning rate, e.g., "1e-3": 1e-3
        输入仿真时长，例如“100”
         input simulating steps, e.g., "100": 50
        输入LIF神经元的时间常数tau，例如“100.0”
         input membrane time constant, tau, for LIF neurons, e.g., "100.0": 100.0
        输入训练轮数，即遍历训练集的次数，例如“100”
         input training epochs, e.g., "100": 100
        输入保存tensorboard日志文件的位置，例如“./”
         input root directory for saving tensorboard logs, e.g., "./": ./logs_lif_fc_mnist
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 0 train_accuracy 0.109375
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 1024 train_accuracy 0.5078125
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 2048 train_accuracy 0.7890625
        ...
        cuda:15 ./mnist 128 0.001 50 100.0 100 ./logs_lif_fc_mnist
        train_times 46080 train_accuracy 0.9296875


    '''
    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”\n input membrane time constant, tau, for LIF neurons, e.g., "100.0": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')


    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # 初始化网络
    net = Net(tau=tau).to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    for epoch in range(train_epoch):
        net.train()
        for img, label in train_data_loader:
            img = img.to(device)
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float())
                else:
                    out_spikes_counter += net(encoder(img).float())

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

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
            if train_times % 1024 == 0:
                print(device, dataset_dir, batch_size, learning_rate, T, tau, train_epoch, log_dir)
                print('train_times', train_times, 'train_accuracy', accuracy)
            train_times += 1
        net.eval()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device)
                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float())
                    else:
                        out_spikes_counter += net(encoder(img).float())

                correct_sum += (out_spikes_counter.max(1)[1] == label.to(device)).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)

            writer.add_scalar('test_accuracy', correct_sum / test_sum, epoch)

if __name__ == '__main__':
    main()




