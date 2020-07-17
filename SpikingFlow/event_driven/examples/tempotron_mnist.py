import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from torch.utils.tensorboard import SummaryWriter
import SpikingFlow.event_driven.encoding as encoding
import SpikingFlow.event_driven.neuron as neuron
import readline
import math

class Net(nn.Module):
    def __init__(self, m, T):
        # m是高斯调谐曲线编码器编码一个像素点所使用的神经元数量
        super().__init__()
        self.tempotron = neuron.Tempotron(784*m, 10, T)
    def forward(self, x: torch.Tensor):
        # 返回的是输出层10个Tempotron在仿真时长内的电压峰值
        return self.tempotron(x, 'v_max')

def main():
    '''
    :return: None

    使用高斯调谐曲线编码器编码图像为脉冲，单层Tempotron进行MNIST识别。运行示例：
    
    .. code-block:: python

        >>> import SpikingFlow.event_driven.examples.tempotron_mnist as tempotron_mnist
        >>> tempotron_mnist.main()
        输入运行的设备，例如“cpu”或“cuda:0”
         input device, e.g., "cpu" or "cuda:0": cuda:15
        输入保存MNIST数据集的位置，例如“./”
         input root directory for saving MNIST dataset, e.g., "./": ./mnist
        输入batch_size，例如“64”
         input batch_size, e.g., "64": 64
        输入学习率，例如“1e-3”
         input learning rate, e.g., "1e-3": 1e-3
        输入仿真时长，例如“100”
         input simulating steps, e.g., "100": 100
        输入训练轮数，即遍历训练集的次数，例如“100”
         input training epochs, e.g., "100": 10
        输入使用高斯调谐曲线编码每个像素点使用的神经元数量，例如“16”
         input neuron number for encoding a piexl in GaussianTuning encoder, e.g., "16": 16
        输入保存tensorboard日志文件的位置，例如“./”
         input root directory for saving tensorboard logs, e.g., "./": ./logs_tempotron_mnist
        cuda:15 ./mnist 64 0.001 100 100 16 ./logs_tempotron_mnist
        train_acc 0.09375 0
        cuda:15 ./mnist 64 0.001 100 100 16 ./logs_tempotron_mnist
        train_acc 0.78125 512
        ...



    '''
    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    m = int(input('输入使用高斯调谐曲线编码每个像素点使用的神经元数量，例如“16”\n input neuron number for encoding a piexl in GaussianTuning encoder, e.g., "16": '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')

    # 每个像素点用m个神经元来编码
    encoder = encoding.GaussianTuning(n=1, m=m, x_min=torch.zeros(size=[1]).to(device), x_max=torch.ones(size=[1]).to(device))

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
    net = Net(m, T).to(device)
    # 使用Adam优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)


    train_times = 0
    for epoch in range(train_epoch):
        net.train()
        for img, label in train_data_loader:
            img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]

            in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
            in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]
            v_max = net(in_spikes)
            train_acc = (v_max.argmax(dim=1) == label.to(device)).float().mean().item()
            if train_times % 256 == 0:
                writer.add_scalar('train_acc', train_acc, train_times)
            if train_times % 512 == 0:
                print(device, dataset_dir, batch_size, learning_rate, T, train_epoch, m, log_dir)
                print('train_acc', train_acc, train_times)
            loss = neuron.Tempotron.mse_loss(v_max, net.tempotron.v_threshold, label.to(device), 10)
            loss.backward()
            optimizer.step()
            train_times += 1
        net.eval()
        with torch.no_grad():
            correct_num = 0
            img_num = 0
            for img, label in train_data_loader:
                img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]

                in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
                in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]
                v_max = net(in_spikes)
                correct_num += (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
                img_num += img.shape[0]
            test_acc = correct_num / img_num
            writer.add_scalar('test_acc', test_acc, epoch)
            print('test_acc', test_acc, train_times, log_dir)






if __name__ == '__main__':
    main()




