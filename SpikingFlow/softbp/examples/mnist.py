import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import SpikingFlow.softbp as softbp
import SpikingFlow.encoding as encoding
import sys
from torch.utils.tensorboard import SummaryWriter
import readline

class Net(nn.Module):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(28 * 28, 14 * 14, bias=False),
            softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
            nn.Linear(14 * 14, 10, bias=False),
            softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
        )

    def forward(self, x):
        return self.fc(x)

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()

if __name__ == '__main__':
    device = input('输入运行的设备，例如“CPU”或“cuda:0”  ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”  ')
    batch_size = int(input('输入batch_size，例如“64”  '))
    learning_rate = float(input('输入学习率，例如“1e-3”  '))
    T = int(input('输入仿真时长，例如“50”  '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”  '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

    writer = SummaryWriter(log_dir)
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

    net = Net(tau=tau).to(device)

    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)

    encoder = encoding.PoissonEncoder()

    train_times = 0
    for _ in range(train_epoch):

        for img, label in train_data_loader:
            optimizer.zero_grad()

            in_spikes = encoder(img.to(device)).float()
            for t in range(T):
                if t == 0:
                    out_spikes = net(in_spikes)
                else:
                    out_spikes += net(in_spikes)

            out_spikes_frequency = out_spikes / T
            loss = F.cross_entropy(out_spikes_frequency, label.to(device))
            loss.backward()
            optimizer.step()
            net.reset_()

            correct_rate = (out_spikes_frequency.max(1)[1] == label.to(device)).float().mean().item()
            writer.add_scalar('train_correct_rate', correct_rate, train_times)
            if train_epoch % 32 == 0:
                print('train_times', train_times, 'train_correct_rate', correct_rate)
            train_times += 1

        with torch.no_grad():
            test_num = 0
            for img, label in test_data_loader:

                in_spikes = encoder(img.to(device)).float()
                for t in range(T):
                    if t == 0:
                        out_spikes = net(in_spikes)
                    else:
                        out_spikes += net(in_spikes)

                out_spikes_frequency = out_spikes / T
                correct_rate = (out_spikes_frequency.max(1)[1] == label.to(device)).float().sum().item()
                test_num += label.numel()
                net.reset_()
                writer.add_scalar('test_correct_rate', correct_rate, train_times)




