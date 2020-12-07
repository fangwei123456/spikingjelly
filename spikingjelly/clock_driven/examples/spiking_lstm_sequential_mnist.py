import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from spikingjelly.clock_driven import rnn
from torch.utils.tensorboard import SummaryWriter
import sys
if sys.platform != 'win32':
    import readline
import torchvision
import tqdm
class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.lstm = rnn.SpikingLSTM(28, 1024, 1)
        self.fc = nn.Linear(1024, 10)

    def forward(self, x):
        x, _ = self.lstm(x)
        return self.fc(x[-1])

def main():

    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
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
    net = Net().to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    train_times = 0
    max_test_accuracy = 0
    for epoch in range(train_epoch):
        net.train()
        for img, label in tqdm.tqdm(train_data_loader):
            img = img.to(device)  # [N, 1, 28, 28]
            label = label.to(device)
            label_one_hot = F.one_hot(label, 10).float()

            img.squeeze_()  # [N, 28, 28]
            img = img.permute(1, 0, 2)  # [28, N, 28]

            optimizer.zero_grad()

            out_spikes_counter_frequency = net(img)

            loss = F.mse_loss(out_spikes_counter_frequency, label_one_hot)
            loss.backward()
            optimizer.step()


            accuracy = (out_spikes_counter_frequency.max(1)[1] == label).float().mean().item()
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
                label = label.to(device)

                img.squeeze_()  # [N, 28, 28]
                img = img.permute(1, 0, 2)  # [28, N, 28]
                out_spikes_counter_frequency = net(img)

                correct_sum += (out_spikes_counter_frequency.argmax(dim=1) == label).float().sum().item()
                test_sum += label.numel()
            test_accuracy = correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            # if max_test_accuracy < test_accuracy:
            #     max_test_accuracy = test_accuracy
            #     print('saving net...')
            #     torch.save(net, log_dir + '/net_max_acc.pt')
            #     print('saved')

        print(
            'device={}, dataset_dir={}, batch_size={}, learning_rate={}, log_dir={}, max_test_accuracy={}, train_times={}'.format(
                device, dataset_dir, batch_size, learning_rate, log_dir, max_test_accuracy, train_times
            ))


if __name__ == '__main__':
    main()