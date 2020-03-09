import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import sys
sys.path.append('.')
import SpikingFlow.softbp as softbp
import SpikingFlow.encoding as encoding
from torch.utils.tensorboard import SummaryWriter
import readline

class Net(softbp.ModelPipeline):
    def __init__(self, gpu_list, tau=100.0, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        # 网络结构，卷积-卷积-最大池化堆叠，最后接一个全连接层

        self.append(
            nn.Sequential(
                nn.Conv2d(3, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256)
            ),
            gpu_list[0]
        )

        self.append(
            nn.Sequential(
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
            ),
            gpu_list[1]
        )

        self.append(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256)
            ),
            gpu_list[2]
        )

        self.append(
            nn.Sequential(
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)  # 16 * 16
            ),
            gpu_list[3]
        )

        self.append(
            nn.Sequential(
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 8 * 8
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.BatchNorm2d(256),
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),
                nn.Conv2d(256, 256, kernel_size=3, padding=1),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(256),
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset),  # 4 * 4
                nn.Flatten(),
                nn.Linear(256 * 4 * 4, 10, bias=False),
                softbp.LIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset)
            ),
            gpu_list[4]
        )

    def reset_(self):
        for item in self.modules():
            if hasattr(item, 'reset'):
                item.reset()
def main():
    gpu_list = input('输入使用的5个gpu，例如“0,1,2,0,3”  ').split(',')
    dataset_dir = input('输入保存CIFAR10数据集的位置，例如“./”  ')
    batch_size = int(input('输入batch_size，例如“64”  '))
    split_sizes = int(input('输入split_sizes，例如“16”  '))
    learning_rate = float(input('输入学习率，例如“1e-3”  '))
    T = int(input('输入仿真时长，例如“50”  '))
    tau = float(input('输入LIF神经元的时间常数tau，例如“100.0”  '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”  '))
    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.CIFAR10(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)

    # 初始化网络
    net = Net(gpu_list=gpu_list, tau=tau)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    for _ in range(train_epoch):

        for img, label in train_data_loader:

            label = label.to(net.gpu_list[-1])
            optimizer.zero_grad()
            # 将图像编码为脉冲数据

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            for t in range(T):
                if t == 0:
                    out_spikes_counter = net(encoder(img).float(), split_sizes)
                else:
                    out_spikes_counter += net(encoder(img).float(), split_sizes)

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / T

            # 损失函数为输出层神经元的脉冲发放频率，与真实类别的交叉熵
            # 这样的损失函数会使，当类别i输入时，输出层中第i个神经元的脉冲发放频率趋近1，而其他神经元的脉冲发放频率趋近0
            loss = F.cross_entropy(out_spikes_counter_frequency, label)
            loss.backward()
            optimizer.step()
            # 优化一次参数后，需要重置网络的状态，因为SNN的神经元是有“记忆”的
            net.reset_()

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            correct_rate = (out_spikes_counter_frequency.max(1)[1] == label).float().mean().item()
            writer.add_scalar('train_correct_rate', correct_rate, train_times)
            if train_times % 1024 == 0:
                print('train_times', train_times, 'train_correct_rate', correct_rate)
            train_times += 1

        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                label = label.to(net.gpu_list[-1])

                for t in range(T):
                    if t == 0:
                        out_spikes_counter = net(encoder(img).float(), split_sizes)
                    else:
                        out_spikes_counter += net(encoder(img).float(), split_sizes)

                correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
                test_sum += label.numel()
                net.reset_()

            writer.add_scalar('test_correct_rate', correct_sum / test_sum, train_times)

if __name__ == '__main__':
    main()




