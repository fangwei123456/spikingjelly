import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data as data
import torchvision
from torch.utils.tensorboard import SummaryWriter
import spikingjelly.event_driven.encoding as encoding
import spikingjelly.event_driven.neuron as neuron
from tqdm import tqdm


parser = argparse.ArgumentParser(description='spikingjelly Tempotron MNIST Training')

parser.add_argument('--device', default='cuda:0', help='运行的设备，例如“cpu”或“cuda:0”\n Device, e.g., "cpu" or "cuda:0"')

parser.add_argument('--dataset-dir', default='./', help='保存MNIST数据集的位置，例如“./”\n Root directory for saving MNIST dataset, e.g., "./"')
parser.add_argument('--log-dir', default='./', help='保存tensorboard日志文件的位置，例如“./”\n Root directory for saving tensorboard logs, e.g., "./"')
parser.add_argument('--model-output-dir', default='./', help='模型保存路径，例如“./”\n Model directory for saving, e.g., "./"')

parser.add_argument('-b', '--batch-size', default=64, type=int, help='Batch 大小，例如“64”\n Batch size, e.g., "64"')
parser.add_argument('-T', '--timesteps', default=100, type=int, dest='T', help='仿真时长，例如“100”\n Simulating timesteps, e.g., "100"')
parser.add_argument('--lr', '--learning-rate', default=1e-3, type=float, metavar='LR', help='学习率，例如“1e-3”\n Learning rate, e.g., "1e-3": ', dest='lr')
# parser.add_argument('--tau', default=2.0, type=float, help='LIF神经元的时间常数tau，例如“100.0”\n Membrane time constant, tau, for LIF neurons, e.g., "100.0"')
parser.add_argument('-N', '--epoch', default=100, type=int, help='训练epoch，例如“100”\n Training epoch, e.g., "100"')
parser.add_argument('-m', default=16, type=int, help='使用高斯调谐曲线编码每个像素点使用的神经元数量，例如“16”\n input neuron number for encoding a piexl in GaussianTuning encoder, e.g., "16"')


class Net(nn.Module):
    def __init__(self, m, T):
        # m是高斯调谐曲线编码器编码一个像素点所使用的神经元数量
        super().__init__()
        self.tempotron = neuron.Tempotron(28*28*m, 10, T)     # mnist 28*28=784
    
    def forward(self, x: torch.Tensor):
        # 返回的是输出层10个Tempotron在仿真时长内的电压峰值
        return self.tempotron(x, 'v_max')


def main():
    '''
    :return: None

    * :ref:`API in English <tempotron_mnist.main-en>`

    .. _tempotron_mnist.main-cn:

    使用高斯调谐曲线编码器编码图像为脉冲，单层Tempotron进行MNIST识别。\n
    这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <tempotron_mnist.main-cn>`

    .. _tempotron_mnist.main-en:

    Use Gaussian tuned activation function encoder to encode the images to spikes.\n
    The network with single Tempotron structure for classifying MNIST.\n
    This function initials the network, starts training and shows accuracy on test dataset.
    '''

    args = parser.parse_args()
    print("########## Configurations ##########")
    print('\n'.join(f'{k}={v}' for k, v in vars(args).items()))
    print("####################################")

    device = args.device

    dataset_dir = args.dataset_dir
    log_dir = args.log_dir
    model_output_dir = args.model_output_dir

    batch_size = args.batch_size
    T = args.T
    learning_rate = args.lr
    train_epoch = args.epoch
    m = args.m

    # 每个像素点用m个神经元来编码
    encoder = encoding.GaussianTuning(n=1, m=m, x_min=torch.zeros(size=[1]).to(device), x_max=torch.ones(size=[1]).to(device))

    writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    train_data_loader = data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = data.DataLoader(
        dataset=torchvision.datasets.MNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    # 初始化网络
    net = Net(m, T).to(device)
    # 使用Adam优化器
    optimizer = torch.optim.SGD(net.parameters(), lr=learning_rate)

    train_times = 0
    max_test_accuracy = 0

    for epoch in range(train_epoch):
        print("Epoch {}:".format(epoch))
        print("Training...")
        train_correct_sum = 0
        train_sum = 0
        net.train()
        for img, label in tqdm(train_data_loader):
            img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]
            in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
            in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]

            optimizer.zero_grad()

            v_max = net(in_spikes)
            loss = neuron.Tempotron.mse_loss(v_max, net.tempotron.v_threshold, label.to(device), 10)
            loss.backward()
            optimizer.step()

            train_correct_sum += (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
            train_sum += label.numel()

            train_batch_acc = (v_max.argmax(dim=1) == label.to(device)).float().mean().item()
            writer.add_scalar('train_batch_acc', train_batch_acc, train_times)

            train_times += 1
        train_accuracy = train_correct_sum / train_sum

        print("Testing...")
        net.eval()
        with torch.no_grad():
            correct_num = 0
            img_num = 0
            for img, label in tqdm(test_data_loader):
                img = img.view(img.shape[0], -1).unsqueeze(1)  # [batch_size, 1, 784]

                in_spikes = encoder.encode(img.to(device), T)  # [batch_size, 1, 784, m]
                in_spikes = in_spikes.view(in_spikes.shape[0], -1)  # [batch_size, 784*m]
                v_max = net(in_spikes)
                correct_num += (v_max.argmax(dim=1) == label.to(device)).float().sum().item()
                img_num += img.shape[0]
            test_accuracy = correct_num / img_num
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print("Epoch {}: train_acc = {}, test_acc={}, max_test_acc={}, train_times={}".format(epoch, train_accuracy, test_accuracy, max_test_accuracy, train_times))
        print()
    
    # 保存模型
    torch.save(net, model_output_dir + "/tempotron_snn_mnist.ckpt")
    # 读取模型
    # net = torch.load(model_output_dir + "/tempotron_snn_mnist.ckpt")


if __name__ == '__main__':
    main()
