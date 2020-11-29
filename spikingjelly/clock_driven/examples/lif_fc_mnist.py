import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import numpy as np
from spikingjelly.clock_driven import neuron, encoding, functional
from torch.utils.tensorboard import SummaryWriter
import readline
from tqdm import tqdm


def main():
    '''
    * :ref:`API in English <lif_fc_mnist.main-en>`

    .. _lif_fc_mnist.main-cn:

    :return: None

    使用全连接-LIF-全连接-LIF的网络结构，进行MNIST识别。这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。

    * :ref:`中文API <lif_fc_mnist.main-cn>`

    .. _lif_fc_mnist.main-en:

    The network with FC-LIF-FC-LIF structure for classifying MNIST. This function initials the network, starts training
    and shows accuracy on test dataset.
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
    train_dataset = torchvision.datasets.MNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True
    )
    test_dataset = torchvision.datasets.MNIST(root=dataset_dir,train=False,transform=torchvision.transforms.ToTensor(),download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_dataset,
        batch_size=batch_size,
        shuffle=False,
        drop_last=False)

    # 定义并初始化网络
    net = nn.Sequential(
        nn.Flatten(),
        nn.Linear(28 * 28, 10, bias=False),
        neuron.LIFNode(tau=tau)
    )
    net = net.to(device)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    # 使用泊松编码器
    encoder = encoding.PoissonEncoder()
    train_times = 0
    max_test_accuracy = 0

    test_accs = []
    train_accs = []

    for epoch in range(train_epoch):
        net.train()
        for img, label in tqdm(train_data_loader):
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
            
            writer.add_scalar('train_accuracy', accuracy, train_times)
            train_accs.append(accuracy)

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
            test_accuracy = correct_sum / test_sum
            writer.add_scalar('test_accuracy', test_accuracy, epoch)
            test_accs.append(test_accuracy)
            max_test_accuracy = max(max_test_accuracy, test_accuracy)
        print(f'Epoch {epoch}: device={device}, dataset_dir={dataset_dir}, batch_size={batch_size}, learning_rate={learning_rate}, T={T}, log_dir={log_dir}, max_test_accuracy={max_test_accuracy}, train_times={train_times}')
    
    # 保存绘图用数据
    net.eval()
    functional.set_monitor(net, True)
    with torch.no_grad():
        img, label = test_dataset[0]        
        img = img.to(device)
        for t in range(T):
            if t == 0:
                out_spikes_counter = net(encoder(img).float())
            else:
                out_spikes_counter += net(encoder(img).float())
        out_spikes_counter_frequency = (out_spikes_counter / T).cpu().numpy()
        print(f'Firing rate: {out_spikes_counter_frequency}')
        output_layer = net[-1] # 输出层
        v_t_array = np.asarray(output_layer.monitor['v']).squeeze().T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        np.save("v_t_array.npy",v_t_array)
        s_t_array = np.asarray(output_layer.monitor['s']).squeeze().T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        np.save("s_t_array.npy",s_t_array)

    train_accs = np.array(train_accs)
    np.save('train_accs.npy', train_accs)
    test_accs = np.array(test_accs)
    np.save('test_accs.npy', test_accs)


if __name__ == '__main__':
    main()




