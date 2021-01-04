from matplotlib import pyplot as plt
import numpy as np
from spikingjelly.clock_driven.examples.conv_fashion_mnist import Net
from spikingjelly import visualizing
import torch
import torch.nn as nn
import torchvision
def plot_log(csv_file, title, x_label, y_label, figsize=(12, 8), plot_max=False):
    log_data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=(1, 2))
    x = log_data[:, 0]
    y = log_data[:, 1]
    fig = plt.figure(figsize=figsize)
    plt.plot(x, y)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=20)
    plt.grid(linestyle='-.')

    if plot_max:
        # 画最大值
        index = y.argmax()
        plt.text(x[index], y[index], '({}, {})'.format(int(x[index]), round(y[index], 3)), fontsize=14)
        plt.scatter(x[index], y[index], marker='1', alpha=0.8, linewidths=0.1, c='r')

    # plt.show()
if __name__ == '__main__':
    plt.style.use(['science', 'muted'])
    plot_log('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/run-logs-tag-train_accuracy.csv', 'Accuracy on train batch',
             'iteration', 'accuracy')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/train.svg')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/train.pdf')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/train.png')
    plt.clf()
    plot_log('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/run-logs-tag-test_accuracy.csv', 'Accuracy on test dataset',
             'epoch', 'accuracy', plot_max=True)
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/test.svg')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/test.pdf')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/test.png')
    exit()
    dataset_dir = input('输入保存Fashion MNIST数据集的位置，例如“./”\n input root directory for saving Fashion MNIST dataset, e.g., "./": ')

    log_dir = input('输入保存tensorboard日志文件的位置，例如“./”\n input root directory for saving tensorboard logs, e.g., "./": ')
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=1,
        shuffle=True,
        drop_last=False)
    net = torch.load('./net_max_acc.pt', 'cpu')
    encoder = nn.Sequential(
        net.static_conv,
        net.conv[0]
    )
    encoder.eval()

    with torch.no_grad():
        # 每遍历一次全部数据集，就在测试集上测试一次
        for img, label in test_data_loader:
            fig = plt.figure(dpi=200)
            plt.imshow(img.squeeze().numpy(), cmap='gray')
            plt.title('Input image', fontsize=20)
            plt.xticks([])
            plt.yticks([])
            plt.show()
            out_spikes = 0
            for t in range(net.T):
                out_spikes += encoder(img).squeeze()
                if t == 0 or t == net.T - 1:
                    out_spikes_c = out_spikes.clone()
                    for i in range(out_spikes_c.shape[0]):
                        if out_spikes_c[i].max().item() > out_spikes_c[i].min().item():
                            out_spikes_c[i] = (out_spikes_c[i] - out_spikes_c[i].min()) / (out_spikes_c[i].max() - out_spikes_c[i].min())
                    visualizing.plot_2d_spiking_feature_map(out_spikes_c, 8, 16, 1, None)
                    plt.title('$\\sum_{t} S_{t}$ at $t = ' + str(t) + '$', fontsize=20)
                    plt.show()
