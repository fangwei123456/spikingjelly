from matplotlib import pyplot as plt
import numpy as np
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
    plt.style.use(['science'])
    plot_log('./run-T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp_cext_dvsg_logs-tag-test_acc.csv', 'Test accuracy', 'iteration', 'accuracy')
    plt.savefig('./test_acc.svg')
    plt.savefig('./test_acc.pdf')
    plt.savefig('./test_acc.png')

    plt.clf()
    plot_log('./run-T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp_cext_dvsg_logs-tag-test_loss.csv', 'Test loss', 'iteration', 'accuracy')
    plt.savefig('./test_loss.svg')
    plt.savefig('./test_loss.pdf')
    plt.savefig('./test_loss.png')

    plt.clf()
    plot_log('./run-T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp_cext_dvsg_logs-tag-train_acc.csv', 'Train accuracy', 'iteration', 'accuracy')
    plt.savefig('./train_acc.svg')
    plt.savefig('./train_acc.pdf')
    plt.savefig('./train_acc.png')

    plt.clf()
    plot_log('./run-T_16_b_16_c_128_Adam_lr_0.001_CosALR_64_amp_cext_dvsg_logs-tag-train_loss.csv', 'Train loss', 'iteration', 'accuracy')
    plt.savefig('./train_loss.svg')
    plt.savefig('./train_loss.pdf')
    plt.savefig('./train_loss.png')

    exit()
