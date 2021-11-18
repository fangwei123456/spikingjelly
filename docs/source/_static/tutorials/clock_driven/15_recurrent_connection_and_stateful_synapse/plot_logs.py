from matplotlib import pyplot as plt
import numpy as np
from spikingjelly import visualizing
import torch
import torch.nn as nn
import torchvision
def plot_log(csv_file, title, x_label, y_label, plot_max=False, label=None):
    log_data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=(1, 2))
    x = log_data[:, 0]
    y = log_data[:, 1]
    plt.plot(x, y, label=label)
    plt.xlabel(x_label, fontsize=20)
    plt.ylabel(y_label, fontsize=20)
    plt.title(title, fontsize=20)
    plt.grid(linestyle='-.')

    if plot_max:
        # 画最大值
        index = y.argmax()
        # plt.text(x[index], y[index], '({}, {})'.format(int(x[index]), round(y[index], 3)), fontsize=14)
        plt.scatter(x[index], y[index], marker='1', alpha=0.8, linewidths=0.1, c='r')

if __name__ == '__main__':
    plt.style.use(['science'])
    name_list = ['plain', 'feedback', 'stateful-synapse']
    figsize = (12, 8)
    fig = plt.figure(figsize=figsize, dpi=200)


    for fname in name_list:
        plot_log(f'./run-{fname}-tag-test_acc1.csv', 'Test accuracy', 'iteration', 'accuracy', label=fname)
    plt.legend(frameon=True, fontsize=20)
    plt.savefig('./test_acc.svg')
    plt.savefig('./test_acc.pdf')
    plt.savefig('./test_acc.png')


    plt.clf()
    for fname in name_list:
        plot_log(f'./run-{fname}-tag-train_acc1.csv', 'Train accuracy', 'iteration', 'accuracy', label=fname)
    plt.legend(frameon=True, fontsize=20)
    plt.savefig('./train_acc.svg')
    plt.savefig('./train_acc.pdf')
    plt.savefig('./train_acc.png')

    plt.clf()
    for fname in name_list:
        plot_log(f'./run-{fname}-tag-train_loss.csv', 'Train loss', 'iteration', 'accuracy', label=fname)
    plt.legend(frameon=True, fontsize=20)
    plt.savefig('./train_loss.svg')
    plt.savefig('./train_loss.pdf')
    plt.savefig('./train_loss.png')

    exit()
