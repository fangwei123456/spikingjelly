from matplotlib import pyplot as plt
import numpy as np
import os
def plot_log(csv_file, x_label=None, y_label=None, plot_max=False, label=None, fontsize=20):
    log_data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=(1, 2))
    x = log_data[:, 0]
    y = log_data[:, 1]
    plt.plot(x, y, label=label)
    if x_label is not None:
        plt.xlabel(x_label, fontsize=fontsize)
    if y_label is not None:
        plt.ylabel(y_label, fontsize=fontsize)
    plt.grid(linestyle='-.')

    if plot_max:
        # 画最大值
        index = y.argmax()
        plt.text(x[index], y[index], '({}, {})'.format(int(x[index]), round(y[index], 3)), fontsize=fontsize // 2)
        plt.scatter(x[index], y[index], marker='1', alpha=0.8, linewidths=1., c='r')

if __name__ == '__main__':
    plt.style.use(['science'])
    root = 'C:/Users/fw/Desktop/代码/spikingjelly/docs/source/_static/tutorials/activation_based/conv_fashion_mnist'
    figsize = (12, 8)
    dpi = 200
    fontsize = 20
    fig = plt.figure(figsize=figsize, dpi=dpi)

    title = 'Training curves on FashionMNIST'
    fname_label_xlabel_ylabel = {
        'run-T4_b256_sgd_lr0.1_c128_amp_cupy-tag-train_acc.csv': ('Training accuracy', 'epoch', 'accuracy'),
        'run-T4_b256_sgd_lr0.1_c128_amp_cupy-tag-test_acc.csv': ('Test accuracy', 'epoch', 'accuracy'),
    }

    for fname in fname_label_xlabel_ylabel.keys():
        label, x_label, y_label = fname_label_xlabel_ylabel[fname]
        plot_log(os.path.join(root, fname), x_label=x_label, y_label=y_label, label=label, plot_max= (label == 'Test accuracy'))
    plt.title(title, fontsize=fontsize)
    plt.legend(frameon=True, fontsize=fontsize)
    plt.savefig(os.path.join(root, f'fmnist_logs.pdf'))
    plt.savefig(os.path.join(root, f'fmnist_logs.svg'))
    plt.savefig(os.path.join(root, f'fmnist_logs.png'))
