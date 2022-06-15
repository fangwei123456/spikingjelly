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
    root = 'C:/Users/fw/Desktop/代码/spikingjelly/docs/source/_static/tutorials/activation_based/recurrent_connection_and_stateful_synapse'
    figsize = (12, 8)
    dpi = 200
    fontsize = 20
    fig = plt.figure(figsize=figsize, dpi=dpi)

    title = 'Training curves on Sequential FashionMNIST'
    fname_label_xlabel_ylabel = {
        'run-plain_b256_sgd_lr0.1_amp_cupy-tag-train_acc.csv': ('Plain', 'epoch', 'accuracy'),
        'run-ss_b256_sgd_lr0.1_amp_cupy-tag-train_acc.csv': ('StatefulSynapse', 'epoch', 'accuracy'),
        'run-fb_b256_sgd_lr0.1_amp_cupy-tag-train_acc.csv': ('FeedBack', 'epoch', 'accuracy'),
    }

    for fname in fname_label_xlabel_ylabel.keys():
        label, x_label, y_label = fname_label_xlabel_ylabel[fname]
        plot_log(os.path.join(root, fname), x_label=x_label, y_label=y_label, label=label)
    plt.title(title, fontsize=fontsize)
    plt.legend(frameon=True, fontsize=fontsize)
    plt.savefig(os.path.join(root, f'rsnn_train_acc.pdf'))
    plt.savefig(os.path.join(root, f'rsnn_train_acc.svg'))
    plt.savefig(os.path.join(root, f'rsnn_train_acc.png'))
    plt.clf()

    title = 'Test curves on Sequential FashionMNIST'
    fname_label_xlabel_ylabel = {
        'run-plain_b256_sgd_lr0.1_amp_cupy-tag-test_acc.csv': ('Plain', 'epoch', 'accuracy'),
        'run-ss_b256_sgd_lr0.1_amp_cupy-tag-test_acc.csv': ('StatefulSynapse', 'epoch', 'accuracy'),
        'run-fb_b256_sgd_lr0.1_amp_cupy-tag-test_acc.csv': ('FeedBack', 'epoch', 'accuracy'),
    }

    for fname in fname_label_xlabel_ylabel.keys():
        label, x_label, y_label = fname_label_xlabel_ylabel[fname]
        plot_log(os.path.join(root, fname), x_label=x_label, y_label=y_label, label=label)
    plt.title(title, fontsize=fontsize)
    plt.legend(frameon=True, fontsize=fontsize)
    plt.savefig(os.path.join(root, f'rsnn_test_acc.pdf'))
    plt.savefig(os.path.join(root, f'rsnn_test_acc.svg'))
    plt.savefig(os.path.join(root, f'rsnn_test_acc.png'))