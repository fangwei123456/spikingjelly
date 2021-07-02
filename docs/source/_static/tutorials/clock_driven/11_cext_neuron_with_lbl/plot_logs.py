from matplotlib import pyplot as plt
import numpy as np
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

    # plt.show()
if __name__ == '__main__':
    figsize = (12, 8)
    plt.style.use(['science', 'muted'])
    fig = plt.figure(figsize=figsize, dpi=200)

    plot_log('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/run-4_conv_fashion_mnist-tag-train_acc.csv', 'Accuracy on train set',
             'iteration', 'accuracy', label='Naive PyTorch with step-by-step')
    plot_log('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/run-11_cext_neuron_with_lbl-tag-train_acc.csv', 'Accuracy on train set',
             'iteration', 'accuracy', label='CUDA Multi-Step with layer-by-layer')
    plt.legend(frameon=True)
    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.svg')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.pdf')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.png')
    plt.clf()
    plot_log('./docs/source/_static/tutorials/clock_driven/4_conv_fashion_mnist/run-4_conv_fashion_mnist-tag-test_acc.csv', 'Accuracy on test set',
             'epoch', 'accuracy', plot_max=True, label='Naive PyTorch with step-by-step')
    plot_log('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/run-11_cext_neuron_with_lbl-tag-test_acc.csv', 'Accuracy on test set',
             'epoch', 'accuracy', plot_max=True, label='CUDA Multi-Step with layer-by-layer')
    plt.legend(frameon=True)

    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.svg')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.pdf')
    plt.savefig('./docs/source/_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.png')
    exit()
