from matplotlib import pyplot as plt
import numpy as np
def plot_log(csv_file, title, x_label, y_label, dpi=200, plot_max=False):
    log_data = np.loadtxt(csv_file, delimiter=',', skiprows=1, usecols=(1, 2))
    x = log_data[:, 0]
    y = log_data[:, 1]
    fig = plt.figure(dpi=dpi)
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

    plt.show()
if __name__ == '__main__':
    plt.style.use(['science', 'muted'])
    plot_log('run-logs_conv_fashion_mnist-tag-train_accuracy.csv', 'Accuracy on train batch',
             'iteration', 'accuracy')
    plot_log('run-logs_conv_fashion_mnist-tag-test_accuracy.csv', 'Accuracy on test dataset',
             'epoch', 'accuracy', plot_max=True)