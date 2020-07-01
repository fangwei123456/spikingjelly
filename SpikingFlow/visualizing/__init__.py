import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_neurons_voltage_heatmap(v:np.ndarray):
    '''
    :param v: shape=[N, T]的np数组，表示N个神经元在T个时刻的电压值
    :return: 一个figure，画出了N个神经元在T个时刻的电压的热力图

    画出N个神经元在T个时刻的电压的热力图，示例代码：

    .. code-block:: python

        neuron_num = 32
        T = 50
        lif_node = neuron.LIFNode(monitor=True)
        w = torch.rand([neuron_num]) * 50
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
        visualizing.plot_neurons_voltage_heatmap(np.asarray(lif_node.monitor['v']).T)
        plt.show()

    .. image:: ./_static/API/plot_neurons_voltage_heatmap.png
    '''
    fig, heatmap = plt.subplots(dpi=200)
    im = heatmap.imshow(v)
    heatmap.set_title('voltage of neurons')
    heatmap.set_xlabel('simulating step')
    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    heatmap.set_ylabel('neuron index')
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    cbar = heatmap.figure.colorbar(im)
    cbar.ax.set_ylabel('voloage magnitude', rotation=90, va='top')
    return fig

def plot_neurons_spikes(spikes:np.asarray, plot_spiking_rate=True):
    '''
    :param spikes: shape=[N, T]的np数组，其中的元素只为0或1，表示N个神经元在T个时刻的脉冲
    :param plot_spiking_rate: 是否画出各个神经元的脉冲发放频率
    :return: 一个figure，画出了N个神经元在T个时刻的脉冲发放情况

    画出N个神经元在T个时刻的脉冲发放情况，示例代码：

    .. code-block:: python

        neuron_num = 32
        T = 50
        lif_node = neuron.LIFNode(monitor=True)
        w = torch.rand([neuron_num]) * 50
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
        visualizing.plot_neurons_spikes(np.asarray(lif_node.monitor['s']).T)
        plt.show()

    .. image:: ./_static/API/plot_neurons_spikes.png
    '''
    if plot_spiking_rate:
        fig = plt.figure(tight_layout=True, dpi=200)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        spiking_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()



    spikes_map.set_title('spikes of neurons')
    spikes_map.set_xlabel('simulating step')
    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    spikes_map.set_xlim(-0.5, spikes.shape[1] + 0.5)

    spikes_map.set_ylabel('neuron index')
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    spikes_map.set_ylim(-0.5, spikes.shape[0] + 0.5)
    spikes_map.invert_yaxis()
    N = spikes.shape[0]
    T = spikes.shape[1]
    t = np.arange(0, T)
    t_spike = spikes * t
    mask = (spikes == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    for i in range(N):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10))

    if plot_spiking_rate:
        spiking_rate = np.mean(spikes, axis=1, keepdims=True)
        spiking_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        spiking_rate_map.imshow(spiking_rate, cmap='plasma', aspect='auto')
        for i in range(spiking_rate.shape[0]):
            spiking_rate_map.text(0, i, spiking_rate[i][0], ha='center', va='center', color='w')
        spiking_rate_map.get_xaxis().set_visible(False)
        spiking_rate_map.set_title('spiking rate')


def plot_2d_feature_map_spikes(spikes:np.asarray, nrows, ncols):
    '''
    :param spikes: shape=[C, W, H]，C个尺寸为W * H的脉冲矩阵，矩阵中的元素为0或1。这样的矩阵一般来源于卷积层后的脉冲神经元的输出
    :param nrows: 画成多少行
    :param ncols: 画成多少列
    :return: 一个figure，将C个矩阵全部画出，然后排列成nrows行ncols列

    将C个尺寸为W * H的脉冲矩阵，全部画出，然后排列成nrows行ncols列。这样的矩阵一般来源于卷积层后的脉冲神经元的输出，通过这个函数\\
    可以对输出进行可视化。示例代码：

    .. code-block:: python

        C = 48
        W = 8
        H = 8
        spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
        visualizing.plot_2d_feature_map_spikes(spikes, 6, 8)
        plt.show()

    .. image:: ./_static/API/plot_2d_feature_map_spikes.png
    '''
    C = spikes.shape[0]

    assert nrows * ncols == C, 'nrows * ncols != C'

    fig, axs = plt.subplots(nrows=nrows, ncols=ncols, dpi=200)
    fig.suptitle('feature map spikes')
    for i in range(nrows):
        for j in range(ncols):
            axs[i][j].imshow(spikes[i * ncols + j], cmap='gray')
            axs[i][j].get_xaxis().set_visible(False)
            axs[i][j].get_yaxis().set_visible(False)