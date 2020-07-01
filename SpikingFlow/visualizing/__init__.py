import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_voltage_heatmap(v:np.ndarray):
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
        plot_voltage_heatmap(np.asarray(lif_node.monitor['v']).T)
        plt.show()
    .. image:: ./_static/API/plot_voltage_heatmap.png

    '''
    fig, heatmap = plt.subplots()
    im = heatmap.imshow(v)
    heatmap.set_title('voloage of neurons')
    heatmap.set_xlabel('simulating step')
    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    heatmap.set_ylabel('neuron index')
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))

    cbar = heatmap.figure.colorbar(im)
    cbar.ax.set_ylabel('voloage magnitude', rotation=90, va='top')
    return fig

def plot_spikes(spikes:np.asarray, plot_spiking_rate=True):
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
        plot_spikes(np.asarray(lif_node.monitor['s']).T)
        plt.show()

    .. image:: ./_static/API/plot_spikes.png


    '''
    if plot_spiking_rate:
        fig = plt.figure(tight_layout=True)
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

    colormap = plt.get_cmap('brg')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    # 使得颜色交叉分布，便于查看
    for i in range(0, N, 2):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i / N))
    for i in range(1, N, 2):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(1 - i / N))

    if plot_spiking_rate:
        spiking_rate = np.mean(spikes, axis=1, keepdims=True)
        spiking_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        spiking_rate_map.imshow(spiking_rate, cmap='cool', aspect='auto')
        for i in range(spiking_rate.shape[0]):
            spiking_rate_map.text(0, i, spiking_rate[i][0], ha='center', va='center', color='w')
        spiking_rate_map.get_xaxis().set_visible(False)
        spiking_rate_map.set_title('spiking rate')