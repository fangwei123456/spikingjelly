import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', dpi=200):
    '''
    :param array: shape=[N, M]的任意数组
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_colorbar: 是否画出显示颜色和数值对应关系的colorbar
    :param colorbar_y_label: colorbar的y轴label
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    绘制一张二维的热力图。可以用来绘制一张表示多个神经元在不同时刻的电压的热力图，示例代码：

    .. code-block:: python

        neuron_num = 32
        T = 50
        lif_node = SpikingFlow.event_driven.neuron.LIFNode(monitor=True)
        w = torch.rand([neuron_num]) * 50
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
        v_t_array = np.asarray(lif_node.monitor['v']).T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        visualizing.plot_2d_heatmap(array=v_t_array, title='voltage of neurons', xlabel='simulating step',
                                    ylabel='neuron index', int_x_ticks=True, int_y_ticks=True,
                                    plot_colorbar=True, colorbar_y_label='voltage magnitude', dpi=200)
        plt.show()

    .. image:: ./_static/API/plot_2d_heatmap.png
    '''
    fig, heatmap = plt.subplots(dpi=dpi)
    im = heatmap.imshow(array, aspect='auto')
    heatmap.set_title(title)
    heatmap.set_xlabel(xlabel)
    heatmap.set_ylabel(ylabel)
    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
    return fig

def plot_2d_bar_in_3d(array: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str, int_x_ticks=True, int_y_ticks=True, int_z_ticks=False, dpi=200):
    '''
    :param array: shape=[N, M]的任意数组
    :param title: 图的标题
    :param xlabel: x轴的label
    :param ylabel: y轴的label
    :param zlabel: z轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param int_z_ticks: z轴上是否只显示整数刻度
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    将shape=[N, M]的任意数组，绘制为三维的柱状图。可以用来绘制多个神经元的脉冲发放频率，随着时间的变化情况，示例代码：

    .. code-block:: python

        Epochs = 5
        N = 10
        spiking_rate = torch.zeros(N, Epochs)
        init_spiking_rate = torch.rand(size=[N])
        for i in range(Epochs):
            spiking_rate[:, i] = torch.softmax(init_spiking_rate * (i + 1) ** 2, dim=0)
        visualizing.plot_2d_bar_in_3d(spiking_rate.numpy(), title='spiking rates of output layer', xlabel='neuron index',
                                      ylabel='training epoch', zlabel='spiking rate', int_x_ticks=True, int_y_ticks=True,
                                      int_z_ticks=False, dpi=200)
        plt.show()

    .. image:: ./_static/API/plot_2d_bar_in_3d.png

    也可以用来绘制一张表示多个神经元在不同时刻的电压的热力图，示例代码：

    .. code-block:: python

        neuron_num = 4
        T = 50
        lif_node = SpikingFlow.event_driven.neuron.LIFNode(monitor=True)
        w = torch.rand([neuron_num]) * 10
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
        v_t_array = np.asarray(lif_node.monitor['v']).T  # v_t_array[i][j]表示神经元i在j时刻的电压值
        visualizing.plot_2d_bar_in_3d(v_t_array, title='voltage of neurons', xlabel='neuron index',
                                      ylabel='simulating step', zlabel='voltage', int_x_ticks=True, int_y_ticks=True,
                                      int_z_ticks=False, dpi=200)
        plt.show()

    .. image:: ./_static/API/plot_2d_bar_in_3d_1.png

    '''

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    xs = np.arange(array.shape[1])
    for i in range(array.shape[0]):
        ax.bar(xs, array[i], i, zdir='x', color=colormap(i % 10), alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_z_ticks))
    return fig

def plot_1d_spikes(spikes: np.asarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                   plot_spiking_rate=True, spiking_rate_map_title='spiking rate', dpi=200):
    '''


    :param spikes: shape=[N, T]的np数组，其中的元素只为0或1，表示N个时长为T的脉冲数据
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_spiking_rate: 是否画出各个脉冲发放频率
    :param spiking_rate_map_title: 脉冲频率发放图的标题
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    画出N个时长为T的脉冲数据。可以用来画N个神经元在T个时刻的脉冲发放情况，示例代码：

    .. code-block:: python

        neuron_num = 32
        T = 50
        lif_node = SpikingFlow.event_driven.neuron.LIFNode(monitor=True)
        w = torch.rand([neuron_num]) * 50
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
        s_t_array = np.asarray(lif_node.monitor['s']).T  # s_t_array[i][j]表示神经元i在j时刻释放的脉冲，为0或1
        visualizing.plot_1d_spikes(spikes=s_t_array, title='spikes of neurons', xlabel='simulating step',
                                    ylabel='neuron index', int_x_ticks=True, int_y_ticks=True,
                                    plot_spiking_rate=True, spiking_rate_map_title='spiking rate', dpi=200)
        plt.show()

    .. image:: ./_static/API/plot_1d_spikes.png
    '''
    if plot_spiking_rate:
        fig = plt.figure(tight_layout=True, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        spiking_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))

    spikes_map.set_xlim(-0.5, spikes.shape[1] + 0.5)
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
        spiking_rate_map.set_title(spiking_rate_map_title)
    return fig

def plot_2d_spiking_feature_map(spikes: np.asarray, nrows, ncols, space, title: str, dpi=200):
    '''
    :param spikes: shape=[C, W, H]，C个尺寸为W * H的脉冲矩阵，矩阵中的元素为0或1。这样的矩阵一般来源于卷积层后的脉冲神经元的输出
    :param nrows: 画成多少行
    :param ncols: 画成多少列
    :param space: 矩阵之间的间隙
    :param title: 图的标题
    :param dpi: 绘图的dpi
    :return: 一个figure，将C个矩阵全部画出，然后排列成nrows行ncols列

    将C个尺寸为W * H的脉冲矩阵，全部画出，然后排列成nrows行ncols列。这样的矩阵一般来源于卷积层后的脉冲神经元的输出，通过这个函数\\
    可以对输出进行可视化。示例代码：

    .. code-block:: python

        C = 48
        W = 8
        H = 8
        spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
        visualizing.plot_2d_spiking_feature_map(spikes=spikes, nrows=6, ncols=8, space=2, title='spiking feature map', dpi=200)
        plt.show()

    .. image:: ./_static/API/plot_2d_spiking_feature_map.png
    '''
    C = spikes.shape[0]

    assert nrows * ncols == C, 'nrows * ncols != C'

    h = spikes.shape[1]
    w = spikes.shape[2]
    y = np.ones(shape=[(h + space) * nrows, (w + space) * ncols]) * spikes.max().item()
    index = 0
    for i in range(space // 2, y.shape[0], h + space):
        for j in range(space // 2, y.shape[1], w + space):
            y[i:i + h, j:j + w] = spikes[index]
            index += 1
    fig, maps = plt.subplots(dpi=dpi)
    fig.suptitle(title)
    maps.imshow(y, cmap='gray')

    maps.get_xaxis().set_visible(False)
    maps.get_yaxis().set_visible(False)
    return fig, maps
