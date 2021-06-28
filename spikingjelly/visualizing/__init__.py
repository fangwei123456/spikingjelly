import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def plot_2d_heatmap(array: np.ndarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                    plot_colorbar=True, colorbar_y_label='magnitude', x_max=None, dpi=200):
    '''
    :param array: shape=[T, N]的任意数组
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_colorbar: 是否画出显示颜色和数值对应关系的colorbar
    :param colorbar_y_label: colorbar的y轴label
    :param x_max: 横轴的最大刻度。若设置为 ``None``，则认为横轴的最大刻度是 ``array.shape[1]``
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    绘制一张二维的热力图。可以用来绘制一张表示多个神经元在不同时刻的电压的热力图，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.clock_driven import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt
        import numpy as np

        lif = neuron.LIFNode(tau=100.)
        x = torch.rand(size=[32]) * 4
        T = 50
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))
            v_list.append(lif.v.unsqueeze(0))

        s_list = torch.cat(s_list)
        v_list = torch.cat(v_list)

        visualizing.plot_2d_heatmap(array=np.asarray(v_list), title='Membrane Potentials', xlabel='Simulating Step',
                                    ylabel='Neuron Index', int_x_ticks=True, x_max=T, dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_2d_heatmap.*
        :width: 100%

    '''
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, heatmap = plt.subplots(dpi=dpi)
    if x_max is not None:
        im = heatmap.imshow(array.T, aspect='auto', extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5])
    else:
        im = heatmap.imshow(array.T, aspect='auto')

    heatmap.set_title(title)
    heatmap.set_xlabel(xlabel)
    heatmap.set_ylabel(ylabel)

    heatmap.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    heatmap.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    heatmap.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    heatmap.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if plot_colorbar:
        cbar = heatmap.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va='top')
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return fig

def plot_2d_bar_in_3d(array: np.ndarray, title: str, xlabel: str, ylabel: str, zlabel: str, int_x_ticks=True, int_y_ticks=True, int_z_ticks=False, dpi=200):
    '''
    :param array: shape=[T, N]的任意数组
    :param title: 图的标题
    :param xlabel: x轴的label
    :param ylabel: y轴的label
    :param zlabel: z轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param int_z_ticks: z轴上是否只显示整数刻度
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    将shape=[T, N]的任意数组，绘制为三维的柱状图。可以用来绘制多个神经元的脉冲发放频率，随着时间的变化情况，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt

        Epochs = 5
        N = 10
        firing_rate = torch.zeros(Epochs, N)
        init_firing_rate = torch.rand(size=[N])
        for i in range(Epochs):
            firing_rate[i] = torch.softmax(init_firing_rate * (i + 1) ** 2, dim=0)
        visualizing.plot_2d_bar_in_3d(firing_rate.numpy(), title='spiking rates of output layer', xlabel='neuron index',
                                      ylabel='training epoch', zlabel='spiking rate', int_x_ticks=True, int_y_ticks=True,
                                      int_z_ticks=False, dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_2d_bar_in_3d.png

    也可以用来绘制一张表示多个神经元在不同时刻的电压的热力图，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt
        from spikingjelly.clock_driven import neuron

        neuron_num = 4
        T = 50
        lif_node = neuron.LIFNode(tau=100.)
        w = torch.rand([neuron_num]) * 10
        v_list = []
        for t in range(T):
            lif_node(w * torch.rand(size=[neuron_num]))
            v_list.append(lif_node.v.unsqueeze(0))

        v_list = torch.cat(v_list)
        visualizing.plot_2d_bar_in_3d(v_list, title='voltage of neurons', xlabel='neuron index',
                                      ylabel='simulating step', zlabel='voltage', int_x_ticks=True, int_y_ticks=True,
                                      int_z_ticks=False, dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_2d_bar_in_3d_1.png

    '''
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection='3d')
    ax.set_title(title)
    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    array_T = array.T
    xs = np.arange(array_T.shape[1])
    for i in range(array_T.shape[0]):
        ax.bar(xs, array_T[i], i, zdir='x', color=colormap(i % 10), alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_z_ticks))
    return fig

def plot_1d_spikes(spikes: np.asarray, title: str, xlabel: str, ylabel: str, int_x_ticks=True, int_y_ticks=True,
                   plot_firing_rate=True, firing_rate_map_title='Firing Rate', dpi=200):
    '''


    :param spikes: shape=[T, N]的np数组，其中的元素只为0或1，表示N个时长为T的脉冲数据
    :param title: 热力图的标题
    :param xlabel: 热力图的x轴的label
    :param ylabel: 热力图的y轴的label
    :param int_x_ticks: x轴上是否只显示整数刻度
    :param int_y_ticks: y轴上是否只显示整数刻度
    :param plot_firing_rate: 是否画出各个脉冲发放频率
    :param firing_rate_map_title: 脉冲频率发放图的标题
    :param dpi: 绘图的dpi
    :return: 绘制好的figure

    画出N个时长为T的脉冲数据。可以用来画N个神经元在T个时刻的脉冲发放情况，示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.clock_driven import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt
        import numpy as np

        lif = neuron.LIFNode(tau=100.)
        x = torch.rand(size=[32]) * 4
        T = 50
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))
            v_list.append(lif.v.unsqueeze(0))

        s_list = torch.cat(s_list)
        v_list = torch.cat(v_list)

        visualizing.plot_1d_spikes(spikes=np.asarray(s_list), title='Membrane Potentials', xlabel='Simulating Step',
                                   ylabel='Neuron Index', dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_1d_spikes.*
        :width: 100%

    '''
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spikes.ndim}D array instead")

    spikes_T = spikes.T
    if plot_firing_rate:
        fig = plt.figure(tight_layout=True, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        firing_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots()

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    spikes_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))

    spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    spikes_map.set_xlim(-0.5, spikes_T.shape[1] - 0.5)
    spikes_map.set_ylim(-0.5, spikes_T.shape[0] - 0.5)
    spikes_map.invert_yaxis()
    N = spikes_T.shape[0]
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = (spikes_T == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出

    colormap = plt.get_cmap('tab10')  # cmap的种类参见https://matplotlib.org/gallery/color/colormap_reference.html

    for i in range(N):
        spikes_map.eventplot(t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10))

    if plot_firing_rate:
        firing_rate = np.mean(spikes_T, axis=1, keepdims=True)

        max_rate = firing_rate.max()
        min_rate = firing_rate.min()

        firing_rate_map.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
        firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        firing_rate_map.imshow(firing_rate, cmap='magma', aspect='auto')
        for i in range(firing_rate.shape[0]):
            firing_rate_map.text(0, i, f'{firing_rate[i][0]:.2f}', ha='center', va='center', color='w' if firing_rate[i][0] < 0.7 * max_rate or min_rate == max_rate else 'black')
        firing_rate_map.get_xaxis().set_visible(False)
        firing_rate_map.set_title(firing_rate_map_title)
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

        from spikingjelly import visualizing
        import numpy as np
        from matplotlib import pyplot as plt

        C = 48
        W = 8
        H = 8
        spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
        visualizing.plot_2d_spiking_feature_map(spikes=spikes, nrows=6, ncols=8, space=2, title='Spiking Feature Maps', dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_2d_spiking_feature_map.*
        :width: 100%

    '''
    if spikes.ndim != 3:
        raise ValueError(f"Expected 3D array, got {spikes.ndim}D array instead")

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
    maps.set_title(title)
    maps.imshow(y, cmap='gray')

    maps.get_xaxis().set_visible(False)
    maps.get_yaxis().set_visible(False)
    return fig, maps

def plot_one_neuron_v_s(v: np.ndarray, s: np.ndarray, v_threshold=1.0, v_reset=0.0,
                        title='$V_{t}$ and $S_{t}$ of the neuron', dpi=200):
    '''
    :param v: shape=[T], 存放神经元不同时刻的电压
    :param s: shape=[T], 存放神经元不同时刻释放的脉冲
    :param v_threshold: 神经元的阈值电压
    :param v_reset: 神经元的重置电压。也可以为 ``None``
    :param title: 图的标题
    :param dpi: 绘图的dpi
    :return: 一个figure

    绘制单个神经元的电压、脉冲随着时间的变化情况。示例代码：

    .. code-block:: python

        import torch
        from spikingjelly.clock_driven import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt

        lif = neuron.LIFNode(tau=100.)
        x = torch.Tensor([2.0])
        T = 150
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x))
            v_list.append(lif.v)
        visualizing.plot_one_neuron_v_s(v_list, s_list, v_threshold=lif.v_threshold, v_reset=lif.v_reset,
                                        dpi=200)
        plt.show()

    .. image:: ./_static/API/visualizing/plot_one_neuron_v_s.*
        :width: 100%
    '''
    fig = plt.figure(dpi=dpi)
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax0.set_title(title)
    T = s.shape[0]
    t = np.arange(0, T)
    ax0.plot(t, v)
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylabel('voltage')
    ax0.axhline(v_threshold, label='$V_{threshold}$', linestyle='-.', c='r')
    if v_reset is not None:
        ax0.axhline(v_reset, label='$V_{reset}$', linestyle='-.', c='g')
    ax0.legend()
    t_spike = s * t
    mask = (s == 1)  # eventplot中的数值是时间发生的时刻，因此需要用mask筛选出
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ax1.eventplot(t_spike[mask], lineoffsets=0, colors='r')
    ax1.set_xlim(-0.5, T - 0.5)

    ax1.set_xlabel('simulating step')
    ax1.set_ylabel('spike')
    ax1.set_yticks([])

    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return fig, ax0, ax1
