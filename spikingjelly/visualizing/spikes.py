from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ._utils import _to_numpy

__all__ = ["plot_1d_spikes", "plot_one_neuron_v_s"]


def plot_1d_spikes(
    spikes: Union[np.ndarray, torch.Tensor],
    title: str,
    xlabel: str,
    ylabel: str,
    int_x_ticks: bool = True,
    int_y_ticks: bool = True,
    plot_firing_rate: bool = True,
    firing_rate_map_title: str = "firing rate",
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""
    **API Language** - :ref:`中文 <plot_1d_spikes-cn>` | :ref:`English <plot_1d_spikes-en>`

    ----

    .. _plot_1d_spikes-cn:

    * **中文**

    画出 N 个时长为 T 的脉冲数据。可以用来画 N 个神经元在 T 个时刻的脉冲发放情况。

    :param spikes: shape=[T, N] 的数组，元素只能为 0 或 1，表示 N 个时长为 T 的脉冲数据。
        支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type spikes: Union[np.ndarray, torch.Tensor]

    :param title: 图的标题
    :type title: str

    :param xlabel: x轴标签
    :type xlabel: str

    :param ylabel: y轴标签
    :type ylabel: str

    :param int_x_ticks: x轴是否只显示整数刻度
    :type int_x_ticks: bool

    :param int_y_ticks: y轴是否只显示整数刻度
    :type int_y_ticks: bool

    :param plot_firing_rate: 是否画出各脉冲发放频率
    :type plot_firing_rate: bool

    :param firing_rate_map_title: 脉冲频率发放图的标题
    :type firing_rate_map_title: str

    :param figsize: 图片尺寸
    :type figsize: Tuple[float, float]

    :param dpi: 绘图 dpi
    :type dpi: int

    :return: ``(fig, ax)`` 元组，其中 ``ax`` 是脉冲图的 axes
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: 当 ``spikes`` 不是二维数组时

    ----

    .. _plot_1d_spikes-en:

    * **English**

    Plot spike data for N neurons over T time steps.

    :param spikes: Array of shape=[T, N] with values in {0, 1}, representing spike trains
        of N neurons over T steps. Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type spikes: Union[np.ndarray, torch.Tensor]

    :param title: Title of the plot.
    :type title: str

    :param xlabel: Label of the x-axis.
    :type xlabel: str

    :param ylabel: Label of the y-axis.
    :type ylabel: str

    :param int_x_ticks: Whether to show only integer ticks on the x-axis.
    :type int_x_ticks: bool

    :param int_y_ticks: Whether to show only integer ticks on the y-axis.
    :type int_y_ticks: bool

    :param plot_firing_rate: Whether to draw a firing rate bar beside the spike plot.
    :type plot_firing_rate: bool

    :param firing_rate_map_title: Title of the firing rate subplot.
    :type firing_rate_map_title: str

    :param figsize: Figure size.
    :type figsize: Tuple[float, float]

    :param dpi: Dots per inch.
    :type dpi: int

    :return: ``(fig, ax)`` tuple where ``ax`` is the spike plot axes.
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: If ``spikes`` is not 2-dimensional.

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        from spikingjelly.activation_based import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt

        lif = neuron.LIFNode(tau=100.0)
        x = torch.rand(size=[32]) * 4
        T = 50
        s_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))

        s_list = torch.cat(s_list)

        fig, ax = visualizing.plot_1d_spikes(
            spikes=s_list,
            title="Spikes",
            xlabel="Simulating Step",
            ylabel="Neuron Index",
            dpi=200,
        )
        plt.show()

    .. image:: ../_static/API/visualizing/plot_1d_spikes.*
        :width: 100%
    """
    spikes = _to_numpy(spikes)
    if spikes.ndim != 2:
        raise ValueError(f"Expected 2D array, got {spikes.ndim}D array instead")

    spikes_T = spikes.T
    if plot_firing_rate:
        fig = plt.figure(tight_layout=True, figsize=figsize, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(1, 5)
        spikes_map = fig.add_subplot(gs[0, 0:4])
        firing_rate_map = fig.add_subplot(gs[0, 4])
    else:
        fig, spikes_map = plt.subplots(figsize=figsize, dpi=dpi)

    spikes_map.set_title(title)
    spikes_map.set_xlabel(xlabel)
    spikes_map.set_ylabel(ylabel)

    spikes_map.xaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=int_x_ticks)
    )
    spikes_map.yaxis.set_major_locator(
        matplotlib.ticker.MaxNLocator(integer=int_y_ticks)
    )

    spikes_map.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    spikes_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    spikes_map.set_xlim(-0.5, spikes_T.shape[1] - 0.5)
    spikes_map.set_ylim(-0.5, spikes_T.shape[0] - 0.5)
    spikes_map.invert_yaxis()
    N = spikes_T.shape[0]
    T = spikes_T.shape[1]
    t = np.arange(0, T)
    t_spike = spikes_T * t
    mask = spikes_T == 1

    colormap = plt.get_cmap("tab10")

    for i in range(N):
        spikes_map.eventplot(
            t_spike[i][mask[i]], lineoffsets=i, colors=colormap(i % 10)
        )

    if plot_firing_rate:
        firing_rate = np.mean(spikes_T, axis=1, keepdims=True)

        max_rate = firing_rate.max()
        min_rate = firing_rate.min()

        firing_rate_map.yaxis.set_major_locator(
            matplotlib.ticker.MaxNLocator(integer=True)
        )
        firing_rate_map.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        firing_rate_map.imshow(firing_rate, cmap="magma", aspect="auto")
        for i in range(firing_rate.shape[0]):
            firing_rate_map.text(
                0,
                i,
                f"{firing_rate[i][0]:.2f}",
                ha="center",
                va="center",
                color="w"
                if firing_rate[i][0] < 0.7 * max_rate or min_rate == max_rate
                else "black",
            )
        firing_rate_map.get_xaxis().set_visible(False)
        firing_rate_map.set_title(firing_rate_map_title)
    return fig, spikes_map


def plot_one_neuron_v_s(
    v: Union[np.ndarray, torch.Tensor],
    s: Union[np.ndarray, torch.Tensor],
    v_threshold: float = 1.0,
    v_reset: Optional[float] = 0.0,
    title: str = "$V[t]$ and $S[t]$ of the neuron",
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]:
    r"""
    **API Language** - :ref:`中文 <plot_one_neuron_v_s-cn>` | :ref:`English <plot_one_neuron_v_s-en>`

    ----

    .. _plot_one_neuron_v_s-cn:

    * **中文**

    绘制单个神经元的电压、脉冲随着时间的变化情况。

    :param v: shape=[T] 的数组，存放神经元不同时刻的电压。支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type v: Union[np.ndarray, torch.Tensor]

    :param s: shape=[T] 的数组，存放神经元不同时刻释放的脉冲。支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type s: Union[np.ndarray, torch.Tensor]

    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float

    :param v_reset: 神经元的重置电压。可以为 ``None``
    :type v_reset: Optional[float]

    :param title: 图的标题
    :type title: str

    :param figsize: 图片尺寸
    :type figsize: Tuple[float, float]

    :param dpi: 绘图 dpi
    :type dpi: int

    :return: ``(fig, ax_voltage, ax_spike)`` 三元组
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]

    :raises ValueError: 当 ``v`` 或 ``s`` 不是一维数组时

    ----

    .. _plot_one_neuron_v_s-en:

    * **English**

    Plot the membrane voltage and spike train of a single neuron over time.

    :param v: Array of shape=[T] storing membrane voltage at each time step.
        Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type v: Union[np.ndarray, torch.Tensor]

    :param s: Array of shape=[T] storing spikes emitted at each time step.
        Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type s: Union[np.ndarray, torch.Tensor]

    :param v_threshold: Threshold voltage of the neuron.
    :type v_threshold: float

    :param v_reset: Reset voltage of the neuron. Can be ``None``.
    :type v_reset: Optional[float]

    :param title: Title of the plot.
    :type title: str

    :param figsize: Figure size.
    :type figsize: Tuple[float, float]

    :param dpi: Dots per inch.
    :type dpi: int

    :return: ``(fig, ax_voltage, ax_spike)`` triple.
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes, matplotlib.axes.Axes]

    :raises ValueError: If ``v`` or ``s`` is not 1-dimensional.

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        from spikingjelly.activation_based import neuron
        from spikingjelly import visualizing
        from matplotlib import pyplot as plt

        lif = neuron.LIFNode(tau=100.0)
        x = torch.Tensor([2.0])
        T = 150
        s_list = []
        v_list = []
        for t in range(T):
            s_list.append(lif(x))
            v_list.append(lif.v)
        fig, ax_v, ax_s = visualizing.plot_one_neuron_v_s(
            v_list, s_list, v_threshold=lif.v_threshold, v_reset=lif.v_reset
        )
        plt.show()

    .. image:: ../_static/API/visualizing/plot_one_neuron_v_s.*
        :width: 100%
    """
    v = _to_numpy(v)
    s = _to_numpy(s)
    if v.ndim != 1:
        raise ValueError(f"Expected 1D array for v, got {v.ndim}D array instead")
    if s.ndim != 1:
        raise ValueError(f"Expected 1D array for s, got {s.ndim}D array instead")

    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax0 = plt.subplot2grid((3, 1), (0, 0), rowspan=2)
    ax0.set_title(title)
    T = s.shape[0]
    t = np.arange(0, T)
    ax0.plot(t, v)
    ax0.set_xlim(-0.5, T - 0.5)
    ax0.set_ylabel("voltage")
    ax0.axhline(v_threshold, label="$V_{threshold}$", linestyle="-.", c="r")
    if v_reset is not None:
        ax0.axhline(v_reset, label="$V_{reset}$", linestyle="-.", c="g")
    ax0.legend(frameon=True)
    t_spike = s * t
    mask = s == 1
    ax1 = plt.subplot2grid((3, 1), (2, 0))
    ax1.eventplot(t_spike[mask], lineoffsets=0, colors="r")
    ax1.set_xlim(-0.5, T - 0.5)

    ax1.set_xlabel("simulating step")
    ax1.set_ylabel("spike")
    ax1.set_yticks([])

    ax1.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=True))
    return fig, ax0, ax1
