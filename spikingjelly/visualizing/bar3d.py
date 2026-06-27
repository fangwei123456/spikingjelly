from __future__ import annotations

from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ._utils import _to_numpy

__all__ = ["plot_2d_bar_in_3d"]


def plot_2d_bar_in_3d(
    array: Union[np.ndarray, torch.Tensor],
    title: str,
    xlabel: str,
    ylabel: str,
    zlabel: str,
    int_x_ticks: bool = True,
    int_y_ticks: bool = True,
    int_z_ticks: bool = False,
    dpi: int = 200,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""
    **API Language** - :ref:`中文 <plot_2d_bar_in_3d-cn>` | :ref:`English <plot_2d_bar_in_3d-en>`

    ----

    .. _plot_2d_bar_in_3d-cn:

    * **中文**

    将 shape=[T, N] 的数组绘制为三维柱状图。可以用来绘制多个神经元的脉冲发放频率随时间的变化情况。

    :param array: shape=[T, N]的数组，支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type array: Union[np.ndarray, torch.Tensor]

    :param title: 图的标题
    :type title: str

    :param xlabel: x轴标签
    :type xlabel: str

    :param ylabel: y轴标签
    :type ylabel: str

    :param zlabel: z轴标签
    :type zlabel: str

    :param int_x_ticks: x轴是否只显示整数刻度
    :type int_x_ticks: bool

    :param int_y_ticks: y轴是否只显示整数刻度
    :type int_y_ticks: bool

    :param int_z_ticks: z轴是否只显示整数刻度
    :type int_z_ticks: bool

    :param dpi: 绘图 dpi
    :type dpi: int

    :return: ``(fig, ax)`` 元组
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: 当 ``array`` 不是二维数组时

    ----

    .. _plot_2d_bar_in_3d-en:

    * **English**

    Plot a shape=[T, N] array as a 3D bar chart. Useful for visualizing firing rates of
    multiple neurons changing over time.

    :param array: Array of shape=[T, N]. Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type array: Union[np.ndarray, torch.Tensor]

    :param title: Title of the plot.
    :type title: str

    :param xlabel: Label of the x-axis.
    :type xlabel: str

    :param ylabel: Label of the y-axis.
    :type ylabel: str

    :param zlabel: Label of the z-axis.
    :type zlabel: str

    :param int_x_ticks: Whether to show only integer ticks on the x-axis.
    :type int_x_ticks: bool

    :param int_y_ticks: Whether to show only integer ticks on the y-axis.
    :type int_y_ticks: bool

    :param int_z_ticks: Whether to show only integer ticks on the z-axis.
    :type int_z_ticks: bool

    :param dpi: Dots per inch.
    :type dpi: int

    :return: ``(fig, ax)`` tuple.
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: If ``array`` is not 2-dimensional.

    ----

    * **代码示例 | Example**

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
        fig, ax = visualizing.plot_2d_bar_in_3d(
            firing_rate,
            title="spiking rates of output layer",
            xlabel="neuron index",
            ylabel="training epoch",
            zlabel="spiking rate",
        )
        plt.show()

    .. image:: ../_static/API/visualizing/plot_2d_bar_in_3d.png
    """
    array = _to_numpy(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig = plt.figure(dpi=dpi)
    ax = fig.add_subplot(111, projection="3d")
    ax.set_title(title)
    colormap = plt.get_cmap("tab10")

    array_T = array.T
    xs = np.arange(array_T.shape[1])
    for i in range(array_T.shape[0]):
        ax.bar(xs, array_T[i], i, zdir="x", color=colormap(i % 10), alpha=0.8)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_zlabel(zlabel)
    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    ax.zaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_z_ticks))
    return fig, ax
