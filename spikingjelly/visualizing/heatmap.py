from __future__ import annotations

from typing import Optional, Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ._utils import _to_numpy

__all__ = ["plot_2d_heatmap"]


def plot_2d_heatmap(
    array: Union[np.ndarray, torch.Tensor],
    title: str,
    xlabel: str,
    ylabel: str,
    int_x_ticks: bool = True,
    int_y_ticks: bool = True,
    plot_colorbar: bool = True,
    colorbar_y_label: str = "magnitude",
    x_max: Optional[float] = None,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""
    **API Language** - :ref:`中文 <plot_2d_heatmap-cn>` | :ref:`English <plot_2d_heatmap-en>`

    ----

    .. _plot_2d_heatmap-cn:

    * **中文**

    绘制一张二维热力图。可以用来绘制多个神经元在不同时刻的电压。

    :param array: shape=[T, N]的数组，支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type array: Union[np.ndarray, torch.Tensor]

    :param title: 热力图标题
    :type title: str

    :param xlabel: x轴标签
    :type xlabel: str

    :param ylabel: y轴标签
    :type ylabel: str

    :param int_x_ticks: x轴是否只显示整数刻度
    :type int_x_ticks: bool

    :param int_y_ticks: y轴是否只显示整数刻度
    :type int_y_ticks: bool

    :param plot_colorbar: 是否画出颜色-数值对应关系的 colorbar
    :type plot_colorbar: bool

    :param colorbar_y_label: colorbar 的 y 轴标签
    :type colorbar_y_label: str

    :param x_max: 横轴最大刻度。若为 ``None``，则为 ``array.shape[1]``
    :type x_max: Optional[float]

    :param figsize: 图片尺寸
    :type figsize: Tuple[float, float]

    :param dpi: 绘图 dpi
    :type dpi: int

    :return: ``(fig, ax)`` 元组
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: 当 ``array`` 不是二维数组时

    ----

    .. _plot_2d_heatmap-en:

    * **English**

    Plot a 2D heatmap. Useful for visualizing membrane potentials of multiple neurons over time.

    :param array: Array of shape=[T, N]. Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type array: Union[np.ndarray, torch.Tensor]

    :param title: Title of the heatmap.
    :type title: str

    :param xlabel: Label of the x-axis.
    :type xlabel: str

    :param ylabel: Label of the y-axis.
    :type ylabel: str

    :param int_x_ticks: Whether to show only integer ticks on the x-axis.
    :type int_x_ticks: bool

    :param int_y_ticks: Whether to show only integer ticks on the y-axis.
    :type int_y_ticks: bool

    :param plot_colorbar: Whether to draw a colorbar showing the color-value mapping.
    :type plot_colorbar: bool

    :param colorbar_y_label: Label of the colorbar y-axis.
    :type colorbar_y_label: str

    :param x_max: Maximum tick on the x-axis. If ``None``, defaults to ``array.shape[1]``.
    :type x_max: Optional[float]

    :param figsize: Figure size.
    :type figsize: Tuple[float, float]

    :param dpi: Dots per inch.
    :type dpi: int

    :return: ``(fig, ax)`` tuple.
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: If ``array`` is not 2-dimensional.

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
        v_list = []
        for t in range(T):
            s_list.append(lif(x).unsqueeze(0))
            v_list.append(lif.v.unsqueeze(0))

        s_list = torch.cat(s_list)
        v_list = torch.cat(v_list)

        fig, ax = visualizing.plot_2d_heatmap(
            array=v_list,
            title="Membrane Potentials",
            xlabel="Simulating Step",
            ylabel="Neuron Index",
            int_x_ticks=True,
            x_max=T,
            dpi=200,
        )
        plt.show()

    .. image:: ../_static/API/visualizing/plot_2d_heatmap.*
        :width: 100%
    """
    array = _to_numpy(array)
    if array.ndim != 2:
        raise ValueError(f"Expected 2D array, got {array.ndim}D array instead")

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    if x_max is not None:
        im = ax.imshow(
            array.T, aspect="auto", extent=[-0.5, x_max, array.shape[1] - 0.5, -0.5]
        )
    else:
        im = ax.imshow(array.T, aspect="auto")

    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)

    ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_x_ticks))
    ax.yaxis.set_major_locator(matplotlib.ticker.MaxNLocator(integer=int_y_ticks))
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())

    if plot_colorbar:
        cbar = ax.figure.colorbar(im)
        cbar.ax.set_ylabel(colorbar_y_label, rotation=90, va="top")
        cbar.ax.yaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    return fig, ax
