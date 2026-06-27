from __future__ import annotations

from typing import Tuple, Union

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import torch

from ._utils import _to_numpy

__all__ = ["plot_2d_feature_map"]


def plot_2d_feature_map(
    x3d: Union[np.ndarray, torch.Tensor],
    nrows: int,
    ncols: int,
    space: int,
    title: str,
    figsize: Tuple[float, float] = (12, 8),
    dpi: int = 200,
) -> Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]:
    r"""
    **API Language** - :ref:`中文 <plot_2d_feature_map-cn>` | :ref:`English <plot_2d_feature_map-en>`

    ----

    .. _plot_2d_feature_map-cn:

    * **中文**

    将 C 个尺寸为 W x H 的矩阵全部画出，排列成 nrows 行 ncols 列。这样的矩阵一般来源于卷积层后脉冲神经元的输出。

    :param x3d: shape=[C, W, H] 的数组，支持 ``np.ndarray`` 或 ``torch.Tensor``
    :type x3d: Union[np.ndarray, torch.Tensor]

    :param nrows: 画成多少行
    :type nrows: int

    :param ncols: 画成多少列
    :type ncols: int

    :param space: 矩阵之间的间隙（像素）
    :type space: int

    :param title: 图的标题
    :type title: str

    :param figsize: 图片尺寸
    :type figsize: Tuple[float, float]

    :param dpi: 绘图 dpi
    :type dpi: int

    :return: ``(fig, ax)`` 元组
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: 当 ``x3d`` 不是三维数组时，或 ``nrows * ncols != C`` 时

    ----

    .. _plot_2d_feature_map-en:

    * **English**

    Plot C matrices of size W x H arranged in a grid of ``nrows`` rows and ``ncols`` columns.
    These matrices typically come from the output of convolutional spiking layers.

    :param x3d: Array of shape=[C, W, H]. Accepts ``np.ndarray`` or ``torch.Tensor``.
    :type x3d: Union[np.ndarray, torch.Tensor]

    :param nrows: Number of rows in the grid.
    :type nrows: int

    :param ncols: Number of columns in the grid.
    :type ncols: int

    :param space: Gap (in pixels) between adjacent matrices.
    :type space: int

    :param title: Title of the plot.
    :type title: str

    :param figsize: Figure size.
    :type figsize: Tuple[float, float]

    :param dpi: Dots per inch.
    :type dpi: int

    :return: ``(fig, ax)`` tuple.
    :rtype: Tuple[matplotlib.figure.Figure, matplotlib.axes.Axes]

    :raises ValueError: If ``x3d`` is not 3-dimensional, or ``nrows * ncols != C``.

    ----

    * **代码示例 | Example**

    .. code-block:: python

        from spikingjelly import visualizing
        import numpy as np
        from matplotlib import pyplot as plt

        C = 48
        W = 8
        H = 8
        spikes = (np.random.rand(C, W, H) > 0.8).astype(float)
        fig, ax = visualizing.plot_2d_feature_map(
            x3d=spikes, nrows=6, ncols=8, space=2, title="Spiking Feature Maps", dpi=200
        )
        plt.show()

    .. image:: ../_static/API/visualizing/plot_2d_feature_map.*
        :width: 100%
    """
    x3d = _to_numpy(x3d)
    if x3d.ndim != 3:
        raise ValueError(f"Expected 3D array, got {x3d.ndim}D array instead")

    C = x3d.shape[0]
    if nrows * ncols != C:
        raise ValueError(
            f"nrows * ncols ({nrows} * {ncols} = {nrows * ncols}) != C ({C})"
        )

    h = x3d.shape[1]
    w = x3d.shape[2]
    y = np.ones(shape=[(h + space) * nrows, (w + space) * ncols]) * x3d.max().item()
    index = 0
    for i in range(space // 2, y.shape[0], h + space):
        for j in range(space // 2, y.shape[1], w + space):
            y[i : i + h, j : j + w] = x3d[index]
            index += 1
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.set_title(title)
    ax.imshow(y, cmap="gray")

    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    return fig, ax
