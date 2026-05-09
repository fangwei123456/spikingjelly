import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from .. import base
from .net_config import detach_net


__all__ = [
    "fptt_online_training_init_w_ra",
    "fptt_online_training",
    "ottt_online_training",
]


def fptt_online_training_init_w_ra(optimizer: torch.optim.Optimizer) -> list:
    """
    **API Language:**
    :ref:`中文 <fptt_online_training_init_w_ra-cn>` | :ref:`English <fptt_online_training_init_w_ra-en>`

    ----

    .. _fptt_online_training_init_w_ra-cn:

    * **中文**

    初始化 :func:`fptt_online_training` 使用的 ``w_ra`` 列表。返回列表中的元素顺序与
    ``optimizer.param_groups`` 中参数的遍历顺序一致，列表元素是各参数当前的 ``w.data``。

    :param optimizer: 网络使用的优化器
    :type optimizer: torch.optim.Optimizer

    :return: 与优化器参数顺序对齐的运行平均列表
    :rtype: list

    ----

    .. _fptt_online_training_init_w_ra-en:

    * **English**

    Initialize the ``w_ra`` list used by :func:`fptt_online_training`. The
    returned list follows the traversal order of parameters in
    ``optimizer.param_groups`` and stores the current ``w.data`` of each
    parameter.

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :return: a list containing running averages of parameters
    :rtype: list
    """
    w_ra = []
    for item in optimizer.param_groups:
        for w in item["params"]:
            w_ra.append(w.data)

    return w_ra


def fptt_online_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    target_seq: torch.Tensor,
    f_loss_t: Callable,
    alpha: float,
    w_ra: list,
) -> None:
    """
    **API Language:**
    :ref:`中文 <fptt_online_training-cn>` | :ref:`English <fptt_online_training-en>`

    ----

    .. _fptt_online_training-cn:

    * **中文**

    使用 FPTT 在线训练方法沿 ``x_seq.shape[0]`` 对应的时间维逐步训练网络。每个时间步都会执行一次
    前向、损失计算、参数更新与 ``detach_net``，并对
    :class:`spikingjelly.activation_based.base.MemoryModule` 的内部状态进行保存和恢复。

    :param model: 神经网络
    :type model: nn.Module

    :param optimizer: 网络使用的优化器
    :type optimizer: torch.optim.Optimizer

    :param x_seq: 输入序列
    :type x_seq: torch.Tensor

    :param target_seq: 目标序列
    :type target_seq: torch.Tensor

    :param f_loss_t: 单个时间步的损失函数，调用形式应为 ``f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param alpha: FPTT 使用的超参数
    :type alpha: float

    :param w_ra: 由 :func:`fptt_online_training_init_w_ra` 初始化的运行平均列表
    :type w_ra: list

    :return: None
    :rtype: None

    ----

    .. _fptt_online_training-en:

    * **English**

    The FPTT online learning method proposed by `Training Recurrent Neural Networks via Forward Propagation Through Time <https://proceedings.mlr.press/v139/kag21a.html>`_ and used for SNN in `Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time <https://arxiv.org/abs/2112.11231>`_ .
    This function iterates over the time dimension ``x_seq.shape[0]`` and
    performs forward, loss computation, parameter update, and ``detach_net`` at
    every time step. It also stores and restores the internal states of
    :class:`spikingjelly.activation_based.base.MemoryModule`.

    :param model: the neural network
    :type model: nn.Module

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :param x_seq: the input sequence
    :type x_seq: torch.Tensor

    :param target_seq: the target sequence
    :type target_seq: torch.Tensor

    :param f_loss_t: the loss function, which should have the formulation of
        ``def f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param alpha: the hyper-parameter
    :type alpha: float

    :param w_ra: the running average of parameters, which can be initialized by
        :func:`fptt_online_training_init_w_ra`
    :type w_ra: list

    :return: None
    :rtype: None

    ----

    * **代码示例 | Example**

    .. code-block:: python

        from spikingjelly.activation_based import neuron

        net = nn.Sequential(
            nn.Linear(8, 4), neuron.IFNode(), nn.Linear(4, 2), neuron.IFNode()
        )

        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        T = 4
        N = 2
        w_ra = fptt_online_training_init_w_ra(optimizer)
        for epoch in range(2):
            x_seq = torch.rand([T, N, 8])
            target_seq = torch.rand([T, N, 2])

            fptt_online_training(
                model=net,
                optimizer=optimizer,
                x_seq=x_seq,
                target_seq=target_seq,
                f_loss_t=F.mse_loss,
                alpha=0.1,
                w_ra=w_ra,
            )
            functional.reset_net(net)
    """
    T = x_seq.shape[0]

    grad__l_t_last__to__w_t = []

    for item in optimizer.param_groups:
        for w in item["params"]:
            grad__l_t_last__to__w_t.append(0.0)

    for t in range(T):
        optimizer.zero_grad()

        y_t = model(x_seq[t])
        loss_t = f_loss_t(y_t, target_seq[t])
        loss_reg = 0.0
        i = 0
        for item in optimizer.param_groups:
            for w in item["params"]:
                loss_reg = loss_reg + F.mse_loss(
                    w, w_ra[i] + grad__l_t_last__to__w_t[i] / (2.0 * alpha)
                )
                i += 1

        loss_reg = loss_reg * (alpha / 2.0)

        loss = loss_t + loss_reg
        loss.backward()

        # update params
        optimizer.step()
        detach_net(model)

        # store hidden states
        states = []
        i = 0
        for m in model.modules():
            if isinstance(m, base.MemoryModule):
                states.append(copy.deepcopy(m._memories))
                i += 1

        # update w_ra
        optimizer.zero_grad()
        if t < T - 1:
            y_t = model(x_seq[t])
            loss_t = f_loss_t(y_t, target_seq[t])
            loss_t.backward()
            with torch.no_grad():
                i = 0
                for item in optimizer.param_groups:
                    for w in item["params"]:
                        grad__l_t_last__to__w_t[i] = w.grad
                        w_ra[i] = (w_ra[i] + w) / 2.0 - w.grad / (2.0 * alpha)
                        i += 1
        optimizer.zero_grad()

        # recover hidden states
        i = 0
        for m in model.modules():
            if isinstance(m, base.MemoryModule):
                m._memories = states[i]
                i += 1


def ottt_online_training(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    x_seq: torch.Tensor,
    target_seq: torch.Tensor,
    f_loss_t: Callable,
    online: bool,
) -> None:
    """
    **API Language:**
    :ref:`中文 <ottt_online_training-cn>` | :ref:`English <ottt_online_training-en>`

    ----

    .. _ottt_online_training-cn:

    * **中文**

    使用 OTTT 在线训练方法训练网络，也可用于文献中提到的 SLTT 训练。函数会先将
    ``x_seq`` 和 ``target_seq`` 从 ``[B, T, ...]`` 转置为 ``[T, B, ...]``，然后沿时间维逐步执行
    前向与反向传播。若 ``online`` 为 ``True``，则每个时间步都会执行一次参数更新；否则先累积整段序列的梯度，
    再在最后统一更新。

    :param model: 神经网络
    :type model: nn.Module

    :param optimizer: 网络使用的优化器
    :type optimizer: torch.optim.Optimizer

    :param x_seq: 输入序列，形状为 ``[B, T, ...]``
    :type x_seq: torch.Tensor

    :param target_seq: 目标序列，形状为 ``[B, T, ...]``
    :type target_seq: torch.Tensor

    :param f_loss_t: 单个时间步的损失函数，调用形式应为 ``f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param online: 是否在每个时间步在线更新参数；若为 ``False``，则仅在整段序列结束后更新一次
    :type online: bool

    :return: ``(batch_loss, y_all)``，其中 ``batch_loss`` 是各时间步损失之和，
        ``y_all`` 是形状为 ``[B, T, ...]`` 的按时间堆叠且已 detach 的输出
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _ottt_online_training-en:

    * **English**

    The OTTT online training method is proposed by `Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>`_.
    This function can also be used for SLTT training method proposed by `Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks <https://openaccess.thecvf.com/content/ICCV2023/html/Meng_Towards_Memory-_and_Time-Efficient_Backpropagation_for_Training_Spiking_Neural_Networks_ICCV_2023_paper.html>`_ .
    It first transposes ``x_seq`` and ``target_seq`` from ``[B, T, ...]`` to
    ``[T, B, ...]`` and then runs forward and backward passes step by step along
    the time dimension. If ``online`` is ``True``, the optimizer updates
    parameters at every time step; otherwise, gradients are accumulated through
    the whole sequence and applied once at the end.

    :param model: the neural network
    :type model: nn.Module

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :param x_seq: the input sequence with ``shape=[B, T, ...]``
    :type x_seq: torch.Tensor

    :param target_seq: the target sequence with ``shape=[B, T, ...]``
    :type target_seq: torch.Tensor

    :param f_loss_t: the loss function, which should have the formulation of
        ``def f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param online: whether to update parameters online at each time step or to
        accumulate gradients through time steps
    :type online: bool

    :return: ``(batch_loss, y_all)``, where ``batch_loss`` is the sum of per-step
        losses and ``y_all`` is the detached stacked output with
        ``shape=[B, T, ...]``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    * **代码示例 | Example**

    .. code-block:: python

        from spikingjelly.activation_based import neuron, layer, functional

        net = layer.OTTTSequential(
            nn.Linear(8, 4), neuron.OTTTLIFNode(), nn.Linear(4, 2), neuron.LIFNode()
        )

        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        T = 4
        N = 2
        online = True
        for epoch in range(2):
            x_seq = torch.rand([N, T, 8])
            target_seq = torch.rand([N, T, 2])

            functional.ottt_online_training(
                model=net,
                optimizer=optimizer,
                x_seq=x_seq,
                target_seq=target_seq,
                f_loss_t=F.mse_loss,
                online=online,
            )
            functional.reset_net(net)
    """

    # input x_seq/target_seq: [B, T, ...]
    # transpose to [T, B, ...]
    x_seq = x_seq.transpose(0, 1)
    target_seq = target_seq.transpose(0, 1)
    T = x_seq.shape[0]

    batch_loss = 0.0
    y_all = []
    if not online:
        optimizer.zero_grad()
    for t in range(T):
        if online:
            optimizer.zero_grad()

        y_t = model(x_seq[t])
        loss = f_loss_t(y_t, target_seq[t].contiguous())

        loss.backward()

        # update params
        if online:
            optimizer.step()

        batch_loss += loss.data
        y_all.append(y_t.detach())

    if not online:
        optimizer.step()

    # y_all: [B, T, ...]
    y_all = torch.stack(y_all, dim=1)

    return batch_loss, y_all
