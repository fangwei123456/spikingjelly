import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Callable

from .. import base
from .net_config import detach_net


def fptt_online_training_init_w_ra(optimizer: torch.optim.Optimizer) -> list:
    """
    .. _fptt_online_training_init_w_ra-en:

    * **English**

    Used to initialize ``w_ra`` for :func:`fptt_online_training` .

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :return: a list containing running averages of parameters
    :rtype: list
    """
    w_ra = []
    for item in optimizer.param_groups:
        for w in item['params']:
            w_ra.append(w.data)

    return w_ra


def fptt_online_training(model: nn.Module, optimizer: torch.optim.Optimizer, x_seq: torch.Tensor, target_seq: torch.Tensor, f_loss_t: Callable, alpha: float, w_ra: list) -> None:
    """
    .. _fptt_online_training-en:

    * **English**

    The FPTT online learning method proposed by `Training Recurrent Neural Networks via Forward Propagation Through Time <https://proceedings.mlr.press/v139/kag21a.html>`_ and used for SNN in `Accurate online training of dynamical spiking neural networks through Forward Propagation Through Time <https://arxiv.org/abs/2112.11231>`_ .

    :param model: the neural network
    :type model: nn.Module

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :param x_seq: the input sequence
    :type x_seq: torch.Tensor

    :param target_seq: the output sequence
    :type target_seq: torch.Tensor

    :param f_loss_t: the loss function, which should has the formulation of ``def f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param alpha: the hyper-parameter
    :type alpha: float

    :param w_ra: the running average of params, which can be initialized by :class:`spikingjelly.activation_based.functional.fptt_online_training_init_w_ra`
    :type w_ra: list

    ----

    * **代码示例 | Example**

    .. code-block:: python

        from spikingjelly.activation_based import neuron

        net = nn.Sequential(
            nn.Linear(8, 4),
            neuron.IFNode(),
            nn.Linear(4, 2),
            neuron.IFNode()
        )

        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        T = 4
        N = 2
        w_ra = fptt_online_training_init_w_ra(optimizer)
        for epoch in range(2):

            x_seq = torch.rand([T, N, 8])
            target_seq = torch.rand([T, N, 2])

            fptt_online_training(model=net, optimizer=optimizer, x_seq=x_seq, target_seq=target_seq, f_loss_t=F.mse_loss, alpha=0.1, w_ra=w_ra)
            functional.reset_net(net)
    """
    T = x_seq.shape[0]

    grad__l_t_last__to__w_t = []

    for item in optimizer.param_groups:
        for w in item['params']:
            grad__l_t_last__to__w_t.append(0.)

    for t in range(T):
        optimizer.zero_grad()

        y_t = model(x_seq[t])
        loss_t = f_loss_t(y_t, target_seq[t])
        loss_reg = 0.
        i = 0
        for item in optimizer.param_groups:
            for w in item['params']:
                loss_reg = loss_reg + F.mse_loss(w, w_ra[i] + grad__l_t_last__to__w_t[i] / (2. * alpha))
                i += 1

        loss_reg = loss_reg * (alpha / 2.)

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
                    for w in item['params']:
                        grad__l_t_last__to__w_t[i] = w.grad
                        w_ra[i] = (w_ra[i] + w) / 2. - w.grad / (2. * alpha)
                        i += 1
        optimizer.zero_grad()

        # recover hidden states
        i = 0
        for m in model.modules():
            if isinstance(m, base.MemoryModule):
                m._memories = states[i]
                i += 1


def ottt_online_training(model: nn.Module, optimizer: torch.optim.Optimizer, x_seq: torch.Tensor, target_seq: torch.Tensor, f_loss_t: Callable, online: bool) -> None:
    """
    .. _ottt_online_training-en:

    * **English**

    The OTTT online training method is proposed by `Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>`_.
    This function can also be used for SLTT training method proposed by `Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks <https://openaccess.thecvf.com/content/ICCV2023/html/Meng_Towards_Memory-_and_Time-Efficient_Backpropagation_for_Training_Spiking_Neural_Networks_ICCV_2023_paper.html>`_ .

    :param model: the neural network
    :type model: nn.Module

    :param optimizer: the optimizer for the network
    :type optimizer: torch.optim.Optimizer

    :param x_seq: the input sequence
    :type x_seq: torch.Tensor

    :param target_seq: the output sequence
    :type target_seq: torch.Tensor

    :param f_loss_t: the loss function, which should has the formulation of ``def f_loss_t(x_t, y_t) -> torch.Tensor``
    :type f_loss_t: Callable

    :param online: whether online update parameters or accumulate gradients through time steps
    :type online: bool

    ----

    * **代码示例 | Example**

    .. code-block:: python

        from spikingjelly.activation_based import neuron, layer, functional

        net = layer.OTTTSequential(
            nn.Linear(8, 4),
            neuron.OTTTLIFNode(),
            nn.Linear(4, 2),
            neuron.LIFNode()
        )

        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)

        T = 4
        N = 2
        online = True
        for epoch in range(2):

            x_seq = torch.rand([N, T, 8])
            target_seq = torch.rand([N, T, 2])

            functional.ottt_online_training(model=net, optimizer=optimizer, x_seq=x_seq, target_seq=target_seq, f_loss_t=F.mse_loss, online=online)
            functional.reset_net(net)
    """

    # input x_seq/target_seq: [B, T, ...]
    # transpose to [T, B, ...]
    x_seq = x_seq.transpose(0, 1)
    target_seq = target_seq.transpose(0, 1)
    T = x_seq.shape[0]

    batch_loss = 0.
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

