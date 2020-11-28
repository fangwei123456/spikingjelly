import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def lif_hard_reset_forward(x: torch.Tensor, v:torch.Tensor, v_threshold: float, v_reset: float, tau: float):
    '''
    * :ref:`API in English <lif_hard_reset_forward-en>`

    .. _lif_hard_reset_forward-cn:

    :param x: 给神经元的输入
    :type x: torch.Tensor
    :param v: 神经元的膜电位
    :type v: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param v_reset: 神经元的重置电压
    :type v_reset: float
    :param tau: LIF神经元的膜时间常数
    :type tau: float
    :return: ``(spike, v_next)``，其中 ``spike`` 是释放的脉冲，``v_next`` 是经过充电、放电、重置后的电压
    :rtype: tuple

    对LIF神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。更新的方程为

    .. math::
        H_{t} & = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    * :ref:`中文API <lif_hard_reset_forward-cn>`

    .. _lif_hard_reset_forward-en:

    :param x: the input to the neuron
    :type x: torch.Tensor
    :param v: the membrane potential of the neuron
    :type v: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param v_reset: the reset voltage of the neuron
    :type v_reset: float
    :param tau: the membrane time constant of the LIF neuron
    :type tau: float
    :return: ``(spike, v_next)``,where ``spike`` is the output spike,and ``v_next`` is the membrane potential of the LIF neuron in next time step
    :rtype: tuple

    Update the membrane potential of the LIF neuron by one time step with hard reset. The update is calculated by

    .. math::
        H_{t} & = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}
    '''

    pass

def lif_soft_reset_forward(x: torch.Tensor, v:torch.Tensor, v_threshold: float, tau: float):
    '''
    * :ref:`API in English <lif_soft_reset_forward-en>`

    .. _lif_soft_reset_forward-cn:

    :param x: 给神经元的输入
    :type x: torch.Tensor
    :param v: 神经元的膜电位
    :type v: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param tau: LIF神经元的膜时间常数
    :type tau: float
    :return: ``(spike, v_next)``，其中 ``spike`` 是释放的脉冲，``v_next`` 是经过充电、放电、重置后的电压
    :rtype: tuple

    对LIF神经元进行单步的电压更新，其中电压重置方式是软重置(soft reset)。更新的方程为

    .. math::
        H_{t} & = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    * :ref:`中文API <lif_soft_reset_forward-cn>`

    .. _lif_soft_reset_forward-en:

    :param x: the input to the neuron
    :type x: torch.Tensor
    :param v: the membrane potential of the neuron
    :type v: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param tau: the membrane time constant of the LIF neuron
    :type tau: float
    :return: ``(spike, v_next)``,where ``spike`` is the output spike,and ``v_next`` is the membrane potential of the LIF neuron in next time step
    :rtype: tuple

    Update the membrane potential of the LIF neuron by one time step with soft reset. The update is calculated by

    .. math::
        H_{t} & = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}
    '''

    pass


def lif_hard_reset_backward(grad_spike: torch.Tensor, grad_v_next:torch.Tensor, v_threshold: float, v_reset: float, tau: float):
    '''
    * :ref:`API in English <lif_hard_reset_backward-en>`

    .. _lif_hard_reset_backward-cn:

    :param grad_spike: 损失对脉冲的梯度
    :type grad_spike: torch.Tensor
    :param grad_v_next: 损失对LIF神经元膜电位的梯度
    :type grad_v_next: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param v_reset: 神经元的重置电压
    :type v_reset: float
    :param tau: LIF神经元的膜时间常数
    :type tau: float
    :return: ``(grad_x, grad_v)``，其中 ``grad_x`` 是损失对输入 ``x`` 的梯度，``grad_v`` 是损失对上一个时刻LIF神经元膜电位的梯度
    :rtype: tuple

    :ref:`lif_hard_reset_forward-cn` 的反向传播。梯度的计算按照

    .. math::


    * :ref:`中文API <lif_hard_reset_backward-cn>`

    .. _lif_hard_reset_backward-en:

    :param grad_x: the input to the neuron
    :type grad_x: torch.Tensor
    :param grad_v_next: the membrane potential of the neuron
    :type grad_v_next: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param v_reset: the reset voltage of the neuron
    :type v_reset: float
    :param tau: the membrane time constant of the LIF neuron
    :type tau: float
    :return: ``(spike, v_next)``,where ``spike`` is the output spike,and ``v_next`` is the membrane potential of the LIF neuron in next time step
    :rtype: tuple

    The backward of :ref:`lif_hard_reset_forward-en`. The gradient is calculated by

    .. math::

    '''

    pass




