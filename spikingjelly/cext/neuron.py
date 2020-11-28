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





