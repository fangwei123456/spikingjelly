import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import _C_neuron as cext_neuron

def hard_reset_forward_template(x: torch.Tensor, v:torch.Tensor, v_threshold: float, v_reset: float, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_forward_template-en>`

    .. _hard_reset_forward_template-cn:

    :param x: 给神经元的输入
    :type x: torch.Tensor
    :param v: 神经元的膜电位
    :type v: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param v_reset: 神经元的重置电压
    :type v_reset: float
    :return: ``(h, spike, v_next)``，其中 ``h`` 是经过充电后的电压，``spike`` 是释放的脉冲，`v_nest` 是重置后的电压
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。

    * :ref:`中文API <hard_reset_forward_template-cn>`

    .. _hard_reset_forward_template-en:

    :param x: the input to the neuron
    :type x: torch.Tensor
    :param v: the membrane potential of the neuron
    :type v: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param v_reset: the reset voltage of the neuron
    :type v_reset: float
    :return: ``(h, spike, v_next)``, where ``h`` is the membrane potential after charging, ``spike`` is the output spike,and ``v_next`` is the membrane potential of the neuron in next time step
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with hard reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters.
    '''

    pass

def soft_reset_forward_template(x: torch.Tensor, v:torch.Tensor, v_threshold: float, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_forward_template-en>`

    .. _soft_reset_forward_template-cn:

    :param x: 给神经元的输入
    :type x: torch.Tensor
    :param v: 神经元的膜电位
    :type v: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :return: ``(h, spike, v_next)``，其中 ``h`` 是经过充电后的电压，``spike`` 是释放的脉冲，`v_nest` 是重置后的电压
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是软重置(soft reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。

    * :ref:`中文API <soft_reset_forward_template-cn>`

    .. _soft_reset_forward_template-en:

    :param x: the input to the neuron
    :type x: torch.Tensor
    :param v: the membrane potential of the neuron
    :type v: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :return: ``(h, spike, v_next)``, where ``h`` is the membrane potential after charging, ``spike`` is the output spike,and ``v_next`` is the membrane potential of the neuron in next time step
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with soft reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters.
    '''

    pass

def LIF_hard_reset_forward(x: torch.Tensor, v:torch.Tensor, v_threshold: float, v_reset: float, tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_forward-en>`

    .. _LIF_hard_reset_forward-cn:

    :param tau: LIF神经元的膜时间常数
    :type tau: float

    其余的参数参见 :ref:`hard_reset_forward_template <hard_reset_forward_template-cn>`。

    对LIF神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。充电的方程为

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    * :ref:`中文API <LIF_hard_reset_forward-cn>`

    .. _LIF_hard_reset_forward-en:

    :param tau: the membrane time constant of the LIF neuron
    :type tau: float

    See :ref:`hard_reset_forward_template <hard_reset_forward_template-en>` for more details about other args。

    Update the membrane potential of the LIF neuron by one time step with hard reset. The charging equation is

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    '''
    return cext_neuron.LIF_hard_reset_forward(x, v, v_threshold, v_reset, tau)

def LIF_soft_reset_forward(x: torch.Tensor, v:torch.Tensor, v_threshold: float, tau: float):
    '''
    * :ref:`API in English <LIF_soft_reset_forward-en>`

    .. _LIF_soft_reset_forward-cn:

    :param tau: LIF神经元的膜时间常数
    :type tau: float

    其余的参数参见 :ref:`soft_reset_forward_template <soft_reset_forward_template-cn>`。

    对LIF神经元进行单步的电压更新，其中电压重置方式是软重置(soft reset)。充电的方程为

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    * :ref:`中文API <LIF_soft_reset_forward-cn>`

    .. _LIF_soft_reset_forward-en:

    :param tau: the membrane time constant of the LIF neuron
    :type tau: float

    See :ref:`soft_reset_forward_template <soft_reset_forward_template-en>` for more details about other args。

    Update the membrane potential of the LIF neuron by one time step with soft reset. The charging equation is

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    '''

    raise NotImplementedError

def hard_reset_backward_template(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, h: torch.Tensor, spike: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_backward_template-en>`

    .. _hard_reset_backward_template-cn:

    :param grad_spike: 损失对脉冲的梯度
    :type grad_spike: torch.Tensor
    :param grad_v_next: 损失对神经元膜电位的梯度
    :type grad_v_next: torch.Tensor
    :param h: 充电后的膜电位
    :type h: torch.Tensor
    :param spike: 释放的脉冲
    :type spike: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param v_reset: 神经元的重置电压
    :type v_reset: float
    :param alpha: 梯度替代函数的参数
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(grad_x, grad_v)``，其中 ``grad_x`` 是损失对输入 ``x`` 的梯度，``grad_v`` 是损失对上一个时刻神经元膜电位的梯度
    :rtype: tuple

    :ref:`hard_reset_forward_template <hard_reset_forward_template-cn>` 的反向传播。梯度的计算按照

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - S_{t} + (V_{reset} - H_{t})\\frac{\\partial S_{t}}{\\partial H_{t}}

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}
    * :ref:`中文API <hard_reset_backward_template-cn>`

    .. _hard_reset_backward_template-en:

    :param grad_x: the input to the neuron
    :type grad_x: torch.Tensor
    :param grad_v_next: the membrane potential of the neuron
    :type grad_v_next: torch.Tensor
    :param h: the membrane potential after charging
    :type h: torch.Tensor
    :param spike: the output spikes
    :type spike: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param v_reset: the reset voltage of the neuron
    :type v_reset: float
    :param alpha: argument of the gradient surrogate function
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(grad_x, grad_v)``, where ``grad_x`` is the gradient of ``x`` and ``grad_v`` is the gradient of membrane potential at last time step
    :rtype: tuple

    The backward of :ref:`hard_reset_forward_template <hard_reset_forward_template-en>`. The gradients are calculated by

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - S_{t} + (V_{reset} - H_{t})\\frac{\\partial S_{t}}{\\partial H_{t}}

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}


    '''

    pass

def soft_reset_backward_template(grad_spike: torch.Tensor, grad_v_next:torch.Tensor, h: torch.Tensor, spike: torch.Tensor, v_threshold: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_backward_template-en>`

    .. _soft_reset_backward_template-cn:

    :param grad_spike: 损失对脉冲的梯度
    :type grad_spike: torch.Tensor
    :param grad_v_next: 损失对神经元膜电位的梯度
    :type grad_v_next: torch.Tensor
    :param h: 充电后的膜电位
    :type h: torch.Tensor
    :param spike: 释放的脉冲
    :type spike: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param alpha: 梯度替代函数的参数
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(grad_x, grad_v)``，其中 ``grad_x`` 是损失对输入 ``x`` 的梯度，``grad_v`` 是损失对上一个时刻神经元膜电位的梯度
    :rtype: tuple

    :ref:`soft_reset_forward_template <soft_reset_forward_template-cn>` 的反向传播。梯度的计算按照

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - V_{threshold} \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}

    * :ref:`中文API <soft_reset_backward_template-cn>`

    .. _soft_reset_backward_template-en:

    :param grad_x: the input to the neuron
    :type grad_x: torch.Tensor
    :param grad_v_next: the membrane potential of the neuron
    :type grad_v_next: torch.Tensor
    :param h: the membrane potential after charging
    :type h: torch.Tensor
    :param spike: the output spikes
    :type spike: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param alpha: argument of the gradient surrogate function
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(grad_x, grad_v)``, where ``grad_x`` is the gradient of ``x`` and ``grad_v`` is the gradient of membrane potential at last time step
    :rtype: tuple

    The backward of :ref:`soft_reset_forward_template <soft_reset_forward_template-en>`. The gradients are calculated by

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - V_{threshold} \\Theta'(H_{t} - V_{threshold})

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}


    '''

    pass

def LIF_hard_reset_backward(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, h: torch.Tensor, spike: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_backward-en>`

    .. _LIF_hard_reset_backward-cn:

    :param tau: LIF神经元的膜时间常数
    :type tau: float

    其余的参数参见 :ref:`hard_reset_backward_template<hard_reset_backward_template-cn>`。

    :ref:`LIF_hard_reset_forward <LIF_hard_reset_forward-cn>` 的反向传播。梯度的计算按照

    .. math::
        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}



    * :ref:`中文API <LIF_hard_reset_backward-cn>`

    .. _LIF_hard_reset_backward-en:

    :param tau: the membrane time constant of the LIF neuron
    :type tau: float

    See :ref:`hard_reset_forward_template <hard_reset_forward_template-en>` for more details about other args。

    The backward of :ref:`LIF_hard_reset_forward <LIF_hard_reset_forward-en>`. The gradients are calculated by

    .. math::

        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}

    '''
    return cext_neuron.LIF_hard_reset_backward(grad_spike, grad_v_next, h, spike, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, tau)

def hard_reset_bptt_template(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, h: torch.Tensor, spike: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_bptt_template-en>`

    .. _hard_reset_bptt_template-cn:

    :param grad_spike: 损失对脉冲的梯度
    :type grad_spike: torch.Tensor
    :param grad_v_next: 损失对 :math:`V_{T-1}` 的梯度
    :type grad_v_next: torch.Tensor
    :param h: 充电后的膜电位
    :type h: torch.Tensor
    :param spike: 释放的脉冲
    :type spike: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param v_reset: 神经元的重置电压
    :type v_reset: float
    :param alpha: 梯度替代函数的参数
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``grad_x``，``grad_x`` 是损失对输入 ``x`` 的梯度
    :rtype: torch.Tensor

    :ref:`hard_reset_forward_template <hard_reset_forward_template-cn>` 的BPTT。记

    .. math::
        M_t &= \\frac{\\partial H_{t+1}}{\\partial V_{t}}\\left[(1-S_{t}+(V_{reset} - H_{t})\\Theta'(H_{t} - V_{threshold})\\right], 

        M_{T-1} &= \\frac{\\partial L}{\\partial V_{T-1}}\\left[(1-S_{T-1}+(V_{reset} - H_{T-1})\\Theta'(H_{T-1} - V_{threshold})\\right],

        N_t &= \\frac{\\partial L}{\\partial S_{t}}\\Theta'(H_{t} - V_{threshold}),

    其中 :math:`t \\in [0,T-1]`。梯度的计算按照

    .. math::
        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial H_{t}}{\\partial X_{t}}\\left[N_t+\\sum_{i=t+1}^{T-1}N_{i}\\left(\\prod_{j=t}^{i-1}M_j\\right)+\\prod_{i=t}^{T-1}M_i\\right]

        \\frac{\\partial L}{\\partial V_{init}} &= \\frac{\\partial H_{0}}{\\partial V_{init}}\\left[N_0+\\sum_{i=1}^{T-1}N_{i}\\left(\\prod_{j=0}^{i-1}M_j\\right)+\\prod_{i=0}^{T-1}M_i\\right]

    实际使用迭代的计算方式：

    .. math::
            \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

            \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold})

            \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - S_{t} + (V_{reset} - H_{t})\\frac{\\partial S_{t}}{\\partial H_{t}}

            \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

            \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}

    其中 :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, \\frac{\\partial V_{t}}{\\partial H_{t}}, \\frac{\\partial H_{t}}{\\partial X_{t}}, \\frac{\\partial H_{t}}{\\partial V_{t-1}}` 可以先并行求解。

    * :ref:`中文API <hard_reset_bptt_template-cn>`

    .. _hard_reset_bptt_template-en:

    :param grad_spike: the gradient of output spikes
    :type grad_spike: torch.Tensor
    :param grad_v_next: the gradient of :math:`V_{T-1}`
    :type grad_v_next: torch.Tensor
    :param h: the membrane potential after charging
    :type h: torch.Tensor
    :param spike: the output spikes
    :type spike: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param v_reset: the reset voltage of the neuron
    :type v_reset: float
    :param alpha: argument of the gradient surrogate function
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``grad_x``, where ``grad_x`` is the gradient of ``x``
    :rtype: torch.Tensor

    The BPTT of :ref:`hard_reset_forward_template <hard_reset_forward_template-en>`. Let

    .. math::
        M_t &= \\frac{\\partial H_{t+1}}{\\partial V_{t}}\\left[(1-S_{t}+(V_{reset} - H_{t})\\Theta'(H_{t} - V_{threshold})\\right], 

        M_{T-1} &= \\frac{\\partial L}{\\partial V_{T-1}}\\left[(1-S_{T-1}+(V_{reset} - H_{T-1})\\Theta'(H_{T-1} - V_{threshold})\\right],

        N_t &= \\frac{\\partial L}{\\partial S_{t}}\\Theta'(H_{t} - V_{threshold}),
    
    where :math:`t \\in [0,T-1]`. The gradients are calculated by

    .. math::
        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial H_{t}}{\\partial X_{t}}\\left[N_t+\\sum_{i=t+1}^{T-1}N_{i}\\left(\\prod_{j=t}^{i-1}M_j\\right)+\\prod_{i=t}^{T-1}M_i\\right]

        \\frac{\\partial L}{\\partial V_{init}} &= \\frac{\\partial H_{0}}{\\partial V_{init}}\\left[N_0+\\sum_{i=1}^{T-1}N_{i}\\left(\\prod_{j=0}^{i-1}M_j\\right)+\\prod_{i=0}^{T-1}M_i\\right]

    '''
    raise NotImplementedError

def soft_reset_bptt_template(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, h: torch.Tensor, spike: torch.Tensor, v_threshold: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_bptt_template-en>`

    .. _soft_reset_bptt_template-cn:

    :param grad_spike: 损失对脉冲的梯度
    :type grad_spike: torch.Tensor
    :param grad_v_next: 损失对 :math:`V_{T-1}` 的梯度
    :type grad_v_next: torch.Tensor
    :param h: 充电后的膜电位
    :type h: torch.Tensor
    :param spike: 释放的脉冲
    :type spike: torch.Tensor
    :param v_threshold: 神经元的阈值电压
    :type v_threshold: float
    :param alpha: 梯度替代函数的参数
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``grad_x``，``grad_x`` 是损失对输入 ``x`` 的梯度
    :rtype: torch.Tensor

    :ref:`soft_reset_forward_template <soft_reset_forward_template-cn>` 的BPTT。记

    .. math::
        M_t &= \\frac{\\partial H_{t+1}}{\\partial V_{t}}\\left[1 - V_{threshold} \\Theta'(H_{t} - V_{threshold})\\right],

        M_{T-1} &= \\frac{\\partial L}{\\partial V_{T-1}}\\left[1 - V_{threshold} \\Theta'(H_{T-1} - V_{threshold})\\right],

        N_t &= \\frac{\\partial L}{\\partial S_{t}}\\Theta'(H_{t} - V_{threshold}),

    其中 :math:`t \\in [0,T-1]`。梯度的计算按照

    .. math::
        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial H_{t}}{\\partial X_{t}}\\left[N_t+\\sum_{i=t+1}^{T-1}N_{i}\\left(\\prod_{j=t}^{i-1}M_j\\right)+\\prod_{i=t}^{T-1}M_i\\right]

        \\frac{\\partial L}{\\partial V_{init}} &= \\frac{\\partial H_{0}}{\\partial V_{init}}\\left[N_0+\\sum_{i=1}^{T-1}N_{i}\\left(\\prod_{j=0}^{i-1}M_j\\right)+\\prod_{i=0}^{T-1}M_i\\right]

    * :ref:`中文API <soft_reset_bptt_template-cn>`

    .. _soft_reset_bptt_template-en:

    :param grad_spike: the gradient of output spikes
    :type grad_spike: torch.Tensor
    :param grad_v_next: the gradient of :math:`V_{T-1}`
    :type grad_v_next: torch.Tensor
    :param h: the membrane potential after charging
    :type h: torch.Tensor
    :param spike: the output spikes
    :type spike: torch.Tensor
    :param v_threshold: the threshold voltage of the neuron
    :type v_threshold: float
    :param alpha: argument of the gradient surrogate function
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``grad_x``, where ``grad_x`` is the gradient of ``x``
    :rtype: torch.Tensor

    The BPTT of :ref:`soft_reset_forward_template <soft_reset_forward_template-en>`. Let

    .. math::
        M_t &= \\frac{\\partial H_{t+1}}{\\partial V_{t}}\\left[1 - V_{threshold} \\Theta'(H_{t} - V_{threshold})\\right],

        M_{T-1} &= \\frac{\\partial L}{\\partial V_{T-1}}\\left[1 - V_{threshold} \\Theta'(H_{T-1} - V_{threshold})\\right],

        N_t &= \\frac{\\partial L}{\\partial S_{t}}\\Theta'(H_{t} - V_{threshold}),

    where :math:`t \\in [0,T-1]`. The gradients are calculated by

    .. math::
        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial H_{t}}{\\partial X_{t}}\\left[N_t+\\sum_{i=t+1}^{T-1}N_{i}\\left(\\prod_{j=t}^{i-1}M_j\\right)+\\prod_{i=t}^{T-1}M_i\\right]

        \\frac{\\partial L}{\\partial V_{init}} &= \\frac{\\partial H_{0}}{\\partial V_{init}}\\left[N_0+\\sum_{i=1}^{T-1}N_{i}\\left(\\prod_{j=0}^{i-1}M_j\\right)+\\prod_{i=0}^{T-1}M_i\\right]

    '''
    raise NotImplementedError

class LIFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, tau):
        if v_reset is None:
            raise NotImplementedError
        h, spike, v_next = cext_neuron.LIF_hard_reset_forward(x, v, v_threshold, v_reset, tau)
        if x.requires_grad:
            ctx.save_for_backward(h, spike)
            ctx.v_threshold = v_threshold
            ctx.v_reset = v_reset
            ctx.alpha = alpha
            ctx.detach_reset = detach_reset
            ctx.grad_surrogate_function_index = grad_surrogate_function_index
            ctx.tau = tau
        return h, spike, v_next

    @staticmethod
    def backward(ctx, grad_h, grad_spike, grad_v_next):
        grad_x, grad_v = cext_neuron.LIF_hard_reset_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.v_threshold, ctx.v_reset, ctx.alpha, ctx.detach_reset, ctx.grad_surrogate_function_index, ctx.tau)
        return grad_x, grad_v, None, None, None, None, None

class LIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, tau):
        if v_reset is None:
            raise NotImplementedError
        h_seq, spike_seq, v_next = cext_neuron.LIF_hard_reset_fptt(x_seq, v, v_threshold, v_reset, tau)
        if x_seq.requires_grad:
            ctx.save_for_backward(h_seq, spike_seq)
            ctx.v_threshold = v_threshold
            ctx.v_reset = v_reset
            ctx.alpha = alpha
            ctx.detach_reset = detach_reset
            ctx.grad_surrogate_function_index = grad_surrogate_function_index
            ctx.tau = tau
        return h_seq, spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_h_seq, grad_spike_seq, grad_v_next_seq):
        grad_x, grad_v = cext_neuron.LIF_hard_reset_bptt(grad_spike_seq, grad_v_next_seq, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.v_threshold, ctx.v_reset, ctx.alpha, ctx.detach_reset, ctx.grad_surrogate_function_index, ctx.tau)
        return grad_x, grad_v, None, None, None, None, None, None


surrogate_function_dict = {
    'ATan': 0,
    'Sigmoid': 1
}
class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False, update_function=None):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.grad_surrogate_function_index = surrogate_function_dict[surrogate_function]
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.update_function = update_function

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            h, spike, self.v = self.update_function(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index)
            return spike

    def reset(self):
        self.v = self.v_reset

class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset, LIFStep.apply)
        self.tau = tau

class MultiStepLIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset, LIFMultiStep.apply)
        self.tau = tau

