import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import _C_neuron

def hard_reset_forward_template(x: torch.Tensor, v:torch.Tensor, v_threshold: float, v_reset: float, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_forward_template-en>`

    .. _hard_reset_forward_template-cn:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :return: ``(spike, v_next)``，其中 ``spike`` 是 :math:`S_{t}`，`v_next` 是 :math:`V_{t}`
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。

    * :ref:`中文API <hard_reset_forward_template-cn>`

    .. _hard_reset_forward_template-en:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :return: ``(spike, v_next)``, where ``spike`` is :math:`S_{t}`, and ``v_next`` is :math:`V_{t}`
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with hard reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters.
    '''

    pass

def hard_reset_fptt_template(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_fptt_template-en>`

    .. _hard_reset_fptt_template-cn:

    :param x: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :return: ``(spike_seq, v_next)``，其中 ``spike`` 是 :math:`S_{t}, t=0,1,...,T-1`，`v_next` 是 :math:`V_{T-1}`
    :rtype: tuple

    :ref:`hard_reset_forward_template <hard_reset_forward_template-cn>` 的多步版本。

    * :ref:`中文API <hard_reset_fptt_template-cn>`

    .. _hard_reset_fptt_template-en:

    :param x: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :return: ``(spike_seq, v_next)``, where ``spike`` is :math:`S_{t}, t=0,1,...,T-1`, `v_next` is :math:`V_{T-1}`
    :rtype: tuple

    The multi-step version of :ref:`hard_reset_forward_template <hard_reset_forward_template-en>`.
    '''
    pass

def soft_reset_forward_template(x: torch.Tensor, v:torch.Tensor, v_threshold: float, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_forward_template-en>`

    .. _soft_reset_forward_template-cn:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :return: ``(spike, v_next)``，其中 ``spike`` 是 :math:`S_{t}`，`v_next` 是 :math:`V_{t}`
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是软重置(soft reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。

    * :ref:`中文API <soft_reset_forward_template-cn>`

    .. _soft_reset_forward_template-en:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :return: ``(spike, v_next)``, where ``spike`` is :math:`S_{t}`, and ``v_next`` is :math:`V_{t}`
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with soft reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters.
    '''

    pass

def soft_reset_fptt_template(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_fptt_template-en>`

    .. _soft_reset_fptt_template-cn:

    :param x: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :return: ``(spike_seq, v_next)``，其中 ``spike`` 是 :math:`S_{t}, t=0,1,...,T-1`，`v_next` 是 :math:`V_{T-1}`
    :rtype: tuple

    :ref:`soft_reset_forward_template <soft_reset_forward_template-cn>` 的多步版本。

    * :ref:`中文API <soft_reset_fptt_template-cn>`

    .. _soft_reset_fptt_template-en:

    :param x: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :return: ``(spike_seq, v_next)``, where ``spike`` is :math:`S_{t}, t=0,1,...,T-1`, `v_next` is :math:`V_{T-1}`
    :rtype: tuple

    The multi-step version of :ref:`soft_reset_forward_template <soft_reset_forward_template-en>`.
    '''
    pass

def hard_reset_forward_with_grad_template(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_forward_with_grad_template-en>`

    .. _hard_reset_forward_with_grad_template-cn:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(spike, v_next, grad_s_to_h, grad_v_to_h)``，其中 ``spike`` 是 :math:`S_{t}`，`v_next` 是 :math:`V_{t}`，``grad_s_to_h`` 是 :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`，``grad_v_to_h`` 是 :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。并且会计算出反向传播所需的梯度

    .. math::

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold}) = \\sigma(\\alpha(H_{t} - V_{threshold}))

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - S_{t} + (V_{reset} - H_{t})\\frac{\\partial S_{t}}{\\partial H_{t}}

    * :ref:`中文API <hard_reset_forward_with_grad_template-cn>`

    .. _hard_reset_forward_with_grad_template-en:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(spike, v_next, grad_s_to_h, grad_v_to_h)``, where ``spike`` is :math:`S_{t}`, `v_next` is :math:`V_{t}`, ``grad_s_to_h`` is :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`, ``grad_v_to_h`` is :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with hard reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = S_{t}V_{reset} + (1 - S_{t})H_{t}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters. This function will also calculate the gradients which the backward function needs

    .. math::
        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold}) = \\sigma(\\alpha(H_{t} - V_{threshold}))

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - S_{t} + (V_{reset} - H_{t})\\frac{\\partial S_{t}}{\\partial H_{t}}
    '''

    pass

def hard_reset_fptt_with_grad_template(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <hard_reset_fptt_with_grad_template-en>`

    .. _hard_reset_fptt_with_grad_template-cn:

    :param x_seq: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(spike_seq, v_next, grad_s_to_h, grad_v_to_h)``，其中 ``spike_seq`` 是 :math:`S_{t}, t=0,1,...,T-1`，`v_next` 是 :math:`V_{T-1}`，``grad_s_to_h`` 是 :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`，``grad_v_to_h`` 是 :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :rtype: tuple

    :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-cn>` 的多步版本。

    * :ref:`中文API <hard_reset_fptt_with_grad_template-cn>`

    .. _hard_reset_fptt_with_grad_template-en:

    :param x_seq: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(spike_seq, v_next, grad_s_to_h, grad_v_to_h)``, where ``spike_seq`` is :math:`S_{t}, t=0,1,...,T-1`, `v_next` is :math:`V_{T-1}`, ``grad_s_to_h`` is :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`, ``grad_v_to_h`` is :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :rtype: tuple

    The multi-step version of :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-en>`.
    '''

    pass

def soft_reset_forward_with_grad_template(x: torch.Tensor, v: torch.Tensor, v_threshold: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_forward_with_grad_template-en>`

    .. _soft_reset_forward_with_grad_template-cn:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(spike, v_next, grad_s_to_h, grad_v_to_h)``，其中 ``spike`` 是 :math:`S_{t}`，`v_next` 是 :math:`V_{t}`，``grad_s_to_h`` 是 :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`，``grad_v_to_h`` 是 :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :rtype: tuple

    对神经元进行单步的电压更新，其中电压重置方式是软重置(soft reset)。更新的方程为

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    其中 :math:`f(\\cdot)` 是充电方程，:math:`\\theta` 是神经元自身的参数。并且会计算出反向传播所需的梯度

    .. math::

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold}) = \\sigma(\\alpha(H_{t} - V_{threshold}))

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - V_{threshold} \\frac{\\partial S_{t}}{\\partial H_{t}}


    * :ref:`中文API <soft_reset_forward_with_grad_template-cn>`

    .. _soft_reset_forward_with_grad_template-en:

    :param x: :math:`X_{t}`
    :type x: torch.Tensor
    :param v: :math:`V_{t-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(spike, v_next, grad_s_to_h, grad_v_to_h)``, where ``spike`` is :math:`S_{t}`, `v_next` is :math:`V_{t}`, ``grad_s_to_h`` is :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`, ``grad_v_to_h`` is :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :rtype: tuple

    Update the membrane potential of the neuron by one time step with soft reset. The update is calculated by

    .. math::
        H_{t} & = f(X_{t}, V_{t-1}; \\theta)

        S_{t} & = \\Theta(H_{t} - V_{threshold})

        V_{t} & = H_{t} - S_{t}V_{threshold}

    where :math:`f(\\cdot)` is the charging equation and :math:`\\theta` is the neuron's parameters. This function will also calculate the gradients which the backward function needs

     .. math::

        \\frac{\\partial S_{t}}{\\partial H_{t}} & = \\Theta'(H_{t} - V_{threshold}) = \\sigma(\\alpha(H_{t} - V_{threshold}))

        \\frac{\\partial V_{t}}{\\partial H_{t}} & = 1 - V_{threshold} \\frac{\\partial S_{t}}{\\partial H_{t}}
    '''

    pass

def soft_reset_fptt_with_grad_template(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, *args, **kwargs):
    '''
    * :ref:`API in English <soft_reset_fptt_with_grad_template-en>`

    .. _soft_reset_fptt_with_grad_template-cn:

    :param x_seq: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: 是否在反向传播的计算图中断开重置过程
    :type detach_reset: bool
    :param grad_surrogate_function_index: 梯度替代函数的索引
    :type grad_surrogate_function_index: int
    :return: ``(spike_seq, v_next, grad_s_to_h, grad_v_to_h)``，其中 ``spike_seq`` 是 :math:`S_{t}, t=0,1,...,T-1`，`v_next` 是 :math:`V_{T-1}`，``grad_s_to_h`` 是 :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`，``grad_v_to_h`` 是 :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :rtype: tuple

    :ref:`soft_reset_forward_with_grad_template <soft_reset_forward_with_grad_template-cn>` 的多步版本。

    * :ref:`中文API <soft_reset_fptt_with_grad_template-cn>`

    .. _soft_reset_fptt_with_grad_template-en:

    :param x_seq: :math:`X_{t}, t=0,1,...,T-1`
    :type x: torch.Tensor
    :param v: :math:`V_{-1}`
    :type v: torch.Tensor
    :param v_threshold: :math:`V_{threshold}`
    :type v_threshold: float
    :param v_reset: :math:`V_{reset}`
    :type v_reset: float
    :param alpha: :math:`\\alpha`
    :type alpha: float
    :param detach_reset: whether detach the neuronal reset during backward
    :type detach_reset: bool
    :param grad_surrogate_function_index: index of the gradient surrogate function
    :type grad_surrogate_function_index: int
    :return: ``(spike_seq, v_next, grad_s_to_h, grad_v_to_h)``, where ``spike_seq`` is :math:`S_{t}, t=0,1,...,T-1`, `v_next` is :math:`V_{T-1}`, ``grad_s_to_h`` is :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`, ``grad_v_to_h`` is :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :rtype: tuple

    The multi-step version of :ref:`soft_reset_forward_with_grad_template <soft_reset_forward_with_grad_template-en>`.
    '''

    pass

def backward_template(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, grad_s_to_h: torch.Tensor, grad_v_to_h: float, *args, **kwargs):
    '''
    * :ref:`API in English <backward_template-en>`

    .. _backward_template-cn:

    :param grad_spike: :math:`\\frac{\\partial L}{\\partial S_{t}}`
    :type grad_spike: torch.Tensor
    :param grad_v_next: :math:`\\frac{\\partial L}{\\partial V_{t}}`
    :type grad_v_next: torch.Tensor
    :param grad_s_to_h: :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`
    :type grad_s_to_h: torch.Tensor
    :param grad_v_to_h: :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :type grad_v_to_h: torch.Tensor
    :return: ``(grad_x, grad_v)``，其中 ``grad_x`` 是 :math:`\\frac{\\partial L}{\\partial X_{t}}`，``grad_v`` 是 :math:`\\frac{\\partial L}{\\partial V_{t-1}}`
    :rtype: tuple

    :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-cn>` 和 :ref:`soft_reset_forward_with_grad_template <soft_reset_forward_with_grad_template-cn>` 的反向传播。梯度的计算按照

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}


    * :ref:`中文API <backward_template-cn>`

    .. _backward_template-en:

    :param grad_spike: :math:`\\frac{\\partial L}{\\partial S_{t}}`
    :type grad_spike: torch.Tensor
    :param grad_v_next: :math:`\\frac{\\partial L}{\\partial V_{t}}`
    :type grad_v_next: torch.Tensor
    :param grad_s_to_h: :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}`
    :type grad_s_to_h: torch.Tensor
    :param grad_v_to_h: :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}`
    :type grad_v_to_h: torch.Tensor
    :return: ``(grad_x, grad_v)``, where ``grad_x`` is :math:`\\frac{\\partial L}{\\partial X_{t}}`, ``grad_v`` is :math:`\\frac{\\partial L}{\\partial V_{t-1}}`
    :rtype: tuple

    The backward of :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-en>` and :ref:`soft_reset_forward_with_grad_template <soft_reset_forward_with_grad_template-en>`. The gradients are calculated by

    .. math::
        \\frac{\\partial L}{\\partial H_{t}} & = \\frac{\\partial L}{\\partial S_{t}} \\frac{\\partial S_{t}}{\\partial H_{t}} + \\frac{\\partial L}{\\partial V_{t}} \\frac{\\partial V_{t}}{\\partial H_{t}}

        \\frac{\\partial L}{\\partial X_{t}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial X_{t}}

        \\frac{\\partial L}{\\partial V_{t-1}} &= \\frac{\\partial L}{\\partial H_{t}} \\frac{\\partial H_{t}}{\\partial V_{t-1}}
    '''
    pass

def bptt_template(grad_spike_seq: torch.Tensor, grad_v_next: torch.Tensor, grad_s_to_h: torch.Tensor, grad_v_to_h: torch.Tensor, *args, **kwargs):
    '''
    * :ref:`API in English <bptt_template-en>`

    .. _bptt_template-cn:

    :param grad_spike_seq: :math:`\\frac{\\partial L}{\\partial S_{t}}, t=0,1,...,T-1`
    :type grad_spike_seq: torch.Tensor
    :param grad_v_next: :math:`\\frac{\\partial L}{\\partial V_{T-1}}`
    :type grad_v_next: torch.Tensor
    :param grad_s_to_h: :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :type grad_s_to_h: torch.Tensor
    :param grad_v_to_h: :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :type grad_v_to_h: torch.Tensor
    :return: ``(grad_x_seq, grad_v)``，其中 ``grad_x_seq`` 是 :math:`\\frac{\\partial L}{\\partial X_{t}}, t=0,1,...,T-1`，``grad_v`` 是 :math:`\\frac{\\partial L}{\\partial V_{-1}}`
    :rtype: tuple

    :ref:`backward_template <backward_template-cn>` 的多步版本。


    * :ref:`中文API <bptt_template-cn>`

    .. _bptt_template-en:

    :param grad_spike_seq: :math:`\\frac{\\partial L}{\\partial S_{t}}, t=0,1,...,T-1`
    :type grad_spike_seq: torch.Tensor
    :param grad_v_next: :math:`\\frac{\\partial L}{\\partial V_{T-1}}`
    :type grad_v_next: torch.Tensor
    :param grad_s_to_h: :math:`\\frac{\\partial S_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :type grad_s_to_h: torch.Tensor
    :param grad_v_to_h: :math:`\\frac{\\partial V_{t}}{\\partial H_{t}}, t=0,1,...,T-1`
    :type grad_v_to_h: torch.Tensor
    :return: ``(grad_x_seq, grad_v)``, where ``grad_x_seq`` is :math:`\\frac{\\partial L}{\\partial X_{t}}, t=0,1,...,T-1`, ``grad_v`` is :math:`\\frac{\\partial L}{\\partial V_{-1}}`
    :rtype: tuple

    The multi-step version of :ref:`backward_template <backward_template-en>`.

    '''
    raise NotImplementedError

def LIF_hard_reset_forward(x: torch.Tensor, v:torch.Tensor, v_threshold: float, v_reset: float, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_forward-en>`

    .. _LIF_hard_reset_forward-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`hard_reset_forward_template <hard_reset_forward_template-cn>`。

    对LIF神经元进行单步的电压更新，其中电压重置方式是硬重置(hard reset)。充电的方程为

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    * :ref:`中文API <LIF_hard_reset_forward-cn>`

    .. _LIF_hard_reset_forward-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`hard_reset_forward_template <hard_reset_forward_template-en>` for more details about other args。

    Update the membrane potential of the LIF neuron by one time step with hard reset. The charging equation is

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    '''
    return _C_neuron.LIF_hard_reset_forward(x, v, v_threshold, v_reset, reciprocal_tau)

def LIF_hard_reset_fptt(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_fptt-en>`

    .. _LIF_hard_reset_fptt-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`hard_reset_fptt_template <hard_reset_fptt_template-cn>`。

    :ref:`LIF_hard_reset_forward <LIF_hard_reset_forward-cn>` 的多步版本。

    * :ref:`中文API <LIF_hard_reset_fptt-cn>`

    .. _LIF_hard_reset_fptt-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`hard_reset_fptt_template <hard_reset_fptt_template-en>` for more details about other args。

    The multi-step version of :ref:`LIF_hard_reset_forward <LIF_hard_reset_forward-en>`.
    '''
    return _C_neuron.LIF_hard_reset_fptt(x_seq, v, v_threshold, v_reset, reciprocal_tau)

def LIF_hard_reset_forward_with_grad(x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, alpha: float, detach_reset: bool, grad_surrogate_function_index: int, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_forward_with_grad-en>`

    .. _LIF_hard_reset_forward_with_grad-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-cn>`。

    对LIF神经元进行单步的电压更新并计算反向传播所需的梯度，其中电压重置方式是硬重置(hard reset)。充电的方程为

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    * :ref:`中文API <LIF_hard_reset_forward_with_grad-cn>`

    .. _LIF_hard_reset_forward_with_grad-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`hard_reset_forward_with_grad_template <hard_reset_forward_with_grad_template-en>` for more details about other args。

    Update the membrane potential of the LIF neuron by one time step with hard reset and calculate the gradients that the backward function needs. The charging equation is

    .. math::
        H_{t} = V_{t-1} + \\frac{1}{\\tau}(X_{t} -(V_{t-1} - V_{reset}))

    '''
    return _C_neuron.LIF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, reciprocal_tau)

def LIF_hard_reset_fptt_with_grad(x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_hard_reset_fptt_with_grad-en>`

    .. _LIF_hard_reset_fptt_with_grad-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`hard_reset_fptt_with_grad_template <hard_reset_fptt_with_grad_template-cn>`。

    :ref:`LIF_hard_reset_forward_with_grad <LIF_hard_reset_forward_with_grad-cn>` 的多步版本。

    * :ref:`中文API <LIF_hard_reset_fptt_with_grad-cn>`

    .. _LIF_hard_reset_fptt_with_grad-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`hard_reset_fptt_with_grad_template <hard_reset_fptt_with_grad_template-en>` for more details about other args。

    The multi-step version of :ref:`LIF_hard_reset_forward_with_grad <LIF_hard_reset_forward_with_grad-en>`.
    '''
    return _C_neuron.LIF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, reciprocal_tau)

def LIF_backward(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, grad_s_to_h: torch.Tensor, grad_v_to_h: float, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_backward-en>`

    .. _LIF_backward-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`backward_template <backward_template-cn>`。

    梯度的计算按照

    .. math::
        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}

    * :ref:`中文API <LIF_backward-cn>`

    .. _LIF_backward-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`backward_template <backward_template-en>` for more details about other args。

    The gradients are calculated by

    .. math::
        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}

    '''
    return _C_neuron.LIF_backward(grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h, reciprocal_tau)


def LIF_bptt(grad_spike: torch.Tensor, grad_v_next: torch.Tensor, grad_s_to_h: torch.Tensor, grad_v_to_h: float, reciprocal_tau: float):
    '''
    * :ref:`API in English <LIF_bptt-en>`

    .. _LIF_bptt-cn:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    其余的参数参见 :ref:`bptt_template <bptt_template-cn>`。

    :ref:`LIF_backward <LIF_backward-cn>` 的多步版本。

    梯度的计算按照

    .. math::
        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}

    * :ref:`中文API <LIF_bptt-cn>`

    .. _LIF_bptt-en:

    :param reciprocal_tau: :math:`\\frac{1}{\\tau}`
    :type reciprocal_tau: float

    See :ref:`bptt_template <bptt_template-en>` for more details about other args。

    The multi-step version of :ref:`LIF_backward <LIF_backward-en>`.

    The gradients are calculated by

    .. math::
        \\frac{\\partial H_{t}}{\\partial X_{t}} & = \\frac{1}{\\tau}

        \\frac{\\partial H_{t}}{\\partial V_{t-1}} & = 1 - \\frac{1}{\\tau}

    '''
    return _C_neuron.LIF_bptt(grad_spike, grad_v_next, grad_s_to_h, grad_v_to_h, reciprocal_tau)

class LIFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input):
        if v_reset is None:
            raise NotImplementedError

        spike, v_next, grad_s_to_h, grad_v_to_h = _C_neuron.LIF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        ctx.reciprocal_tau = reciprocal_tau
        ctx.detach_input = detach_input

        return spike, v_next

    @staticmethod
    def backward(ctx, grad_spike, grad_v_next):
        grad_x, grad_v = _C_neuron.LIF_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.reciprocal_tau, ctx.detach_input)
        return grad_x, grad_v, None, None, None, None, None, None, None

class LIFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input):
        if v_reset is None:
            raise NotImplementedError

        spike_seq, v_next, grad_s_to_h, grad_v_to_h = _C_neuron.LIF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index, reciprocal_tau, detach_input)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        ctx.reciprocal_tau = reciprocal_tau
        ctx.detach_input = detach_input
        return spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_next):
        grad_x, grad_v = _C_neuron.LIF_bptt(grad_spike_seq, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1], ctx.reciprocal_tau, ctx.detach_input)
        return grad_x, grad_v, None, None, None, None, None, None, None

class IFStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index):
        if v_reset is None:
            raise NotImplementedError

        spike, v_next, grad_s_to_h, grad_v_to_h = _C_neuron.IF_hard_reset_forward_with_grad(x, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        return spike, v_next

    @staticmethod
    def backward(ctx, grad_spike, grad_v_next):
        grad_x, grad_v = _C_neuron.IF_backward(grad_spike, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1])
        return grad_x, grad_v, None, None, None, None, None

class IFMultiStep(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index):
        if v_reset is None:
            raise NotImplementedError

        spike_seq, v_next, grad_s_to_h, grad_v_to_h = _C_neuron.IF_hard_reset_fptt_with_grad(x_seq, v, v_threshold, v_reset, alpha, detach_reset, grad_surrogate_function_index)
        ctx.save_for_backward(grad_s_to_h, grad_v_to_h)
        return spike_seq, v_next

    @staticmethod
    def backward(ctx, grad_spike_seq, grad_v_next):
        grad_x, grad_v = _C_neuron.IF_bptt(grad_spike_seq, grad_v_next, ctx.saved_tensors[0], ctx.saved_tensors[1])
        return grad_x, grad_v, None, None, None, None, None

surrogate_function_dict = {
    'ATan': 0,
    'Sigmoid': 1
}

class BaseNode(nn.Module):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.grad_surrogate_function_index = surrogate_function_dict[surrogate_function]
        self.alpha = alpha
        self.detach_reset = detach_reset
        self.reset()
    
    def reset(self):
        self.v = self.v_reset

    def extra_repr(self):
        return f'v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, surrogate_function={self.surrogate_function}, alpha={self.alpha}'

class LIFNode(BaseNode):
    def __init__(self, tau=100.0, detach_input=False, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0, detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)
        self.reciprocal_tau = 1 / tau
        self.detach_input = detach_input
    
    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike, self.v = LIFStep.apply(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index, self.reciprocal_tau, self.detach_input)
            else:
                spike, self.v = _C_neuron.LIF_hard_reset_forward(dv, self.v, self.v_threshold, self.v_reset, self.reciprocal_tau, self.detach_input)
            return spike

    def extra_repr(self):
        return super().extra_repr() + f' tau={1 / self.reciprocal_tau}'

class MultiStepLIFNode(LIFNode):
    def forward(self, dv_seq: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv_seq[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike_seq, self.v = LIFMultiStep.apply(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset, self.grad_surrogate_function_index, self.reciprocal_tau, self.detach_input)
            else:
                spike_seq, self.v = _C_neuron.LIF_hard_reset_fptt(dv_seq, self.v, self.v_threshold, self.v_reset, self.reciprocal_tau, self.detach_input)
            return spike_seq


class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0, surrogate_function='ATan', alpha=2.0,
                 detach_reset=False):
        super().__init__(v_threshold, v_reset, surrogate_function, alpha, detach_reset)

    def forward(self, dv: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv.data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike, self.v = IFStep.apply(dv, self.v, self.v_threshold, self.v_reset, self.alpha, self.detach_reset,
                                              self.grad_surrogate_function_index)
            else:
                spike, self.v = _C_neuron.IF_hard_reset_forward(dv, self.v, self.v_threshold, self.v_reset)
            return spike


class MultiStepIFNode(IFNode):
    def forward(self, dv_seq: torch.Tensor):
        if self.v_reset is None:
            raise NotImplementedError
        else:
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.zeros_like(dv_seq[0].data)
                if self.v_reset != 0.0:
                    self.v.fill_(self.v_reset)
            if self.training:
                spike_seq, self.v = IFMultiStep.apply(dv_seq, self.v, self.v_threshold, self.v_reset, self.alpha,
                                                       self.detach_reset, self.grad_surrogate_function_index)
            else:
                spike_seq, self.v = _C_neuron.IF_hard_reset_fptt(dv_seq, self.v, self.v_threshold, self.v_reset)
            return spike_seq