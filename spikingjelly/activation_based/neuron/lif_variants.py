from typing import Callable, Optional
import math

import numpy as np
import torch
import torch.nn as nn

from .. import surrogate, base
from .base_node import BaseNode
from .lif import LIFNode


__all__ = ["GatedLIFNode", "KLIFNode", "CUBALIFNode", "LIAFNode"]


class GatedLIFNode(base.MemoryModule):
    def __init__(
        self,
        T: int,
        inplane=None,
        init_linear_decay=None,
        init_v_subreset=None,
        init_tau: float = 0.25,
        init_v_threshold: float = 0.5,
        init_conduct: float = 0.5,
        surrogate_function: Callable = surrogate.Sigmoid(),
        step_mode="m",
        backend="torch",
    ):
        """
        **API Language:**
        :ref:`中文 <GatedLIFNode.__init__-cn>` | :ref:`English <GatedLIFNode.__init__-en>`

        ----

        .. _GatedLIFNode.__init__-cn:

        * **中文**

        Gated LIF 神经元（GLIF），由
        `GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
        <https://openreview.net/forum?id=UmFSx2c4ubT>`_ 提出。
        该模型对 LIF 神经元进行统一门控建模，膜电位相关参数（包括门控系数）均为可学习参数。

        :param T: 时间步数
        :type T: int

        :param inplane: 输入张量的通道数。
            若为 ``None``，则使用 layer-wise GLIF；否则使用 channel-wise GLIF
        :type inplane: int

        :param init_linear_decay: 膜电位线性衰减系数的初始值。
            若不设置，默认值为 ``init_v_threshold / (T * 2)``
        :type init_linear_decay: float

        :param init_v_subreset: 膜电位软复位电压的初始值
        :type init_v_subreset: float

        :param init_tau: 膜电位指数衰减时间常数的初始值
        :type init_tau: float

        :param init_v_threshold: 神经元阈值电压的初始值
        :type init_v_threshold: float

        :param init_conduct: 膜电位电导率的初始值
        :type init_conduct: float

        :param surrogate_function: 反向传播中用于计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param step_mode: 步进模式，仅支持 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 使用的后端。不同 ``step_mode`` 支持的后端可能不同。
            可通过 ``self.supported_backends`` 查看当前步进模式支持的后端。
            Gated LIF 仅支持 ``'torch'`` 后端
        :type backend: str

        ----

        .. _GatedLIFNode.__init__-en:

        * **English**

        Gated LIF neuron (GLIF), proposed in
        `GLIF: A Unified Gated Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
        <https://openreview.net/forum?id=UmFSx2c4ubT>`_.
        This model introduces unified gating mechanisms into LIF neurons.
        All membrane-related parameters, including gating coefficients, are learnable.

        :param T: number of time-steps
        :type T: int

        :param inplane: number of channels of the input tensor.
            If ``None``, layer-wise GLIF is used; otherwise, channel-wise GLIF is applied
        :type inplane: int

        :param init_linear_decay: initial value of the linear decay coefficient.
            Defaults to ``init_v_threshold / (T * 2)`` if not specified
        :type init_linear_decay: float

        :param init_v_subreset: initial soft-reset voltage of the membrane potential
        :type init_v_subreset: float

        :param init_tau: initial exponential decay time constant of the membrane potential
        :type init_tau: float

        :param init_v_threshold: initial membrane potential threshold
        :type init_v_threshold: float

        :param init_conduct: initial membrane conductance
        :type init_conduct: float

        :param surrogate_function: surrogate function used to compute spike gradients during backpropagation
        :type surrogate_function: Callable

        :param step_mode: step mode, only `'m'` (multi-step) is supported
        :type step_mode: str

        :param backend: backend of this neuron layer. Supported backends depend on ``step_mode``.
            Users can print ``self.supported_backends`` to check availability.
            Gated LIF only supports the ``'torch'`` backend
        :type backend: str
        """
        assert isinstance(init_tau, float) and init_tau < 1.0
        assert isinstance(T, int) and T is not None
        assert isinstance(inplane, int) or inplane is None
        assert (
            isinstance(init_linear_decay, float) and init_linear_decay < 1.0
        ) or init_linear_decay is None
        assert (
            isinstance(init_v_subreset, float) and init_v_subreset < 1.0
        ) or init_v_subreset is None

        assert step_mode == "m"
        super().__init__()
        self.surrogate_function = surrogate_function
        self.backend = backend
        self.step_mode = step_mode
        self.T = T
        self.register_memory("v", 0.0)
        self.register_memory("u", 0.0)
        self.channel_wise = inplane is not None
        if self.channel_wise:  # channel-wise learnable params
            self.alpha, self.beta, self.gamma = [
                nn.Parameter(
                    torch.tensor(
                        0.2 * (np.random.rand(inplane) - 0.5), dtype=torch.float
                    )
                )
                for i in range(3)
            ]
            self.tau = nn.Parameter(
                -math.log(1 / init_tau - 1) * torch.ones(inplane, dtype=torch.float)
            )
            self.v_threshold = nn.Parameter(
                -math.log(1 / init_v_threshold - 1)
                * torch.ones(inplane, dtype=torch.float)
            )
            init_linear_decay = (
                init_v_threshold / (T * 2)
                if init_linear_decay is None
                else init_linear_decay
            )
            self.linear_decay = nn.Parameter(
                -math.log(1 / init_linear_decay - 1)
                * torch.ones(inplane, dtype=torch.float)
            )
            init_v_subreset = (
                init_v_threshold if init_v_subreset is None else init_v_subreset
            )
            self.v_subreset = nn.Parameter(
                -math.log(1 / init_v_subreset - 1)
                * torch.ones(inplane, dtype=torch.float)
            )
            self.conduct = nn.Parameter(
                -math.log(1 / init_conduct - 1)
                * torch.ones((T, inplane), dtype=torch.float)
            )

        else:  # layer-wise learnable params
            self.alpha, self.beta, self.gamma = [
                nn.Parameter(
                    torch.tensor(0.2 * (np.random.rand() - 0.5), dtype=torch.float)
                )
                for i in range(3)
            ]
            self.tau = nn.Parameter(
                torch.tensor(-math.log(1 / init_tau - 1), dtype=torch.float)
            )
            self.v_threshold = nn.Parameter(
                torch.tensor(-math.log(1 / init_v_threshold - 1), dtype=torch.float)
            )
            init_linear_decay = (
                init_v_threshold / (T * 2)
                if init_linear_decay is None
                else init_linear_decay
            )
            self.linear_decay = nn.Parameter(
                torch.tensor(-math.log(1 / init_linear_decay - 1), dtype=torch.float)
            )
            init_v_subreset = (
                init_v_threshold if init_v_subreset is None else init_v_subreset
            )
            self.v_subreset = nn.Parameter(
                torch.tensor(-math.log(1 / init_v_subreset - 1), dtype=torch.float)
            )
            self.conduct = nn.Parameter(
                -math.log(1 / init_conduct - 1) * torch.ones(T, dtype=torch.float)
            )

    @property
    def supported_backends(self):
        return ("torch",)

    def extra_repr(self):
        with torch.no_grad():
            tau = self.tau
            v_subreset = self.v_subreset
            linear_decay = self.linear_decay
            conduct = self.conduct
        return (
            super().extra_repr()
            + f", tau={tau}"
            + f", v_subreset={v_subreset}"
            + f", linear_decay={linear_decay}"
            + f", conduct={conduct}"
        )

    def neuronal_charge(
        self, x: torch.Tensor, alpha: torch.Tensor, beta: torch.Tensor, t
    ):
        input = x * (1 - beta * (1 - self.conduct[t].view(1, -1, 1, 1).sigmoid()))
        self.u = (
            (1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid())) * self.v
            - (1 - alpha) * self.linear_decay.view(1, -1, 1, 1).sigmoid()
        ) + input

    def neuronal_reset(self, spike, alpha: torch.Tensor, gamma: torch.Tensor):
        self.u = (
            self.u
            - (1 - alpha * (1 - self.tau.view(1, -1, 1, 1).sigmoid()))
            * self.v
            * gamma
            * spike
            - (1 - gamma) * self.v_subreset.view(1, -1, 1, 1).sigmoid() * spike
        )

    def neuronal_fire(self):
        return self.surrogate_function(
            self.u - self.v_threshold.view(1, -1, 1, 1).sigmoid()
        )

    def multi_step_forward(self, x_seq: torch.Tensor):
        alpha, beta, gamma = (
            self.alpha.view(1, -1, 1, 1).sigmoid(),
            self.beta.view(1, -1, 1, 1).sigmoid(),
            self.gamma.view(1, -1, 1, 1).sigmoid(),
        )
        y_seq = []
        spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
        for t in range(self.T):
            self.neuronal_charge(x_seq[t], alpha, beta, t)
            self.neuronal_reset(spike, alpha, gamma)
            spike = self.neuronal_fire()
            self.v = self.u
            y_seq.append(spike)
        return torch.stack(y_seq)


class KLIFNode(BaseNode):
    def __init__(
        self,
        scale_reset: bool = False,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language:**
        :ref:`中文 <KLIFNode.__init__-cn>` | :ref:`English <KLIFNode.__init__-en>`

        ----

        .. _KLIFNode.__init__-cn:

        **中文 API**

        K-based Leaky Integrate-and-Fire（KLIF）神经元的构造函数。

        KLIF 神经元模型源自
        `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_，
        可视为一种带漏电项的积分器，其在阈下阶段与放电 / 重置阶段均具有不同于传统 LIF 的动力学形式。

        **阈下动力学方程**

        若 ``decay_input == True``：

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``：

        .. math::
            H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        **放电与重置机制**

        KLIF 神经元的放电与重置形式如下：

        .. math::
            F[t] &= \\mathrm{ReLU}(kH[t]) \\\\
            S[t] &= \\Theta(F[t] - V_{th})

        若 ``scale_reset == False``：

        .. math::
            V[t] =
            \\begin{cases}
                F[t](1-S[t]) + V_{reset}S[t], & \\text{hard reset} \\\\
                F[t] - S[t]V_{th}, & \\text{soft reset}
            \\end{cases}

        若 ``scale_reset == True``：

        .. math::
            V[t] =
            \\begin{cases}
                \\frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], & \\text{hard reset} \\\\
                \\frac{1}{k}(F[t] - S[t]V_{th}), & \\text{soft reset}
            \\end{cases}

        :param scale_reset: 是否在 ``neuronal_reset`` 阶段对膜电位 ``v`` 进行缩放
        :type scale_reset: bool

        :param tau: 膜电位的时间常数
        :type tau: float

        :param decay_input: 输入项是否参与膜电位衰减
        :type decay_input: bool

        :param v_threshold: 神经元的放电阈值
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。若不为 ``None``，放电后膜电位将被重置为 ``v_reset``；
            若为 ``None``，则放电后膜电位减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播中用于近似阶跃函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否在反向传播时将 reset 过程从计算图中分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可选 ``'s'`` （单步）或 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 计算后端。不同 ``step_mode`` 支持的后端可能不同，
            可通过 ``self.supported_backends`` 查看当前步进模式支持的后端。
            在支持的情况下，``'cupy'`` 或 ``'triton'`` 后端通常具有最高的执行效率
        :type backend: str

        :param store_v_seq: 当 ``step_mode = 'm'`` 且输入形状为 ``[T, N, *]`` 时，
            是否保存所有时间步的膜电位序列 ``self.v_seq``（形状为 ``[T, N, *]``）。
            若为 ``False``，仅保留最后一个时间步的膜电位 ``self.v``（形状为 ``[N, *]``），
            以降低内存开销
        :type store_v_seq: bool

        ----

        .. _KLIFNode.__init__-en:

        **English API**

        Constructor of the K-based Leaky Integrate-and-Fire (KLIF) neuron.

        The KLIF neuron is proposed in  
        `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_.
        It can be regarded as a leaky integrator with a modified firing and reset mechanism compared to conventional LIF neurons.

        **Sub-threshold neuronal dynamics**

        If ``decay_input == True``:

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

        .. math::
            H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        **Firing and reset mechanism**

        The firing and reset equations of KLIF are as follows:

        .. math::
            F[t] &= \\mathrm{ReLU}(kH[t]) \\\\
            S[t] &= \\Theta(F[t] - V_{th})

        If ``scale_reset == False``:

        .. math::
            V[t] =
            \\begin{cases}
                F[t](1-S[t]) + V_{reset}S[t], & \\text{hard reset} \\\\
                F[t] - S[t]V_{th}, & \\text{soft reset}
            \\end{cases}

        If ``scale_reset == True``:

        .. math::
            V[t] =
            \\begin{cases}
                \\frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], & \\text{hard reset} \\\\
                \\frac{1}{k}(F[t] - S[t]V_{th}), & \\text{soft reset}
            \\end{cases}

        :param scale_reset: whether to scale the membrane potential ``v`` during ``neuronal_reset``
        :type scale_reset: bool

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input term participates in decay
        :type decay_input: bool

        :param v_threshold: firing threshold of the neuron
        :type v_threshold: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane potential
            will be reset to ``v_reset`` after firing; otherwise, ``v_threshold`` will be subtracted
        :type v_reset: Optional[float]

        :param surrogate_function: surrogate function used to approximate the gradient
            of the Heaviside step function during backpropagation
        :type surrogate_function: Callable

        :param detach_reset: whether to detach the reset operation from the computation graph
        :type detach_reset: bool

        :param step_mode: step mode, either ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str

        :param backend: backend for this neuron. Different ``step_mode`` may support different backends.
            Supported backends can be queried via ``self.supported_backends``.
            If available, ``'cupy'`` or ``'triton'`` usually provides the fastest execution
        :type backend: str

        :param store_v_seq: when ``step_mode = 'm'`` and input shape is ``[T, N, *]``,
            whether to store the membrane potential at all time steps in ``self.v_seq``.
            If ``False``, only the final membrane potential ``self.v`` is kept to reduce memory usage
        :type store_v_seq: bool
        """
        assert isinstance(tau, float) and tau > 1.0
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )

        self.scale_reset = scale_reset
        self.tau = tau
        self.decay_input = decay_input

        self.k = nn.Parameter(torch.as_tensor(1.0))

    @property
    def supported_backends(self):
        return ("torch",)

    @staticmethod
    @torch.jit.script
    def neuronal_charge_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor
    ):
        v = v + (x - (v - v_reset)) / tau
        v = torch.relu_(k * v)
        return v

    @staticmethod
    @torch.jit.script
    def neuronal_charge_no_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float, k: torch.Tensor
    ):
        v = v - (v - v_reset) / tau + x
        v = torch.relu_(k * v)
        return v

    def neuronal_charge(self, x: torch.Tensor):
        if self.v_reset is None:
            v_reset = 0.0
        else:
            v_reset = self.v_reset
        if self.decay_input:
            self.v = self.neuronal_charge_decay_input(
                x, self.v, v_reset, self.tau, self.k
            )

        else:
            self.v = self.neuronal_charge_no_decay_input(
                x, self.v, v_reset, self.tau, self.k
            )

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.scale_reset:
            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold) / self.k

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v / self.k, spike_d, self.v_reset)

        else:
            if self.v_reset is None:
                # soft reset
                self.v = self.jit_soft_reset(self.v, spike_d, self.v_threshold)

            else:
                # hard reset
                self.v = self.jit_hard_reset(self.v, spike_d, self.v_reset)


class CUBALIFNode(BaseNode):
    def __init__(
        self,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        """
        CUrrent-BAsed LIF neuron.

        .. warning::

            ``CLIFNode`` is renamed to ``CUBALIFNode`` in version ``0.0.0.1.0``.

        :param c_decay: decay factor for input current. Defaults to 0.5
        :type c_decay: float

        :param v_decay: decay factor for membrane potential. Defaults to 0.75
        :type v_decay: float

        :param v_threshold: firing threshold of the neuron
        :type v_threshold: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane potential
            will be reset to ``v_reset`` after firing; otherwise, ``v_threshold`` will be subtracted
        :type v_reset: Optional[float]

        :param surrogate_function: surrogate function used to compute spike gradients during backpropagation
        :type surrogate_function: Callable
        """
        super().__init__(v_threshold, v_reset, surrogate_function)

        self.register_memory("c", 0.0)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.c_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            spike = self.single_step_forward(x_seq[t])
            spike_seq.append(spike)

        return torch.stack(spike_seq)

    def c_float_to_tensor(self, c: torch.Tensor):
        if isinstance(self.c, float):
            c_init = self.c
            self.c = torch.full_like(c.data, fill_value=c_init)


class LIAFNode(LIFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <LIAFNode.__init__-cn>` | :ref:`English <LIAFNode.__init__-en>`

        ----

        .. _LIAFNode.__init__-cn:

        **中文 API**

        LIAF（Leaky Integrate and Analog Fire）神经元的构造函数。

        LIAF 神经元由
        `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_
        提出，其行为与 LIF 神经元相同，但输出经过连续激活函数而非二值脉冲。

        .. admonition:: 警告
            :class: warning

            该神经元层的输出不是二值脉冲，而是连续值。

        :param act: 激活函数
        :type act: Callable

        :param threshold_related: 是否使用阈值依赖模式（TR mode）。若为 ``True``，输出为 ``y = act(h - v_th)``，
            否则为 ``y = act(h)``
        :type threshold_related: bool

        其他参数请参考 :class:`LIFNode`。

        ----

        .. _LIAFNode.__init__-en:

        **English API**

        Constructor of the LIAF (Leaky Integrate and Analog Fire) neuron.

        The LIAF neuron is proposed in
        `LIAF-Net: Leaky Integrate and Analog Fire Network for Lightweight and Efficient Spatiotemporal Information Processing <https://arxiv.org/abs/2011.06176>`_.
        It behaves like a LIF neuron, but the output passes through a continuous activation function instead of generating binary spikes.

        .. admonition:: Warning
            :class: warning

            The outputs of this neuron layer are not binary spikes.

        :param act: the activation function
        :type act: Callable

        :param threshold_related: whether the neuron uses threshold-related (TR) mode. If ``True``, the output is ``y = act(h - v_th)``,
            otherwise ``y = act(h)``
        :type threshold_related: bool

        Other parameters in `*args, **kwargs` are the same as :class:`LIFNode`.
        """
        super().__init__(*args, **kwargs)
        self.act = act
        self.threshold_related = threshold_related

        assert self.backend == "torch", "LIAFNode only supports for backend='torch'!"
        assert self.single_step_cupy_fp32_inference == False, (
            "LIAFNode does not support for single_step_cupy_fp32_inference!"
        )

    @property
    def supported_backends(self):
        return ("torch",)

    def single_step_forward(self, x: torch.Tensor):
        self.neuronal_charge(x)
        if self.threshold_related:
            y = self.act(self.v - self.v_threshold)
        else:
            y = self.act(self.v)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return y
