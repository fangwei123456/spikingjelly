import math
from typing import Callable, Optional

import numpy as np
import torch
import torch.nn as nn

from .. import base, functional, surrogate
from .base_node import BaseNode
from .lif import LIFNode

__all__ = [
    "GatedLIFNode",
    "KLIFNode",
    "ComplementaryLIFNode",
    "CUBALIFNode",
    "LIAFNode",
]


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
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        step_mode="m",
        backend="torch",
    ):
        r"""
        **API Language** - :ref:`中文 <GatedLIFNode.__init__-cn>` | :ref:`English <GatedLIFNode.__init__-en>`

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
        :type surrogate_function: surrogate.SurrogateFunctionBase

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
        :type surrogate_function: surrogate.SurrogateFunctionBase

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
        param_shape = () if inplane is None else (inplane,)
        self.alpha, self.beta, self.gamma = [
            nn.Parameter(
                torch.tensor(
                    0.2 * (np.random.rand(*param_shape) - 0.5),
                    dtype=torch.float,
                )
            )
            for _ in range(3)
        ]
        init_linear_decay = (
            init_v_threshold / (T * 2)
            if init_linear_decay is None
            else init_linear_decay
        )
        init_v_subreset = (
            init_v_threshold if init_v_subreset is None else init_v_subreset
        )
        self.tau = nn.Parameter(
            torch.full(param_shape, -math.log(1 / init_tau - 1), dtype=torch.float)
        )
        self.v_threshold = nn.Parameter(
            torch.full(
                param_shape,
                -math.log(1 / init_v_threshold - 1),
                dtype=torch.float,
            )
        )
        self.linear_decay = nn.Parameter(
            torch.full(
                param_shape,
                -math.log(1 / init_linear_decay - 1),
                dtype=torch.float,
            )
        )
        self.v_subreset = nn.Parameter(
            torch.full(
                param_shape,
                -math.log(1 / init_v_subreset - 1),
                dtype=torch.float,
            )
        )
        self.conduct = nn.Parameter(
            torch.full(
                (T, *param_shape),
                -math.log(1 / init_conduct - 1),
                dtype=torch.float,
            )
        )

    @property
    def supported_backends(self):
        return ("torch",)

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", tau={self.tau}"
            + f", v_subreset={self.v_subreset}"
            + f", linear_decay={self.linear_decay}"
            + f", conduct={self.conduct}"
        )

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        v = states[0]
        if not isinstance(v, torch.Tensor):
            v = torch.full_like(inputs[0][0], v)
        return v, states[1]

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        v = states[0]
        alpha = self.alpha.view(1, -1, 1, 1).sigmoid()
        beta = self.beta.view(1, -1, 1, 1).sigmoid()
        gamma = self.gamma.view(1, -1, 1, 1).sigmoid()
        tau = self.tau.view(1, -1, 1, 1).sigmoid()
        v_threshold = self.v_threshold.view(1, -1, 1, 1).sigmoid()
        linear_decay = self.linear_decay.view(1, -1, 1, 1).sigmoid()
        v_subreset = self.v_subreset.view(1, -1, 1, 1).sigmoid()

        spike = torch.zeros(x_seq.shape[1:], device=x_seq.device)
        spike_seq = []
        for t in range(self.T):
            spike, v = functional.gated_lif_step(
                x_seq[t],
                v,
                spike,
                alpha,
                beta,
                gamma,
                tau,
                v_threshold,
                linear_decay,
                v_subreset,
                self.conduct[t].view(1, -1, 1, 1).sigmoid(),
                self.surrogate_function,
            )
            spike_seq.append(spike)
        return (torch.stack(spike_seq),), (v, v)


class KLIFNode(BaseNode):
    def __init__(
        self,
        scale_reset: bool = False,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <KLIFNode.__init__-cn>` | :ref:`English <KLIFNode.__init__-en>`

        ----

        .. _KLIFNode.__init__-cn:

        * **中文**

        K-based Leaky Integrate-and-Fire（KLIF）神经元的构造函数。

        KLIF 神经元模型源自
        `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_，
        可视为一种带漏电项的积分器，其在阈下阶段与放电 / 重置阶段均具有不同于传统 LIF 的动力学形式。

        **阈下动力学方程**

        若 ``decay_input == True``：

        .. math::
            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``：

        .. math::
            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        **放电与重置机制**

        KLIF 神经元的放电与重置形式如下：

        .. math::
            :nowrap:

            \begin{align*}
            F[t] &= \mathrm{ReLU}(kH[t]) \\
            S[t] &= \Theta(F[t] - V_{th})
            \end{align*}

        若 ``scale_reset == False``：

        .. math::
            V[t] =
            \begin{cases}
                F[t](1-S[t]) + V_{reset}S[t], & \text{hard reset} \\
                F[t] - S[t]V_{th}, & \text{soft reset}
            \end{cases}

        若 ``scale_reset == True``：

        .. math::
            V[t] =
            \begin{cases}
                \frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], & \text{hard reset} \\
                \frac{1}{k}(F[t] - S[t]V_{th}), & \text{soft reset}
            \end{cases}

        :param scale_reset: 是否在重置阶段对膜电位 ``v`` 进行缩放
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
        :type surrogate_function: surrogate.SurrogateFunctionBase

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

        * **English**

        Constructor of the K-based Leaky Integrate-and-Fire (KLIF) neuron.

        The KLIF neuron is proposed in
        `KLIF: An optimized spiking neuron unit for tuning surrogate gradient slope and membrane potential <https://arxiv.org/abs/2302.09238>`_.
        It can be regarded as a leaky integrator with a modified firing and reset mechanism compared to conventional LIF neurons.

        **Sub-threshold neuronal dynamics**

        If ``decay_input == True``:

        .. math::
            H[t] = V[t-1] + \frac{1}{\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

        .. math::
            H[t] = V[t-1] - \frac{1}{\tau}(V[t-1] - V_{reset}) + X[t]

        **Firing and reset mechanism**

        The firing and reset equations of KLIF are as follows:

        .. math::
            :nowrap:

            \begin{align*}
            F[t] &= \mathrm{ReLU}(kH[t]) \\
            S[t] &= \Theta(F[t] - V_{th})
            \end{align*}

        If ``scale_reset == False``:

        .. math::
            V[t] =
            \begin{cases}
                F[t](1-S[t]) + V_{reset}S[t], & \text{hard reset} \\
                F[t] - S[t]V_{th}, & \text{soft reset}
            \end{cases}

        If ``scale_reset == True``:

        .. math::
            V[t] =
            \begin{cases}
                \frac{F[t]}{k}(1-S[t]) + V_{reset}S[t], & \text{hard reset} \\
                \frac{1}{k}(F[t] - S[t]V_{th}), & \text{soft reset}
            \end{cases}

        :param scale_reset: whether to scale the membrane potential ``v`` during reset
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
        :type surrogate_function: surrogate.SurrogateFunctionBase

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

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        spike, v = functional.klif_step(
            x,
            v,
            self.k,
            self.tau,
            self.decay_input,
            self.scale_reset,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])


class ComplementaryLIFNode(BaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(alpha=1.0),
        step_mode: str = "s",
        backend: str = "torch",
        store_state_seqs: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ComplementaryLIFNode.__init__-cn>` | :ref:`English <ComplementaryLIFNode.__init__-en>`

        ----

        .. _ComplementaryLIFNode.__init__-cn:

        * **中文**

        Complementary Leaky Integrate-and-Fire（CLIF）神经元，由
        `CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
        <https://proceedings.mlr.press/v235/huang24n.html>`_ 提出。

        CLIF 在 LIF 的膜电位 :math:`V[t]` 之外维护互补电位 :math:`M[t]`。
        对当前输入 :math:`X[t]`，状态按照以下顺序更新：

        .. math::
            H[t] = \left(1 - \frac{1}{\tau}\right)V[t-1] + X[t]

        .. math::
            S[t] = \Theta(H[t] - V_{th})

        .. math::
            M[t] = M[t-1] \odot \sigma\left(\frac{H[t]}{\tau}\right) + S[t]

        .. math::
            V[t] = H[t] - S[t] \odot \left(V_{th} + \sigma(M[t])\right)

        该实现仅包含论文定义的输入不衰减和软重置动力学，不引入 CLIF
        专属可学习参数。``v`` 和 ``m`` 是持久状态，调用 :meth:`reset` 会将两者
        恢复为零。默认替代函数的前向输出为二值脉冲；若显式传入非脉冲替代函数，
        则输出遵循该替代函数的定义。

        :param tau: 膜电位时间常数，必须为大于 ``1.0`` 的浮点数
        :type tau: float
        :param v_threshold: 放电阈值
        :type v_threshold: float
        :param surrogate_function: 反向传播中用于近似阶跃函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: 步进模式，``"s"`` 表示单步，``"m"`` 表示多步
        :type step_mode: str
        :param backend: 计算后端，仅支持 ``"torch"``
        :type backend: str
        :param store_state_seqs: 在多步模式下是否保存完整状态轨迹。若为 ``True``，
            ``state_seqs`` 按 ``[v_seq, m_seq]`` 保存两个形状为 ``[T, N, *]`` 的张量；
            functional forward 不写入该缓存。本类不使用父类的 ``store_v_seq`` 和
            ``v_seq``，所有状态轨迹统一由 ``store_state_seqs`` 控制
        :type store_state_seqs: bool
        :raises AssertionError: ``tau`` 不是大于 ``1.0`` 的浮点数时抛出
        :raises ValueError: ``step_mode`` 不是 ``"s"`` 或 ``"m"`` 时抛出
        :raises NotImplementedError: ``backend`` 不是 ``"torch"`` 时抛出

        ----

        .. _ComplementaryLIFNode.__init__-en:

        * **English**

        The Complementary Leaky Integrate-and-Fire (CLIF) neuron proposed in
        `CLIF: Complementary Leaky Integrate-and-Fire Neuron for Spiking Neural Networks
        <https://proceedings.mlr.press/v235/huang24n.html>`_.

        In addition to the LIF membrane potential :math:`V[t]`, CLIF maintains
        the complementary potential :math:`M[t]`. For the current input
        :math:`X[t]`, the states are updated in the following order:

        .. math::
            H[t] = \left(1 - \frac{1}{\tau}\right)V[t-1] + X[t]

        .. math::
            S[t] = \Theta(H[t] - V_{th})

        .. math::
            M[t] = M[t-1] \odot \sigma\left(\frac{H[t]}{\tau}\right) + S[t]

        .. math::
            V[t] = H[t] - S[t] \odot \left(V_{th} + \sigma(M[t])\right)

        This implementation contains only the non-decayed-input and soft-reset
        dynamics defined by the paper and adds no CLIF-specific learnable
        parameters. ``v`` and ``m`` are persistent states; :meth:`reset`
        restores both to zero. The default surrogate produces binary spikes in
        forward propagation. If a non-spiking surrogate is supplied explicitly,
        the output follows that surrogate's definition.

        :param tau: Membrane time constant, which must be a float greater than ``1.0``
        :type tau: float
        :param v_threshold: Firing threshold
        :type v_threshold: float
        :param surrogate_function: Surrogate function used to approximate the gradient of the step function
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param step_mode: Step mode, ``"s"`` for single-step or ``"m"`` for multi-step
        :type step_mode: str
        :param backend: Execution backend; only ``"torch"`` is supported
        :type backend: str
        :param store_state_seqs: Whether to store complete state trajectories in
            multi-step mode. If ``True``, ``state_seqs`` contains two tensors in
            ``[v_seq, m_seq]`` order, each with shape ``[T, N, *]``. Functional
            forward does not write this cache. This class does not use the parent
            ``store_v_seq`` or ``v_seq`` interface; ``store_state_seqs`` controls
            all state trajectories
        :type store_state_seqs: bool
        :raises AssertionError: If ``tau`` is not a float greater than ``1.0``
        :raises ValueError: If ``step_mode`` is neither ``"s"`` nor ``"m"``
        :raises NotImplementedError: If ``backend`` is not ``"torch"``
        """
        assert isinstance(tau, float) and tau > 1.0
        super().__init__(
            v_threshold,
            None,
            surrogate_function,
            False,
            step_mode,
            backend,
            False,
        )
        self.tau = tau
        self.register_memory("m", 0.0)
        self.store_state_seqs = store_state_seqs

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch",)

    @property
    def store_state_seqs(self) -> bool:
        r"""
        **API Language** - :ref:`中文 <ComplementaryLIFNode.store_state_seqs-cn>` | :ref:`English <ComplementaryLIFNode.store_state_seqs-en>`

        ----

        .. _ComplementaryLIFNode.store_state_seqs-cn:

        * **中文**

        :return: 是否在常规多步前向后保存 ``[v_seq, m_seq]``。修改该属性会清除
            已保存的 ``state_seqs``
        :rtype: bool

        ----

        .. _ComplementaryLIFNode.store_state_seqs-en:

        * **English**

        :return: Whether to store ``[v_seq, m_seq]`` after a regular multi-step
            forward. Assigning this property clears the cached ``state_seqs``
        :rtype: bool
        """
        return self._store_state_seqs

    @store_state_seqs.setter
    def store_state_seqs(self, value: bool) -> None:
        self._store_state_seqs = value
        self.state_seqs = None

    def reset(self) -> None:
        super().reset()
        self.state_seqs = None

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        states = super().materialize_states(inputs, states, step_mode)
        v, m = states
        if not isinstance(m, torch.Tensor):
            m = torch.full_like(v, m, requires_grad=False)
        elif m.ndim == 0:
            m = m.to(dtype=v.dtype, device=v.device).expand_as(v)
        elif m.shape != v.shape:
            m = torch.zeros_like(v, requires_grad=False)
        elif m.dtype != v.dtype or m.device != v.device:
            m = m.to(dtype=v.dtype, device=v.device)
        return v, m

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v, m = states
        v = functional.lif_charge(x, v, self.tau, False, None)
        m = m * torch.sigmoid(v / self.tau)
        spike_function = (
            self.surrogate_function
            if self.training or not self.surrogate_function.spiking
            else surrogate.heaviside
        )
        spike = spike_function(v - self.v_threshold)
        m = m + spike
        v = v - spike * (self.v_threshold + torch.sigmoid(m))
        return (spike,), (v, m)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        if not self.store_state_seqs:
            return base.MemoryModule.multi_step_forward(self, x_seq)

        states = self.materialize_states((x_seq,), tuple(self._memories.values()), "m")
        spike_steps = []
        v_steps = []
        m_steps = []
        for x in x_seq:
            (spike,), states = self.single_step_functional_forward((x,), states)
            spike_steps.append(spike)
            v_steps.append(states[0])
            m_steps.append(states[1])

        self.v, self.m = states
        self.state_seqs = [torch.stack(v_steps), torch.stack(m_steps)]
        return torch.stack(spike_steps)

    def extra_repr(self) -> str:
        return super().extra_repr() + f", tau={self.tau}"


class CUBALIFNode(BaseNode):
    def __init__(
        self,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
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
        :type surrogate_function: surrogate.SurrogateFunctionBase
        """
        super().__init__(v_threshold, v_reset, surrogate_function)

        self.register_memory("c", 0.0)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        states = super().materialize_states(inputs, states, step_mode)
        x = states[0]
        c = states[1]
        if isinstance(c, float):
            c = torch.full_like(x, c, requires_grad=False)
        elif isinstance(c, torch.Tensor):
            if c.shape != x.shape:
                c = torch.zeros_like(x, requires_grad=False)
            elif c.dtype != x.dtype or c.device != x.device:
                c = c.to(dtype=x.dtype, device=x.device)
        return (states[0], c, *states[2:])

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v, c = states
        spike, c, v = functional.cuba_lif_step(
            x,
            c,
            v,
            self.c_decay,
            self.v_decay,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, c)


class LIAFNode(LIFNode):
    def __init__(self, act: Callable, threshold_related: bool, *args, **kwargs):
        """
        **API Language** - :ref:`中文 <LIAFNode.__init__-cn>` | :ref:`English <LIAFNode.__init__-en>`

        ----

        .. _LIAFNode.__init__-cn:

        * **中文**

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

        * **English**

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

    @property
    def supported_backends(self):
        return ("torch",)

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]

        y, v = functional.liaf_step(
            x,
            v,
            self.tau,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.act,
            self.threshold_related,
            self.surrogate_function,
            self.detach_reset,
        )
        return (y,), (v, *states[1:])
