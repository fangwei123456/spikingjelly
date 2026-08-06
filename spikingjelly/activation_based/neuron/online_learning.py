from typing import Optional

import torch

from .. import functional, surrogate
from .lif import LIFNode

__all__ = ["OTTTLIFNode", "SLTTLIFNode"]


class _OnlineLIFNode(LIFNode):
    @property
    def supported_backends(self):
        return "torch"

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        spike, v = functional.lif_step(
            x,
            v.detach() if self.training else v,
            self.tau,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function if self.training else surrogate.heaviside,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])


class OTTTLIFNode(_OnlineLIFNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = False,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = None,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = True,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <OTTTLIFNode.__init__-cn>` | :ref:`English <OTTTLIFNode.__init__-en>`

        ----

        .. _OTTTLIFNode.__init__-cn:

        * **中文**

        用于 OTTT 训练的单步 LIF 神经元。该类继承 :class:`LIFNode` 的放电行为，但仅支持
        ``step_mode='s'`` 和 ``backend='torch'``，并在训练时额外维护迹以供后续模块使用。

        OTTT LIF 神经元模型，来源于
        `Online Training Through Time for Spiking Neural Networks
        <https://arxiv.org/pdf/2210.04195.pdf>`_。
        其正向传播过程与 Leaky Integrate-and-Fire（LIF）神经元相同。

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None`` ，当神经元释放脉冲后，
            电压会被重置为 ``v_reset`` ；如果设置为 ``None`` ，当神经元释放脉冲后，
            电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: 是否将 reset 过程的计算图分离。
            该参数在本模块中不起作用，仅为保持代码接口统一而保留
        :type detach_reset: bool

        :param step_mode: 步进模式。为了保证神经元的显存占用较小，仅支持 ``'s'`` （单步）
        :type step_mode: str

        :param backend: 使用的后端。当前实现仅支持 ``'torch'``，其他取值会触发
            ``ValueError``
        :type backend: str

        :param store_v_seq: 为保持与 :class:`LIFNode` 接口一致而保留的参数。本类仅支持
            单步模式，不会使用 ``self.v_seq``
        :type store_v_seq: bool

        ----

        .. _OTTTLIFNode.__init__-en:

        * **English**

        Single-step LIF neuron for OTTT training. This class inherits the firing
        behavior of :class:`LIFNode`, but only supports ``step_mode='s'`` and
        ``backend='torch'``. During training it also maintains a trace for
        downstream modules.

        OTTT LIF neuron, proposed in
        `Online Training Through Time for Spiking Neural Networks
        <https://arxiv.org/pdf/2210.04195.pdf>`_.
        This neuron is designed for OTTT.
        Its forward propagation is identical to that of the
        Leaky Integrate-and-Fire (LIF) neuron.

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of the neuron
        :type v_threshold: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane
            potential will be reset to ``v_reset`` after firing a spike.
            If ``None``, the membrane potential will subtract ``v_threshold``
            after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function used to compute surrogate gradients
            of the Heaviside step function in backward propagation
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: whether to detach the computation graph of the reset
            operation in backward propagation. This parameter has no effect in
            this module and is retained solely for interface consistency
        :type detach_reset: bool

        :param step_mode: step mode. To guarantee memory-efficient computation,
            only ``'s'`` (single-step) mode is supported
        :type step_mode: str

        :param backend: backend for this neuron layer. The current
            implementation only supports ``'torch'``; other values raise
            ``ValueError``
        :type backend: str

        :param store_v_seq: retained for interface consistency with
            :class:`LIFNode`. This class only supports single-step mode and does
            not use ``self.v_seq``
        :type store_v_seq: bool
        """

        super().__init__(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        assert step_mode == "s", (
            "Please use single-step mode to enable memory-efficient training."
        )
        self.register_memory("trace", None)

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        states = super().materialize_states(inputs, states, step_mode)
        trace = states[1]
        if self.training and trace is None:
            trace = torch.zeros_like(states[0])
        return states[0], trace

    @staticmethod
    def track_trace(spike: torch.Tensor, trace: torch.Tensor, tau: float):
        with torch.no_grad():
            trace = trace * (1.0 - 1.0 / tau) + spike
        return trace

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        (spike,), (v, trace) = super().single_step_functional_forward(
            inputs, states, **kwargs
        )
        if not self.training:
            return (spike,), (v, trace)
        trace = self.track_trace(spike, trace, self.tau)
        return (spike, trace), (v, trace)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor | list[torch.Tensor]:
        outputs = super().single_step_forward(x)
        return list(outputs) if isinstance(outputs, tuple) else outputs


class SLTTLIFNode(_OnlineLIFNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = True,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <SLTTLIFNode.__init__-cn>` | :ref:`English <SLTTLIFNode.__init__-en>`

        ----

        .. _SLTTLIFNode.__init__-cn:

        * **中文**

        用于 SLTT 训练的单步 LIF 神经元。该类继承 :class:`LIFNode` 的放电行为，但仅支持
        ``step_mode='s'`` 和 ``backend='torch'``，并通过截断时间梯度来降低训练的时间与显存开销。

        SLTT LIF 神经元模型，来源于
        `Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks
        <https://arxiv.org/pdf/2302.14311.pdf>`_。
        该模型在正向传播过程中与 Leaky Integrate-and-Fire（LIF）神经元相同，
        通过截断时间梯度实现更高的时间与显存效率。

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None`` ，当神经元释放脉冲后，
            电压会被重置为 ``v_reset`` ；如果设置为 ``None`` ，当神经元释放脉冲后，
            电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: 是否将 reset 过程的计算图分离。
            该参数在本模块中不起作用，仅为保持代码接口统一而保留
        :type detach_reset: bool

        :param step_mode: 步进模式。为了保证神经元的显存占用较小，仅支持 ``'s'`` （单步）
        :type step_mode: str

        :param backend: 使用的后端。当前实现仅支持 ``'torch'``，其他取值会触发
            ``ValueError``
        :type backend: str

        :param store_v_seq: 为保持与 :class:`LIFNode` 接口一致而保留的参数。本类仅支持
            单步模式，不会使用 ``self.v_seq``
        :type store_v_seq: bool

        ----

        .. _SLTTLIFNode.__init__-en:

        * **English**

        Single-step LIF neuron for SLTT training. This class inherits the firing
        behavior of :class:`LIFNode`, but only supports ``step_mode='s'`` and
        ``backend='torch'``. It reduces training time and memory cost by truncating
        temporal gradients.

        SLTT LIF neuron, proposed in
        `Towards Memory- and Time-Efficient Backpropagation for Training Spiking Neural Networks
        <https://arxiv.org/pdf/2302.14311.pdf>`_.
        The forward propagation of this neuron is identical to that of the
        Leaky Integrate-and-Fire (LIF) neuron, while it truncates temporal gradients to enable more
        memory- and time-efficient training.

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold voltage of the neuron
        :type v_threshold: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane
            potential will be reset to ``v_reset`` after firing a spike.
            If ``None``, the membrane potential will subtract ``v_threshold``
            after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function used to compute surrogate gradients
            of the Heaviside step function in backward propagation
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: whether to detach the computation graph of the reset
            operation in backward propagation. This parameter has no effect in
            this module and is retained solely for interface consistency
        :type detach_reset: bool

        :param step_mode: step mode. To guarantee memory-efficient computation,
            only ``'s'`` (single-step) mode is supported
        :type step_mode: str

        :param backend: backend for this neuron layer. The current
            implementation only supports ``'torch'``; other values raise
            ``ValueError``
        :type backend: str

        :param store_v_seq: retained for interface consistency with
            :class:`LIFNode`. This class only supports single-step mode and does
            not use ``self.v_seq``
        :type store_v_seq: bool
        """
        super().__init__(
            tau,
            decay_input,
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        assert step_mode == "s", (
            "Please use single-step mode to enable memory-efficient training."
        )
