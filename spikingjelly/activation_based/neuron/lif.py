from typing import Optional

import torch

from .. import functional, surrogate
from .base_node import BaseNode, NonSpikingBaseNode, SimpleBaseNode


__all__ = ["SimpleLIFNode", "LIFNode", "NonSpikingLIFNode"]


class SimpleLIFNode(SimpleBaseNode):
    def __init__(
        self,
        tau: float,
        decay_input: bool,
        v_threshold: float = 1.0,
        v_reset: float = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        """
        **API Language** - :ref:`中文 <SimpleLIFNode.__init__-cn>` | :ref:`English <SimpleLIFNode.__init__-en>`

        ----

        .. _SimpleLIFNode.__init__-cn:

        * **中文**

        基于 :class:`SimpleBaseNode` 充电-放电-重置接口的纯 PyTorch LIF 实现。

        ----

        .. _SimpleLIFNode.__init__-en:

        * **English**

        A pure-PyTorch LIF implementation built on the charge-fire-reset interface
        of :class:`SimpleBaseNode`.

        :param tau: 膜电位时间常数（详见父类 :class:`LIFNode`）
        :type tau: float
        :param decay_input: 输入是否参与衰减（详见父类）
        :type decay_input: bool
        :param v_threshold: 神经元的阈值电压（详见父类）
        :type v_threshold: float
        :param v_reset: 神经元的重置电压（详见父类）
        :type v_reset: float
        :param surrogate_function: 替代梯度函数（详见父类）
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否将 reset 过程的计算图分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可为 ``\"s\"`` 或 ``\"m\"``
        :type step_mode: str

        :param tau: Membrane time constant (see parent class :class:`LIFNode`)
        :type tau: float
        :param decay_input: Whether input participates in decay (see parent)
        :type decay_input: bool
        :param v_threshold: Threshold voltage of the neuron (see parent)
        :type v_threshold: float
        :param v_reset: Reset voltage of the neuron (see parent)
        :type v_reset: float
        :param surrogate_function: Surrogate gradient function (see parent)
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach reset graph in backward
        :type detach_reset: bool
        :param step_mode: Step mode, either ``\"s\"`` or ``\"m\"``
        :type step_mode: str
        """
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )
        self.tau = tau
        self.decay_input = decay_input

    def neuronal_charge(self, x: torch.Tensor):
        """
        If ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]
        """
        if self.decay_input:
            self.v = self.v + (self.v_reset - self.v + x) / self.tau
        else:
            self.v = self.v + (self.v_reset - self.v) / self.tau + x


class LIFNode(BaseNode):
    def __init__(
        self,
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
        """
        **API Language** - :ref:`中文 <LIFNode.__init__-cn>` | :ref:`English <LIFNode.__init__-en>`

        ----

        .. _LIFNode.__init__-cn:

        * **中文**

        Leaky Integrate-and-Fire 神经元模型，可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        :param tau: 膜电位时间常数
        :type tau: float

        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: 是否将 reset 过程的计算图分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。该参数是显式执行后端选择：设置为 ``'torch'``、``'cupy'`` 或 ``'triton'`` 时，将分别使用
            对应后端，不会隐式切换到其他后端。在支持的情况下，使用 ``'cupy'`` 或 ``'triton'`` 后端通常更快。
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
        :type store_v_seq: bool

        ----

        .. _LIFNode.__init__-en:

        * **English**

        The Leaky Integrate-and-Fire neuron, which can be seen as a leaky integrator.
        The subthreshold neural dynamics of it is as followed:

        If ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        If ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        :param tau: membrane time constant
        :type tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float

        :param v_reset: reset voltage of this neurons layer. If not ``None``, the neuron's voltage will be set to ``v_reset``
            after firing a spike. If ``None``, the neuron's voltage will subtract ``v_threshold`` after firing a spike
        :type v_reset: Optional[float]

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend for this neurons layer. Different ``step_mode`` may support different backends. Users can
            print ``self.supported_backends`` to check what backends are supported by the current ``step_mode``. This argument
            is an explicit execution-backend choice: ``'torch'``, ``'cupy'``, and ``'triton'`` each use their own backend and
            are not silently upgraded to another backend. If supported, ``'cupy'`` or ``'triton'`` is usually faster
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
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

        self.tau = tau
        self.decay_input = decay_input

    @property
    def supported_backends(self):
        if self.step_mode == "s":
            return ("torch", "cupy")
        elif self.step_mode == "m":
            return ("torch", "cupy", "triton", "inductor")
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return super().extra_repr() + f", tau={self.tau}"

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]

        if self.backend == "torch":
            surrogate_function = (
                self.surrogate_function
                if self.training
                or not getattr(self.surrogate_function, "spiking", True)
                else surrogate.heaviside
            )
            spike, v = functional.lif_step(
                x,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                surrogate_function,
                self.detach_reset,
            )
        elif self.backend == "cupy":
            spike, v = functional.lif_step_cupy(
                x,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
            )
        else:
            raise ValueError(self.backend)
        return (spike,), (v, *states[1:])

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        v = states[0]

        if self.backend == "inductor":
            spike_seq, v, _ = functional.lif_multi_step_inductor(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                False,
            )
        elif self.backend == "cupy":
            spike_seq, v, _ = functional.lif_multi_step_cupy(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                False,
            )
        elif self.backend == "triton":
            if not self.training and not getattr(
                self.surrogate_function, "spiking", True
            ):
                raise NotImplementedError(
                    "Triton backend only supports spiking surrogate functions. "
                    "Use backend='torch' for non-spiking surrogate functions."
                )
            spike_seq, v, _ = functional.lif_multi_step_triton(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                False,
            )
        elif self.backend == "torch":
            return super().multi_step_functional_forward(inputs, states, **kwargs)
        else:
            raise ValueError(self.backend)

        return (spike_seq,), (v,)

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if not self.store_v_seq or self.backend == "torch":
            return super().multi_step_forward(x_seq, *args, **kwargs)

        states = self.materialize_states(
            (x_seq, *args), tuple(self._memories.values()), "m"
        )
        v = states[0]
        if self.backend == "inductor":
            spike_seq, v, v_seq = functional.lif_multi_step_inductor(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                True,
            )
        elif self.backend == "cupy":
            spike_seq, v, v_seq = functional.lif_multi_step_cupy(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                True,
            )
        elif self.backend == "triton":
            if not self.training and not getattr(
                self.surrogate_function, "spiking", True
            ):
                raise NotImplementedError(
                    "Triton backend only supports spiking surrogate functions. "
                    "Use backend='torch' for non-spiking surrogate functions."
                )
            spike_seq, v, v_seq = functional.lif_multi_step_triton(
                x_seq,
                v,
                self.tau,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                True,
            )
        else:
            raise ValueError(self.backend)
        self.v = v
        self.v_seq = v_seq
        return spike_seq


class NonSpikingLIFNode(NonSpikingBaseNode):
    def __init__(self, tau: float = 2.0, decode: Optional[str] = None):
        """Non-spiking version of :class:`LIFNode` that outputs continuous-valued membrane potentials instead of spikes.
        See also: :class:`spikingjelly.activation_based.layer.misc.SynapseFilter`.

        :param tau: 膜电位时间常数
        :type tau: float
        :param decode: 解码方式
        :type decode: Optional[str]

        :param tau: Membrane time constant
        :type tau: float
        :param decode: Decoding method
        :type decode: Optional[str]
        """
        super().__init__(decode)

        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) / self.tau
