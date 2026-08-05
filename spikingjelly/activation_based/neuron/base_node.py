from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .. import base, surrogate

__all__ = ["BaseNode", "NonSpikingBaseNode", "SimpleBaseNode"]


class SimpleBaseNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        """
        **API Language** - :ref:`中文 <SimpleBaseNode.__init__-cn>` | :ref:`English <SimpleBaseNode.__init__-en>`

        ----

        .. _SimpleBaseNode.__init__-cn:

        * **中文**

        面向教学和神经元动力学自定义的纯 PyTorch 接口。``Simple`` 描述的是接口
        目标，并不表示一种神经元数学模型。该类将 SNN 神经元的充电、放电和重置
        职责直接表达为 :meth:`neuronal_charge`、:meth:`neuronal_fire` 和
        :meth:`neuronal_reset`；用户通常只需要重写充电方程。该类不提供原生
        functional 状态转移，
        :func:`spikingjelly.activation_based.base.to_functional_forward` 会通过通用
        状态替换路径转换其实例。

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float
        :param v_reset: 神经元的重置电压
        :type v_reset: Optional[float]
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否将 reset 过程的计算图分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _SimpleBaseNode.__init__-en:

        * **English**

        A pure-PyTorch interface for teaching and customizing neuron dynamics.
        ``Simple`` describes the interface goal; it is not a neuron mathematical
        model. The class exposes the charge, fire, and reset responsibilities of an
        SNN neuron directly as :meth:`neuronal_charge`, :meth:`neuronal_fire`, and
        :meth:`neuronal_reset`; users normally only need to override the charge
        equation. This class does not provide a native functional transition.
        Instances are converted by
        :func:`spikingjelly.activation_based.base.to_functional_forward` through
        the general state-substitution path.

        :param v_threshold: threshold of this neurons layer
        :type v_threshold: float
        :param v_reset: reset voltage of this neurons layer
        :type v_reset: Optional[float]
        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: whether detach the computation graph of reset in backward
        :type detach_reset: bool
        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.surrogate_function = surrogate_function
        self.detach_reset = detach_reset
        self.step_mode = step_mode
        self.register_memory(name="v", value=0.0)

    def single_step_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SimpleBaseNode.single_step_forward-cn>` | :ref:`English <SimpleBaseNode.single_step_forward-en>`

        ----

        .. _SimpleBaseNode.single_step_forward-cn:

        * **中文**

        依次执行充电、放电和重置，完成一个时间步的前向传播。

        :param x: 单步输入张量
        :type x: torch.Tensor
        :return: 输出脉冲
        :rtype: torch.Tensor

        ----

        .. _SimpleBaseNode.single_step_forward-en:

        * **English**

        Run one time step by applying charge, fire, and reset in order.

        :param x: Single-step input tensor
        :type x: torch.Tensor
        :return: Output spikes
        :rtype: torch.Tensor
        """
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SimpleBaseNode.multi_step_forward-cn>` | :ref:`English <SimpleBaseNode.multi_step_forward-en>`

        ----

        .. _SimpleBaseNode.multi_step_forward-cn:

        * **中文**

        沿时间维逐步调用 :meth:`single_step_forward`。

        :param x_seq: 输入序列，形状约定为 ``[T, N, *]``
        :type x_seq: torch.Tensor
        :return: 沿时间维堆叠的输出脉冲序列
        :rtype: torch.Tensor

        ----

        .. _SimpleBaseNode.multi_step_forward-en:

        * **English**

        Apply :meth:`single_step_forward` successively along the time dimension.

        :param x_seq: Input sequence with conventional shape ``[T, N, *]``
        :type x_seq: torch.Tensor
        :return: Output spike sequence stacked along the time dimension
        :rtype: torch.Tensor
        """
        return torch.stack([self.single_step_forward(x) for x in x_seq])

    def neuronal_charge(self, x: torch.Tensor) -> None:
        r"""
        **API Language** - :ref:`中文 <SimpleBaseNode.neuronal_charge-cn>` | :ref:`English <SimpleBaseNode.neuronal_charge-en>`

        ----

        .. _SimpleBaseNode.neuronal_charge-cn:

        * **中文**

        使用单步输入更新 ``self.v``。继承 :class:`SimpleBaseNode` 自定义动力学时，
        通常只需要实现本方法。

        :param x: 单步输入张量
        :type x: torch.Tensor
        :raises NotImplementedError: 子类未实现充电方程时抛出

        ----

        .. _SimpleBaseNode.neuronal_charge-en:

        * **English**

        Update ``self.v`` from one input step. Subclasses of
        :class:`SimpleBaseNode` normally only need to implement this method to
        customize their dynamics.

        :param x: Single-step input tensor
        :type x: torch.Tensor
        :raises NotImplementedError: If the subclass does not implement a charge equation
        """
        raise NotImplementedError

    def neuronal_fire(self) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <SimpleBaseNode.neuronal_fire-cn>` | :ref:`English <SimpleBaseNode.neuronal_fire-en>`

        ----

        .. _SimpleBaseNode.neuronal_fire-cn:

        * **中文**

        根据当前膜电位和阈值计算脉冲。

        :return: 输出脉冲
        :rtype: torch.Tensor

        ----

        .. _SimpleBaseNode.neuronal_fire-en:

        * **English**

        Calculate spikes from the current membrane potential and threshold.

        :return: Output spikes
        :rtype: torch.Tensor
        """
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike: torch.Tensor) -> None:
        r"""
        **API Language** - :ref:`中文 <SimpleBaseNode.neuronal_reset-cn>` | :ref:`English <SimpleBaseNode.neuronal_reset-en>`

        ----

        .. _SimpleBaseNode.neuronal_reset-cn:

        * **中文**

        根据输出脉冲对 ``self.v`` 执行硬重置或软重置。

        :param spike: 当前时间步的输出脉冲
        :type spike: torch.Tensor

        ----

        .. _SimpleBaseNode.neuronal_reset-en:

        * **English**

        Apply hard or soft reset to ``self.v`` according to the output spikes.

        :param spike: Output spikes of the current time step
        :type spike: torch.Tensor
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v = self.v - self.v_threshold * spike_d

        else:
            # hard reset
            self.v = spike_d * self.v_reset + (1.0 - spike_d) * self.v


class BaseNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language** - :ref:`中文 <BaseNode.__init__-cn>` | :ref:`English <BaseNode.__init__-en>`

        ----

        .. _BaseNode.__init__-cn:

        * **中文**

        生产级可微分 SNN 神经元的基类。常规前向由显式状态的 functional
        状态转移支持；子类应实现 ``single_step_functional_forward``，并仅在存在
        独立序列实现或专用 kernel 时重写 ``multi_step_functional_forward``。
        若需要通过 ``neuronal_charge`` 修改 Python 神经元方程，请继承
        :class:`SimpleBaseNode`。

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。如果不为 ``None``，当神经元释放脉冲后，电压会被重置为 ``v_reset``；
            如果设置为 ``None``，当神经元释放脉冲后，电压会被减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase

        :param detach_reset: 是否将reset过程的计算图分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。该参数是显式执行后端选择：设置为 ``'torch'``、``'cupy'`` 或 ``'triton'`` 时，
            将分别使用对应后端，不会隐式切换到其他后端。在支持的情况下，``'cupy'`` 或 ``'triton'`` 后端通常更快。
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
        :type store_v_seq: bool

        ----

        .. _BaseNode.__init__-en:

        * **English**

        Base class for production differentiable spiking neurons. Regular forward
        is backed by an explicit-state functional transition. Subclasses should
        implement ``single_step_functional_forward`` and override
        ``multi_step_functional_forward`` only for an independent sequence
        implementation or specialized kernel. Inherit :class:`SimpleBaseNode`
        instead to modify Python neuron equations through ``neuronal_charge``.

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
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        assert isinstance(detach_reset, bool)
        super().__init__()

        if v_reset is None:
            self.register_memory("v", 0.0)
        else:
            self.register_memory("v", v_reset)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function

        self.step_mode = step_mode
        self.backend = backend

        self.store_v_seq = store_v_seq

        # Lava conversion reads this instance value so callers can tune its scale.
        self.lava_s_cale = 1 << 6

        # Kept empty for compatibility; compiled graphs live in inductor_cache.
        self._inductor_compiled_graphs = {}

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value and "v_seq" not in self._memories:
            self.register_memory("v_seq", None)
        elif not value and "v_seq" in self._memories:
            del self.v_seq

    @staticmethod
    def apply_hard_reset(v: torch.Tensor, spike: torch.Tensor, v_reset: float):
        v = (1.0 - spike) * v + spike * v_reset
        return v

    @staticmethod
    def apply_soft_reset(v: torch.Tensor, spike: torch.Tensor, v_threshold: float):
        v = v - spike * v_threshold
        return v

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <BaseNode.single_step_functional_forward-cn>` | :ref:`English <BaseNode.single_step_functional_forward-en>`

        ----

        .. _BaseNode.single_step_functional_forward-cn:

        * **中文**

        使用显式状态执行一个神经元时间步。生产级神经元子类必须实现本方法，
        且不得修改模块 memory 或传入的 ``states``。

        :param inputs: 单步输入张量元组
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 当前神经元的显式状态元组
        :type states: tuple[object, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        :raises NotImplementedError: 子类未实现 functional 状态转移时抛出

        ----

        .. _BaseNode.single_step_functional_forward-en:

        * **English**

        Run one neuron time step with explicit states. Production neuron
        subclasses must implement this method without mutating module memories
        or the supplied ``states``.

        :param inputs: Tuple of single-step input tensors
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Tuple of explicit neuron states
        :type states: tuple[object, ...]
        :return: ``(outputs, updated_states)``
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        :raises NotImplementedError: If the subclass does not implement a functional state transition
        """
        raise NotImplementedError(
            f"{self._get_name()} must implement single_step_functional_forward."
        )

    def extra_repr(self):
        return f"v_threshold={self.v_threshold}, v_reset={self.v_reset}, detach_reset={self.detach_reset}, step_mode={self.step_mode}, backend={self.backend}"

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        output_steps = []
        v_steps = []
        for t in range(inputs[0].shape[0]):
            outputs, states = self.single_step_functional_forward(
                tuple(x[t] for x in inputs), states, **kwargs
            )
            output_steps.append(outputs)
            if self.store_v_seq:
                v_steps.append(states[0])

        if self.store_v_seq:
            states = (states[0], torch.stack(v_steps), *states[2:])
        return (
            tuple(
                torch.stack(output_seq)
                for output_seq in zip(*output_steps, strict=True)
            ),
            states,
        )

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0]
        if step_mode == "m" and x.dim() > 0 and x.shape[0] > 0:
            x = x[0]
        v = states[0]
        if isinstance(v, float):
            v = torch.full_like(x, v, requires_grad=False)
        elif isinstance(v, torch.Tensor):
            if v.shape != x.shape:
                v = torch.full_like(
                    x,
                    self.v_reset if self.v_reset is not None else 0.0,
                    requires_grad=False,
                )
            elif v.dtype != x.dtype or v.device != x.device:
                v = v.to(dtype=x.dtype, device=x.device)
        return (v, *states[1:])


class NonSpikingBaseNode(nn.Module, base.MultiStepModule):
    def __init__(self, decode: Optional[str] = None):
        """
        :param decode: 解码方式。若不为 ``None``，在 ``forward`` 中将使用该方式对膜电位序列进行解码
        :type decode: Optional[str]
        """
        super().__init__()
        self.decode = decode

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def forward(self, x_seq: torch.Tensor):
        self.v = torch.full_like(x_seq[0].data, fill_value=0.0)

        T = x_seq.shape[0]
        v_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            v_seq.append(self.v)

        if self.decode == "max-mem":
            return torch.max(torch.stack(v_seq, 0), 0).values
        elif self.decode == "max-abs-mem":
            v_stack = torch.stack(v_seq, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (
                max_mem.abs() <= min_mem.abs()
            )
            return mem
        elif self.decode == "mean-mem":
            return torch.mean(torch.stack(v_seq, 0), 0)
        elif self.decode == "last_mem":
            return v_seq[-1]
        else:
            return v_seq
