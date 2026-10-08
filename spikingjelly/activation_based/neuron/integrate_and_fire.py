import numbers
from typing import Optional, Union

import torch

from .. import base, functional, surrogate
from .base_node import BaseNode, NonSpikingBaseNode, SimpleBaseNode

__all__ = [
    "SimpleIFNode",
    "IFNode",
    "HalfThresholdIFNode",
    "ActivationAwareIFNode",
    "NonSpikingIFNode",
]


class SimpleIFNode(SimpleBaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
    ):
        """
        **API Language** - :ref:`中文 <SimpleIFNode.__init__-cn>` | :ref:`English <SimpleIFNode.__init__-en>`

        ----

        .. _SimpleIFNode.__init__-cn:

        * **中文**

        基于 :class:`SimpleBaseNode` 充电-放电-重置接口的纯 PyTorch IF 实现。

        :param v_threshold: 神经元阈值电压
        :type v_threshold: float
        :param v_reset: 神经元重置电压
        :type v_reset: Optional[float]
        :param surrogate_function: 替代梯度函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否在反向传播时分离 reset 计算图
        :type detach_reset: bool
        :param step_mode: 步进模式，可为 ``"s"`` 或 ``"m"``
        :type step_mode: str

        ----

        .. _SimpleIFNode.__init__-en:

        * **English**

        A pure-PyTorch IF implementation built on the charge-fire-reset interface
        of :class:`SimpleBaseNode`.

        :param v_threshold: Threshold voltage of the neuron
        :type v_threshold: float
        :param v_reset: Reset voltage of the neuron
        :type v_reset: Optional[float]
        :param surrogate_function: Surrogate gradient function
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach reset graph in backward
        :type detach_reset: bool
        :param step_mode: Step mode, either ``"s"`` or ``"m"``
        :type step_mode: str
        """
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <SimpleIFNode.neuronal_charge-cn>` | :ref:`English <SimpleIFNode.neuronal_charge-en>`

        ----

        .. _SimpleIFNode.neuronal_charge-cn:

        * **中文**

        神经元充电的微分方程：

        .. math::
            H[t] = V[t-1] + X[t]

        :param x: 输入电压
        :type x: torch.Tensor
        :return: None（膜电位更新存储在 ``self.v`` 中）

        ----

        .. _SimpleIFNode.neuronal_charge-en:

        * **English**

        The differential equation for neuronal charge:

        .. math::
            H[t] = V[t-1] + X[t]

        :param x: Input voltage
        :type x: torch.Tensor
        :return: None (membrane potential is stored in ``self.v``)
        """
        self.v = self.v + x


class IFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        store_v_seq: bool = False,
    ):
        """
        **API Language** - :ref:`中文 <IFNode.__init__-cn>` | :ref:`English <IFNode.__init__-en>`

        ----

        .. _IFNode.__init__-cn:

        * **中文**

        Integrate-and-Fire 神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像 LIF 神经元那样衰减。其阈下神经动力学方程为：

        .. math::
            H[t] = V[t-1] + X[t]

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

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存。在使用 ``step_mode = 's'`` 时，每个时间步结束后的电压会被追加到 ``self.v_seq`` ，
            直到调用 ``reset()`` ；每一步都会复制整个序列，因此该选项主要用于监控和调试
        :type store_v_seq: bool

        ----

        .. _IFNode.__init__-en:

        * **English**

        The Integrate-and-Fire neuron, which can be seen as an ideal integrator. The voltage of the IF neuron will not decay
        as that of the LIF neuron. The sub-threshold neural dynamics of it is as followed:

        .. math::
            H[t] = V[t-1] + X[t]

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

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption. When using ``step_mode = 's'``, the voltage after each time-step is appended to
            ``self.v_seq`` until ``reset()`` is called; every step copies the whole sequence, so this option is meant for
            monitoring and debugging
        :type store_v_seq: bool
        """
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
        )

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        spike, v = functional.if_step(
            x,
            v,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        spikes, v, _ = functional.if_multi_step(
            inputs[0],
            states[0],
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
            False,
            neuron_storage=(
                None if self._neuron_precision is None else self._neuron_precision[0]
            ),
            neuron_fwd=(
                "fp32" if self._neuron_precision is None else self._neuron_precision[1]
            ),
            neuron_bwd=(
                "fp32" if self._neuron_precision is None else self._neuron_precision[2]
            ),
        )
        return (spikes,), (v,)

    def multi_step_forward(
        self, x_seq: torch.Tensor, *args: torch.Tensor, **kwargs: object
    ) -> torch.Tensor | tuple[torch.Tensor, ...]:
        if (
            type(self).single_step_functional_forward
            is not IFNode.single_step_functional_forward
            or type(self).multi_step_functional_forward
            is not IFNode.multi_step_functional_forward
        ):
            return super().multi_step_forward(x_seq, *args, **kwargs)
        inputs = (x_seq, *args)
        states = self.materialize_states(inputs, tuple(self._memories.values()), "m")
        precision = self._neuron_precision
        spikes, v, v_seq = functional.if_multi_step(
            x_seq,
            states[0],
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
            self.store_v_seq,
            neuron_storage=None if precision is None else precision[0],
            neuron_fwd="fp32" if precision is None else precision[1],
            neuron_bwd="fp32" if precision is None else precision[2],
        )
        self._memories["v"] = v
        self.v_seq = v_seq
        return spikes


class HalfThresholdIFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <HalfThresholdIFNode.__init__-cn>` | :ref:`English <HalfThresholdIFNode.__init__-en>`

        ----

        .. _HalfThresholdIFNode.__init__-cn:

        * **中文**

        半阈值初始膜电位的 Integrate-and-Fire 神经元。每次调用 ``reset()``
        后膜电位会恢复为 ``v_threshold / 2``。单步前向中的脉冲后重置仍使用
        标准软重置。除此之外，其充电、放电和重置动力学与软重置 IF 神经元一致：

        .. math::

            H[t] = V[t-1] + X[t]

        .. math::

            S[t] = \Theta(H[t] - V_{threshold})

        .. math::

            V[t] = H[t] - S[t] V_{threshold}

        训练时使用 ``surrogate_function`` 为脉冲函数提供替代梯度；前向输出仍为
        离散脉冲。

        :param v_threshold: 神经元阈值电压，必须为正实数或单元素张量
        :type v_threshold: float or torch.Tensor
        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否在反向传播时分离 reset 计算图
        :type detach_reset: bool
        :param step_mode: 步进模式，可以为 ``"s"`` 或 ``"m"``
        :type step_mode: str
        :param store_v_seq: 是否将每个时间步的膜电位序列保存到 ``self.v_seq``。在
            ``step_mode="s"`` 时膜电位会逐步追加，直到调用 ``reset()``；每一步都会
            复制整个序列，因此该选项主要用于监控和调试
        :type store_v_seq: bool
        :raises TypeError: 当 ``v_threshold`` 不是实数或张量时抛出
        :raises ValueError: 当 ``v_threshold`` 不是单元素有限正数时抛出

        ----

        .. _HalfThresholdIFNode.__init__-en:

        * **English**

        An Integrate-and-Fire neuron with half-threshold initial membrane
        potential. After each explicit ``reset()``, its membrane potential is
        restored to ``v_threshold / 2``. The per-step post-spike reset still
        uses the standard soft reset. Apart from the initial reset value, its
        charge, fire, and reset dynamics are the same as a soft-reset IF neuron:

        .. math::

            H[t] = V[t-1] + X[t]

        .. math::

            S[t] = \Theta(H[t] - V_{threshold})

        .. math::

            V[t] = H[t] - S[t] V_{threshold}

        During training, ``surrogate_function`` provides surrogate gradients for
        the spike function; the forward output remains discrete spikes.

        :param v_threshold: Threshold voltage of the neuron, which must be a
            finite positive real number or a scalar tensor
        :type v_threshold: float or torch.Tensor
        :param surrogate_function: Surrogate gradient function for the spike
            function in backward propagation
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach the reset computation graph in
            backward propagation
        :type detach_reset: bool
        :param step_mode: Step mode, either ``"s"`` or ``"m"``
        :type step_mode: str
        :param store_v_seq: Whether to store the membrane potentials at every time
            step in ``self.v_seq``. When ``step_mode="s"`` the voltage is appended
            step by step until ``reset()`` is called and every step copies the whole
            sequence, so this option is meant for monitoring and debugging
        :type store_v_seq: bool
        :raises TypeError: Raised when ``v_threshold`` is not a real number or
            tensor
        :raises ValueError: Raised when ``v_threshold`` is not scalar finite
            positive
        """
        if isinstance(v_threshold, torch.Tensor):
            if v_threshold.numel() != 1:
                raise ValueError("v_threshold must be scalar finite positive.")
            v_threshold = float(v_threshold)
        elif not isinstance(v_threshold, numbers.Real):
            raise TypeError("v_threshold must be a real number.")
        v_threshold = float(v_threshold)
        if not torch.isfinite(torch.tensor(v_threshold)) or v_threshold <= 0.0:
            raise ValueError("v_threshold must be finite positive.")
        super().__init__(
            v_threshold=v_threshold,
            v_reset=None,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
        )
        half_threshold = self.v_threshold / 2.0
        self.set_reset_value("v", half_threshold)
        self.v = half_threshold

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0][0] if step_mode == "m" else inputs[0]
        v = states[0]
        if isinstance(v, float):
            v = torch.full_like(x, v, requires_grad=False)
        elif isinstance(v, torch.Tensor):
            if v.shape != x.shape:
                v = torch.full_like(x, self.v_threshold / 2.0, requires_grad=False)
            elif v.dtype != x.dtype or v.device != x.device:
                v = v.to(dtype=x.dtype, device=x.device)
        return (v, *states[1:])

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        spike, v = functional.if_step(
            x,
            v,
            self.v_threshold,
            None,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])


class ActivationAwareIFNode(base.MemoryModule):
    def __init__(
        self,
        v_threshold: Union[float, torch.Tensor] = 1.0,
        v_offset: Union[float, torch.Tensor] = 0.0,
        channel_dim: int = -1,
        v_reset: Optional[float] = None,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode: str = "s",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.__init__-cn>` | :ref:`English <ActivationAwareIFNode.__init__-en>`

        ----

        .. _ActivationAwareIFNode.__init__-cn:

        * **中文**

        Activation-aware IF 神经元，用于 ANN2SNN 中
        Activation-Aware Redistribution (AAR) 风格的最小垂直切片。该神经元
        支持标量或 1D channel-wise 的发放阈值 ``v_threshold`` 和膜电位偏移
        ``v_offset``。当 ``v_threshold`` 或 ``v_offset`` 为 1D 张量时，会沿
        ``channel_dim`` 广播到输入张量。

        它不继承 :class:`BaseNode`，也不改变现有 :class:`IFNode` /
        :class:`BaseNode` 的标量 ``v_threshold`` 约定。该实现用于研究和转换，不表示默认 ANN2SNN 路径支持多元素阈值。

        单步动力学为：

        .. math::

            H[t] = V[t-1] + X[t]

        .. math::

            S[t] = \Theta(H[t] + O - V_{th})

        其中 ``O`` 为 ``v_offset``。软复位时：

        .. math::

            V[t] = H[t] - S[t] V_{th}

        硬复位时：

        .. math::

            V[t] = S[t] V_{reset} + (1 - S[t]) H[t]

        :param v_threshold: 发放阈值。必须为有限正标量，或有限正 1D 张量。
        :type v_threshold: float or torch.Tensor
        :param v_offset: 膜电位偏移。必须为有限标量，或有限 1D 张量。
        :type v_offset: float or torch.Tensor
        :param channel_dim: 1D ``v_threshold`` / ``v_offset`` 对应的输入通道维。
        :type channel_dim: int
        :param v_reset: 硬复位电压。``None`` 表示软复位。
            若不为 ``None``，``reset()`` 会将膜电位 ``v`` 恢复为 ``v_reset``，
            与 :class:`BaseNode` 的硬复位语义一致。
        :type v_reset: Optional[float]
        :param surrogate_function: 反向传播时使用的替代函数。
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否在反向传播时分离 reset 计算图。
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步，``"m"`` 为多步。
        :type step_mode: str
        :param store_v_seq: 多步模式下是否保存每个时间步的膜电位。本类仅在多步
            模式下保存 ``v_seq`` ，单步模式下不会累积。
        :type store_v_seq: bool
        :raises ValueError: 当 step_mode、channel_dim、threshold、offset、多步输入形状
            或逐通道参数长度非法时抛出。

        ----

        .. _ActivationAwareIFNode.__init__-en:

        * **English**

        Activation-aware IF neuron for an ANN2SNN
        Activation-Aware Redistribution (AAR) style minimal vertical slice. This
        neuron supports scalar or 1D channel-wise firing threshold
        ``v_threshold`` and membrane offset ``v_offset``. A 1D ``v_threshold`` or
        ``v_offset`` is broadcast to the input tensor along ``channel_dim``.

        It does not inherit from :class:`BaseNode` and does not change the scalar
        ``v_threshold`` convention of existing :class:`IFNode` / :class:`BaseNode`.
        It is intended for research and conversion workloads; the default ANN2SNN path
        still does not support multi-element thresholds.

        The single-step dynamics are:

        .. math::

            H[t] = V[t-1] + X[t]

        .. math::

            S[t] = \Theta(H[t] + O - V_{th})

        where ``O`` is ``v_offset``. With soft reset:

        .. math::

            V[t] = H[t] - S[t] V_{th}

        With hard reset:

        .. math::

            V[t] = S[t] V_{reset} + (1 - S[t]) H[t]

        :param v_threshold: Firing threshold. It must be a finite positive
            scalar or a finite positive 1D tensor.
        :type v_threshold: float or torch.Tensor
        :param v_offset: Membrane offset. It must be a finite scalar or a finite
            1D tensor.
        :type v_offset: float or torch.Tensor
        :param channel_dim: Input channel dimension for 1D ``v_threshold`` /
            ``v_offset``.
        :type channel_dim: int
        :param v_reset: Hard-reset voltage. ``None`` means soft reset.
            If it is not ``None``, ``reset()`` restores membrane voltage ``v`` to
            ``v_reset``, matching the hard-reset semantics of :class:`BaseNode`.
        :type v_reset: Optional[float]
        :param surrogate_function: Surrogate function used in backward.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach the reset graph during backward.
        :type detach_reset: bool
        :param step_mode: Step mode, ``"s"`` for single-step and ``"m"`` for
            multi-step.
        :type step_mode: str
        :param store_v_seq: Whether to store membrane voltage at each time step
            in multi-step mode. This class stores ``v_seq`` only in multi-step
            mode and does not accumulate it in single-step mode.
        :type store_v_seq: bool
        :raises ValueError: If step_mode, channel_dim, threshold, offset,
            multi-step input shape, or channel-wise parameter length is invalid.
        """
        super().__init__()
        if step_mode not in ("s", "m"):
            raise ValueError("step_mode must be 's' or 'm'.")
        if v_reset is not None and not isinstance(v_reset, float):
            raise ValueError(
                f"v_reset must be a float or None, got {type(v_reset).__name__}."
            )
        if not isinstance(detach_reset, bool):
            raise ValueError("detach_reset must be bool.")
        if not isinstance(store_v_seq, bool):
            raise ValueError("store_v_seq must be bool.")
        if not isinstance(channel_dim, int):
            raise ValueError("channel_dim must be int.")

        threshold = torch.as_tensor(v_threshold)
        offset = torch.as_tensor(v_offset)
        self._check_threshold(threshold)
        self._check_offset(offset)

        self.register_buffer("v_threshold", threshold.clone().detach())
        self.register_buffer("v_offset", offset.clone().detach())
        self.channel_dim = channel_dim
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self.surrogate_function = surrogate_function
        if v_reset is None:
            self.register_memory("v", 0.0)
        else:
            self.register_memory("v", v_reset)
        self.store_v_seq = store_v_seq
        self.step_mode = step_mode

    @staticmethod
    def _check_threshold(v_threshold: torch.Tensor) -> None:
        if v_threshold.dim() > 1:
            raise ValueError(
                "v_threshold must be a scalar or 1D tensor, "
                f"but got shape {tuple(v_threshold.shape)}."
            )
        if not torch.is_floating_point(v_threshold):
            v_threshold = v_threshold.to(torch.float)
        if not torch.isfinite(v_threshold).all() or not (v_threshold > 0).all():
            raise ValueError("v_threshold must contain finite positive values.")

    @staticmethod
    def _check_offset(v_offset: torch.Tensor) -> None:
        if v_offset.dim() > 1:
            raise ValueError(
                "v_offset must be a scalar or 1D tensor, "
                f"but got shape {tuple(v_offset.shape)}."
            )
        if not torch.is_floating_point(v_offset):
            v_offset = v_offset.to(torch.float)
        if not torch.isfinite(v_offset).all():
            raise ValueError("v_offset must contain finite values.")

    @property
    def store_v_seq(self) -> bool:
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.store_v_seq-cn>` | :ref:`English <ActivationAwareIFNode.store_v_seq-en>`

        ----

        .. _ActivationAwareIFNode.store_v_seq-cn:

        * **中文**

        返回多步前向后是否保存完整膜电位序列。将该属性从 ``True`` 设为
        ``False`` 会立即释放之前由 ``self.v_seq`` 引用的序列张量。

        :return: 是否保存完整膜电位序列。
        :rtype: bool

        ----

        .. _ActivationAwareIFNode.store_v_seq-en:

        * **English**

        Return whether the full membrane-voltage sequence is stored after a
        multi-step forward. Changing this property from ``True`` to ``False``
        immediately releases the sequence tensor previously referenced by
        ``self.v_seq``.

        :return: Whether to store the full membrane-voltage sequence.
        :rtype: bool
        """
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool) -> None:
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.store_v_seq-setter-cn>` | :ref:`English <ActivationAwareIFNode.store_v_seq-setter-en>`

        ----

        .. _ActivationAwareIFNode.store_v_seq-setter-cn:

        * **中文**

        设置是否保存完整膜电位序列。禁用时将 ``self.v_seq`` 置为 ``None``，
        以免保留之前多步前向的完整 storage。

        :param value: 是否保存完整膜电位序列。
        :type value: bool
        ----

        .. _ActivationAwareIFNode.store_v_seq-setter-en:

        * **English**

        Set whether to store the full membrane-voltage sequence. Disabling it
        sets ``self.v_seq`` to ``None`` so storage from a previous multi-step
        forward is not retained.

        :param value: Whether to store the full membrane-voltage sequence.
        :type value: bool
        """
        self._store_v_seq = value
        self.v_seq = None

    def reset(self):
        super().reset()
        self.v_seq = None

    def _canonical_channel_dim(self, x: torch.Tensor) -> int:
        channel_dim = self.channel_dim
        if channel_dim < 0:
            channel_dim += x.dim()
        if channel_dim < 0 or channel_dim >= x.dim():
            raise ValueError(
                f"channel_dim={self.channel_dim} is out of range for input "
                f"with {x.dim()} dimensions."
            )
        return channel_dim

    def _broadcast_parameter(
        self, param: torch.Tensor, x: torch.Tensor, name: str
    ) -> torch.Tensor:
        param = param.to(device=x.device, dtype=x.dtype)
        if param.dim() == 0:
            return param

        channel_dim = self._canonical_channel_dim(x)
        if param.numel() != x.shape[channel_dim]:
            raise ValueError(
                f"{name} has length {param.numel()}, but input shape "
                f"{tuple(x.shape)} has {x.shape[channel_dim]} channels at "
                f"channel_dim={self.channel_dim}."
            )
        shape = [1] * x.dim()
        shape[channel_dim] = param.numel()
        return param.view(shape)

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        x = inputs[0][0] if step_mode == "m" else inputs[0]
        v = states[0]
        if isinstance(v, float):
            v = torch.full_like(x, v, requires_grad=False)
        elif isinstance(v, torch.Tensor):
            if v.shape != x.shape:
                fill_value = self.v_reset if self.v_reset is not None else 0.0
                v = torch.full_like(x, fill_value, requires_grad=False)
            elif v.dtype != x.dtype or v.device != x.device:
                v = v.to(dtype=x.dtype, device=x.device)
        return (v, *states[1:])

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.single_step_functional_forward-cn>` | :ref:`English <ActivationAwareIFNode.single_step_functional_forward-en>`

        ----

        .. _ActivationAwareIFNode.single_step_functional_forward-cn:

        * **中文**

        使用显式膜电位执行一个 activation-aware IF 时间步。
        本方法不修改模块状态。

        :param inputs: 仅包含单步输入 ``x`` 的元组，``x`` 形状为 ``[N, *]``。
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 显式状态 ``(v,)``。
        :type states: tuple
        :return: ``((spike,), updated_states)``。
        :rtype: tuple[tuple[torch.Tensor, ...], tuple]

        ----

        .. _ActivationAwareIFNode.single_step_functional_forward-en:

        * **English**

        Run one activation-aware IF time step with explicit membrane voltage.
        This method does not mutate module state.

        :param inputs: Tuple containing only the single-step input ``x`` with shape ``[N, *]``.
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Explicit state ``(v,)``.
        :type states: tuple
        :return: ``((spike,), updated_states)``.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple]
        """
        x = inputs[0]
        v = states[0]

        threshold = self._broadcast_parameter(self.v_threshold, x, "v_threshold")
        offset = self._broadcast_parameter(self.v_offset, x, "v_offset")
        spike, v = functional.activation_aware_if_step(
            x,
            v,
            threshold,
            offset,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])

    def _registered_multi_step_functional_forward(
        self, x_seq: torch.Tensor, v, store_v_seq: bool
    ) -> tuple[torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        threshold = self.v_threshold.to(device=x_seq.device, dtype=torch.float32)
        offset = self.v_offset.to(device=x_seq.device, dtype=torch.float32)
        if threshold.dim() == 1 or offset.dim() == 1:
            channel_dim = self._canonical_channel_dim(x_seq[0])
            channel_size = x_seq.shape[1 + channel_dim]
            if threshold.dim() == 1 and threshold.numel() != channel_size:
                raise ValueError(
                    f"v_threshold has length {threshold.numel()}, but input shape "
                    f"{tuple(x_seq.shape[1:])} has {channel_size} channels at "
                    f"channel_dim={self.channel_dim}."
                )
            if offset.dim() == 1 and offset.numel() != channel_size:
                raise ValueError(
                    f"v_offset has length {offset.numel()}, but input shape "
                    f"{tuple(x_seq.shape[1:])} has {channel_size} channels at "
                    f"channel_dim={self.channel_dim}."
                )
            inner_size = 1
            for size in x_seq.shape[2 + channel_dim :]:
                inner_size *= size
        else:
            channel_size = 1
            inner_size = x_seq[0].numel()

        spike_seq, v, v_seq = functional.activation_aware_if_multi_step(
            x_seq,
            v,
            threshold,
            offset,
            channel_size,
            inner_size,
            self.v_reset,
            store_v_seq,
        )
        return spike_seq, v, v_seq

    def _can_use_registered_multi_step(
        self, x_seq: torch.Tensor, v: torch.Tensor
    ) -> bool:
        from ..._ops.surrogate import _surrogate_spec

        return (
            x_seq.dtype in (torch.float32, torch.float16, torch.bfloat16)
            and v.dtype == torch.float32
            and _surrogate_spec(self.surrogate_function) is not None
            and not any(
                tensor.requires_grad
                for tensor in (x_seq, v, self.v_threshold, self.v_offset)
            )
        )

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.multi_step_functional_forward-cn>` | :ref:`English <ActivationAwareIFNode.multi_step_functional_forward-en>`

        ----

        .. _ActivationAwareIFNode.multi_step_functional_forward-cn:

        * **中文**

        使用显式状态执行 activation-aware IF 多步前向。可由注册算子执行的
        推理输入会依据张量 device 自动分发；训练和其他不兼容输入保留 Torch
        参考状态转移。

        :param inputs: 仅包含 ``x_seq`` 的元组，``x_seq`` 形状为 ``[T, N, *]``。
        :type inputs: tuple[torch.Tensor, ...]
        :param states: 显式状态 ``(v,)``。
        :type states: tuple
        :return: ``((spike_seq,), updated_states)``。
        :rtype: tuple[tuple[torch.Tensor, ...], tuple]
        :raises ValueError: 当输入形状、T 或逐通道参数长度非法时抛出。

        ----

        .. _ActivationAwareIFNode.multi_step_functional_forward-en:

        * **English**

        Run the multi-step activation-aware IF forward pass with explicit state.
        Eligible inference inputs are dispatched by tensor device; training and
        other unsupported inputs use the Torch reference transition.

        :param inputs: Tuple containing only ``x_seq`` with shape ``[T, N, *]``.
        :type inputs: tuple[torch.Tensor, ...]
        :param states: Explicit state ``(v,)``.
        :type states: tuple
        :return: ``((spike_seq,), updated_states)``.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple]
        :raises ValueError: If the input shape, T, or channel-wise parameter
            length is invalid.
        """
        x_seq, v = inputs[0], states[0]
        if self._can_use_registered_multi_step(x_seq, v):
            spike_seq, v, _ = self._registered_multi_step_functional_forward(
                x_seq, v, False
            )
            return (spike_seq,), (v,)
        outputs = []
        for x in x_seq:
            step_outputs, states = self.single_step_functional_forward((x,), states)
            outputs.append(step_outputs[0])
        return (torch.stack(outputs),), states

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        states = self.materialize_states(
            (x_seq, *args), tuple(self._memories.values()), "m"
        )
        if self._can_use_registered_multi_step(x_seq, states[0]):
            spike_seq, v, v_seq = self._registered_multi_step_functional_forward(
                x_seq, states[0], self.store_v_seq
            )
        else:
            spike_steps = []
            voltage_steps = []
            for t in range(x_seq.shape[0]):
                outputs, states = self.single_step_functional_forward(
                    (x_seq[t],), states, **kwargs
                )
                spike_steps.append(outputs[0])
                if self.store_v_seq:
                    voltage_steps.append(states[0])
            spike_seq = torch.stack(spike_steps)
            v = states[0]
            v_seq = torch.stack(voltage_steps) if self.store_v_seq else None
        self.v = v
        self.v_seq = v_seq if self.store_v_seq else None
        return spike_seq

    def extra_repr(self):
        return (
            f"v_threshold_shape={tuple(self.v_threshold.shape)}, "
            f"v_offset_shape={tuple(self.v_offset.shape)}, "
            f"channel_dim={self.channel_dim}, v_reset={self.v_reset}, "
            f"detach_reset={self.detach_reset}, step_mode={self.step_mode}"
        )


class NonSpikingIFNode(NonSpikingBaseNode):
    def __init__(self, decode: Optional[str] = None):
        """
        **API Language** - :ref:`中文 <NonSpikingIFNode.__init__-cn>` | :ref:`English <NonSpikingIFNode.__init__-en>`

        ----

        .. _NonSpikingIFNode.__init__-cn:

        * **中文**

        不发放脉冲的 IF 节点，输出膜电位（或根据 ``decode`` 进行解码）。

        :param decode: 非脉冲输出解码方式，见 :class:`NonSpikingBaseNode`
        :type decode: Optional[str]

        ----

        .. _NonSpikingIFNode.__init__-en:

        * **English**

        Non-spiking IF node that outputs membrane potential (or decoded outputs specified by ``decode``).

        :param decode: Decoding mode for non-spiking outputs, see :class:`NonSpikingBaseNode`
        :type decode: Optional[str]
        """
        super().__init__(decode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
