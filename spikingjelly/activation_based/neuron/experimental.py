import math
from typing import Optional, Union

import torch

from ..._ops.if_ import if_multi_step
from ..._ops.lif import lif
from ..._ops.plif import plif
from ..._ops.surrogate import _SURROGATE_IDS
from .. import surrogate

__all__ = [
    "ExperimentalIFNode",
    "ExperimentalLIFNode",
    "ExperimentalParametricLIFNode",
    "ExperimentalQIFNode",
    "ExperimentalEIFNode",
    "ExperimentalIzhikevichNode",
    "ExperimentalILIFNode",
    "ExperimentalActivationAwareIFNode",
    "ExperimentalSTBIFNode",
]


class _ExperimentalNeuron(torch.nn.Module):
    def __init__(
        self, v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
    ):
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.detach_reset = detach_reset
        self._surrogate_id = 0
        if surrogate_function is not None:
            supported = {
                getattr(surrogate, name): index
                for name, index in _SURROGATE_IDS.items()
            }
            if type(surrogate_function) not in supported:
                raise TypeError("Unsupported experimental surrogate type")
            if not surrogate_function.spiking:
                raise ValueError("Experimental neurons require spiking=True")
            alpha = surrogate_function.alpha
            self._surrogate_id = supported[type(surrogate_function)]
        if isinstance(alpha, torch.Tensor):
            raise TypeError("Surrogate alpha must be a fixed Python scalar")
        if not math.isfinite(alpha) or alpha <= 0:
            raise ValueError("Surrogate alpha must be finite and positive")
        self.alpha = float(alpha)
        self.store_v_seq = store_v_seq
        self.register_buffer("v", None, persistent=False)
        self.register_buffer("v_seq", None, persistent=False)

    def _initial_voltage(self, x_seq):
        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("expected [T, ...] with T >= 1")
        if self.v is None or self.v.shape != x_seq.shape[1:]:
            return torch.full_like(x_seq[0], self.v_reset or 0.0, dtype=torch.float32)
        return self.v.to(device=x_seq.device, dtype=torch.float32)

    def _store_voltage(self, voltages):
        self.v = voltages[-1].clone() if self.store_v_seq else voltages
        self.v_seq = voltages if self.store_v_seq else None

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-voltage-reset-cn>` | :ref:`English <experimental-voltage-reset-en>`

        ----

        .. _experimental-voltage-reset-cn:

        * **中文**

        清空电位和轨迹及本模块持有的计算图引用。下次调用重新初始化。

        ----

        .. _experimental-voltage-reset-en:

        * **English**

        Clear voltage, trace, and graph references held by this module.
        The next call initializes fresh state.
        """
        self.v = None
        self.v_seq = None


class ExperimentalLIFNode(_ExperimentalNeuron):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-lif-init-cn>` | :ref:`English <experimental-lif-init-en>`

        ----

        .. _experimental-lif-init-cn:

        * **中文**

        实验性多步 LIF，支持 CPU/NVIDIA CUDA 的 FP32、FP16、BF16 输入和七种替代梯度。
        CUDA 实现在首次使用时按设备选择并缓存。输入为非空 ``[T, ...]``
        序列，输出同形状脉冲。设备分派由 PyTorch 完成，无 backend 参数。
        ``v`` 和可选 ``v_seq`` 为非持久 buffer；``reset()`` 清空二者。
        只支持一阶反向传播，支持 autocast 输入；不支持可学习的替代梯度参数。
        脉冲及输入梯度跟随输入 dtype；膜电位及跨时间梯度累积使用 FP32。

        :param tau: 有限且大于 1 的膜时间常数，以时间步为单位，默认 2。
        :type tau: float
        :param decay_input: 是否对输入乘以 ``1/tau``，默认 True。
        :type decay_input: bool
        :param v_threshold: 有限的发放阈值，默认 1。
        :type v_threshold: float
        :param v_reset: 有限的硬重置电位，默认 0；None 使用软重置并从 0 初始化。
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置分支的脉冲梯度，默认 False。
        :type detach_reset: bool
        :param alpha: surrogate_function=None 时的有限正 Sigmoid 斜率，默认 4。
        :type alpha: float
        :param store_v_seq: 是否保留本次调用的完整膜电位轨迹，默认 False。
        :type store_v_seq: bool
        :param surrogate_function: 默认 None 使用 Sigmoid(alpha)。支持 Sigmoid、ATan、
            PiecewiseQuadratic、PiecewiseExp、SoftSign、SuperSpike、Erf 的精确类型，
            必须 spiking=True 且 alpha 为有限正 Python 标量。构造时读取类型和 alpha，
            覆盖 alpha 参数；随后修改原对象无效。不保留替代梯度模块。
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: 替代梯度类型不受支持，或 alpha 为 Tensor。
        :raises ValueError: 替代梯度未启用 spiking 或其 alpha 非有限正数。

        ----

        .. _experimental-lif-init-en:

        * **English**

        Experimental multi-step LIF for CPU/NVIDIA CUDA FP32, FP16 or BF16 inputs
        with seven supported surrogates. The CUDA implementation is selected once per device on first
        use. Nonempty ``[T, ...]`` sequences produce equally shaped spikes. PyTorch
        dispatches by device without a backend argument. ``v`` and optional
        ``v_seq`` are nonpersistent buffers; ``reset()`` clears both. Only first-order
        reverse-mode gradients are supported, including autocast inputs, with fixed
        surrogate parameters. Spikes/input gradients follow the input dtype; voltage
        and temporal-gradient accumulation use FP32.

        :param tau: Finite membrane time constant greater than 1, in steps; default 2.
        :type tau: float
        :param decay_input: Scale the input by ``1/tau``; default True.
        :type decay_input: bool
        :param v_threshold: Finite firing threshold; default 1.
        :type v_threshold: float
        :param v_reset: Finite hard-reset voltage, default 0; None selects soft reset
            with zero initial voltage.
        :type v_reset: Optional[float]
        :param detach_reset: Detach spikes in the reset branch; default False.
        :type detach_reset: bool
        :param alpha: Finite positive Sigmoid slope when surrogate_function=None; default 4.
        :type alpha: float
        :param store_v_seq: Retain the complete voltage trace of this call; default False.
        :type store_v_seq: bool
        :param surrogate_function: None (default) uses Sigmoid(alpha). Accepts exact
            Sigmoid, ATan, PiecewiseQuadratic, PiecewiseExp, SoftSign, SuperSpike or
            Erf types with spiking=True and finite positive Python scalar alpha.
            Construction snapshots its type and alpha, overriding the alpha argument;
            later changes to that object have no effect. The module is not retained.
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: Unsupported surrogate type or tensor-valued alpha.
        :raises ValueError: Surrogate spiking is disabled or alpha is not finite and positive.
        """
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )
        self.tau = tau
        self.decay_input = decay_input

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <experimental-lif-forward-cn>` | :ref:`English <experimental-lif-forward-en>`

        ----

        .. _experimental-lif-forward-cn:

        * **中文**

        从当前膜电位执行多步 LIF，更新 ``v``，按配置保存 ``v_seq``。
        状态缺失或形状变化时重新初始化；否则将状态移至输入 device 并使用 FP32。
        train/eval 均发放硬脉冲；启用梯度时，两种模式均使用所选替代梯度，
        与生产级 LIFNode 的 eval 语义不同。是否记录梯度由 autograd 控制。

        :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 非空 ``[T, ...]`` 输入，T >= 1；
            后续维度表示独立神经元，允许非连续 strided 布局。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和 device 的脉冲序列。
        :rtype: torch.Tensor
        :raises ValueError: 输入没有时间及神经元维度、T 为零，或标量参数不在所述范围内。
        :raises RuntimeError: dtype 不是 FP32/FP16/BF16、神经元维度为空、布局不是 strided，
            或所需 CUDA 实现不可用。

        ----

        .. _experimental-lif-forward-en:

        * **English**

        Advance the current voltage through a sequence, update ``v``, and retain
        ``v_seq`` if requested. Missing or differently shaped state is initialized;
        otherwise state is moved to the input device in FP32. Both train/eval emit
        hard spikes and use the selected surrogate gradients when enabled, unlike
        production LIFNode evaluation semantics. Autograd controls recording.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 ``[T, ...]`` input with T >= 1.
            Remaining dimensions identify independent neurons; noncontiguous
            strided layouts are accepted.
        :type x_seq: torch.Tensor
        :return: Spike sequence with the input shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: Input lacks time/neuron dimensions, T is zero, or a
            scalar parameter is outside its stated range.
        :raises RuntimeError: The dtype is not FP32/FP16/BF16, neuron dimensions are empty,
            the layout is not strided, or the required CUDA implementation is unavailable.
        """
        v = self._initial_voltage(x_seq)
        spikes, voltages, _ = lif(
            x_seq,
            v,
            self.tau,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltages)
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-lif-reset-cn>` | :ref:`English <experimental-lif-reset-en>`

        ----

        .. _experimental-lif-reset-cn:

        * **中文**

        清空膜电位和轨迹，释放本模块持有的对应计算图引用。下次调用重新初始化状态。

        ----

        .. _experimental-lif-reset-en:

        * **English**

        Clear voltage and trace, releasing their graph references held by this
        module. The next call initializes fresh state.
        """
        self.v = None
        self.v_seq = None


class ExperimentalIFNode(_ExperimentalNeuron):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-if-init-cn>` | :ref:`English <experimental-if-init-en>`

        ----

        .. _experimental-if-init-cn:

        * **中文**

        实验性多步 IF，支持 CPU/NVIDIA CUDA 的 FP32、FP16、BF16 输入和七种替代梯度。
        CUDA 实现在首次使用时按设备选择并缓存。输入为非空 ``[T, ...]``
        序列，输出同形状脉冲。设备分派由 PyTorch 完成，无 backend 参数。
        ``v`` 和可选 ``v_seq`` 为非持久 buffer；``reset()`` 清空二者。
        充电方程为 ``h = v + x``。只支持一阶反向传播，支持 autocast 输入；不支持可学习的替代梯度参数。
        脉冲及输入梯度跟随输入 dtype；膜电位及跨时间梯度累积使用 FP32。

        :param v_threshold: 有限的发放阈值，默认 1。
        :type v_threshold: float
        :param v_reset: 有限的硬重置电位，默认 0；None 使用软重置并从 0 初始化。
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置分支的脉冲梯度，默认 False。
        :type detach_reset: bool
        :param alpha: surrogate_function=None 时的有限正 Sigmoid 斜率，默认 4。
        :type alpha: float
        :param store_v_seq: 是否保留本次调用的完整膜电位轨迹，默认 False。
        :type store_v_seq: bool
        :param surrogate_function: 默认 None 使用 Sigmoid(alpha)。支持 Sigmoid、ATan、
            PiecewiseQuadratic、PiecewiseExp、SoftSign、SuperSpike、Erf 的精确类型，
            必须 spiking=True 且 alpha 为有限正 Python 标量。构造时读取类型和 alpha，
            覆盖 alpha 参数；随后修改原对象无效。不保留替代梯度模块。
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: 替代梯度类型不受支持，或 alpha 为 Tensor。
        :raises ValueError: 替代梯度未启用 spiking 或其 alpha 非有限正数。

        ----

        .. _experimental-if-init-en:

        * **English**

        Experimental multi-step IF for CPU/NVIDIA CUDA FP32, FP16 or BF16 inputs
        with seven supported surrogates. The CUDA implementation is selected once per device on first
        use. Nonempty ``[T, ...]`` sequences produce equally shaped spikes. PyTorch
        dispatches by device without a backend argument. ``v`` and optional
        ``v_seq`` are nonpersistent buffers; ``reset()`` clears both. Only first-order
        reverse-mode gradients are supported, including autocast inputs, with fixed
        surrogate parameters. Spikes/input gradients follow the input dtype; voltage
        and temporal-gradient accumulation use FP32. Charging follows ``h = v + x``.

        :param v_threshold: Finite firing threshold; default 1.
        :type v_threshold: float
        :param v_reset: Finite hard-reset voltage, default 0; None selects soft reset
            with zero initial voltage.
        :type v_reset: Optional[float]
        :param detach_reset: Detach spikes in the reset branch; default False.
        :type detach_reset: bool
        :param alpha: Finite positive Sigmoid slope when surrogate_function=None; default 4.
        :type alpha: float
        :param store_v_seq: Retain the complete voltage trace of this call; default False.
        :type store_v_seq: bool
        :param surrogate_function: None (default) uses Sigmoid(alpha). Accepts exact
            Sigmoid, ATan, PiecewiseQuadratic, PiecewiseExp, SoftSign, SuperSpike or
            Erf types with spiking=True and finite positive Python scalar alpha.
            Construction snapshots its type and alpha, overriding the alpha argument;
            later changes to that object have no effect. The module is not retained.
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: Unsupported surrogate type or tensor-valued alpha.
        :raises ValueError: Surrogate spiking is disabled or alpha is not finite and positive.
        """
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <experimental-if-forward-cn>` | :ref:`English <experimental-if-forward-en>`

        ----

        .. _experimental-if-forward-cn:

        * **中文**

        从当前膜电位执行多步 IF，更新 ``v``，按配置保存 ``v_seq``。
        状态缺失或形状变化时重新初始化；否则将状态移至输入 device 并使用 FP32。
        train/eval 均发放硬脉冲；启用梯度时，两种模式均使用所选替代梯度，
        与生产级 IFNode 的 eval 语义不同。是否记录梯度由 autograd 控制。

        :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 非空 ``[T, ...]`` 输入，T >= 1；
            后续维度表示独立神经元，允许非连续 strided 布局。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和 device 的脉冲序列。
        :rtype: torch.Tensor
        :raises ValueError: 输入没有时间及神经元维度、T 为零，或标量参数不在所述范围内。
        :raises RuntimeError: dtype 不是 FP32/FP16/BF16、神经元维度为空、布局不是 strided，
            或所需 CUDA 实现不可用。

        ----

        .. _experimental-if-forward-en:

        * **English**

        Advance the current voltage through a sequence, update ``v``, and retain
        ``v_seq`` if requested. Missing or differently shaped state is initialized;
        otherwise state is moved to the input device in FP32. Both train/eval emit
        hard spikes and use the selected surrogate gradients when enabled, unlike
        production IFNode evaluation semantics. Autograd controls recording.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 ``[T, ...]`` input with T >= 1.
            Remaining dimensions identify independent neurons; noncontiguous
            strided layouts are accepted.
        :type x_seq: torch.Tensor
        :return: Spike sequence with the input shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: Input lacks time/neuron dimensions, T is zero, or a
            scalar parameter is outside its stated range.
        :raises RuntimeError: The dtype is not FP32/FP16/BF16, neuron dimensions are empty,
            the layout is not strided, or the required CUDA implementation is unavailable.
        """
        v = self._initial_voltage(x_seq)
        spikes, voltages, _ = if_multi_step(
            x_seq,
            v,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltages)
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-if-reset-cn>` | :ref:`English <experimental-if-reset-en>`

        ----

        .. _experimental-if-reset-cn:

        * **中文**

        清空膜电位和轨迹，释放本模块持有的对应计算图引用。下次调用重新初始化状态。

        ----

        .. _experimental-if-reset-en:

        * **English**

        Clear voltage and trace, releasing their graph references held by this
        module. The next call initializes fresh state.
        """
        self.v = None
        self.v_seq = None


class ExperimentalParametricLIFNode(_ExperimentalNeuron):
    def __init__(
        self,
        init_tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-plif-init-cn>` | :ref:`English <experimental-plif-init-en>`

        ----

        .. _experimental-plif-init-cn:

        * **中文**

        实验性多步 PLIF，支持 CPU/NVIDIA CUDA 的 FP32、FP16、BF16 输入和七种替代梯度。
        CUDA 实现在首次使用时按设备选择并缓存。输入为非空 ``[T, ...]``
        序列，输出同形状脉冲。设备分派由 PyTorch 完成，无 backend 参数。
        ``v`` 和可选 ``v_seq`` 为非持久 buffer；``reset()`` 清空二者。
        共享的零维 FP32 参数 ``w`` 初始化为 ``-log(init_tau - 1)``，每次调用使用
        ``sigmoid(w.float())`` 作为 ``1/tau``。``w`` 支持 FP32/FP16/BF16，
        必须与输入同 device；使用
        ``.to(device)`` 移动整个模块，不会自动移动参数。``state_dict`` 保存 ``w``。
        只支持一阶反向传播，支持 autocast 输入；``reset()`` 不改变 ``w``。
        膜电位、跨时间梯度及 w 的归约使用 FP32；脉冲及输入梯度跟随输入 dtype，
        w 的梯度最后转为 w 的 dtype。

        :param init_tau: 有限且大于 1 的初始膜时间常数，以时间步为单位，默认 2。
        :type init_tau: float
        :param decay_input: 是否对输入乘以 ``1/tau``，默认 True。
        :type decay_input: bool
        :param v_threshold: 有限的发放阈值，默认 1。
        :type v_threshold: float
        :param v_reset: 有限的硬重置电位，默认 0；None 使用软重置并从 0 初始化。
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置分支的脉冲梯度，默认 False。
        :type detach_reset: bool
        :param alpha: surrogate_function=None 时的有限正 Sigmoid 斜率，默认 4。
        :type alpha: float
        :param store_v_seq: 是否保留本次调用的完整膜电位轨迹，默认 False。
        :type store_v_seq: bool
        :param surrogate_function: 默认 None 使用 Sigmoid(alpha)。支持 Sigmoid、ATan、
            PiecewiseQuadratic、PiecewiseExp、SoftSign、SuperSpike、Erf 的精确类型，
            必须 spiking=True 且 alpha 为有限正 Python 标量。构造时读取类型和 alpha，
            覆盖 alpha 参数；随后修改原对象无效。不保留替代梯度模块。
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: 替代梯度类型不受支持，或 alpha 为 Tensor。
        :raises ValueError: 替代梯度未启用 spiking 或其 alpha 非有限正数。
        :raises ValueError: ``init_tau`` 非有限值或不大于 1。

        ----

        .. _experimental-plif-init-en:

        * **English**

        Experimental multi-step PLIF for CPU/NVIDIA CUDA FP32, FP16 or BF16 inputs
        with seven supported surrogates. The CUDA implementation is selected once per device on first
        use. Nonempty ``[T, ...]`` sequences produce equally shaped spikes. PyTorch
        dispatches by device without a backend argument. ``v`` and optional
        ``v_seq`` are nonpersistent buffers; ``reset()`` clears both. The shared zero-dimensional FP32 parameter ``w`` starts at
        ``-log(init_tau - 1)``; each call uses ``sigmoid(w.float())`` as ``1/tau``.
        It accepts FP32/FP16/BF16 and must share the input device. Move the module with ``.to(device)``;
        parameters are not moved automatically. ``state_dict`` includes ``w``.
        Only first-order gradients are supported, including autocast inputs. Voltage,
        temporal gradients and the w reduction use FP32. Spikes/input gradients follow
        the input dtype; the w gradient is finally cast to the parameter dtype.
        ``reset()`` preserves ``w``.

        :param init_tau: Finite initial membrane time constant greater than 1, in steps; default 2.
        :type init_tau: float
        :param decay_input: Scale the input by ``1/tau``; default True.
        :type decay_input: bool
        :param v_threshold: Finite firing threshold; default 1.
        :type v_threshold: float
        :param v_reset: Finite hard-reset voltage, default 0; None selects soft reset
            with zero initial voltage.
        :type v_reset: Optional[float]
        :param detach_reset: Detach spikes in the reset branch; default False.
        :type detach_reset: bool
        :param alpha: Finite positive Sigmoid slope when surrogate_function=None; default 4.
        :type alpha: float
        :param store_v_seq: Retain the complete voltage trace of this call; default False.
        :type store_v_seq: bool
        :param surrogate_function: None (default) uses Sigmoid(alpha). Accepts exact
            Sigmoid, ATan, PiecewiseQuadratic, PiecewiseExp, SoftSign, SuperSpike or
            Erf types with spiking=True and finite positive Python scalar alpha.
            Construction snapshots its type and alpha, overriding the alpha argument;
            later changes to that object have no effect. The module is not retained.
        :type surrogate_function: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        :raises TypeError: Unsupported surrogate type or tensor-valued alpha.
        :raises ValueError: Surrogate spiking is disabled or alpha is not finite and positive.
        :raises ValueError: ``init_tau`` is not finite or is not greater than 1.
        """
        if not math.isfinite(init_tau) or init_tau <= 1:
            raise ValueError("init_tau must be finite and greater than one")
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )
        self.w = torch.nn.Parameter(
            torch.tensor(-math.log(init_tau - 1), dtype=torch.float32)
        )
        self.decay_input = decay_input

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <experimental-plif-forward-cn>` | :ref:`English <experimental-plif-forward-en>`

        ----

        .. _experimental-plif-forward-cn:

        * **中文**

        从当前膜电位执行多步 LIF，更新 ``v``，按配置保存 ``v_seq``。
        状态缺失或形状变化时重新初始化；否则将状态移至输入 device 并使用 FP32。
        train/eval 均发放硬脉冲；启用梯度时，两种模式均使用所选替代梯度，
        与生产级 ParametricLIFNode 的 eval 语义不同。是否记录梯度由 autograd 控制。

        :param x_seq: CPU/NVIDIA CUDA FP32/FP16/BF16 非空 ``[T, ...]`` 输入，T >= 1；
            后续维度表示独立神经元，允许非连续 strided 布局。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和 device 的脉冲序列。
        :rtype: torch.Tensor
        :raises ValueError: 输入没有时间及神经元维度、T 为零，或标量参数不在所述范围内。
        :raises RuntimeError: dtype 不是 FP32/FP16/BF16、神经元维度为空、布局不是 strided，
            参数 ``w`` 不是与输入同 device 的零维 FP32/FP16/BF16 Tensor，或所需 CUDA 实现不可用。

        ----

        .. _experimental-plif-forward-en:

        * **English**

        Advance the current voltage through a sequence, update ``v``, and retain
        ``v_seq`` if requested. Missing or differently shaped state is initialized;
        otherwise state is moved to the input device in FP32. Both train/eval emit
        hard spikes and use the selected surrogate gradients when enabled, unlike
        production ParametricLIFNode evaluation semantics. Autograd controls recording.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 ``[T, ...]`` input with T >= 1.
            Remaining dimensions identify independent neurons; noncontiguous
            strided layouts are accepted.
        :type x_seq: torch.Tensor
        :return: Spike sequence with the input shape, dtype, and device.
        :rtype: torch.Tensor
        :raises ValueError: Input lacks time/neuron dimensions, T is zero, or a
            scalar parameter is outside its stated range.
        :raises RuntimeError: The dtype is not FP32/FP16/BF16, neuron dimensions are empty,
            the layout is not strided, ``w`` is not a scalar FP32/FP16/BF16 tensor on the input
            device, or the required CUDA implementation is unavailable.
        """
        v = self._initial_voltage(x_seq)
        spikes, voltages, _ = plif(
            x_seq,
            v,
            self.w,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltages)
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <experimental-plif-reset-cn>` | :ref:`English <experimental-plif-reset-en>`

        ----

        .. _experimental-plif-reset-cn:

        * **中文**

        清空膜电位和轨迹，释放本模块持有的对应计算图引用。下次调用重新初始化状态，不改变可学习参数 ``w``。

        ----

        .. _experimental-plif-reset-en:

        * **English**

        Clear voltage and trace, releasing their graph references held by this
        module. The next call initializes fresh state; the learnable parameter ``w`` is unchanged.
        """
        self.v = None
        self.v_seq = None


class ExperimentalQIFNode(_ExperimentalNeuron):
    def __init__(
        self,
        tau: float = 2.0,
        v_rest: float = 0.0,
        v_c: float = 0.8,
        a0: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalQIFNode-init-cn>` | :ref:`English <registered-ExperimentalQIFNode-init-en>`

        ----

        .. _registered-ExperimentalQIFNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 支持一阶梯度；训练及 eval 都保留替代梯度。

        :param tau: 有限时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
        :type tau: float
        :param v_rest: 有限静息电位。 默认 ``0.0``.
        :type v_rest: float
        :param v_c: 有限临界电位。 默认 ``0.8``.
        :type v_c: float
        :param a0: 有限二次项系数。 默认 ``1.0``.
        :type a0: float
        :param v_threshold: 有限发放阈值；I-LIF 必须为正；ActivationAwareIF 接受不需梯度的标量或通道张量。 默认 ``1.0``.
        :type v_threshold: float
        :param v_reset: 有限硬重置值；None 为软重置。 默认 ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置脉冲梯度；Izhikevich 保留恢复脉冲及 spike*v_reset 梯度。 默认 ``False``.
        :type detach_reset: bool
        :param alpha: 默认 Sigmoid 的有限正 alpha；显式 surrogate_function 覆盖它。 默认 ``4.0``.
        :type alpha: float
        :param store_v_seq: 是否保存最近一次调用的完整 FP32 电位轨迹；Izhikevich 同时保存 w_seq。 默认 ``False``.
        :type store_v_seq: bool
        :param surrogate_function: 七种受支持的固定替代梯度，None 使用 Sigmoid(alpha)；I-LIF 仅接受 MultiLevelSpikeCount，默认计数上限 4。参数在构造时复制。 默认 ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: 参数超出有效范围。
        :raises TypeError: 替代梯度类型或参数不受支持。

        ----

        .. _registered-ExperimentalQIFNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. First-order gradients are supported in both training and eval.

        :param tau: Finite time constant in time steps, greater than one. Default: ``2.0``.
        :type tau: float
        :param v_rest: Finite resting voltage. Default: ``0.0``.
        :type v_rest: float
        :param v_c: Finite critical voltage. Default: ``0.8``.
        :type v_c: float
        :param a0: Finite quadratic coefficient. Default: ``1.0``.
        :type a0: float
        :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF accepts scalar/channel tensors without gradients. Default: ``1.0``.
        :type v_threshold: float
        :param v_reset: Finite hard reset; None selects soft reset. Default: ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: Detach reset spikes; Izhikevich retains recovery-spike and spike*v_reset gradients. Default: ``False``.
        :type detach_reset: bool
        :param alpha: Finite positive default Sigmoid alpha, overridden by surrogate_function. Default: ``4.0``.
        :type alpha: float
        :param store_v_seq: Save the latest complete FP32 voltage trace; Izhikevich also saves w_seq. Default: ``False``.
        :type store_v_seq: bool
        :param surrogate_function: One of seven fixed binary surrogates; None uses Sigmoid(alpha). I-LIF accepts only MultiLevelSpikeCount, default maximum 4. Parameters are snapshotted at construction. Default: ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: Parameters are outside their valid ranges.
        :raises TypeError: Unsupported surrogate type or parameters.
        """
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )
        if not all(math.isfinite(v) for v in (tau, v_rest, v_c, a0)) or tau <= 1:
            raise ValueError("QIF requires finite parameters and tau > 1")
        self.tau, self.v_rest, self.v_c, self.a0 = tau, v_rest, v_c, a0

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalQIFNode-forward-cn>` | :ref:`English <registered-ExperimentalQIFNode-forward-en>`

        ----

        .. _registered-ExperimentalQIFNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalQIFNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.qif import _forward

        spikes, voltage, _, _ = _forward(
            x_seq,
            self._initial_voltage(x_seq),
            self.tau,
            self.v_rest,
            self.v_c,
            self.a0,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltage)
        return spikes


class ExperimentalEIFNode(_ExperimentalNeuron):
    def __init__(
        self,
        tau: float = 2.0,
        v_rest: float = 0.0,
        theta_rh: float = 1.0,
        delta_t: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalEIFNode-init-cn>` | :ref:`English <registered-ExperimentalEIFNode-init-en>`

        ----

        .. _registered-ExperimentalEIFNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 支持一阶梯度；训练及 eval 都保留替代梯度。

        :param tau: 有限时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
        :type tau: float
        :param v_rest: 有限静息电位。 默认 ``0.0``.
        :type v_rest: float
        :param theta_rh: 有限流变阈值。 默认 ``1.0``.
        :type theta_rh: float
        :param delta_t: 有限正指数电位宽度。 默认 ``1.0``.
        :type delta_t: float
        :param v_threshold: 有限发放阈值；I-LIF 必须为正；ActivationAwareIF 接受不需梯度的标量或通道张量。 默认 ``1.0``.
        :type v_threshold: float
        :param v_reset: 有限硬重置值；None 为软重置。 默认 ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置脉冲梯度；Izhikevich 保留恢复脉冲及 spike*v_reset 梯度。 默认 ``False``.
        :type detach_reset: bool
        :param alpha: 默认 Sigmoid 的有限正 alpha；显式 surrogate_function 覆盖它。 默认 ``4.0``.
        :type alpha: float
        :param store_v_seq: 是否保存最近一次调用的完整 FP32 电位轨迹；Izhikevich 同时保存 w_seq。 默认 ``False``.
        :type store_v_seq: bool
        :param surrogate_function: 七种受支持的固定替代梯度，None 使用 Sigmoid(alpha)；I-LIF 仅接受 MultiLevelSpikeCount，默认计数上限 4。参数在构造时复制。 默认 ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: 参数超出有效范围。
        :raises TypeError: 替代梯度类型或参数不受支持。

        ----

        .. _registered-ExperimentalEIFNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. First-order gradients are supported in both training and eval.

        :param tau: Finite time constant in time steps, greater than one. Default: ``2.0``.
        :type tau: float
        :param v_rest: Finite resting voltage. Default: ``0.0``.
        :type v_rest: float
        :param theta_rh: Finite rheobase threshold. Default: ``1.0``.
        :type theta_rh: float
        :param delta_t: Finite positive exponential voltage width. Default: ``1.0``.
        :type delta_t: float
        :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF accepts scalar/channel tensors without gradients. Default: ``1.0``.
        :type v_threshold: float
        :param v_reset: Finite hard reset; None selects soft reset. Default: ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: Detach reset spikes; Izhikevich retains recovery-spike and spike*v_reset gradients. Default: ``False``.
        :type detach_reset: bool
        :param alpha: Finite positive default Sigmoid alpha, overridden by surrogate_function. Default: ``4.0``.
        :type alpha: float
        :param store_v_seq: Save the latest complete FP32 voltage trace; Izhikevich also saves w_seq. Default: ``False``.
        :type store_v_seq: bool
        :param surrogate_function: One of seven fixed binary surrogates; None uses Sigmoid(alpha). I-LIF accepts only MultiLevelSpikeCount, default maximum 4. Parameters are snapshotted at construction. Default: ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: Parameters are outside their valid ranges.
        :raises TypeError: Unsupported surrogate type or parameters.
        """
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )
        if (
            not all(math.isfinite(v) for v in (tau, v_rest, theta_rh, delta_t))
            or tau <= 1
            or delta_t <= 0
        ):
            raise ValueError("EIF requires finite parameters, tau > 1, and delta_t > 0")
        self.tau, self.v_rest, self.theta_rh, self.delta_t = (
            tau,
            v_rest,
            theta_rh,
            delta_t,
        )

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalEIFNode-forward-cn>` | :ref:`English <registered-ExperimentalEIFNode-forward-en>`

        ----

        .. _registered-ExperimentalEIFNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalEIFNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.eif import _forward

        spikes, voltage, _, _ = _forward(
            x_seq,
            self._initial_voltage(x_seq),
            self.tau,
            self.v_rest,
            self.theta_rh,
            self.delta_t,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltage)
        return spikes


class ExperimentalIzhikevichNode(_ExperimentalNeuron):
    def __init__(
        self,
        tau: float = 2.0,
        v_rest: float = 0.0,
        v_c: float = 0.8,
        a0: float = 1.0,
        a: float = 0.1,
        b: float = 0.2,
        tau_w: float = 2.0,
        init_w: float = 0.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        detach_reset: bool = False,
        alpha: float = 4.0,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.SurrogateFunctionBase] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalIzhikevichNode-init-cn>` | :ref:`English <registered-ExperimentalIzhikevichNode-init-en>`

        ----

        .. _registered-ExperimentalIzhikevichNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 支持一阶梯度；训练及 eval 都保留替代梯度。

        :param tau: 有限时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
        :type tau: float
        :param v_rest: 有限静息电位。 默认 ``0.0``.
        :type v_rest: float
        :param v_c: 有限临界电位。 默认 ``0.8``.
        :type v_c: float
        :param a0: 有限二次项系数。 默认 ``1.0``.
        :type a0: float
        :param a: 有限恢复变量耦合系数。 默认 ``0.1``.
        :type a: float
        :param b: 有限发放恢复增量。 默认 ``0.2``.
        :type b: float
        :param tau_w: 有限正恢复时间常数，以时间步为单位。 默认 ``2.0``.
        :type tau_w: float
        :param init_w: 首次或形状改变后的恢复初态。 默认 ``0.0``.
        :type init_w: float
        :param v_threshold: 有限发放阈值；I-LIF 必须为正；ActivationAwareIF 接受不需梯度的标量或通道张量。 默认 ``1.0``.
        :type v_threshold: float
        :param v_reset: 有限硬重置值；None 为软重置。 默认 ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: 是否分离重置脉冲梯度；Izhikevich 保留恢复脉冲及 spike*v_reset 梯度。 默认 ``False``.
        :type detach_reset: bool
        :param alpha: 默认 Sigmoid 的有限正 alpha；显式 surrogate_function 覆盖它。 默认 ``4.0``.
        :type alpha: float
        :param store_v_seq: 是否保存最近一次调用的完整 FP32 电位轨迹；Izhikevich 同时保存 w_seq。 默认 ``False``.
        :type store_v_seq: bool
        :param surrogate_function: 七种受支持的固定替代梯度，None 使用 Sigmoid(alpha)；I-LIF 仅接受 MultiLevelSpikeCount，默认计数上限 4。参数在构造时复制。 默认 ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: 参数超出有效范围。
        :raises TypeError: 替代梯度类型或参数不受支持。

        ----

        .. _registered-ExperimentalIzhikevichNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. First-order gradients are supported in both training and eval.

        :param tau: Finite time constant in time steps, greater than one. Default: ``2.0``.
        :type tau: float
        :param v_rest: Finite resting voltage. Default: ``0.0``.
        :type v_rest: float
        :param v_c: Finite critical voltage. Default: ``0.8``.
        :type v_c: float
        :param a0: Finite quadratic coefficient. Default: ``1.0``.
        :type a0: float
        :param a: Finite recovery coupling coefficient. Default: ``0.1``.
        :type a: float
        :param b: Finite post-spike recovery increment. Default: ``0.2``.
        :type b: float
        :param tau_w: Finite positive recovery time constant in time steps. Default: ``2.0``.
        :type tau_w: float
        :param init_w: Initial recovery state on first call or shape change. Default: ``0.0``.
        :type init_w: float
        :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF accepts scalar/channel tensors without gradients. Default: ``1.0``.
        :type v_threshold: float
        :param v_reset: Finite hard reset; None selects soft reset. Default: ``0.0``.
        :type v_reset: Optional[float]
        :param detach_reset: Detach reset spikes; Izhikevich retains recovery-spike and spike*v_reset gradients. Default: ``False``.
        :type detach_reset: bool
        :param alpha: Finite positive default Sigmoid alpha, overridden by surrogate_function. Default: ``4.0``.
        :type alpha: float
        :param store_v_seq: Save the latest complete FP32 voltage trace; Izhikevich also saves w_seq. Default: ``False``.
        :type store_v_seq: bool
        :param surrogate_function: One of seven fixed binary surrogates; None uses Sigmoid(alpha). I-LIF accepts only MultiLevelSpikeCount, default maximum 4. Parameters are snapshotted at construction. Default: ``None``.
        :type surrogate_function: Optional[surrogate.SurrogateFunctionBase]
        :raises ValueError: Parameters are outside their valid ranges.
        :raises TypeError: Unsupported surrogate type or parameters.
        """
        super().__init__(
            v_threshold, v_reset, detach_reset, alpha, store_v_seq, surrogate_function
        )
        if (
            not all(
                math.isfinite(v) for v in (tau, v_rest, v_c, a0, a, b, tau_w, init_w)
            )
            or tau <= 1
            or tau_w <= 0
        ):
            raise ValueError(
                "Izhikevich requires finite parameters, tau > 1, and tau_w > 0"
            )
        self.tau, self.v_rest, self.v_c, self.a0 = tau, v_rest, v_c, a0
        self.a, self.b, self.tau_w, self.init_w = a, b, tau_w, init_w
        self.register_buffer("w", None, persistent=False)
        self.register_buffer("w_seq", None, persistent=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalIzhikevichNode-forward-cn>` | :ref:`English <registered-ExperimentalIzhikevichNode-forward-en>`

        ----

        .. _registered-ExperimentalIzhikevichNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalIzhikevichNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.izhikevich import _forward

        v = self._initial_voltage(x_seq)
        w = (
            torch.full_like(v, self.init_w)
            if self.w is None or self.w.shape != v.shape
            else self.w.to(device=x_seq.device, dtype=torch.float32)
        )
        spikes, voltage, recovery, _, _ = _forward(
            x_seq,
            v,
            w,
            self.tau,
            self.v_rest,
            self.v_c,
            self.a0,
            self.a,
            self.b,
            self.tau_w,
            self.v_threshold,
            self.v_reset,
            self.detach_reset,
            self.alpha,
            self.store_v_seq,
            self._surrogate_id,
        )
        self._store_voltage(voltage)
        self.w = recovery[-1].clone() if self.store_v_seq else recovery
        self.w_seq = recovery if self.store_v_seq else None
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalIzhikevichNode-reset-cn>` | :ref:`English <registered-ExperimentalIzhikevichNode-reset-en>`

        ----

        .. _registered-ExperimentalIzhikevichNode-reset-cn:

        * **中文**

        清空所有非持久状态和轨迹，释放本模块持有的计算图；保留配置与参数。下次调用重新初始化。

        ----

        .. _registered-ExperimentalIzhikevichNode-reset-en:

        * **English**

        Clear all nonpersistent state/traces and graph references held by this module, preserving configuration and parameters. The next call reinitializes state.
        """
        super().reset()
        self.w = None
        self.w_seq = None


class ExperimentalILIFNode(torch.nn.Module):
    def __init__(
        self,
        tau: float = 2.0,
        v_threshold: float = 1.0,
        detach_reset: bool = False,
        store_v_seq: bool = False,
        surrogate_function: Optional[surrogate.MultiLevelSpikeCount] = None,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalILIFNode-init-cn>` | :ref:`English <registered-ExperimentalILIFNode-init-en>`

        ----

        .. _registered-ExperimentalILIFNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 支持一阶梯度；训练及 eval 都保留替代梯度。

        :param tau: 有限时间常数，以时间步为单位，必须大于 1。 默认 ``2.0``.
        :type tau: float
        :param v_threshold: 有限发放阈值；I-LIF 必须为正；ActivationAwareIF 接受不需梯度的标量或通道张量。 默认 ``1.0``.
        :type v_threshold: float
        :param detach_reset: 是否分离重置脉冲梯度；Izhikevich 保留恢复脉冲及 spike*v_reset 梯度。 默认 ``False``.
        :type detach_reset: bool
        :param store_v_seq: 是否保存最近一次调用的完整 FP32 电位轨迹；Izhikevich 同时保存 w_seq。 默认 ``False``.
        :type store_v_seq: bool
        :param surrogate_function: 七种受支持的固定替代梯度，None 使用 Sigmoid(alpha)；I-LIF 仅接受 MultiLevelSpikeCount，默认计数上限 4。参数在构造时复制。 默认 ``None``.
        :type surrogate_function: Optional[surrogate.MultiLevelSpikeCount]
        :raises ValueError: 参数超出有效范围。
        :raises TypeError: 替代梯度类型或参数不受支持。

        ----

        .. _registered-ExperimentalILIFNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. First-order gradients are supported in both training and eval.

        :param tau: Finite time constant in time steps, greater than one. Default: ``2.0``.
        :type tau: float
        :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF accepts scalar/channel tensors without gradients. Default: ``1.0``.
        :type v_threshold: float
        :param detach_reset: Detach reset spikes; Izhikevich retains recovery-spike and spike*v_reset gradients. Default: ``False``.
        :type detach_reset: bool
        :param store_v_seq: Save the latest complete FP32 voltage trace; Izhikevich also saves w_seq. Default: ``False``.
        :type store_v_seq: bool
        :param surrogate_function: One of seven fixed binary surrogates; None uses Sigmoid(alpha). I-LIF accepts only MultiLevelSpikeCount, default maximum 4. Parameters are snapshotted at construction. Default: ``None``.
        :type surrogate_function: Optional[surrogate.MultiLevelSpikeCount]
        :raises ValueError: Parameters are outside their valid ranges.
        :raises TypeError: Unsupported surrogate type or parameters.
        """
        super().__init__()
        function = (
            surrogate.MultiLevelSpikeCount(4)
            if surrogate_function is None
            else surrogate_function
        )
        if type(function) is not surrogate.MultiLevelSpikeCount or not function.spiking:
            raise TypeError("I-LIF requires MultiLevelSpikeCount(spiking=True)")
        if (
            not math.isfinite(tau)
            or tau <= 1
            or not math.isfinite(v_threshold)
            or v_threshold <= 0
        ):
            raise ValueError("I-LIF requires tau > 1 and a finite positive threshold")
        self.tau, self.v_threshold = tau, v_threshold
        self.detach_reset, self.store_v_seq = detach_reset, store_v_seq
        self.max_spike_count = float(function.max_spike_count)
        self.grad_min, self.grad_max = (
            float(function.grad_min),
            float(function.grad_max),
        )
        if (
            not math.isfinite(self.grad_min)
            or not math.isfinite(self.grad_max)
            or self.grad_min > self.grad_max
        ):
            raise ValueError("STE window must have finite ordered endpoints")
        self.register_buffer("v", None, persistent=False)
        self.register_buffer("v_seq", None, persistent=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalILIFNode-forward-cn>` | :ref:`English <registered-ExperimentalILIFNode-forward-en>`

        ----

        .. _registered-ExperimentalILIFNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalILIFNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.ilif import _forward

        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("expected nonempty [T, ...]")
        v = (
            torch.zeros_like(x_seq[0], dtype=torch.float32)
            if self.v is None or self.v.shape != x_seq.shape[1:]
            else self.v.to(device=x_seq.device, dtype=torch.float32)
        )
        spikes, voltage, _ = _forward(
            x_seq,
            v,
            self.tau,
            self.max_spike_count,
            self.grad_min,
            self.grad_max,
            self.v_threshold,
            self.detach_reset,
            self.store_v_seq,
        )
        self.v = voltage[-1].clone() if self.store_v_seq else voltage
        self.v_seq = voltage if self.store_v_seq else None
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalILIFNode-reset-cn>` | :ref:`English <registered-ExperimentalILIFNode-reset-en>`

        ----

        .. _registered-ExperimentalILIFNode-reset-cn:

        * **中文**

        清空所有非持久状态和轨迹，释放本模块持有的计算图；保留配置与参数。下次调用重新初始化。

        ----

        .. _registered-ExperimentalILIFNode-reset-en:

        * **English**

        Clear all nonpersistent state/traces and graph references held by this module, preserving configuration and parameters. The next call reinitializes state.
        """
        self.v = None
        self.v_seq = None


class ExperimentalActivationAwareIFNode(torch.nn.Module):
    def __init__(
        self,
        v_threshold: Union[float, torch.Tensor] = 1.0,
        v_offset: Union[float, torch.Tensor] = 0.0,
        channel_size: int = 1,
        inner_size: int = 1,
        v_reset: Optional[float] = 0.0,
        store_v_seq: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalActivationAwareIFNode-init-cn>` | :ref:`English <registered-ExperimentalActivationAwareIFNode-init-en>`

        ----

        .. _registered-ExperimentalActivationAwareIFNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 仅用于推理，不支持输入/状态/参数梯度。

        :param v_threshold: 有限发放阈值；I-LIF 必须为正；ActivationAwareIF 接受不需梯度的标量或通道张量。 默认 ``1.0``.
        :type v_threshold: Union[float, torch.Tensor]
        :param v_offset: 不需梯度的标量或通道发放偏移，保存为 FP32 buffer。 默认 ``0.0``.
        :type v_offset: Union[float, torch.Tensor]
        :param channel_size: 正通道数；阈值和偏移元素数必须为 1 或此值。 默认 ``1``.
        :type channel_size: int
        :param inner_size: 每个通道内元素数，必须为正；channel_size*inner_size 须整除状态元素数。 默认 ``1``.
        :type inner_size: int
        :param v_reset: 有限硬重置值；None 为软重置。 默认 ``0.0``.
        :type v_reset: Optional[float]
        :param store_v_seq: 是否保存最近一次调用的完整 FP32 电位轨迹；Izhikevich 同时保存 w_seq。 默认 ``False``.
        :type store_v_seq: bool
        :raises ValueError: 参数超出有效范围。

        ----

        .. _registered-ExperimentalActivationAwareIFNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. Inference only; inputs/states/parameters must not require gradients.

        :param v_threshold: Finite firing threshold; positive for I-LIF. ActivationAwareIF accepts scalar/channel tensors without gradients. Default: ``1.0``.
        :type v_threshold: Union[float, torch.Tensor]
        :param v_offset: Scalar/channel firing offset without gradients, saved as an FP32 buffer. Default: ``0.0``.
        :type v_offset: Union[float, torch.Tensor]
        :param channel_size: Positive channel count; thresholds/offsets must contain one or this many elements. Default: ``1``.
        :type channel_size: int
        :param inner_size: Positive elements per channel; channel_size*inner_size must divide state size. Default: ``1``.
        :type inner_size: int
        :param v_reset: Finite hard reset; None selects soft reset. Default: ``0.0``.
        :type v_reset: Optional[float]
        :param store_v_seq: Save the latest complete FP32 voltage trace; Izhikevich also saves w_seq. Default: ``False``.
        :type store_v_seq: bool
        :raises ValueError: Parameters are outside their valid ranges.
        """
        super().__init__()
        if channel_size < 1 or inner_size < 1:
            raise ValueError("channel_size and inner_size must be positive")
        if v_reset is not None and not math.isfinite(v_reset):
            raise ValueError("v_reset must be finite")
        threshold, offset = (
            torch.as_tensor(v_threshold, dtype=torch.float32),
            torch.as_tensor(v_offset, dtype=torch.float32),
        )
        if threshold.requires_grad or offset.requires_grad:
            raise ValueError("ActivationAwareIF parameters are inference-only")
        if threshold.numel() not in (1, channel_size) or offset.numel() not in (
            1,
            channel_size,
        ):
            raise ValueError("threshold/offset must be scalar or channel-wise")
        self.register_buffer("v_threshold", threshold.clone())
        self.register_buffer("v_offset", offset.clone())
        self.channel_size, self.inner_size = channel_size, inner_size
        self.v_reset, self.store_v_seq = v_reset, store_v_seq
        self.register_buffer("v", None, persistent=False)
        self.register_buffer("v_seq", None, persistent=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalActivationAwareIFNode-forward-cn>` | :ref:`English <registered-ExperimentalActivationAwareIFNode-forward-en>`

        ----

        .. _registered-ExperimentalActivationAwareIFNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalActivationAwareIFNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.activation_aware_if import _forward

        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("expected nonempty [T, ...]")
        v = (
            torch.full_like(x_seq[0], self.v_reset or 0.0, dtype=torch.float32)
            if self.v is None or self.v.shape != x_seq.shape[1:]
            else self.v.to(device=x_seq.device, dtype=torch.float32)
        )
        spikes, voltage = _forward(
            x_seq,
            v,
            self.v_threshold.to(device=x_seq.device, dtype=torch.float32),
            self.v_offset.to(device=x_seq.device, dtype=torch.float32),
            self.channel_size,
            self.inner_size,
            self.v_reset,
            self.store_v_seq,
        )
        self.v = voltage[-1].clone() if self.store_v_seq else voltage
        self.v_seq = voltage if self.store_v_seq else None
        return spikes

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalActivationAwareIFNode-reset-cn>` | :ref:`English <registered-ExperimentalActivationAwareIFNode-reset-en>`

        ----

        .. _registered-ExperimentalActivationAwareIFNode-reset-cn:

        * **中文**

        清空所有非持久状态和轨迹，释放本模块持有的计算图；保留配置与参数。下次调用重新初始化。

        ----

        .. _registered-ExperimentalActivationAwareIFNode-reset-en:

        * **English**

        Clear all nonpersistent state/traces and graph references held by this module, preserving configuration and parameters. The next call reinitializes state.
        """
        self.v = None
        self.v_seq = None


class ExperimentalSTBIFNode(torch.nn.Module):
    def __init__(
        self, q_threshold: float = 1.0, pos_max: float = 15.0, neg_min: float = 0.0
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalSTBIFNode-init-cn>` | :ref:`English <registered-ExperimentalSTBIFNode-init-en>`

        ----

        .. _registered-ExperimentalSTBIFNode-init-cn:

        * **中文**

        实验多步节点，使用独立注册算子自动选择 CPU 或 NVIDIA CUDA 实现。输入支持 FP32/FP16/BF16，状态与累积保持 FP32；不继承 MemoryModule，不接受 backend 参数。状态为非持久 buffer，连续调用保留状态，reset() 清空；形状改变时重建，设备改变时转移。 仅用于推理，不支持输入/状态/参数梯度。

        :param q_threshold: 有限非零量化尺度，保存为 FP32 buffer。 默认 ``1.0``.
        :type q_threshold: float
        :param pos_max: 有限累计释放上界。 默认 ``15.0``.
        :type pos_max: float
        :param neg_min: 有限累计释放下界，不大于 pos_max。 默认 ``0.0``.
        :type neg_min: float
        :raises ValueError: 参数超出有效范围。

        ----

        .. _registered-ExperimentalSTBIFNode-init-en:

        * **English**

        Experimental multi-step node using independent registered operators with automatic CPU/NVIDIA CUDA selection. Inputs support FP32/FP16/BF16; state and accumulation stay FP32. Does not inherit MemoryModule or accept backend. Nonpersistent state buffers survive consecutive calls; reset() clears them. Shape changes reinitialize state and device changes move it. Inference only; inputs/states/parameters must not require gradients.

        :param q_threshold: Finite nonzero quantization scale, saved as an FP32 buffer. Default: ``1.0``.
        :type q_threshold: float
        :param pos_max: Finite accumulated release upper bound. Default: ``15.0``.
        :type pos_max: float
        :param neg_min: Finite accumulated release lower bound, at most pos_max. Default: ``0.0``.
        :type neg_min: float
        :raises ValueError: Parameters are outside their valid ranges.
        """
        super().__init__()
        if (
            not all(math.isfinite(v) for v in (q_threshold, pos_max, neg_min))
            or q_threshold == 0
            or neg_min > pos_max
        ):
            raise ValueError(
                "STBIF requires finite scale/bounds, nonzero scale and ordered bounds"
            )
        self.register_buffer(
            "q_threshold", torch.tensor(q_threshold, dtype=torch.float32)
        )
        self.register_buffer("pos_max", torch.tensor(pos_max, dtype=torch.float32))
        self.register_buffer("neg_min", torch.tensor(neg_min, dtype=torch.float32))
        self.register_buffer("q", None, persistent=False)
        self.register_buffer("acc_q", None, persistent=False)
        self.register_buffer("cur_output", None, persistent=False)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalSTBIFNode-forward-cn>` | :ref:`English <registered-ExperimentalSTBIFNode-forward-en>`

        ----

        .. _registered-ExperimentalSTBIFNode-forward-cn:

        * **中文**

        处理 [T, ...] 输入并更新本模块的 FP32 状态；不修改输入。

        :param x_seq: 非空 CPU/NVIDIA CUDA FP32/FP16/BF16 序列，T >= 1；形状为 [T, ...]。
        :type x_seq: torch.Tensor
        :return: 与输入同形状、dtype 和设备的输出；STBIF 为带符号量化输出，其余为脉冲/计数。
        :rtype: torch.Tensor
        :raises ValueError: 输入维度或标量参数无效。
        :raises RuntimeError: 张量约束不满足或所需实现不可用。

        ----

        .. _registered-ExperimentalSTBIFNode-forward-en:

        * **English**

        Process [T, ...] input and update this module's FP32 state without mutating inputs.

        :param x_seq: Nonempty CPU/NVIDIA CUDA FP32/FP16/BF16 sequence [T, ...], T >= 1.
        :type x_seq: torch.Tensor
        :return: Output with input shape/dtype/device; signed quantized output for STBIF, spikes/counts otherwise.
        :rtype: torch.Tensor
        :raises ValueError: Invalid input dimensions or scalar parameters.
        :raises RuntimeError: Tensor constraints violated or required implementation unavailable.
        """
        from ..._ops.stbif import _forward

        if x_seq.ndim < 2 or x_seq.shape[0] == 0:
            raise ValueError("expected nonempty [T, ...]")
        if self.q is None or self.q.shape != x_seq.shape[1:]:
            q = torch.full_like(x_seq[0], 0.5, dtype=torch.float32)
            acc = torch.zeros_like(q)
        else:
            q = self.q.to(device=x_seq.device, dtype=torch.float32)
            acc = self.acc_q.to(device=x_seq.device, dtype=torch.float32)
        output, self.q, self.acc_q, self.cur_output = _forward(
            x_seq,
            q,
            acc,
            self.q_threshold.to(device=x_seq.device, dtype=torch.float32),
            self.pos_max.to(device=x_seq.device, dtype=torch.float32),
            self.neg_min.to(device=x_seq.device, dtype=torch.float32),
        )
        return output

    def reset(self) -> None:
        r"""
        **API Language** - :ref:`中文 <registered-ExperimentalSTBIFNode-reset-cn>` | :ref:`English <registered-ExperimentalSTBIFNode-reset-en>`

        ----

        .. _registered-ExperimentalSTBIFNode-reset-cn:

        * **中文**

        清空所有非持久状态和轨迹，释放本模块持有的计算图；保留配置与参数。下次调用重新初始化。

        ----

        .. _registered-ExperimentalSTBIFNode-reset-en:

        * **English**

        Clear all nonpersistent state/traces and graph references held by this module, preserving configuration and parameters. The next call reinitializes state.
        """
        self.q = None
        self.acc_q = None
        self.cur_output = None
