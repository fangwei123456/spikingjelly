import logging
import numbers
from typing import Optional, Union

import torch

from .. import base, surrogate
from .base_node import BaseNode, NonSpikingBaseNode, SimpleBaseNode

try:
    from ..cuda_kernel.auto_cuda import neuron_kernel as ac_neuron_kernel
    from ..cuda_kernel.auto_cuda import ss_neuron_kernel as ss_ac_neuron_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    ac_neuron_kernel = None
    ss_ac_neuron_kernel = None

try:
    from .. import triton_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    triton_kernel = None


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

        :class:`IFNode` 的简化版实现。

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

        A simple version of :class:`IFNode`.

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
        backend="torch",
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

        :param backend: 使用哪种后端。不同的 ``step_mode`` 可能会带有不同的后端。可以通过打印 ``self.supported_backends`` 查看当前
            使用的步进模式支持的后端。该参数是显式执行后端选择：设置为 ``'torch'``、``'cupy'`` 或 ``'triton'`` 时，将分别使用
            对应后端，不会隐式切换到其他后端。在支持的情况下，使用 ``'cupy'`` 或 ``'triton'`` 后端通常更快。
        :type backend: str

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存
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
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )

    @property
    def supported_backends(self):
        if self.step_mode == "s":
            return ("torch", "cupy")
        elif self.step_mode == "m":
            return ("torch", "cupy", "triton", "inductor")
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    def _eval_single_step_forward(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset
    ):
        """Unified single-step eval (replaces jit_eval_single_step_forward_*)."""
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = (
            (v - spike * v_threshold)
            if v_reset is None
            else (v_reset * spike + (1.0 - spike) * v)
        )
        return spike, v

    @staticmethod
    def _eval_multi_step_forward(
        x_seq: torch.Tensor,
        v: torch.Tensor,
        v_threshold: float,
        v_reset,
        store_v_seq: bool = False,
    ):
        """Unified multi-step eval (replaces jit_eval_multi_step_forward_*)."""
        T = x_seq.shape[0]
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq) if store_v_seq else None
        soft_reset = v_reset is None
        _vr = 0.0 if soft_reset else v_reset
        for t in range(T):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = (
                (v - spike * v_threshold)
                if soft_reset
                else (_vr * spike + (1.0 - spike) * v)
            )
            spike_seq[t] = spike
            if store_v_seq:
                v_seq[t] = v
        if store_v_seq:
            return spike_seq, v, v_seq
        return spike_seq, v

    # kept for subclass backward-compatibility
    @staticmethod
    def jit_eval_single_step_forward_hard_reset(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1.0 - spike) * v
        return spike, v

    @staticmethod
    def jit_eval_single_step_forward_soft_reset(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float
    ):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    def _build_inductor_multi_step_graph(self):
        store_v_seq = self.store_v_seq
        soft_reset = self.v_reset is None
        v_reset = 0.0 if soft_reset else self.v_reset
        surrogate_fn = self.surrogate_function
        v_threshold = self.v_threshold
        detach_reset = self.detach_reset

        def _graph(x_seq: torch.Tensor, v_init: torch.Tensor):
            v = v_init
            spike_seq = torch.empty_like(x_seq)
            if store_v_seq:
                v_seq = torch.empty_like(x_seq)
            for t in range(x_seq.shape[0]):
                v = v + x_seq[t]
                spike = surrogate_fn(v - v_threshold)
                spike_d = spike.detach() if detach_reset else spike
                if soft_reset:
                    v = v - spike_d * v_threshold
                else:
                    v = spike_d * v_reset + (1.0 - spike_d) * v
                spike_seq[t] = spike
                if store_v_seq:
                    v_seq[t] = v
            if store_v_seq:
                return spike_seq, v, v_seq
            return spike_seq, v

        return _graph

    def _inductor_multi_step_forward(self, x_seq: torch.Tensor):
        self.v_float_to_tensor(x_seq[0])
        x_seq = self._canonicalize_inductor_tensor(x_seq)
        v_init = self._canonicalize_inductor_tensor(self.v)
        graph = self._compile_inductor_graph(
            (
                "if",
                self.store_v_seq,
                self.v_threshold,
                self.v_reset,
                self.detach_reset,
                self._surrogate_inductor_cache_key(),
                self._inductor_runtime_cache_key(x_seq, v_init),
            ),
            self._build_inductor_multi_step_graph(),
        )
        out = graph(x_seq, v_init)
        if self.store_v_seq:
            spike_seq, self.v, self.v_seq = out
        else:
            spike_seq, self.v = out
        return spike_seq

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == "inductor":
            return self._inductor_multi_step_forward(x_seq)
        if self.training:
            if self.backend == "torch":
                return super().multi_step_forward(x_seq)
            elif self.backend == "cupy":
                hard_reset = self.v_reset is not None
                if x_seq.dtype == torch.float:
                    dtype = "float"
                elif x_seq.dtype == torch.half:
                    dtype = "half2"
                else:
                    raise NotImplementedError(x_seq.dtype)

                if (
                    self.forward_kernel is None
                    or not self.forward_kernel.check_attributes(
                        hard_reset=hard_reset, dtype=dtype
                    )
                ):
                    self.forward_kernel = ac_neuron_kernel.IFNodeFPTTKernel(
                        hard_reset=hard_reset, dtype=dtype
                    )
                if (
                    self.backward_kernel is None
                    or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )
                ):
                    self.backward_kernel = ac_neuron_kernel.IFNodeBPTTKernel(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )

                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = ac_neuron_kernel.multistep_if(
                    x_seq=x_seq.flatten(1),
                    v_init=self.v.flatten(0),
                    v_threshold=self.v_threshold,
                    v_reset=self.v_reset,
                    detach_reset=self.detach_reset,
                    surrogate_function=self.surrogate_function,
                    forward_kernel=self.forward_kernel,
                    backward_kernel=self.backward_kernel,
                )
                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
                return spike_seq
            elif self.backend == "triton":
                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_out = triton_kernel.multistep_if(
                    x_seq,
                    self.v,
                    self.v_threshold,
                    self.v_reset,
                    self.detach_reset,
                    self.surrogate_function,
                    self.store_v_seq,
                )
                if self.store_v_seq:
                    self.v_seq = v_out
                    self.v = v_out[-1].clone()
                else:
                    self.v = v_out
                return spike_seq
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])

            if self.backend == "triton":
                if not getattr(self.surrogate_function, "spiking", True):
                    raise NotImplementedError(
                        "Triton backend only supports spiking surrogate functions. "
                        "Use backend='torch' for non-spiking surrogate functions."
                    )
                spike_seq, v_out = triton_kernel.multistep_if(
                    x_seq,
                    self.v,
                    self.v_threshold,
                    self.v_reset,
                    self.detach_reset,
                    self.surrogate_function,
                    self.store_v_seq,
                )
                if self.store_v_seq:
                    self.v_seq = v_out
                    self.v = v_out[-1]
                else:
                    self.v = v_out
                return spike_seq
            elif self.backend == "cupy":
                spike_seq, v_seq = ac_neuron_kernel.multistep_if(
                    x_seq=x_seq.flatten(1),
                    v_init=self.v.flatten(0),
                    v_threshold=self.v_threshold,
                    v_reset=self.v_reset,
                    detach_reset=self.detach_reset,
                    surrogate_function=self.surrogate_function,
                )
                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)
                if self.store_v_seq:
                    self.v_seq = v_seq
                    self.v = v_seq[-1]
                else:
                    self.v = v_seq[-1].clone()
                return spike_seq

            # torch backend:
            out = self._eval_multi_step_forward(
                x_seq,
                self.v,
                self.v_threshold,
                self.v_reset,
                store_v_seq=self.store_v_seq,
            )
            if self.store_v_seq:
                spike_seq, self.v, self.v_seq = out
            else:
                spike_seq, self.v = out
            return spike_seq

    def single_step_forward(self, x: torch.Tensor):
        if self.training:
            if self.backend == "torch":
                return super().single_step_forward(x)
            elif self.backend == "cupy":
                hard_reset = self.v_reset is not None
                if x.dtype == torch.float:
                    dtype = "float"
                elif x.dtype == torch.half:
                    dtype = "half2"
                else:
                    raise NotImplementedError(x.dtype)

                if (
                    self.forward_kernel is None
                    or not self.forward_kernel.check_attributes(
                        hard_reset=hard_reset, dtype=dtype
                    )
                ):
                    self.forward_kernel = ss_ac_neuron_kernel.IFNodeFPKernel(
                        hard_reset=hard_reset, dtype=dtype
                    )

                if (
                    self.backward_kernel is None
                    or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )
                ):
                    self.backward_kernel = ss_ac_neuron_kernel.IFNodeBPKernel(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )

                self.v_float_to_tensor(x)
                spike, v = ss_ac_neuron_kernel.ss_if_step(
                    x.flatten(0),
                    self.v.flatten(0),
                    self.v_threshold,
                    self.v_reset,
                    self.forward_kernel,
                    self.backward_kernel,
                )
                spike = spike.reshape(x.shape)
                v = v.reshape(x.shape)
                self.v = v
                return spike
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x)
            spike, self.v = self._eval_single_step_forward(
                x,
                self.v,
                self.v_threshold,
                self.v_reset,
            )
            return spike


class HalfThresholdIFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
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
        :param backend: 后端名称。当前实现支持 ``"torch"``
        :type backend: str
        :param store_v_seq: 在 ``step_mode="m"`` 时是否保存每个时间步的膜电位序列
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
        :param backend: Backend name. The current implementation supports
            ``"torch"``
        :type backend: str
        :param store_v_seq: Whether to store membrane potentials at every time
            step when ``step_mode="m"``
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
            backend=backend,
            store_v_seq=store_v_seq,
        )
        half_threshold = self.v_threshold / 2.0
        self.set_reset_value("v", half_threshold)
        self.v = half_threshold

    @property
    def supported_backends(self):
        return ("torch",)

    def v_float_to_tensor(self, x: torch.Tensor):
        half_threshold = self.v_threshold / 2.0
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v, requires_grad=False)
        elif isinstance(self.v, torch.Tensor):
            if self.v.shape != x.shape:
                self.v = torch.full_like(x, half_threshold, requires_grad=False)
            elif self.v.dtype != x.dtype or self.v.device != x.device:
                self.v = self.v.to(dtype=x.dtype, device=x.device)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x


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
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <ActivationAwareIFNode.__init__-cn>` | :ref:`English <ActivationAwareIFNode.__init__-en>`

        ----

        .. _ActivationAwareIFNode.__init__-cn:

        * **中文**

        实验性的 activation-aware IF 神经元，用于 ANN2SNN 中
        Activation-Aware Redistribution (AAR) 风格的最小垂直切片。该神经元
        支持标量或 1D channel-wise 的发放阈值 ``v_threshold`` 和膜电位偏移
        ``v_offset``。当 ``v_threshold`` 或 ``v_offset`` 为 1D 张量时，会沿
        ``channel_dim`` 广播到输入张量。

        该类只支持 ``backend="torch"``，不继承 :class:`BaseNode`，也不改变
        现有 :class:`IFNode` / :class:`BaseNode` 的标量 ``v_threshold`` 约定。
        它面向研究和转换 POC，不表示默认 ANN2SNN 路径支持多元素阈值。

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
        :param backend: 仅支持 ``"torch"``。
        :type backend: str
        :param store_v_seq: 多步模式下是否保存每个时间步的膜电位。
        :type store_v_seq: bool
        :raises ValueError: 当 backend、step_mode、channel_dim、threshold 或 offset 非法时抛出。

        ----

        .. _ActivationAwareIFNode.__init__-en:

        * **English**

        Experimental activation-aware IF neuron for an ANN2SNN
        Activation-Aware Redistribution (AAR) style minimal vertical slice. This
        neuron supports scalar or 1D channel-wise firing threshold
        ``v_threshold`` and membrane offset ``v_offset``. A 1D ``v_threshold`` or
        ``v_offset`` is broadcast to the input tensor along ``channel_dim``.

        This class supports only ``backend="torch"``. It does not inherit from
        :class:`BaseNode` and does not change the scalar ``v_threshold``
        convention of existing :class:`IFNode` / :class:`BaseNode`. It is meant
        for research and conversion POCs, and does not imply that the default
        ANN2SNN path supports multi-element thresholds.

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
        :param backend: Only ``"torch"`` is supported.
        :type backend: str
        :param store_v_seq: Whether to store membrane voltage at each time step
            in multi-step mode.
        :type store_v_seq: bool
        :raises ValueError: If backend, step_mode, channel_dim, threshold, or
            offset is invalid.
        """
        super().__init__()
        if backend != "torch":
            raise ValueError(
                f"ActivationAwareIFNode only supports backend='torch', got {backend!r}."
            )
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
        self.store_v_seq = store_v_seq
        if v_reset is None:
            self.register_memory("v", 0.0)
        else:
            self.register_memory("v", v_reset)
        self.step_mode = step_mode
        self.backend = backend

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
    def supported_backends(self):
        return ("torch",)

    @property
    def store_v_seq(self):
        return self._store_v_seq

    @store_v_seq.setter
    def store_v_seq(self, value: bool):
        self._store_v_seq = value
        if value and not hasattr(self, "v_seq"):
            self.register_memory("v_seq", None)

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

    def v_float_to_tensor(self, x: torch.Tensor) -> None:
        if isinstance(self.v, float):
            self.v = torch.full_like(x, self.v, requires_grad=False)
        elif isinstance(self.v, torch.Tensor):
            if self.v.shape != x.shape:
                fill_value = self.v_reset if self.v_reset is not None else 0.0
                self.v = torch.full_like(x, fill_value, requires_grad=False)
            elif self.v.dtype != x.dtype or self.v.device != x.device:
                self.v = self.v.to(dtype=x.dtype, device=x.device)

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        threshold = self._broadcast_parameter(self.v_threshold, x, "v_threshold")
        offset = self._broadcast_parameter(self.v_offset, x, "v_offset")
        h = self.v + x
        spike = self.surrogate_function(h + offset - threshold)
        spike_d = spike.detach() if self.detach_reset else spike
        if self.v_reset is None:
            self.v = h - spike_d * threshold
        else:
            self.v = spike_d * self.v_reset + (1.0 - spike_d) * h
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        T = x_seq.shape[0]
        y_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t])
            y_seq.append(y)
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq, dim=0)
        return torch.stack(y_seq, dim=0)

    def extra_repr(self):
        return (
            f"v_threshold_shape={tuple(self.v_threshold.shape)}, "
            f"v_offset_shape={tuple(self.v_offset.shape)}, "
            f"channel_dim={self.channel_dim}, v_reset={self.v_reset}, "
            f"detach_reset={self.detach_reset}, step_mode={self.step_mode}, "
            f"backend={self.backend}"
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
