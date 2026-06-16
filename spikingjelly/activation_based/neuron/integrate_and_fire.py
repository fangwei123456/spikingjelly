import logging
from typing import Optional

import torch

from .. import surrogate
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


__all__ = ["SimpleIFNode", "IFNode", "NonSpikingIFNode"]


def _is_expected_triton_fallback_error(exc: RuntimeError) -> bool:
    message = str(exc).lower()
    expected_markers = (
        "unsupported",
        "not supported",
        "no triton",
        "triton is not installed",
        "failed to import triton",
        "dtype",
        "invalid argument",
    )
    return any(marker in message for marker in expected_markers)


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
        **API Language:**
        :ref:`中文 <SimpleIFNode.__init__-cn>` | :ref:`English <SimpleIFNode.__init__-en>`

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
        :return: None
        :rtype: None
        """
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <SimpleIFNode.neuronal_charge-cn>` | :ref:`English <SimpleIFNode.neuronal_charge-en>`

        ----

        .. _SimpleIFNode.neuronal_charge-cn:
        * **中文**

        * **中文**

        神经元充电的微分方程：

        .. math::
            H[t] = V[t-1] + X[t]

        :param x: 输入电压
        :type x: torch.Tensor
        :return: None（膜电位更新存储在 ``self.v`` 中）
        :rtype: None

        ----

        .. _SimpleIFNode.neuronal_charge-en:
        * **English**

        * **English**

        The differential equation for neuronal charge:

        .. math::
            H[t] = V[t-1] + X[t]

        :param x: Input voltage
        :type x: torch.Tensor
        :return: None (membrane potential is stored in ``self.v``)
        :rtype: None
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
        **API Language:**
        :ref:`中文 <IFNode.__init__-cn>` | :ref:`English <IFNode.__init__-en>`

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
        :return: None
        :rtype: None
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
        x: torch.Tensor,
        v: torch.Tensor,
        v_threshold: float,
        v_reset,
        tau: Optional[float] = None,
        decay_input: Optional[bool] = None,
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
        tau: Optional[float] = None,
        decay_input: Optional[bool] = None,
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
                spike_seq, v_seq = triton_kernel.multistep_if(
                    x_seq,
                    self.v,
                    self.v_threshold,
                    self.v_reset,
                    self.detach_reset,
                    self.surrogate_function,
                )
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
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
                try:
                    spike_seq, v_seq = triton_kernel.multistep_if(
                        x_seq,
                        self.v,
                        self.v_threshold,
                        self.v_reset,
                        self.detach_reset,
                        self.surrogate_function,
                    )
                    if self.store_v_seq:
                        self.v_seq = v_seq
                        self.v = v_seq[-1]
                    else:
                        self.v = v_seq[-1].clone()
                    return spike_seq
                except (NotImplementedError, AttributeError, TypeError, KeyError) as e:
                    logging.debug("Falling back from Triton IF kernel in eval: %s", e)
                except RuntimeError as e:
                    if _is_expected_triton_fallback_error(e):
                        logging.debug(
                            "Falling back from Triton IF kernel in eval: %s", e
                        )
                    else:
                        logging.exception(
                            "Unexpected Triton IF kernel failure in eval "
                            "(dtype=%s, surrogate=%s)",
                            x_seq.dtype,
                            type(self.surrogate_function).__name__,
                        )
                        raise

            # torch & cupy backend:
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


class NonSpikingIFNode(NonSpikingBaseNode):
    def __init__(self, decode: Optional[str] = None):
        """
        **API Language:**
        :ref:`中文 <NonSpikingIFNode.__init__-cn>` | :ref:`English <NonSpikingIFNode.__init__-en>`

        ----

        .. _NonSpikingIFNode.__init__-cn:
        * **中文**

        * **中文**

        不发放脉冲的 IF 节点，输出膜电位（或根据 ``decode`` 进行解码）。

        :param decode: 非脉冲输出解码方式，见 :class:`NonSpikingBaseNode`
        :type decode: Optional[str]

        ----

        .. _NonSpikingIFNode.__init__-en:
        * **English**

        * **English**

        Non-spiking IF node that outputs membrane potential (or decoded outputs specified by ``decode``).

        :param decode: Decoding mode for non-spiking outputs, see :class:`NonSpikingBaseNode`
        :type decode: Optional[str]
        :return: None
        :rtype: None
        """
        super().__init__(decode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
