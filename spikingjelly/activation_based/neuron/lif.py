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


__all__ = ["SimpleLIFNode", "LIFNode", "NonSpikingLIFNode"]


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
        **API Language:**
        :ref:`中文 <SimpleLIFNode.__init__-cn>` | :ref:`English <SimpleLIFNode.__init__-en>`

        ----

        .. _SimpleLIFNode.__init__-cn:

        * **中文**

        :class:`LIFNode` 的简化版实现。

        ----

        .. _SimpleLIFNode.__init__-en:

        * **English**

        A simple version of :class:`LIFNode`.
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
        **API Language:**
        :ref:`中文 <LIFNode.__init__-cn>` | :ref:`English <LIFNode.__init__-en>`

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
            使用的步进模式支持的后端。在支持的情况下，使用 ``'cupy'`` 或 ``'triton'`` 后端速度更快。
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

        :param backend: backend fot this neurons layer. Different ``step_mode`` may support for different backends. The user can
            print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
            using ``'cupy'`` or ``'triton'`` backend will be faster
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

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = self.neuronal_charge_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_decay_input(
                    x, self.v, self.v_reset, self.tau
                )

        else:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = self.neuronal_charge_no_decay_input_reset0(x, self.v, self.tau)
            else:
                self.v = self.neuronal_charge_no_decay_input(
                    x, self.v, self.v_reset, self.tau
                )

    @staticmethod
    def neuronal_charge_decay_input_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v + (x - v) / tau
        return v

    @staticmethod
    def neuronal_charge_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        return v

    @staticmethod
    def neuronal_charge_no_decay_input_reset0(
        x: torch.Tensor, v: torch.Tensor, tau: float
    ):
        v = v * (1.0 - 1.0 / tau) + x
        return v

    @staticmethod
    def neuronal_charge_no_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        return v

    @staticmethod
    def _eval_single_step_forward(
        x: torch.Tensor,
        v: torch.Tensor,
        v_threshold: float,
        v_reset,
        tau: float,
        decay_input: bool,
    ):
        """Unified single-step eval forward (replaces the 4 jit_eval_single_step_* methods)."""
        soft_reset = v_reset is None
        _vr = 0.0 if soft_reset else v_reset
        if decay_input:
            v = v + (x - (v - _vr)) / tau
        else:
            v = v - (v - _vr) / tau + x
        spike = (v >= v_threshold).to(x)
        v = (
            (v - spike * v_threshold)
            if soft_reset
            else (_vr * spike + (1.0 - spike) * v)
        )
        return spike, v

    # ---------- kept for subclass backward-compatibility ----------
    @staticmethod
    def jit_eval_single_step_forward_hard_reset_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, tau: float
    ):
        v = v + (x - (v - v_reset)) / tau
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1.0 - spike) * v
        return spike, v

    @staticmethod
    def jit_eval_single_step_forward_hard_reset_no_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float, tau: float
    ):
        v = v - (v - v_reset) / tau + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1.0 - spike) * v
        return spike, v

    @staticmethod
    def jit_eval_single_step_forward_soft_reset_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v = v + (x - v) / tau
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    def jit_eval_single_step_forward_soft_reset_no_decay_input(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, tau: float
    ):
        v = v * (1.0 - 1.0 / tau) + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    def _eval_multi_step_forward(
        x_seq: torch.Tensor,
        v: torch.Tensor,
        v_threshold: float,
        v_reset,
        tau: float,
        decay_input: bool,
        store_v_seq: bool,
        spiking: bool = True,
        surrogate_fn=None,
    ):
        """Unified fallback for all 4 LIF variants (CPU or unsupported surrogate).

        When *spiking* is False the surrogate primitive function is used to
        compute a continuous spike value instead of the hard Heaviside threshold,
        matching the behaviour of ``single_step_forward`` with ``spiking=False``.
        """
        T = x_seq.shape[0]
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq) if store_v_seq else None
        soft_reset = v_reset is None
        _vr = 0.0 if soft_reset else v_reset
        for t in range(T):
            if decay_input:
                v = v + (x_seq[t] - (v - _vr)) / tau
            else:
                v = v - (v - _vr) / tau + x_seq[t]
            if spiking:
                spike = (v >= v_threshold).to(x_seq)
            else:
                spike = surrogate_fn(v - v_threshold)
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
                        hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input
                    )
                ):
                    self.forward_kernel = ss_ac_neuron_kernel.LIFNodeFPKernel(
                        decay_input=self.decay_input, hard_reset=hard_reset, dtype=dtype
                    )

                if (
                    self.backward_kernel is None
                    or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                        decay_input=self.decay_input,
                    )
                ):
                    self.backward_kernel = ss_ac_neuron_kernel.LIFNodeBPKernel(
                        decay_input=self.decay_input,
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )

                self.v_float_to_tensor(x)
                spike, v = ss_ac_neuron_kernel.ss_lif_step(
                    x.flatten(0),
                    self.v.flatten(0),
                    self.v_threshold,
                    self.v_reset,
                    1.0 / self.tau,
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
                self.tau,
                self.decay_input,
            )
            return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == "inductor":
            return self._inductor_multi_step_forward(x_seq)
        if self.training:
            if self.backend == "torch":
                # On GPU with a supported surrogate, use the unified Triton kernel
                # (much faster than the Python for loop in super()).
                # Falls back to the Python loop for CPU or custom surrogates.
                if x_seq.is_cuda and getattr(self.surrogate_function, "spiking", True):
                    self.v_float_to_tensor(x_seq[0])
                    try:
                        spike_seq, v_seq = triton_kernel.multistep_lif(
                            x_seq,
                            self.v,
                            self.decay_input,
                            self.tau,
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
                    except (
                        NotImplementedError,
                        AttributeError,
                        TypeError,
                        KeyError,
                    ) as e:
                        logging.debug(
                            "Falling back from Triton LIF kernel in training: %s", e
                        )
                    except RuntimeError as e:
                        if _is_expected_triton_fallback_error(e):
                            logging.debug(
                                "Falling back from Triton LIF kernel in training: %s", e
                            )
                        else:
                            logging.exception(
                                "Unexpected Triton LIF kernel failure in training "
                                "(dtype=%s, surrogate=%s)",
                                x_seq.dtype,
                                type(self.surrogate_function).__name__,
                            )
                            raise
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
                        hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input
                    )
                ):
                    self.forward_kernel = ac_neuron_kernel.LIFNodeFPTTKernel(
                        decay_input=self.decay_input, hard_reset=hard_reset, dtype=dtype
                    )
                if (
                    self.backward_kernel is None
                    or not self.backward_kernel.check_attributes(
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                        decay_input=self.decay_input,
                    )
                ):
                    self.backward_kernel = ac_neuron_kernel.LIFNodeBPTTKernel(
                        decay_input=self.decay_input,
                        surrogate_function=self.surrogate_function.cuda_codes,
                        hard_reset=hard_reset,
                        detach_reset=self.detach_reset,
                        dtype=dtype,
                    )

                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = ac_neuron_kernel.multistep_lif(
                    x_seq=x_seq.flatten(1),
                    v_init=self.v.flatten(0),
                    decay_input=self.decay_input,
                    tau=self.tau,
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
                    self.v = v_seq[-1]
                else:
                    self.v = v_seq[-1].clone()
                return spike_seq
            elif self.backend == "triton":
                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = triton_kernel.multistep_lif(
                    x_seq,
                    self.v,
                    self.decay_input,
                    self.tau,
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
            else:
                raise ValueError(self.backend)

        else:
            self.v_float_to_tensor(x_seq[0])

            # All backends: try Triton on GPU first (unified kernel covers all
            # soft/hard reset x decay_input variants via tl.constexpr).
            # Falls back to the unified Python loop for CPU, custom surrogates,
            # or when spiking=False (Triton always emits hard spikes).
            if x_seq.is_cuda and getattr(self.surrogate_function, "spiking", True):
                try:
                    spike_seq, v_seq = triton_kernel.multistep_lif(
                        x_seq,
                        self.v,
                        self.decay_input,
                        self.tau,
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
                    logging.debug("Falling back from Triton LIF kernel in eval: %s", e)
                except RuntimeError as e:
                    if _is_expected_triton_fallback_error(e):
                        logging.debug(
                            "Falling back from Triton LIF kernel in eval: %s", e
                        )
                    else:
                        logging.exception(
                            "Unexpected Triton LIF kernel failure in eval "
                            "(dtype=%s, surrogate=%s)",
                            x_seq.dtype,
                            type(self.surrogate_function).__name__,
                        )
                        raise

            # CPU or unsupported surrogate: unified Python fallback
            # (replaces the 8 separate jit_eval_multi_step_forward_* methods)
            _spiking = getattr(self.surrogate_function, "spiking", True)
            out = self._eval_multi_step_forward(
                x_seq,
                self.v,
                self.v_threshold,
                self.v_reset,
                self.tau,
                self.decay_input,
                self.store_v_seq,
                spiking=_spiking,
                # When spiking=False, SurrogateFunctionBase.forward() returns the
                # primitive (smooth) function, so we can call it directly.
                surrogate_fn=self.surrogate_function if not _spiking else None,
            )
            if self.store_v_seq:
                spike_seq, self.v, self.v_seq = out
            else:
                spike_seq, self.v = out
            return spike_seq

    def _build_inductor_multi_step_graph(self):
        store_v_seq = self.store_v_seq
        soft_reset = self.v_reset is None
        v_reset = 0.0 if soft_reset else self.v_reset
        surrogate_fn = self.surrogate_function
        v_threshold = self.v_threshold
        detach_reset = self.detach_reset
        tau = self.tau
        decay_input = self.decay_input

        def _graph(x_seq: torch.Tensor, v_init: torch.Tensor):
            v = v_init
            spike_seq = torch.empty_like(x_seq)
            if store_v_seq:
                v_seq = torch.empty_like(x_seq)
            for t in range(x_seq.shape[0]):
                if decay_input:
                    v = v + (x_seq[t] - (v - v_reset)) / tau
                else:
                    v = v - (v - v_reset) / tau + x_seq[t]
                spike = surrogate_fn(v - v_threshold)
                spike_d = spike.detach() if detach_reset else spike
                if soft_reset:
                    v = v - spike_d * v_threshold
                else:
                    v = v_reset * spike_d + (1.0 - spike_d) * v
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
                "lif",
                self.store_v_seq,
                self.decay_input,
                self.tau,
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


class NonSpikingLIFNode(NonSpikingBaseNode):
    def __init__(self, tau: float = 2.0, decode: Optional[str] = None):
        """
        See also: :class:`spikingjelly.activation_based.layer.misc.SynapseFilter`.
        """
        super().__init__(decode)

        self.tau = tau

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + (x - self.v) / self.tau
