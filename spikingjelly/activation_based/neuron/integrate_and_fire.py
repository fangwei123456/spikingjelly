from typing import Callable, Optional
import logging

import torch

from .. import surrogate
from .base_node import SimpleBaseNode, BaseNode, NonSpikingBaseNode

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


class SimpleIFNode(SimpleBaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
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

        ----

        .. _SimpleIFNode.__init__-en:

        * **English**

        A simple version of :class:`IFNode`.
        """
        super().__init__(
            v_threshold, v_reset, surrogate_function, detach_reset, step_mode
        )

    def neuronal_charge(self, x: torch.Tensor):
        """
        .. math::
            H[t] = V[t-1] + X[t]
        """
        self.v = self.v + x


class IFNode(BaseNode):
    def __init__(
        self,
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
        :type surrogate_function: Callable

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
        :type surrogate_function: Callable

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
            return ("torch", "cupy", "triton")
        else:
            raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_hard_reset(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v_reset * spike + (1.0 - spike) * v
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_single_step_forward_soft_reset(
        x: torch.Tensor, v: torch.Tensor, v_threshold: float
    ):
        v = v + x
        spike = (v >= v_threshold).to(x)
        v = v - spike * v_threshold
        return spike, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1.0 - spike) * v
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_hard_reset_with_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float, v_reset: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v_reset * spike + (1.0 - spike) * v
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
        return spike_seq, v

    @staticmethod
    @torch.jit.script
    def jit_eval_multi_step_forward_soft_reset_with_v_seq(
        x_seq: torch.Tensor, v: torch.Tensor, v_threshold: float
    ):
        spike_seq = torch.zeros_like(x_seq)
        v_seq = torch.zeros_like(x_seq)
        for t in range(x_seq.shape[0]):
            v = v + x_seq[t]
            spike = (v >= v_threshold).to(x_seq)
            v = v - spike * v_threshold
            spike_seq[t] = spike
            v_seq[t] = v
        return spike_seq, v, v_seq

    def multi_step_forward(self, x_seq: torch.Tensor):
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
                spike_seq, v_seq = ac_neuron_kernel.IFNodeATGF.apply(
                    x_seq.flatten(1),
                    self.v.flatten(0),
                    self.v_threshold,
                    self.v_reset,
                    self.forward_kernel,
                    self.backward_kernel,
                )
                spike_seq = spike_seq.reshape(x_seq.shape)
                v_seq = v_seq.reshape(x_seq.shape)
                if self.store_v_seq:
                    self.v_seq = v_seq
                self.v = v_seq[-1].clone()
                return spike_seq
            elif self.backend == "triton":
                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = triton_kernel.MultiStepIFNodePTT.apply(
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
                self.v_float_to_tensor(x_seq[0])
                spike_seq, v_seq = triton_kernel.MultiStepIFNodePTT.apply(
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

            # torch & cupy backend:
            if self.v_reset is None:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = (
                        self.jit_eval_multi_step_forward_soft_reset_with_v_seq(
                            x_seq, self.v, self.v_threshold
                        )
                    )
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_soft_reset(
                        x_seq, self.v, self.v_threshold
                    )
            else:
                if self.store_v_seq:
                    spike_seq, self.v, self.v_seq = (
                        self.jit_eval_multi_step_forward_hard_reset_with_v_seq(
                            x_seq, self.v, self.v_threshold, self.v_reset
                        )
                    )
                else:
                    spike_seq, self.v = self.jit_eval_multi_step_forward_hard_reset(
                        x_seq, self.v, self.v_threshold, self.v_reset
                    )
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
                spike, v = ss_ac_neuron_kernel.IFNodeATGF.apply(
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
            if self.v_reset is None:
                spike, self.v = self.jit_eval_single_step_forward_soft_reset(
                    x, self.v, self.v_threshold
                )
            else:
                spike, self.v = self.jit_eval_single_step_forward_hard_reset(
                    x, self.v, self.v_threshold, self.v_reset
                )
            return spike


class NonSpikingIFNode(NonSpikingBaseNode):
    def __init__(self, decode: Optional[str] = None):
        super().__init__(decode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
