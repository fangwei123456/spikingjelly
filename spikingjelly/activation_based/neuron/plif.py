from typing import Callable, Optional
import logging
import math

import torch
import torch.nn as nn

from .. import surrogate
from .base_node import BaseNode

try:
    from ..cuda_kernel.auto_cuda import neuron_kernel as ac_neuron_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    ac_neuron_kernel = None

try:
    from .. import triton_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    triton_kernel = None


__all__ = ["ParametricLIFNode"]


class ParametricLIFNode(BaseNode):
    def __init__(
        self,
        init_tau: float = 2.0,
        decay_input: bool = True,
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
        :ref:`中文 <ParametricLIFNode.__init__-cn>` | :ref:`English <ParametricLIFNode.__init__-en>`

        ----

        .. _ParametricLIFNode.__init__-cn:

        * **中文**

        Parametric Leaky Integrate-and-Fire (PLIF) 神经元模型，提出自 `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_。可以看作是带漏电的积分器。其阈下神经动力学方程为：

        若 ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        若 ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        其中 :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`，:math:`w` 是可学习的参数。

        :param init_tau: 膜电位时间常数的初始值
        :type init_tau: float

        :param decay_input: 输入是否也会参与衰减
        :type decay_input: bool

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

        .. _ParametricLIFNode.__init__-en:

        * **English**

        The Parametric Leaky Integrate-and-Fire (PLIF) neuron, proposed in `Incorporating Learnable Membrane Time Constant to Enhance Learning of Spiking Neural Networks <https://arxiv.org/abs/2007.05785>`_, can be seen as a leaky integrator. The subthreshold neural dynamics of it is as followed:

        IF ``decay_input == True``:

            .. math::
                H[t] = V[t-1] + \\frac{1}{\\tau}(X[t] - (V[t-1] - V_{reset}))

        IF ``decay_input == False``:

            .. math::
                H[t] = V[t-1] - \\frac{1}{\\tau}(V[t-1] - V_{reset}) + X[t]

        where :math:`\\frac{1}{\\tau} = {\\rm Sigmoid}(w)`, :math:`w` is a learnable parameter.

        :param init_tau: the initial value of membrane time constant
        :type init_tau: float

        :param decay_input: whether the input will decay
        :type decay_input: bool

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

        :param backend: backend for this neurons layer. Different ``step_mode`` may support for different backends. The user can
            print ``self.supported_backends`` and check what backends are supported by the current ``step_mode``. If supported,
            using ``'cupy'`` or ``'triton'`` backend will have the fastest training speed
        :type backend: str

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption
        :type store_v_seq: bool
        """
        assert isinstance(init_tau, float) and init_tau > 1.0
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        self.decay_input = decay_input
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.as_tensor(init_w))  # as reciprocal_tau

    @property
    def supported_backends(self):
        if self.step_mode == "s":
            return ("torch",)
        elif self.step_mode == "m":
            return ("torch", "cupy", "triton")
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        with torch.no_grad():
            tau = 1.0 / self.w.sigmoid()
        return super().extra_repr() + f", tau={tau}"

    def neuronal_charge(self, x: torch.Tensor):
        if self.decay_input:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = self.v + (x - self.v) * self.w.sigmoid()
            else:
                self.v = self.v + (x - (self.v - self.v_reset)) * self.w.sigmoid()
        else:
            if self.v_reset is None or self.v_reset == 0.0:
                self.v = self.v * (1.0 - self.w.sigmoid()) + x
            else:
                self.v = self.v - (self.v - self.v_reset) * self.w.sigmoid() + x

    def multi_step_forward(self, x_seq: torch.Tensor):
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
            if self.forward_kernel is None or not self.forward_kernel.check_attributes(
                hard_reset=hard_reset, dtype=dtype, decay_input=self.decay_input
            ):
                self.forward_kernel = ac_neuron_kernel.ParametricLIFNodeFPTTKernel(
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
                self.backward_kernel = ac_neuron_kernel.ParametricLIFNodeBPTTKernel(
                    decay_input=self.decay_input,
                    surrogate_function=self.surrogate_function.cuda_codes,
                    hard_reset=hard_reset,
                    detach_reset=self.detach_reset,
                    dtype=dtype,
                )
            self.v_float_to_tensor(x_seq[0])
            spike_seq, v_seq = ac_neuron_kernel.ParametricLIFNodeATGF.apply(
                x_seq.flatten(1),
                self.v.flatten(0),
                self.v_threshold,
                self.v_reset,
                self.w.sigmoid().to(x_seq),
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
            spike_seq, v_seq = triton_kernel.MultiStepParametricLIFNodePTT.apply(
                x_seq,
                self.v,
                self.w.sigmoid().to(x_seq),
                self.decay_input,
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
