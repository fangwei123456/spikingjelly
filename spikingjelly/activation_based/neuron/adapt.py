from typing import Callable, Optional
import logging

import torch

from .. import surrogate
from .base_node import BaseNode

try:
    from .. import cuda_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron: {e}")
    cuda_kernel = None


__all__ = ["AdaptBaseNode", "IzhikevichNode"]


class AdaptBaseNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        v_rest: float = 0.0,
        w_rest: float = 0.0,
        tau_w: float = 2.0,
        a: float = 0.0,
        b: float = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        # b: jump amplitudes
        # a: subthreshold coupling
        assert isinstance(w_rest, float)
        assert isinstance(v_rest, float)
        assert isinstance(tau_w, float)
        assert isinstance(a, float)
        assert isinstance(b, float)

        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )

        self.register_memory("w", w_rest)

        self.w_rest = w_rest
        self.v_rest = v_rest
        self.tau_w = tau_w
        self.a = a
        self.b = b

    @staticmethod
    @torch.jit.script
    def jit_neuronal_adaptation(
        w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor
    ):
        return w + 1.0 / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):
        """
        **API Language:**
        :ref:`中文 <AdaptBaseNode.neuronal_adaptation-cn>` | :ref:`English <AdaptBaseNode.neuronal_adaptation-en>`

        ----

        .. _AdaptBaseNode.neuronal_adaptation-cn:

        * **中文**

        脉冲触发的适应性电流的更新

        ----

        .. _AdaptBaseNode.neuronal_adaptation-en:

        * **English**

        Spike-triggered update of adaptation current.
        """
        self.w = self.jit_neuronal_adaptation(
            self.w, self.tau_w, self.a, self.v_rest, self.v
        )

    @staticmethod
    @torch.jit.script
    def jit_hard_reset(
        v: torch.Tensor,
        w: torch.Tensor,
        spike_d: torch.Tensor,
        v_reset: float,
        b: float,
        spike: torch.Tensor,
    ):
        v = (1.0 - spike_d) * v + spike * v_reset
        w = w + b * spike
        return v, w

    @staticmethod
    @torch.jit.script
    def jit_soft_reset(
        v: torch.Tensor,
        w: torch.Tensor,
        spike_d: torch.Tensor,
        v_threshold: float,
        b: float,
        spike: torch.Tensor,
    ):
        v = v - spike_d * v_threshold
        w = w + b * spike
        return v, w

    def neuronal_reset(self, spike):
        """
        **API Language:**
        :ref:`中文 <AdaptBaseNode.neuronal_reset-cn>` | :ref:`English <AdaptBaseNode.neuronal_reset-en>`

        ----

        .. _AdaptBaseNode.neuronal_reset-cn:

        * **中文**

        根据当前神经元释放的脉冲，对膜电位进行重置。

        ----

        .. _AdaptBaseNode.neuronal_reset-en:

        * **English**

        Reset the membrane potential according to neurons' output spikes.
        """
        if self.detach_reset:
            spike_d = spike.detach()
        else:
            spike_d = spike

        if self.v_reset is None:
            # soft reset
            self.v, self.w = self.jit_soft_reset(
                self.v, self.w, spike_d, self.v_threshold, self.b, spike
            )

        else:
            # hard reset
            self.v, self.w = self.jit_hard_reset(
                self.v, self.w, spike_d, self.v_reset, self.b, spike
            )

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}"
        )

    def single_step_forward(self, x: torch.Tensor):
        """
        **API Language:**
        :ref:`中文 <AdaptBaseNode.single_step_forward-cn>` | :ref:`English <AdaptBaseNode.single_step_forward-en>`

        ----

        .. _AdaptBaseNode.single_step_forward-cn:

        * **中文**

        按照充电、适应、放电、重置的顺序进行前向传播。

        :param x: 输入到神经元的电压增量
        :type x: torch.Tensor

        :return: 神经元的输出脉冲
        :rtype: torch.Tensor

        ----

        .. _AdaptBaseNode.single_step_forward-en:

        * **English**

        Forward by the order of ``neuronal_charge``, ``neuronal_adaptation``, ``neuronal_fire``, and ``neuronal_reset``.

        :param x: increment of voltage inputted to neurons
        :type x: torch.Tensor

        :return: out spikes of neurons
        :rtype: torch.Tensor
        """
        self.v_float_to_tensor(x)
        self.w_float_to_tensor(x)
        self.neuronal_charge(x)
        self.neuronal_adaptation()
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def w_float_to_tensor(self, x: torch.Tensor):
        if isinstance(self.w, float):
            w_init = self.w
            self.w = torch.full_like(x.data, fill_value=w_init)


class IzhikevichNode(AdaptBaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        v_c: float = 0.8,
        a0: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        v_rest: float = -0.1,
        w_rest: float = 0.0,
        tau_w: float = 2.0,
        a: float = 0.0,
        b: float = 0.0,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        assert isinstance(tau, float) and tau > 1.0
        assert a0 > 0

        super().__init__(
            v_threshold,
            v_reset,
            v_rest,
            w_rest,
            tau_w,
            a,
            b,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        self.tau = tau
        self.v_c = v_c
        self.a0 = a0

    def extra_repr(self):
        return super().extra_repr() + f", tau={self.tau}, v_c={self.v_c}, a0={self.a0}"

    def neuronal_charge(self, x: torch.Tensor):
        self.v = (
            self.v
            + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c) - self.w)
            / self.tau
        )

    @property
    def supported_backends(self):
        if self.step_mode == "s":
            return ("torch",)
        elif self.step_mode == "m":
            return ("torch", "cupy")
        else:
            raise ValueError(self.step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == "torch":
            return super().multi_step_forward(x_seq)
        elif self.backend == "cupy":
            self.v_float_to_tensor(x_seq[0])
            self.w_float_to_tensor(x_seq[0])

            spike_seq, v_seq, w_seq = cuda_kernel.MultiStepIzhikevichNodePTT.apply(
                x_seq.flatten(1),
                self.v.flatten(0),
                self.w.flatten(0),
                self.tau,
                self.v_threshold,
                self.v_reset,
                self.v_rest,
                self.a,
                self.b,
                self.tau_w,
                self.v_c,
                self.a0,
                self.detach_reset,
                self.surrogate_function.cuda_code,
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)
            w_seq = w_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()
            self.w = w_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)
