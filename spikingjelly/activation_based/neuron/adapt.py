import logging
from typing import Optional

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
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language** - :ref:`中文 <AdaptBaseNode.__init__-cn>` | :ref:`English <AdaptBaseNode.__init__-en>`

        ----

        .. _AdaptBaseNode.__init__-cn:

        * **中文**

        带适应性电流的脉冲神经元基类。在 :class:`BaseNode` 的基础上增加了膜电位恢复变量 :math:`w`，用于实现神经元适应性和脉冲频率适应性。

        :param v_threshold: 神经元的阈值电压
        :type v_threshold: float
        :param v_reset: 重置电压。若为 ``None`` 则使用软重置
        :type v_reset: Optional[float]
        :param v_rest: 静息电位
        :type v_rest: float
        :param w_rest: 适应性电流的静息值
        :type w_rest: float
        :param tau_w: 适应性电流的时间常数
        :type tau_w: float
        :param a: 阈下耦合参数，控制亚阈值电位对适应电流的影响
        :type a: float
        :param b: 脉冲触发跳跃幅度，控制脉冲后适应电流的增加量
        :type b: float
        :param surrogate_function: 替代梯度函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否将重置过程的计算图分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str
        :param backend: 后端
        :type backend: str
        :param store_v_seq: 是否保存中间电压值
        :type store_v_seq: bool

        ----

        .. _AdaptBaseNode.__init__-en:

        * **English**

        Base neuron with adaptation current. Extends :class:`BaseNode` with a membrane recovery variable :math:`w` that provides spike-frequency adaptation.

        :param v_threshold: Threshold voltage of the neuron
        :type v_threshold: float
        :param v_reset: Reset voltage. If ``None``, uses soft reset
        :type v_reset: Optional[float]
        :param v_rest: Resting potential
        :type v_rest: float
        :param w_rest: Resting value of the adaptation current
        :type w_rest: float
        :param tau_w: Time constant of the adaptation current
        :type tau_w: float
        :param a: Subthreshold coupling parameter, controls subthreshold influence on adaptation current
        :type a: float
        :param b: Spike-triggered jump amplitude, controls adaptation current increase after each spike
        :type b: float
        :param surrogate_function: Surrogate gradient function
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach the reset computation graph
        :type detach_reset: bool
        :param step_mode: Step mode, can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        :param backend: Backend for computation
        :type backend: str
        :param store_v_seq: Whether to store intermediate membrane potentials
        :type store_v_seq: bool
        """
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
    def jit_neuronal_adaptation(
        w: torch.Tensor, tau_w: float, a: float, v_rest: float, v: torch.Tensor
    ):
        return w + 1.0 / tau_w * (a * (v - v_rest) - w)

    def neuronal_adaptation(self):
        """
        **API Language** - :ref:`中文 <AdaptBaseNode.neuronal_adaptation-cn>` | :ref:`English <AdaptBaseNode.neuronal_adaptation-en>`

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
    def apply_hard_reset(
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
    def apply_soft_reset(
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
        **API Language** - :ref:`中文 <AdaptBaseNode.neuronal_reset-cn>` | :ref:`English <AdaptBaseNode.neuronal_reset-en>`

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
            self.v, self.w = self.apply_soft_reset(
                self.v, self.w, spike_d, self.v_threshold, self.b, spike
            )

        else:
            # hard reset
            self.v, self.w = self.apply_hard_reset(
                self.v, self.w, spike_d, self.v_reset, self.b, spike
            )

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", v_rest={self.v_rest}, w_rest={self.w_rest}, tau_w={self.tau_w}, a={self.a}, b={self.b}"
        )

    def single_step_forward(self, x: torch.Tensor):
        """
        **API Language** - :ref:`中文 <AdaptBaseNode.single_step_forward-cn>` | :ref:`English <AdaptBaseNode.single_step_forward-en>`

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
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language** - :ref:`中文 <IzhikevichNode.__init__-cn>` | :ref:`English <IzhikevichNode.__init__-en>`

        ----

        .. _IzhikevichNode.__init__-cn:

        * **中文**

        Izhikevich 脉冲神经元模型。参数 :math:`\\tau` 控制膜电位时间常数，:math:`v_c` 和 :math:`a0` 控制非线性 dynamics。
        继承了 :class:`AdaptBaseNode` 的适应性电流机制。

        :param tau: 膜电位时间常数
        :type tau: float
        :param v_c: 截止电压，控制非线性响应的阈值
        :type v_c: float
        :param a0: 非线性系数
        :type a0: float
        :param v_threshold: 阈值电压
        :type v_threshold: float
        :param v_reset: 重置电压
        :type v_reset: Optional[float]
        :param v_rest: 静息电位
        :type v_rest: float
        :param w_rest: 适应性电流静息值
        :type w_rest: float
        :param tau_w: 适应性电流时间常数
        :type tau_w: float
        :param a: 阈下耦合参数
        :type a: float
        :param b: 脉冲触发跳跃幅度
        :type b: float
        :param surrogate_function: 替代梯度函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否分离重置计算图
        :type detach_reset: bool
        :param step_mode: 步进模式
        :type step_mode: str
        :param backend: 后端
        :type backend: str
        :param store_v_seq: 是否保存中间电压值
        :type store_v_seq: bool

        ----

        .. _IzhikevichNode.__init__-en:

        * **English**

        Izhikevich spiking neuron model. The parameters :math:`\\tau`, :math:`v_c`, and :math:`a0` control membrane dynamics.
        Inherits the adaptation current mechanism from :class:`AdaptBaseNode`.

        :param tau: Membrane time constant
        :type tau: float
        :param v_c: Cutoff voltage controlling the nonlinear response threshold
        :type v_c: float
        :param a0: Nonlinear coefficient
        :type a0: float
        :param v_threshold: Threshold voltage
        :type v_threshold: float
        :param v_reset: Reset voltage
        :type v_reset: Optional[float]
        :param v_rest: Resting potential
        :type v_rest: float
        :param w_rest: Resting value of adaptation current
        :type w_rest: float
        :param tau_w: Time constant of adaptation current
        :type tau_w: float
        :param a: Subthreshold coupling parameter
        :type a: float
        :param b: Spike-triggered jump amplitude
        :type b: float
        :param surrogate_function: Surrogate gradient function
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach reset computation graph
        :type detach_reset: bool
        :param step_mode: Step mode, ``'s'`` or ``'m'``
        :type step_mode: str
        :param backend: Backend
        :type backend: str
        :param store_v_seq: Whether to store intermediate membrane potentials
        :type store_v_seq: bool
        """
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
            spike_seq, v_seq, w_seq = cuda_kernel.multistep_izhikevich_ptt(
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
                self.surrogate_function,
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
