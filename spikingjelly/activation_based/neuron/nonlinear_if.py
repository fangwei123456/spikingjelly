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


__all__ = ["QIFNode", "EIFNode"]


class QIFNode(BaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        v_c: float = 0.8,
        a0: float = 1.0,
        v_threshold: float = 1.0,
        v_rest: float = 0.0,
        v_reset: Optional[float] = -0.1,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language:**
        :ref:`中文 <QIFNode.__init__-cn>` | :ref:`English <QIFNode.__init__-en>`

        ----

        .. _QIFNode.__init__-cn:

        **中文 API**

        QIF（Quadratic Integrate-and-Fire）神经元的构造函数。

        QIF 神经元是一种非线性积分发放神经元模型，也是指数积分发放神经元（EIF）的近似版本。

        **阈下动力学方程**

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c)\\right)

        :param tau: 膜电位时间常数
        :type tau: float

        :param v_c: 临界电压
        :type v_c: float

        :param a0: 二次系数
        :type a0: float

        :param v_threshold: 神经元的放电阈值
        :type v_threshold: float

        :param v_rest: 静息电位
        :type v_rest: float

        :param v_reset: 神经元的重置电压。若不为 ``None``，放电后膜电位将被重置为 ``v_reset``；
            若为 ``None``，则放电后膜电位减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param surrogate_function: 反向传播中用于近似阶跃函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否在反向传播时将 reset 过程从计算图中分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可选 ``'s'`` （单步）或 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 计算后端。不同 ``step_mode`` 支持的后端可能不同，可通过 ``self.supported_backends`` 查看。
            在支持的情况下，``'cupy'`` 或 ``'triton'`` 后端通常具有最高的执行效率
        :type backend: str

        :param store_v_seq: 当 ``step_mode = 'm'`` 且输入形状为 ``[T, N, *]`` 时，是否保存所有时间步的膜电位序列 ``self.v_seq``（形状为 ``[T, N, *]``）。
            若为 ``False``，仅保留最后一个时间步的膜电位 ``self.v``（形状为 ``[N, *]``），以降低内存开销
        :type store_v_seq: bool

        ----

        .. _QIFNode.__init__-en:

        **English API**

        Constructor of the Quadratic Integrate-and-Fire (QIF) neuron.

        The QIF neuron is a nonlinear integrate-and-fire model and an approximation of the Exponential Integrate-and-Fire (EIF) neuron.

        **Sub-threshold neuronal dynamics**

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] + a_0 (V[t-1] - V_{rest})(V[t-1] - V_c)\\right)

        :param tau: membrane time constant
        :type tau: float

        :param v_c: critical voltage
        :type v_c: float

        :param a0: quadratic coefficient
        :type a0: float

        :param v_threshold: firing threshold of the neuron
        :type v_threshold: float

        :param v_rest: resting potential
        :type v_rest: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane potential
            will be reset to ``v_reset`` after firing; otherwise, ``v_threshold`` will be subtracted
        :type v_reset: Optional[float]

        :param surrogate_function: surrogate function used to approximate the gradient
            of the Heaviside step function during backpropagation
        :type surrogate_function: Callable

        :param detach_reset: whether to detach the reset operation from the computation graph
        :type detach_reset: bool

        :param step_mode: step mode, either ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str

        :param backend: backend for this neuron. Different ``step_mode`` may support different backends.
            Supported backends can be queried via ``self.supported_backends``.
            If available, ``'cupy'`` or ``'triton'`` usually provides the fastest execution
        :type backend: str

        :param store_v_seq: when ``step_mode = 'm'`` and input shape is ``[T, N, *]``,
            whether to store the membrane potential at all time steps in ``self.v_seq``.
            If ``False``, only the final membrane potential ``self.v`` is kept to reduce memory usage
        :type store_v_seq: bool
        """
        assert isinstance(tau, float) and tau > 1.0
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert a0 > 0

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
        self.v_c = v_c
        self.v_rest = v_rest
        self.a0 = a0

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", tau={self.tau}, v_c={self.v_c}, a0={self.a0}, v_rest={self.v_rest}"
        )

    def neuronal_charge(self, x: torch.Tensor):
        self.v = (
            self.v
            + (x + self.a0 * (self.v - self.v_rest) * (self.v - self.v_c)) / self.tau
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

            spike_seq, v_seq = cuda_kernel.MultiStepQIFNodePTT.apply(
                x_seq.flatten(1),
                self.v.flatten(0),
                self.tau,
                self.v_threshold,
                self.v_reset,
                self.v_rest,
                self.v_c,
                self.a0,
                self.detach_reset,
                self.surrogate_function.cuda_code,
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)


class EIFNode(BaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        delta_T: float = 1.0,
        theta_rh: float = 0.8,
        v_threshold: float = 1.0,
        v_rest: float = 0.0,
        v_reset: Optional[float] = -0.1,
        surrogate_function: Callable = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language:**
        :ref:`中文 <EIFNode.__init__-cn>` | :ref:`English <EIFNode.__init__-en>`

        ----

        .. _EIFNode.__init__-cn:

        **中文 API**

        EIF（Exponential Integrate-and-Fire）神经元的构造函数。

        EIF 神经元是一种非线性积分发放神经元模型，由 Hodgkin-Huxley 模型简化得到的一维模型。
        当 :math:`\\Delta_T \\to 0` 时，退化为普通的 LIF 神经元。

        **阈下动力学方程**

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)

        :param tau: 膜电位时间常数
        :type tau: float

        :param delta_T: 陡峭度参数
        :type delta_T: float

        :param theta_rh: 基强度电压阈值
        :type theta_rh: float

        :param v_threshold: 神经元的放电阈值
        :type v_threshold: float

        :param v_reset: 神经元的重置电压。若不为 ``None``，放电后膜电位将被重置为 ``v_reset``；
            若为 ``None``，则放电后膜电位减去 ``v_threshold``
        :type v_reset: Optional[float]

        :param v_rest: 静息电位
        :type v_rest: float

        :param surrogate_function: 反向传播中用于近似阶跃函数梯度的替代函数
        :type surrogate_function: Callable

        :param detach_reset: 是否在反向传播时将 reset 过程从计算图中分离
        :type detach_reset: bool

        :param step_mode: 步进模式，可选 ``'s'`` （单步）或 ``'m'`` （多步）
        :type step_mode: str

        :param backend: 计算后端。不同 ``step_mode`` 支持的后端可能不同，可通过 ``self.supported_backends`` 查看。
            在支持的情况下，``'cupy'`` 或 ``'triton'`` 后端通常具有最高的执行效率
        :type backend: str

        :param store_v_seq: 当 ``step_mode = 'm'`` 且输入形状为 ``[T, N, *]`` 时，是否保存所有时间步的膜电位序列 ``self.v_seq``（形状为 ``[T, N, *]``）。
            若为 ``False``，仅保留最后一个时间步的膜电位 ``self.v``（形状为 ``[N, *]``），以降低内存开销
        :type store_v_seq: bool

        ----

        .. _EIFNode.__init__-en:

        **English API**

        Constructor of the Exponential Integrate-and-Fire (EIF) neuron.

        The EIF neuron is a nonlinear integrate-and-fire model derived from the Hodgkin-Huxley model.
        It degenerates to the LIF model when :math:`\\Delta_T \\to 0`.

        **Sub-threshold neuronal dynamics**

        .. math::
            H[t] = V[t-1] + \\frac{1}{\\tau}\\left(X[t] - (V[t-1] - V_{rest}) + \\Delta_T\\exp\\left(\\frac{V[t-1] - \\theta_{rh}}{\\Delta_T}\\right)\\right)

        :param tau: membrane time constant
        :type tau: float

        :param delta_T: sharpness parameter
        :type delta_T: float

        :param theta_rh: rheobase threshold
        :type theta_rh: float

        :param v_threshold: firing threshold of the neuron
        :type v_threshold: float

        :param v_reset: reset voltage of the neuron. If not ``None``, the membrane potential
            will be reset to ``v_reset`` after firing; otherwise, ``v_threshold`` will be subtracted
        :type v_reset: Optional[float]

        :param v_rest: resting potential
        :type v_rest: float

        :param surrogate_function: surrogate function used to approximate the gradient
            of the Heaviside step function during backpropagation
        :type surrogate_function: Callable

        :param detach_reset: whether to detach the reset operation from the computation graph
        :type detach_reset: bool

        :param step_mode: step mode, either ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str

        :param backend: backend for this neuron. Different ``step_mode`` may support different backends.
            Supported backends can be queried via ``self.supported_backends``.
            If available, ``'cupy'`` or ``'triton'`` usually provides the fastest execution
        :type backend: str

        :param store_v_seq: when ``step_mode = 'm'`` and input shape is ``[T, N, *]``,
            whether to store the membrane potential at all time steps in ``self.v_seq``.
            If ``False``, only the final membrane potential ``self.v`` is kept to reduce memory usage
        :type store_v_seq: bool
        """
        assert isinstance(tau, float) and tau > 1.0
        if v_reset is not None:
            assert v_threshold > v_reset
            assert v_rest >= v_reset
        assert delta_T > 0

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
        self.delta_T = delta_T
        self.v_rest = v_rest
        self.theta_rh = theta_rh

    def extra_repr(self):
        return (
            super().extra_repr()
            + f", tau={self.tau}, delta_T={self.delta_T}, theta_rh={self.theta_rh}"
        )

    def neuronal_charge(self, x: torch.Tensor):
        with torch.no_grad():
            if not isinstance(self.v, torch.Tensor):
                self.v = torch.as_tensor(self.v, device=x.device)

        self.v = (
            self.v
            + (
                x
                + self.v_rest
                - self.v
                + self.delta_T * torch.exp((self.v - self.theta_rh) / self.delta_T)
            )
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

            spike_seq, v_seq = cuda_kernel.MultiStepEIFNodePTT.apply(
                x_seq.flatten(1),
                self.v.flatten(0),
                self.tau,
                self.v_threshold,
                self.v_reset,
                self.v_rest,
                self.theta_rh,
                self.delta_T,
                self.detach_reset,
                self.surrogate_function.cuda_code,
            )

            spike_seq = spike_seq.reshape(x_seq.shape)
            v_seq = v_seq.reshape(x_seq.shape)

            if self.store_v_seq:
                self.v_seq = v_seq

            self.v = v_seq[-1].clone()

            return spike_seq
        else:
            raise ValueError(self.backend)
