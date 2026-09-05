from typing import Optional

import torch

from .. import functional, surrogate
from .base_node import BaseNode


__all__ = ["RAFNode"]


class RAFNode(BaseNode):
    def __init__(
        self,
        b: float = -0.2,
        omega: float = 1.0,
        dt: float = 1.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode="s",
        backend="torch",
        store_v_seq: bool = False,
    ):
        """
        **API Language** - :ref:`中文 <RAFNode.__init__-cn>` | :ref:`English <RAFNode.__init__-en>`

        ----

        .. _RAFNode.__init__-cn:

        * **中文**

        Resonate-and-fire (RAF) 神经元的构造函数（Izhikevich, *Resonate-and-fire
        neurons*, Neural Networks 14 (2001) 883-894）。

        与积分发放（IF/LIF/QIF/EIF/Izhikevich）神经元家族不同，RAF 神经元是一个
        二维线性阈下*振荡*系统，而非积分器：等价于一个以固定衰减率 :math:`b < 0`
        和固有角频率 :math:`\\omega` 旋转衰减的复数状态 :math:`z = u + iv`，因此对
        接近 :math:`\\omega` 的输入频率有选择性响应，并表现出阈下阻尼振荡。

        **阈下动力学方程**

        .. math::

            u[t] &= \\alpha (u[t-1]\\cos\\theta - v[t-1]\\sin\\theta) + x[t] \\\\
            v[t] &= \\alpha (u[t-1]\\sin\\theta + v[t-1]\\cos\\theta)

        其中 :math:`\\alpha = \\exp(b\\,dt)`，:math:`\\theta = \\omega\\,dt`。状态
        由两个实数张量 ``self.u``（实部，接收输入）和 ``self.v``（虚部，放电分量）
        表示，而非复数张量。放电依据 ``self.v`` 相对 ``v_threshold`` 判定；放电后
        只重置 ``self.v``（复用 :class:`BaseNode` 的 soft/hard reset 规则），
        ``self.u`` 不受影响 —— 这正是产生放电后反弹（post-inhibitory rebound）
        的原因。

        本次实现固定 :math:`b`、:math:`\\omega`、:math:`dt`（不可学习），仅支持
        ``'torch'`` 后端；可学习参数与其它后端留作后续工作。

        :param b: 衰减率，须为负数
        :type b: float
        :param omega: 固有角频率，须为正数
        :type omega: float
        :param dt: 积分步长，须为正数
        :type dt: float
        :param v_threshold: 神经元的放电阈值
        :type v_threshold: float
        :param v_reset: 神经元的重置电压。若不为 ``None``，放电后 ``self.v``
            将被重置为 ``v_reset``；若为 ``None``，则放电后 ``self.v`` 减去
            ``v_threshold``。``self.u`` 在任一情况下都不重置
        :type v_reset: Optional[float]
        :param surrogate_function: 反向传播中用于近似阶跃函数梯度的替代函数
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否在反向传播时将 reset 过程从计算图中分离
        :type detach_reset: bool
        :param step_mode: 步进模式，可选 ``'s'`` （单步）或 ``'m'`` （多步）
        :type step_mode: str
        :param backend: 计算后端。目前仅支持 ``'torch'``
        :type backend: str
        :param store_v_seq: 当 ``step_mode = 'm'`` 且输入形状为 ``[T, N, *]`` 时，
            是否保存所有时间步的放电分量序列 ``self.v_seq``（形状为
            ``[T, N, *]``）。若为 ``False``，仅保留最后一个时间步的 ``self.v``
            （形状为 ``[N, *]``），以降低内存开销
        :type store_v_seq: bool

        ----

        .. _RAFNode.__init__-en:

        * **English**

        Constructor of the resonate-and-fire (RAF) neuron (Izhikevich,
        *Resonate-and-fire neurons*, Neural Networks 14 (2001) 883-894).

        Unlike the integrate-and-fire family (IF/LIF/QIF/EIF/Izhikevich), the
        RAF neuron is a 2-D linear subthreshold *oscillator*, not an
        integrator: it is equivalent to a complex state :math:`z = u + iv`
        that rotates and decays at a fixed rate :math:`b < 0` and intrinsic
        angular frequency :math:`\\omega`, so it responds preferentially to
        input near :math:`\\omega` and shows damped subthreshold oscillations.

        **Sub-threshold neuronal dynamics**

        .. math::

            u[t] &= \\alpha (u[t-1]\\cos\\theta - v[t-1]\\sin\\theta) + x[t] \\\\
            v[t] &= \\alpha (u[t-1]\\sin\\theta + v[t-1]\\cos\\theta)

        where :math:`\\alpha = \\exp(b\\,dt)` and :math:`\\theta = \\omega\\,dt`.
        State is two real-valued tensors, ``self.u`` (real part, receives the
        input) and ``self.v`` (imaginary part, the firing component), not a
        complex tensor. Firing is decided from ``self.v`` against
        ``v_threshold``; only ``self.v`` is reset after a spike (reusing
        :class:`BaseNode`'s soft/hard reset rule) — ``self.u`` is left
        untouched, which is what produces post-inhibitory rebound.

        This implementation fixes :math:`b`, :math:`\\omega`, :math:`dt`
        (not learnable) and supports the ``'torch'`` backend only; learnable
        parameters and other backends are left for future work.

        :param b: Decay rate, must be negative
        :type b: float
        :param omega: Intrinsic angular frequency, must be positive
        :type omega: float
        :param dt: Integration step size, must be positive
        :type dt: float
        :param v_threshold: Firing threshold of the neuron
        :type v_threshold: float
        :param v_reset: Reset voltage of the neuron. If not ``None``,
            ``self.v`` will be reset to ``v_reset`` after firing; if ``None``,
            ``v_threshold`` will be subtracted from ``self.v``. ``self.u`` is
            never reset either way
        :type v_reset: Optional[float]
        :param surrogate_function: surrogate function used to approximate the
            gradient of the Heaviside step function during backpropagation
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: whether to detach the reset operation from the
            computation graph
        :type detach_reset: bool
        :param step_mode: step mode, either ``'s'`` (single-step) or ``'m'``
            (multi-step)
        :type step_mode: str
        :param backend: backend for this neuron. Only ``'torch'`` is
            currently supported
        :type backend: str
        :param store_v_seq: when ``step_mode = 'm'`` and input shape is
            ``[T, N, *]``, whether to store the firing component at all time
            steps in ``self.v_seq``. If ``False``, only the final ``self.v``
            is kept, to reduce memory usage
        :type store_v_seq: bool
        """
        assert isinstance(b, float) and b < 0.0
        assert isinstance(omega, float) and omega > 0.0
        assert isinstance(dt, float) and dt > 0.0

        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode,
            backend,
            store_v_seq,
        )
        self.register_memory("u", 0.0)
        self.b = b
        self.omega = omega
        self.dt = dt

    def extra_repr(self):
        return super().extra_repr() + f", b={self.b}, omega={self.omega}, dt={self.dt}"

    def materialize_states(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        step_mode: str,
    ) -> tuple[object, ...]:
        states = super().materialize_states(inputs, states, step_mode)
        x = states[0]
        u = states[-1]
        if isinstance(u, float):
            u = torch.full_like(x, u, requires_grad=False)
        elif isinstance(u, torch.Tensor):
            if u.shape != x.shape:
                u = torch.full_like(x, 0.0, requires_grad=False)
            elif u.dtype != x.dtype or u.device != x.device:
                u = u.to(dtype=x.dtype, device=x.device)
        return (*states[:-1], u)

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        u = states[-1]
        spike, u, v = functional.raf_step(
            x,
            u,
            v,
            self.b,
            self.omega,
            self.dt,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, u)

    @property
    def supported_backends(self):
        if self.step_mode in ("s", "m"):
            return ("torch",)
        else:
            raise ValueError(self.step_mode)
