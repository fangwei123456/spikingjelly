import math
from typing import Optional

import torch
import torch.nn as nn

from .. import functional, surrogate
from .base_node import BaseNode


__all__ = ["ParametricLIFNode"]


class ParametricLIFNode(BaseNode):
    def __init__(
        self,
        init_tau: float = 2.0,
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
        **API Language** - :ref:`中文 <ParametricLIFNode.__init__-cn>` | :ref:`English <ParametricLIFNode.__init__-en>`

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

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]
        spike, v = functional.plif_step(
            x,
            v,
            self.w,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        v = states[0]

        if self.backend == "cupy":
            spike_seq, v, _ = functional.plif_multi_step_cupy(
                x_seq,
                v,
                self.w,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                False,
            )
        elif self.backend == "triton":
            spike_seq, v, _ = functional.plif_multi_step_triton(
                x_seq,
                v,
                self.w,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                False,
            )
        elif self.backend == "torch":
            return super().multi_step_functional_forward(inputs, states, **kwargs)
        else:
            raise ValueError(self.backend)

        return (spike_seq,), (v,)

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if not self.store_v_seq or self.backend == "torch":
            return super().multi_step_forward(x_seq, *args, **kwargs)

        states = self.materialize_states(
            (x_seq, *args), tuple(self._memories.values()), "m"
        )
        function = {
            "cupy": functional.plif_multi_step_cupy,
            "triton": functional.plif_multi_step_triton,
        }[self.backend]
        spike_seq, v, v_seq = function(
            x_seq,
            states[0],
            self.w,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
            True,
        )
        self.v = v
        self.v_seq = v_seq
        return spike_seq
