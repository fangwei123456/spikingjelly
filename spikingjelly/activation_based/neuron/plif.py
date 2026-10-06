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

        :param store_v_seq: 在使用 ``step_mode = 'm'`` 时，给与 ``shape = [T, N, *]`` 的输入后，是否保存中间过程的 ``shape = [T, N, *]``
            的各个时间步的电压值 ``self.v_seq`` 。设置为 ``False`` 时计算完成后只保留最后一个时刻的电压，即 ``shape = [N, *]`` 的 ``self.v`` 。
            通常设置成 ``False`` ，可以节省内存。在使用 ``step_mode = 's'`` 时，每个时间步结束后的电压会被追加到 ``self.v_seq`` ，
            直到调用 ``reset()`` ；每一步都会复制整个序列，因此该选项主要用于监控和调试
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

        :param store_v_seq: when using ``step_mode = 'm'`` and given input with ``shape = [T, N, *]``, this option controls
            whether storing the voltage at each time-step to ``self.v_seq`` with ``shape = [T, N, *]``. If set to ``False``,
            only the voltage at last time-step will be stored to ``self.v`` with ``shape = [N, *]``, which can reduce the
            memory consumption. When using ``step_mode = 's'``, the voltage after each time-step is appended to
            ``self.v_seq`` until ``reset()`` is called; every step copies the whole sequence, so this option is meant for
            monitoring and debugging
        :type store_v_seq: bool
        """
        assert isinstance(init_tau, float) and init_tau > 1.0
        super().__init__(
            v_threshold,
            v_reset,
            surrogate_function,
            detach_reset,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
        )
        self.decay_input = decay_input
        init_w = -math.log(init_tau - 1.0)
        self.w = nn.Parameter(torch.as_tensor(init_w))  # as reciprocal_tau

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
        spikes, v, _ = functional.plif_multi_step(
            inputs[0],
            states[0],
            self.w,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
            False,
            neuron_storage=(
                None if self._neuron_precision is None else self._neuron_precision[0]
            ),
            neuron_fwd=(
                "fp32" if self._neuron_precision is None else self._neuron_precision[1]
            ),
            neuron_bwd=(
                "fp32" if self._neuron_precision is None else self._neuron_precision[2]
            ),
        )
        return (spikes,), (v,)

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if (
            type(self).single_step_functional_forward
            is not ParametricLIFNode.single_step_functional_forward
            or type(self).multi_step_functional_forward
            is not ParametricLIFNode.multi_step_functional_forward
        ):
            return super().multi_step_forward(x_seq, *args, **kwargs)
        if self._neuron_precision is not None:
            inputs = (x_seq, *args)
            states = self.materialize_states(
                inputs, tuple(self._memories.values()), "m"
            )
            spikes, v, v_seq = functional.plif_multi_step(
                x_seq,
                states[0],
                self.w,
                self.decay_input,
                self.v_threshold,
                self.v_reset,
                self.surrogate_function,
                self.detach_reset,
                self.store_v_seq,
                neuron_storage=self._neuron_precision[0],
                neuron_fwd=self._neuron_precision[1],
                neuron_bwd=self._neuron_precision[2],
            )
            self._memories["v"] = v
            self.v_seq = v_seq
            return spikes
        if self.store_v_seq:
            return super().multi_step_forward(x_seq, *args, **kwargs)
        inputs = (x_seq, *args)
        states = self.materialize_states(inputs, tuple(self._memories.values()), "m")
        outputs, states = self.multi_step_functional_forward(inputs, states, **kwargs)
        for name, value in zip(self._memories, states, strict=True):
            self._memories[name] = value
        self.v_seq = None
        return outputs[0] if len(outputs) == 1 else outputs
