import numbers

import torch

from .. import surrogate
from .base_node import BaseNode

__all__ = ["ILIFNode"]


class ILIFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        max_spike_count: int = 4,
        decay: float = 0.25,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
    ):
        r"""
        **API Language** - :ref:`中文 <ILIFNode.__init__-cn>` | :ref:`English <ILIFNode.__init__-en>`

        ----

        .. _ILIFNode.__init__-cn:

        * **中文**

        Integer-valued training and strictly binary spike inference I-LIF 神经元。
        训练时单步动力学为：

        .. math::

            U[t] = \beta V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(U[t] / V_{th}, 0, D))

        .. math::

            V[t] = U[t] - C[t] V_{th}

        其中 ``D`` 为 ``max_spike_count``，``beta`` 为 ``decay``。``decay`` 位于
        充电过程，形式与 :class:`LIFNode` 在 ``decay_input=False`` 且
        ``v_reset=0`` 时的 charge 方程一致。训练模式 ``self.training == True``
        返回整数发放计数 ``C[t]``，并使用论文中的矩形窗口直通梯度：仅当
        ``U[t] / V_th`` 位于 ``[0, D]`` 内时传递梯度。

        推理模式 ``self.training == False`` 返回严格二值 ``0/1`` 脉冲：

        .. math::

            S[t] = \Theta(U[t] - V_{th})

        .. math::

            V[t] = U[t] - S[t] V_{th}

        因此推理前向不会返回 ``C[t]`` 这样的整数计数。若需要和整数训练公平
        对齐，用户必须在训练/评测 loop 层级组织时间步。

        .. warning::

            严格二值脉冲推理契约：本神经元没有 ``T`` 参数，也不会自动把
            ``[T, ...]`` 输入展开为 ``[T * D, ...]``。训练时由用户运行 ``T`` 个
            逻辑步；公平推理时由用户运行 ``T * D`` 个虚拟步，并向本神经元传入
            已经按虚拟步构造好的电流/脉冲输入。本神经元在每个推理步只输出
            ``0/1``。下游线性层也必须消费这些 ``0/1`` 脉冲或等价的
            spike-aware AC 数据流；如果将训练输出的整数 ``C[t]`` 直接送入普通
            卷积/线性层，就是整数推理，不是 spike-driven inference。

        :param v_threshold: 神经元阈值电压，必须为有限正数
        :type v_threshold: float
        :param max_spike_count: 训练时单个逻辑步最多释放的整数计数 ``D``
        :type max_spike_count: int
        :param decay: 充电过程中历史膜电位的衰减 ``beta``，必须位于 ``[0, 1]``。
            当 ``decay=1`` 时退化为无泄漏的整数 IF 形式
        :type decay: float
        :param detach_reset: 是否在反向传播时分离 reset 中的发放值
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步，``"m"`` 为多步
        :type step_mode: str
        :param backend: 后端名称。当前仅支持 ``"torch"``
        :type backend: str
        :param store_v_seq: 多步模式下是否保存每个输入步后的膜电位
        :type store_v_seq: bool

        ----

        .. _ILIFNode.__init__-en:

        * **English**

        I-LIF neuron for integer-valued training and strictly binary spike
        inference. The training dynamics are:

        .. math::

            U[t] = \beta V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(U[t] / V_{th}, 0, D))

        .. math::

            V[t] = U[t] - C[t] V_{th}

        ``D`` is ``max_spike_count`` and ``beta`` is ``decay``. ``decay`` belongs
        to the charge process, matching the :class:`LIFNode` charge form when
        ``decay_input=False`` and ``v_reset=0``. In training mode, the forward
        output is the integer spike count ``C[t]``. Its backward path uses the
        rectangular straight-through estimator from the paper: gradients pass only
        where ``U[t] / V_th`` lies in ``[0, D]``.

        In eval mode this neuron returns strictly binary ``0/1`` spikes:

        .. math::

            S[t] = \Theta(U[t] - V_{th})

        .. math::

            V[t] = U[t] - S[t] V_{th}

        Thus inference forward never returns integer counts such as ``C[t]``. To
        evaluate fairly against integer-valued training, organize timesteps at
        the user training/evaluation loop level.

        .. admonition:: Strict binary inference contract
            :class: warning

            This neuron has no ``T`` parameter and does not automatically expand
            ``[T, ...]`` inputs to ``[T * D, ...]``. Run ``T`` logical steps in
            the training loop. For fair inference, run ``T * D`` virtual steps in
            the evaluation loop and feed this neuron inputs that have already
            been constructed for those virtual steps. The neuron emits only
            ``0/1`` at each inference step. Downstream linear layers must consume
            these ``0/1`` spikes, or an equivalent spike-aware AC dataflow.
            Feeding the training-time integer count ``C[t]`` directly into dense
            convolutions or linear layers is integer inference, not spike-driven
            inference.

        :param v_threshold: Threshold voltage, which must be finite and positive
        :type v_threshold: float
        :param max_spike_count: Maximum integer count ``D`` emitted in one
            logical training step
        :type max_spike_count: int
        :param decay: Historical membrane decay ``beta`` in charge, in ``[0, 1]``.
            ``decay=1`` degenerates to a non-leaky integer IF form
        :type decay: float
        :param detach_reset: Whether to detach the emitted value used by reset in
            backward
        :type detach_reset: bool
        :param step_mode: Step mode, ``"s"`` or ``"m"``
        :type step_mode: str
        :param backend: Backend name. Only ``"torch"`` is currently supported
        :type backend: str
        :param store_v_seq: Whether to store membrane voltage after each input
            step in multi-step mode
        :type store_v_seq: bool
        """
        if not isinstance(v_threshold, numbers.Real):
            raise TypeError("v_threshold must be a real number.")
        v_threshold = float(v_threshold)
        if not torch.isfinite(torch.tensor(v_threshold)) or v_threshold <= 0.0:
            raise ValueError("v_threshold must be finite positive.")
        if not isinstance(max_spike_count, numbers.Integral) or isinstance(
            max_spike_count, bool
        ):
            raise TypeError("max_spike_count must be an integer.")
        max_spike_count = int(max_spike_count)
        if max_spike_count < 1:
            raise ValueError("max_spike_count must be >= 1.")
        if not isinstance(decay, numbers.Real):
            raise TypeError("decay must be a real number.")
        decay = float(decay)
        if not torch.isfinite(torch.tensor(decay)) or decay < 0.0 or decay > 1.0:
            raise ValueError("decay must be finite and in [0, 1].")

        super().__init__(
            v_threshold=v_threshold,
            v_reset=None,
            surrogate_function=surrogate.MultiLevelSpikeCount(max_spike_count),
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )
        self.decay = decay

    @property
    def supported_backends(self):
        return ("torch",)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.decay * self.v + x

    def neuronal_fire(self):
        if self.training:
            return self.surrogate_function(self.v / self.v_threshold)
        return (self.v >= self.v_threshold).to(self.v)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike = spike.detach()
        self.v = self.v - spike * self.v_threshold

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        spike_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(self.single_step_forward(x_seq[t]))
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(spike_seq)

    def extra_repr(self):
        return (
            f"v_threshold={self.v_threshold}, "
            f"max_spike_count={self.surrogate_function.max_spike_count}, "
            f"decay={self.decay}, detach_reset={self.detach_reset}, "
            f"step_mode={self.step_mode}, backend={self.backend}"
        )
