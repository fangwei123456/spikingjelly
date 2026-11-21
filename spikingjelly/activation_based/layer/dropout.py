import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .. import base


__all__ = ["Dropout", "Dropout2d", "DropConnectLinear"]


class Dropout(base.MemoryModule):
    def __init__(self, p=0.5, step_mode="s"):
        r"""
        **API Language** - :ref:`中文 <Dropout.__init__-cn>` | :ref:`English <Dropout.__init__-en>`

        ----

        .. _Dropout.__init__-cn:

        * **中文**

        与 ``torch.nn.Dropout`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\
        数后，才会按照概率去重新决定，哪些位置被置0。

        .. tip::

            这种Dropout最早由 `Enabling Spike-based Backpropagation for Training Deep Neural Network Architectures
            <https://arxiv.org/abs/1903.06379>`_ 一文进行详细论述：

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.

        :param p: 每个元素被设置为0的概率
        :type p: float

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _Dropout.__init__-en:

        * **English**

        This layer is almost same with ``torch.nn.Dropout``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        .. admonition:: Tip
            :class: tip

            This kind of Dropout is firstly described in `Enabling Spike-based Backpropagation for Training Deep Neural
            Network Architectures <https://arxiv.org/abs/1903.06379>`_:

            There is a subtle difference in the way dropout is applied in SNNs compared to ANNs. In ANNs, each epoch of
            training has several iterations of mini-batches. In each iteration, randomly selected units (with dropout ratio of :math:`p`)
            are disconnected from the network while weighting by its posterior probability (:math:`1-p`). However, in SNNs, each
            iteration has more than one forward propagation depending on the time length of the spike train. We back-propagate
            the output error and modify the network parameters only at the last time step. For dropout to be effective in
            our training method, it has to be ensured that the set of connected units within an iteration of mini-batch
            data is not changed, such that the neural network is constituted by the same random subset of units during
            each forward propagation within a single iteration. On the other hand, if the units are randomly connected at
            each time-step, the effect of dropout will be averaged out over the entire forward propagation time within an
            iteration. Then, the dropout effect would fade-out once the output error is propagated backward and the parameters
            are updated at the last time step. Therefore, we need to keep the set of randomly connected units for the entire
            time window within an iteration.

        :param p: probability of an element to be zeroed
        :type p: float

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
"""
        super().__init__()
        self.step_mode = step_mode
        assert 0 <= p < 1
        self.register_memory("mask", None)
        self.p = p

    def extra_repr(self):
        return f"p={self.p}"

    def create_mask(self, x: Tensor) -> Tensor:
        return F.dropout(torch.ones_like(x), self.p, training=True)

    def single_step_functional_forward(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        mask = states[0]
        if self.training:
            if mask is None:
                mask = self.create_mask(x)
            x = x * mask
        return (x,), (mask,)

    def multi_step_functional_forward(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        mask = states[0]
        if self.training:
            if mask is None:
                mask = self.create_mask(x_seq[0])
            x_seq = x_seq * mask
        return (x_seq,), (mask,)


class Dropout2d(Dropout):
    def __init__(self, p=0.2, step_mode="s"):
        r"""
        **API Language** - :ref:`中文 <Dropout2d.__init__-cn>` | :ref:`English <Dropout2d.__init__-en>`

        ----

        .. _Dropout2d.__init__-cn:

        * **中文**

        与 ``torch.nn.Dropout2d`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\
        数后，才会按照概率去重新决定，哪些位置被置0。

        关于SNN中Dropout的更多信息，参见 :ref:`layer.Dropout <Dropout.__init__-cn>`。

        :param p: 每个元素被设置为0的概率
        :type p: float

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _Dropout2d.__init__-en:

        * **English**

        This layer is almost same with ``torch.nn.Dropout2d``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        For more information about Dropout in SNN, refer to :ref:`layer.Dropout <Dropout.__init__-en>`.

        :param p: probability of an element to be zeroed
        :type p: float

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
"""
        super().__init__(p, step_mode)

    def create_mask(self, x: Tensor) -> Tensor:
        return F.dropout2d(torch.ones_like(x), self.p, training=True)


class DropConnectLinear(base.MemoryModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        p: float = 0.5,
        samples_num: int = 1024,
        invariant: bool = False,
        activation: Optional[nn.Module] = nn.ReLU(),
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <DropConnectLinear.__init__-cn>` | :ref:`English <DropConnectLinear.__init__-en>`

        ----

        .. _DropConnectLinear.__init__-cn:

        * **中文**

        DropConnect，由 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
        一文提出。DropConnect与Dropout非常类似，区别在于DropConnect是以概率 ``p`` 断开连接，而Dropout是将输入以概率置0。

        .. Note::

            在使用DropConnect进行推理时，输出的tensor中的每个元素，都是先从高斯分布中采样，通过激活层激活，再在采样数量上进行平均得到的。
            详细的流程可以在 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
            一文中的 `Algorithm 2` 找到。激活层 ``activation`` 在中间的步骤起作用，因此我们将其作为模块的成员。

        :param in_features: 每个输入样本的特征数
        :type in_features: int

        :param out_features: 每个输出样本的特征数
        :type out_features: int

        :param bias: 若为 ``False``，则本层不会有可学习的偏置项。默认为 ``True``
        :type bias: bool

        :param p: 每个连接被断开的概率。默认为0.5
        :type p: float

        :param samples_num: 在推理时，从高斯分布中采样的数据数量。默认为1024
        :type samples_num: int

        :param invariant: 若为 ``True``，线性层会在第一次执行前向传播时被按概率断开，断开后的线性层会保持不变，直到 ``reset()`` 函数
            被调用，线性层恢复为完全连接的状态。完全连接的线性层，调用 ``reset()`` 函数后的第一次前向传播时被重新按概率断开。 若为
            ``False``，在每一次前向传播时线性层都会被重新完全连接再按概率断开。 阅读 :ref:`layer.Dropout <Dropout.__init__-cn>` 以
            获得更多关于此参数的信息。默认为 ``False``
            缓存状态为布尔 ``weight_mask`` 和可选的 ``bias_mask``，每次前向使用当前参数。
            旧版本保存的浮点丢弃参数状态需要重新初始化。
        :type invariant: bool

        :param activation: 在线性层后的激活层
        :type activation: Optional[torch.nn.Module]

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _DropConnectLinear.__init__-en:

        * **English**

        DropConnect, which is proposed by `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_,
        is similar with Dropout but drop connections of a linear layer rather than the elements of the input tensor with
        probability ``p``.

        .. admonition:: Note
            :class: note

            When inference with DropConnect, every elements of the output tensor are sampled from a Gaussian distribution,
            activated by the activation layer and averaged over the sample number ``samples_num``.
            See `Algorithm 2` in `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
            for more details. Note that activation is an intermediate process. This is the reason why we include
            ``activation`` as a member variable of this module.

        :param in_features: size of each input sample
        :type in_features: int

        :param out_features: size of each output sample
        :type out_features: int

        :param bias: If set to ``False``, the layer will not learn an additive bias. Default: ``True``
        :type bias: bool

        :param p: probability of an connection to be zeroed. Default: 0.5
        :type p: float

        :param samples_num: number of samples drawn from the Gaussian during inference. Default: 1024
        :type samples_num: int

        :param invariant: If set to ``True``, the connections will be dropped at the first time of forward and the dropped
            connections will remain unchanged until ``reset()`` is called and the connections recovery to fully-connected
            status. Then the connections will be re-dropped at the first time of forward after ``reset()``. If set to
            ``False``, the connections will be re-dropped at every forward. See :ref:`layer.Dropout <Dropout.__init__-en>`
            for more information to understand this parameter. Default: ``False``
            Cached states are boolean ``weight_mask`` and optional ``bias_mask``;
            current parameters are used on every forward. Reinitialize explicit
            functional states saved as dropped parameter values by older versions.
        :type invariant: bool

        :param activation: the activation layer after the linear layer
        :type activation: Optional[nn.Module]

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.step_mode = step_mode
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(Tensor(out_features))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

        self.p = p  # 置0的概率
        self.register_memory("weight_mask", None)
        if self.bias is not None:
            self.register_memory("bias_mask", None)

        self.samples_num = samples_num
        self.invariant = invariant
        self.activation = activation

    def reset_parameters(self) -> None:
        r"""
        **API Language** - :ref:`中文 <DropConnectLinear.reset_parameters-cn>` | :ref:`English <DropConnectLinear.reset_parameters-en>`

        ----

        .. _DropConnectLinear.reset_parameters-cn:

        * **中文**


        初始化模型中的可学习参数。

        ----

        .. _DropConnectLinear.reset_parameters-en:

        * **English**


        Initialize the learnable parameters of this module.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reset(self):
        r"""
        **API Language** - :ref:`中文 <DropConnectLinear.reset-cn>` | :ref:`English <DropConnectLinear.reset-en>`

        ----

        .. _DropConnectLinear.reset-cn:

        * **中文**

        将线性层重置为完全连接的状态，若 ``self.activation`` 也是一个有状态的层，则将其也重置。


        ----

        .. _DropConnectLinear.reset-en:

        * **English**

        Reset the linear layer to fully-connected status. If ``self.activation`` is also stateful, this function will
        also reset it.
        """
        super().reset()
        if hasattr(self.activation, "reset"):
            self.activation.reset()

    def _sample_masks(self, batch_size: int) -> tuple[Tensor, Optional[Tensor]]:
        mask_w = (
            torch.rand_like(
                self.weight.unsqueeze(0).expand(batch_size, *self.weight.shape)
            )
            >= self.p
        )

        if self.bias is not None:
            mask_b = (
                torch.rand_like(
                    self.bias.unsqueeze(0).expand(batch_size, *self.bias.shape)
                )
                >= self.p
            )
        else:
            mask_b = None
        return mask_w, mask_b

    def drop(self, batch_size: int) -> tuple[Tensor, Optional[Tensor]]:
        r"""
        **API Language** - :ref:`中文 <DropConnectLinear.drop-cn>` | :ref:`English <DropConnectLinear.drop-en>`

        ----

        .. _DropConnectLinear.drop-cn:

        * **中文**

        用新采样的连接掩码乘以当前参数，不修改缓存状态。

        :param batch_size: 独立连接掩码的数量。
        :type batch_size: int
        :return: 形状为 ``[batch_size, out_features, in_features]`` 的权重和
            形状为 ``[batch_size, out_features]`` 的偏置；无偏置时第二项为 ``None``。
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]

        ----

        .. _DropConnectLinear.drop-en:

        * **English**

        Multiply current parameters by fresh connection masks without changing cached states.

        :param batch_size: Number of independent connection masks.
        :type batch_size: int
        :return: Weights of shape ``[batch_size, out_features, in_features]`` and
            biases of shape ``[batch_size, out_features]``, or ``None`` without bias.
        :rtype: tuple[torch.Tensor, Optional[torch.Tensor]]
        """
        weight_mask, bias_mask = self._sample_masks(batch_size)
        return (
            self.weight * weight_mask,
            None if self.bias is None else self.bias * bias_mask,
        )

    def single_step_functional_forward(
        self,
        inputs: tuple[Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <DropConnectLinear.single_step_functional_forward-cn>` | :ref:`English <DropConnectLinear.single_step_functional_forward-en>`

        ----

        .. _DropConnectLinear.single_step_functional_forward-cn:

        * **中文**

        训练时用当前参数和连接掩码计算输出；``invariant=True`` 时保留掩码，
        否则重新采样。评估时使用高斯采样近似并保留传入状态。

        :param inputs: 单元素元组，输入形状为 ``[B, in_features]``。
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(weight_mask,)`` 或 ``(weight_mask, bias_mask)``；掩码为
            ``None`` 或布尔张量，形状分别为 ``[B, out_features, in_features]`` 和 ``[B, out_features]``。
        :type states: tuple[object, ...]
        :param kwargs: 预留的额外关键字参数，当前不使用。
        :type kwargs: object
        :return: ``((output,), new_states)``，输出形状为 ``[B, out_features]``。
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        :raises ValueError: 训练时复用的状态不是布尔掩码，或评估时 ``p`` 超出 ``[0, 1]``。

        ----

        .. _DropConnectLinear.single_step_functional_forward-en:

        * **English**

        Training uses current parameters and connection masks, retaining masks
        when ``invariant=True`` and resampling otherwise. Evaluation uses the
        Gaussian sampling approximation and preserves the supplied states.

        :param inputs: Single-element tuple with input shape ``[B, in_features]``.
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(weight_mask,)`` or ``(weight_mask, bias_mask)``; masks are
            ``None`` or boolean tensors of shapes ``[B, out_features, in_features]`` and ``[B, out_features]``.
        :type states: tuple[object, ...]
        :param kwargs: Reserved extra keyword arguments, currently unused.
        :type kwargs: object
        :return: ``((output,), new_states)`` with output shape ``[B, out_features]``.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        :raises ValueError: Reused training states are not boolean masks, or evaluation ``p`` is outside ``[0, 1]``.
        """
        input = inputs[0]
        weight_mask = states[0]
        bias_mask = states[1] if self.bias is not None else None
        if self.training:
            if not self.invariant or weight_mask is None:
                weight_mask, bias_mask = self._sample_masks(input.shape[0])
            if weight_mask.dtype != torch.bool or (
                bias_mask is not None and bias_mask.dtype != torch.bool
            ):
                raise ValueError(
                    "DropConnectLinear states must be boolean connection masks; "
                    "reset the module or initialize fresh functional states."
                )
            dropped_w = self.weight * weight_mask
            ret = torch.bmm(dropped_w, input.unsqueeze(-1)).squeeze(-1)
            if self.bias is not None:
                ret = ret + self.bias * bias_mask
            output = ret if self.activation is None else self.activation(ret)
            updated_states = (
                (weight_mask, bias_mask) if self.bias is not None else (weight_mask,)
            )
            return (output,), updated_states

        if not 0 <= self.p <= 1:
            raise ValueError("p must be between 0 and 1")

        mu = (1 - self.p) * F.linear(input, self.weight, self.bias)
        sigma2 = (
            self.p
            * (1 - self.p)
            * F.linear(
                input.square(),
                self.weight.square(),
                None if self.bias is None else self.bias.square(),
            )
        )
        with torch.no_grad():
            samples = torch.normal(
                mu.expand(self.samples_num, *mu.shape),
                sigma2.sqrt().expand(self.samples_num, *sigma2.shape),
            )
        if self.activation is not None:
            samples = self.activation(samples)
        output = samples.mean(dim=0)
        return (output,), states

    def extra_repr(self) -> str:
        return f"in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, p={self.p}, invariant={self.invariant}"
