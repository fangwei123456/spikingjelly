import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import surrogate, base


__all__ = ["PSN", "MaskedPSN", "SlidingPSN"]


class PSN(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        T: int,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
    ):
        """
        **API Language:**
        :ref:`中文 <PSN.__init__-cn>` | :ref:`English <PSN.__init__-en>`

        ----

        .. _PSN.__init__-cn:

        * **中文**

        并行脉冲神经元（Parallel Spiking Neuron，PSN），由 `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_ 提出。神经元动力学定义如下：

        .. math::
            H &= WX, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N}\\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        其中 :math:`W` 是可学习的权重矩阵，:math:`B` 是可学习的阈值。

        .. admonition:: 注意
            :class: note

            PSN 仅支持多步模式。

        :param T: 时间步数
        :type T: int

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        ----

        .. _PSN.__init__-en:

        * **English**

        The Parallel Spiking Neuron (PSN), proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as:

        .. math::
            H &= WX, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N}\\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        where :math:`W` is the learnable weight matrix, and :math:`B` is the learnable threshold.

        .. admonition:: Note
            :class: note

            The PSN only supports the multi-step mode.

        :param T: the number of time-steps
        :type T: int

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable
        """
        super().__init__()
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.0)

    def forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        h_seq = torch.addmm(self.bias, self.weight, x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq)
        return spike_seq.view(x_seq.shape)

    def extra_repr(self):
        return super().extra_repr() + f"T={self.T}, "


class MaskedPSN(base.MemoryModule):
    def __init__(
        self,
        k: int,
        T: int,
        lambda_init: float = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        step_mode: str = "s",
    ):
        """
        **API Language:**
        :ref:`中文 <MaskedPSN.__init__-cn>` | :ref:`English <MaskedPSN.__init__-en>`

        ----

        .. _MaskedPSN.__init__-cn:

        * **中文**

        Masked Parallel Spiking Neuron，由 `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_ 提出。神经元动力学定义如下：

        .. math::
            H &= (W \\cdot {M}_{k})X, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, {M}_{k} \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N} \\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        其中 :math:`W` 是可学习权重矩阵，:math:`B` 是可学习阈值，:math:`{M}_{k}` 定义为：

        .. math::
            {M}_{k}[i][j] = \\begin{cases}
                1, ~~ j \\leq i \\leq j + k - 1 \\\\
                0, \\mathrm{otherwise}
            \\end{cases}.

        :math:`\\lambda` 用于调节逐步掩码过程：

        .. math::
            M_{k}(\\lambda) = \\lambda \\cdot M_{k} + (1 - \\lambda) \\cdot J,

        其中 :math:`J` 为全 1 矩阵。用户可以在训练中通过 ``self.lambda_ = ...`` 设置 :math:`\\lambda`。

        .. admonition:: 注意
            :class: note

            Masked PSN 支持单步模式和多步模式，但多步模式比单步模式快得多。

        :param k: Masked PSN 的阶数
        :type k: int

        :param T: 时间步数
        :type T: int

        :param lambda_init: :math:`\\lambda` 的初始值，用于调节逐步掩码过程
        :type lambda_init: float

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        ----

        .. _MaskedPSN.__init__-en:

        * **English**

        Masked Parallel Spiking Neuron (Masked PSN), proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as:

        .. math::
            H &= (W \\cdot {M}_{k})X, ~~~~~~~~~~~~~~~W \\in \\mathbb{R}^{T \\times T}, {M}_{k} \\in \\mathbb{R}^{T \\times T}, X \\in \\mathbb{R}^{T \\times N} \\\\
            S &= \\Theta(H - B), ~~~~~B \\in \\mathbb{R}^{T}, S\\in \\{0, 1\\}^{T \\times N}

        where :math:`W` is the learnable weight matrix, :math:`B` is the learnable threshold, and :math:`{M}_{k}` is defined as:

        .. math::
            {M}_{k}[i][j] = \\begin{cases}
                1, ~~ j \\leq i \\leq j + k - 1 \\\\
                0, \\mathrm{otherwise}
            \\end{cases}.

        :math:`\\lambda` is used to adjust the progressive masking process:

        .. math::
            M_{k}(\\lambda) = \\lambda \\cdot M_{k} + (1 - \\lambda) \\cdot J,

        where :math:`J` is an all-one matrix. Users can set :math:`\\lambda` during training by calling ``self.lambda_ = ...``.

        .. admonition:: Note
            :class: note

            The masked PSN supports both single-step and multi-step mode. Multi-step mode is much faster than single-step mode.

        :param k: the order of the Masked PSN
        :type k: int

        :param T: the number of time-steps
        :type T: int

        :param lambda_init: the initial value of :math:`\\lambda` to adjust the progressive masking process
        :type lambda_init: float

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.register_memory("time_step", 0)
        self.register_memory("queue", [])
        self.step_mode = step_mode
        self.k = k
        self.T = T
        self.surrogate_function = surrogate_function
        weight = torch.zeros([T, T])
        bias = torch.zeros([T, 1])
        self.register_buffer("_lambda_", torch.as_tensor(lambda_init))

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(bias)

        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        nn.init.constant_(self.bias, -1.0)

        mask1 = torch.ones([T, T])
        mask0 = torch.tril(mask1) * torch.triu(mask1, -(self.k - 1))
        self.register_buffer("mask0", mask0)
        self.register_buffer("mask1", mask1)

    @staticmethod
    @torch.jit.script
    def gen_masked_weight(
        lambda_: torch.Tensor,
        mask0: torch.Tensor,
        mask1: torch.Tensor,
        weight: torch.Tensor,
    ):
        return (lambda_ * mask0 + (1.0 - lambda_) * mask1) * weight

    def masked_weight(self):
        if self.lambda_ >= 1.0:
            return self.weight * self.mask0
        else:
            return self.gen_masked_weight(
                self.lambda_, self.mask0, self.mask1, self.weight
            )

    def single_step_forward(self, x: torch.Tensor):
        if self.lambda_ < 1.0:
            raise ValueError(
                "The masked PSN can not work in single-step mode when k < 1!"
            )

        self.queue.append(x.flatten())
        if self.queue.__len__() > self.k:
            self.queue.pop(0)

        if self.time_step + 1 > self.T:
            raise OverflowError(
                f"The MaskedPSN(T={self.T}) has run {self.time_step + 1} time-steps!"
            )

        weight = self.masked_weight()[
            self.time_step,
            self.time_step + 1 - self.queue.__len__() : self.time_step + 1,
        ]
        x_seq = torch.stack(self.queue)

        for i in range(x.dim()):
            weight = weight.unsqueeze(-1)

        h = torch.sum(weight * x_seq, 0)
        spike = self.surrogate_function(h + self.bias[self.time_step])

        self.time_step += 1
        return spike.view(x.shape)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x_seq.shape = [T, N, *]
        assert x_seq.shape[0] == self.T
        h_seq = torch.addmm(self.bias, self.masked_weight(), x_seq.flatten(1))
        spike_seq = self.surrogate_function(h_seq).view(x_seq.shape)
        return spike_seq

    @property
    def lambda_(self):
        return self._lambda_

    @lambda_.setter
    def lambda_(self, value: float):
        torch.fill_(self.lambda_, value)

    def extra_repr(self):
        return super().extra_repr() + f", lambda_={self.lambda_}, T={self.T}"


class SlidingPSN(base.MemoryModule):
    def __init__(
        self,
        k: int,
        exp_init: bool = True,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.ATan(),
        step_mode: str = "s",
        backend: str = "gemm",
    ):
        """
        **API Language:**
        :ref:`中文 <SlidingPSN.__init__-cn>` | :ref:`English <SlidingPSN.__init__-en>`

        ----

        .. _SlidingPSN.__init__-cn:

        * **中文**

        Sliding Parallel Spiking Neuron，由 `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_ 提出。神经元动力学定义如下：

        .. math::
            H[t] &= \\sum_{i=0}^{k-1} W_i \\cdot X[t - k + 1 + i], \\\\
            S[t] &= \\Theta(H[t] - B),

        其中 :math:`W = [W_0, W_1, ..., W_{k-1}] \\in \\mathbb{R}^{T}` 是可学习权重，:math:`B` 是可学习阈值。

        .. admonition:: 注意
            :class: note

            Sliding PSN 支持单步模式和多步模式，但多步模式比单步模式快得多。

        :param k: Sliding PSN 的阶数
        :type k: int

        :param exp_init: 如果为 ``True``，权重初始化为 ``(..., 1/4, 1/2, 1)``；如果为 ``False``，权重使用 Kaiming uniform 初始化
        :type exp_init: bool

        :param surrogate_function: 反向传播时用来计算脉冲函数梯度的替代函数
        :type surrogate_function: Callable

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        :param backend: 神经元层使用的后端，可以为 "gemm" 或 "conv"。此选项仅在多步模式下生效
        :type backend: str

        ----

        .. _SlidingPSN.__init__-en:

        * **English**

        Sliding Parallel Spiking Neuron (Sliding PSN), proposed in `Parallel Spiking Neurons with High Efficiency and Long-term Dependencies Learning Ability <https://arxiv.org/abs/2304.12760>`_. The neuronal dynamics are defined as:

        .. math::
            H[t] &= \\sum_{i=0}^{k-1} W_i \\cdot X[t - k + 1 + i], \\\\
            S[t] &= \\Theta(H[t] - B),

        where :math:`W = [W_0, W_1, ..., W_{k-1}] \\in \\mathbb{R}^{T}` is the learnable weight, and :math:`B` is the learnable threshold.

        .. admonition:: Note
            :class: note

            Sliding PSN supports both single-step and multi-step mode. Multi-step mode is much faster than single-step mode.

        :param k: the order of the Sliding PSN
        :type k: int

        :param exp_init: if ``True``, the weight will be initialized as ``(..., 1/4, 1/2, 1)``; if ``False``, the weight will be initialized by Kaiming uniform
        :type exp_init: bool

        :param surrogate_function: the function for calculating surrogate gradients of the heaviside step function in backward
        :type surrogate_function: Callable

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        :param backend: backend for this neuron layer, which can be "gemm" or "conv". This option only works for multi-step mode
        :type backend: str
        """

        super().__init__()
        self.register_memory("queue", [])
        self.step_mode = step_mode
        self.k = k
        self.surrogate_function = surrogate_function
        self.backend = backend

        if exp_init:
            weight = torch.ones([k])
            for i in range(k - 2, -1, -1):
                weight[i] = weight[i + 1] / 2.0
        else:
            weight = torch.ones([1, k])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            weight = weight[0]

        self.weight = nn.Parameter(weight)
        self.bias = nn.Parameter(torch.as_tensor(-1.0))

    @property
    def supported_backends(self):
        return "gemm", "conv"

    def gen_gemm_weight(self, T: int):
        weight = torch.zeros([T, T], device=self.weight.device)
        for i in range(T):
            end = i + 1
            start = max(0, i + 1 - self.k)
            length = min(end - start, self.k)
            weight[i][start:end] = self.weight[self.k - length : self.k]

        return weight

    def single_step_forward(self, x: torch.Tensor):
        self.queue.append(x.flatten())
        if self.queue.__len__() > self.k:
            self.queue.pop(0)

        weight = self.weight[self.k - self.queue.__len__() : self.k]
        x_seq = torch.stack(self.queue)

        weight = weight.unsqueeze(-1)

        h = torch.sum(weight * x_seq, 0)
        spike = self.surrogate_function(h + self.bias)

        return spike.view(x.shape)

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == "gemm":
            weight = self.gen_gemm_weight(x_seq.shape[0])
            h_seq = torch.addmm(self.bias, weight, x_seq.flatten(1)).view(x_seq.shape)
            return self.surrogate_function(h_seq)
        elif self.backend == "conv":
            # x_seq.shape = [T, N, *]
            x_seq_shape = x_seq.shape
            # [T, N, *] -> [T, N] -> [N, T] -> [N, 1, T]
            x_seq = x_seq.flatten(1).t().unsqueeze(1)

            x_seq = F.pad(x_seq, pad=(self.k - 1, 0))
            x_seq = F.conv1d(x_seq, self.weight.view(1, 1, -1), stride=1)

            x_seq = x_seq.squeeze(1).t().view(x_seq_shape)
            return self.surrogate_function(x_seq + self.bias)

        else:
            raise NotImplementedError(self.backend)

    def extra_repr(self):
        return super().extra_repr() + f", order={self.k}"
