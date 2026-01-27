"""This module contains attention layers for deep SNNs.

* Attention for convolutional SNNs
    * :class:`TemporalWiseAttention`
    * :class:`MultiDimensionalAttention`

* Attention for Spiking Transformers
    * :class:`SpikingSelfAttention`
    * :class:`QKAttention`, :class:`TokenQKAttention`, :class:`ChannelQKAttention`

For more information about Spiking Transformers, see :doc:`../tutorials/en/spikformer` .
"""

import torch
import torch.nn as nn
from einops import rearrange

from .. import base, neuron
from .container import SeqToANNContainer


__all__ = [
    "TemporalWiseAttention",
    "MultiDimensionalAttention",
    "SpikingSelfAttention",
    "QKAttention",
    "TokenQKAttention",
    "ChannelQKAttention",
]


class TemporalWiseAttention(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, reduction: int = 16, dimension: int = 4):
        """
        **API Language:**
        :ref:`中文 <TemporalWiseAttention.__init__-cn>` | :ref:`English <TemporalWiseAttention.__init__-en>`

        ----

        .. _TemporalWiseAttention.__init__-cn:

        * **中文**

        `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_ 中提出
        的TemporalWiseAttention层。TemporalWiseAttention层必须放在二维卷积层之后脉冲神经元之前，例如：

        ``Conv2d -> TemporalWiseAttention -> LIF``

        输入的尺寸是 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` ，经过TemporalWiseAttention层，输出为 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` 。

        ``reduction`` 是压缩比，相当于论文中的 :math:`r`。

        :param T: 输入数据的时间步长
        :type T: int

        :param reduction: 压缩比
        :type reduction: int

        :param dimension: 输入数据的维度。当输入数据为[T, N, C, H, W]时， dimension = 4；输入数据维度为[T, N, L]时，dimension = 2。
        :type dimension: int

        ----

        .. _TemporalWiseAttention.__init__-en:

        * **English**

        The TemporalWiseAttention layer is proposed in `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_.

        It should be placed after the convolution layer and before the spiking neurons, e.g.,

        ``Conv2d -> TemporalWiseAttention -> LIF``

        The dimension of the input is ``[T, N, C, H, W]`` or  ``[T, N, L]`` , after the TemporalWiseAttention layer, the output dimension is ``[T, N, C, H, W]`` or  ``[T, N, L]`` .

        ``reduction`` is the reduction ratio，which is :math:`r` in the paper.

        :param T: timewindows of input
        :type T: int

        :param reduction: reduction ratio
        :type reduction: int

        :param dimension: Dimensions of input. If the input dimension is [T, N, C, H, W], dimension = 4; when the input dimension is [T, N, L], dimension = 2.
        :type dimension: int
        """
        super().__init__()
        self.step_mode = "m"
        assert dimension == 4 or dimension == 2, "dimension must be 4 or 2"

        self.dimension = dimension

        # Sequence
        if self.dimension == 2:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        elif self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

        assert T >= reduction, "reduction cannot be greater than T"

        # Excitation
        self.sharedMLP = nn.Sequential(
            nn.Linear(T, T // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(T // reduction, T, bias=False),
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f"expected 3D or 5D input with shape [T, N, M] or [T, N, C, H, W], but got input with shape {x_seq.shape}"
        )
        x_seq = x_seq.transpose(0, 1)
        avgout = self.sharedMLP(
            self.avg_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]])
        )
        maxout = self.sharedMLP(
            self.max_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]])
        )
        scores = self.sigmoid(avgout + maxout)
        if self.dimension == 2:
            y_seq = x_seq * scores[:, :, None]
        elif self.dimension == 4:
            y_seq = x_seq * scores[:, :, None, None, None]
        y_seq = y_seq.transpose(0, 1)
        return y_seq


class MultiDimensionalAttention(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        T: int,
        C: int,
        reduction_t: int = 16,
        reduction_c: int = 16,
        kernel_size=3,
    ):
        """
        **API Language:**
        :ref:`中文 <MultiStepMultiDimensionalAttention.__init__-cn>` | :ref:`English <MultiStepMultiDimensionalAttention.__init__-en>`

        ----

        .. _MultiStepMultiDimensionalAttention.__init__-cn:

        * **中文**

        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_ 中提出
        的MA-SNN模型以及MultiStepMultiDimensionalAttention层。

        您可以从以下链接中找到MA-SNN的示例项目:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        输入的尺寸是 ``[T, N, C, H, W]`` ，经过MultiStepMultiDimensionalAttention层，输出为 ``[T, N, C, H, W]`` 。

        :param T: 输入数据的时间步长

        :param C: 输入数据的通道数
        :type C: int

        :param reduction_t: 时间压缩比
        :type reduction_t: int

        :param reduction_c: 通道压缩比
        :type reduction_c: int

        :param kernel_size: 空间注意力机制的卷积核大小
        :type kernel_size: int

        ----

        .. _MultiStepMultiDimensionalAttention.__init__-en:

        * **English**

        The MA-SNN model and MultiStepMultiDimensionalAttention layer are proposed in
        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_.

        You can find the example projects of MA-SNN in the following links:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        The dimension of the input is ``[T, N, C, H, W]`` , after the MultiStepMultiDimensionalAttention layer, the output dimension is ``[T, N, C, H, W]`` .

        :param T: timewindows of input
        :type T: int

        :param C: channel number of input
        :type C: int

        :param reduction_t: temporal reduction ratio
        :type reduction_t: int

        :param reduction_c: channel reduction ratio
        :type reduction_c: int

        :param kernel_size: convolution kernel size of SpatialAttention
        :type kernel_size: int
        """
        super().__init__()

        assert T >= reduction_t, "reduction_t cannot be greater than T"
        assert C >= reduction_c, "reduction_c cannot be greater than C"

        # Attention
        class TimeAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(TimeAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                return self.sigmoid(avgout + maxout)

        class ChannelAttention(nn.Module):
            def __init__(self, in_planes, ratio=16):
                super(ChannelAttention, self).__init__()
                self.avg_pool = nn.AdaptiveAvgPool3d(1)
                self.max_pool = nn.AdaptiveMaxPool3d(1)
                self.sharedMLP = nn.Sequential(
                    nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                    nn.ReLU(),
                    nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False),
                )
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b c f h w")
                avgout = self.sharedMLP(self.avg_pool(x))
                maxout = self.sharedMLP(self.max_pool(x))
                out = self.sigmoid(avgout + maxout)
                out = rearrange(out, "b c f h w -> b f c h w")
                return out

        class SpatialAttention(nn.Module):
            def __init__(self, kernel_size=3):
                super(SpatialAttention, self).__init__()
                assert kernel_size in (3, 7), "kernel size must be 3 or 7"
                padding = 3 if kernel_size == 7 else 1
                self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
                self.sigmoid = nn.Sigmoid()

            def forward(self, x):
                x = rearrange(x, "b f c h w -> b (f c) h w")
                avgout = torch.mean(x, dim=1, keepdim=True)
                maxout, _ = torch.max(x, dim=1, keepdim=True)
                x = torch.cat([avgout, maxout], dim=1)
                x = self.conv(x)
                x = x.unsqueeze(1)
                return self.sigmoid(x)

        self.ta = TimeAttention(T, reduction_t)
        self.ca = ChannelAttention(C, reduction_c)
        self.sa = SpatialAttention(kernel_size)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x: torch.Tensor):
        assert x.dim() == 5, ValueError(
            f"expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}"
        )
        x = x.transpose(0, 1)
        out = self.ta(x) * x
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        out = out.transpose(0, 1)
        return out


class SpikingSelfAttention(nn.Module, base.MultiStepModule):
    def __init__(self, dim, num_heads=8, backend: str = "torch"):
        """
        **API Language:**
        :ref:`中文 <SpikingSelfAttention.__init__-cn>` | :ref:`English <SpikingSelfAttention.__init__-en>`

        ----

        .. _SpikingSelfAttention.__init__-cn:

        * **中文**

        `Spikformer: When Spiking Neural Network Meets Transformer <https://openreview.net/forum?id=frE4fUwz_h>`_
        中提出的 Spiking Self Attention 层。本模块在 `Spikformer源代码 <https://github.com/ZK-Zhou/spikformer/blob/main/imagenet/model.py>`_
        的基础上做了改进，显著提高了运行效率。关于 Spikformer 和本模块实现方式的更多信息，
        参见教程 :doc:`../tutorials/cn/spikformer` 。

        本模块的输入是尺寸为 ``[T, N, C, L]`` 的脉冲张量，其中 ``T`` 是时间步数，
        ``N`` 是 batch size ，``C`` 是 channel 数量，``L`` 是 token 数量 （对于视觉任务， ``L=H*W`` ）。
        输出是尺寸为 ``[T, N, C, L]`` 的脉冲张量。

        :param dim: channel 数量
        :type dim: int

        :param num_heads: 多头自注意力的头数量，默认为 ``8``
        :type num_heads: int

        :param backend: 本模块内部神经元使用的后端，默认为 ``torch``
        :type backend: str

        ----

        .. _SpikingSelfAttention.__init__-en:

        * **English**

        Spiking Self-Attention layer proposed in
        `Spikformer: When Spiking Neural Network Meets Transformer <https://openreview.net/forum?id=frE4fUwz_h>`_.
        This module is implemented based on
        `Spikformer source code <https://github.com/ZK-Zhou/spikformer/blob/main/imagenet/model.py>`_
        with several improvements that significantly enhance efficiency.
        For more details about Spikformer and the implementation of this module,
        please refer to the tutorial :doc:`../tutorials/en/spikformer`.

        The input to this module is a spike tensor of shape ``[T, N, C, L]``,
        where ``T`` denotes the number of time steps, ``N`` is the batch size,
        ``C`` is the number of channels, and ``L`` is the number of tokens
        (for vision tasks, ``L = H * W``). The output is a spiking tensor with
        the same shape ``[T, N, C, L]``.

        :param dim: number of channels
        :type dim: int

        :param num_heads: number of heads in multi-head self-attention. Default: ``8``
        :type num_heads: int

        :param backend: backend used by the internal neurons of this module. Default: ``torch``
        :type backend: str
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divided by num_heads {num_heads}.")
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = 0.125
        self._backend = backend

        self.qkv_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim * 3),
        )
        self.qkv_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, step_mode="m", backend=backend
        )

        self.attn_lif = neuron.LIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, step_mode="m", backend=backend
        )

        self.proj_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, step_mode="m", backend=backend
        )

    @property
    def backend(self):
        """
        一旦设置，本模块中所有神经元的后端都会被同样地设置。

        Once set, the backend of all the neurons in this module will also be changed.
        """
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = value
        self.qkv_lif.backend = value
        self.attn_lif.backend = value
        self.proj_lif.backend = value

    @staticmethod
    def _ssa_kernel_torch(qkv, scale):  # TODO: add triton implementation
        # qkv.shape = [T, N, 3, NUM_HEADS, Cph, L]
        # qt, kt, vt.shape = [T, N, NUM_HEADS, Cph, L]
        qt, kt, vt = qkv.flatten(2, 3).chunk(3, dim=2)
        x_seq = vt @ kt.transpose(-2, -1)
        x_seq = (x_seq @ qt) * scale
        return x_seq  # [T, N, NUM_HEADS, Cph, L]

    def forward(self, x_seq: torch.Tensor):
        """
        :param x_seq: ``shape=[T, N, C, L]``
        :type x_seq: torch.Tensor

        :return: ``shape=[T, N, C, L]``
        :rtype: torch.Tensor
        """
        if x_seq.ndim != 4:
            raise ValueError(
                f"expected 4D input with shape [T, N, C, L], "
                f"but got input with shape {x_seq.shape}"
            )
        T, N, C, L = x_seq.shape

        qkv = self.qkv_conv_bn(x_seq)
        qkv = self.qkv_lif(qkv)  # [T, N, 3*C, L]
        qkv = qkv.reshape(T, N, 3, self.num_heads, C // self.num_heads, L)

        x_seq = self._ssa_kernel_torch(qkv, self.scale)
        x_seq = self.attn_lif(x_seq).reshape(T, N, C, L)

        x_seq = self.proj_conv_bn(x_seq)
        x_seq = self.proj_lif(x_seq)  # [T, N, C, L]
        return x_seq

    def extra_repr(self):
        return f"dim={self.dim}, num_heads={self.num_heads}, backend={self.backend}"


class QKAttention(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        qka_type: str = "token",
        backend: str = "torch",
    ):
        """
        **API Language:**
        :ref:`中文 <QKAttention.__init__-cn>` | :ref:`English <QKAttention.__init__-en>`

        ----

        .. _QKAttention.__init__-cn:

        * **中文**

        `QKFormer: Hierarchical Spiking Transformer using Q-K Attention <https://openreview.net/forum?id=AVd7DpiooC>`_
        中提出的 Q-K Attention 层。本模块在 `QKFormer源代码 <https://github.com/zhouchenlin2096/QKFormer/blob/master/imagenet/qkformer.py>`_
        的基础上做了改进，显著提高了运行效率；改进思路与 Spikformer 类似，见教程 :doc:`../tutorials/cn/spikformer` 。

        本模块的输入是尺寸为 ``[T, N, C, L]`` 的脉冲张量，其中 ``T`` 是时间步数，
        ``N`` 是 batch size ，``C`` 是 channel 数量，``L`` 是 token 数量 （对于视觉任务， ``L=H*W`` ）。
        输出是尺寸为 ``[T, N, C, L]`` 的脉冲张量。

        :param dim: channel 数量
        :type dim: int

        :param num_heads: 多头自注意力的头数量，默认为 ``8``
        :type num_heads: int

        :param qka_type: QKAttention的类型，可选值为 ``token`` 和 ``channel``。默认为 ``token``，生成逐token的掩码
        :type qka_type: str

        :param backend: 本模块内部神经元使用的后端，默认为 ``torch``
        :type backend: str

        ----

        .. _QKAttention.__init__-en:

        * **English**

        Q-K Attention layer proposed in
        `QKFormer: Hierarchical Spiking Transformer using Q-K Attention <https://openreview.net/forum?id=AVd7DpiooC>`_.
        This module is implemented based on the
        `QKFormer source code <https://github.com/zhouchenlin2096/QKFormer/blob/master/imagenet/qkformer.py>`_,
        with several improvements that significantly enhance efficiency.
        The improvement strategy is similar to that used in Spikformer; see the
        tutorial :doc:`../tutorials/en/spikformer` for details.

        The input to this module is a spike tensor of shape ``[T, N, C, L]``,
        where ``T`` denotes the number of time steps, ``N`` is the batch size,
        ``C`` is the number of channels, and ``L`` is the number of tokens (for
        vision tasks, ``L = H * W``). The output is a spiking tensor with the
        same shape ``[T, N, C, L]``.

        :param dim: number of channels.
        :type dim: int

        :param num_heads: number of heads in multi-head self-attention. Default: ``8``.
        :type num_heads: int

        :param qka_type: type of QKAttention. Available options are ``token`` and ``channel``.
                          The default is ``token``, which generates a token-wise mask.
        :type qka_type: str

        :param backend: backend used by the internal neurons of this module. Default: ``torch``.
        :type backend: str
        """
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim {dim} should be divided by num_heads {num_heads}.")
        if qka_type not in ["token", "channel"]:
            raise ValueError(
                f"qka_type should be either 'token' or 'channel', but got {qka_type}."
            )
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self._qka_type = qka_type
        self._backend = backend

        self.qk_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim * 2, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim * 2),
        )
        self.qk_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, step_mode="m", backend=backend
        )

        self.sum_dim = 3 if qka_type == "token" else 4
        self.attn_lif = neuron.LIFNode(
            tau=2.0, v_threshold=0.5, detach_reset=True, step_mode="m", backend=backend
        )

        self.proj_conv_bn = SeqToANNContainer(
            nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
            nn.BatchNorm1d(dim),
        )
        self.proj_lif = neuron.LIFNode(
            tau=2.0, detach_reset=True, step_mode="m", backend=backend
        )

    @property
    def backend(self):
        """
        一旦设置，本模块中所有神经元的后端都会被同样地设置。

        Once set, the backend of all the neurons in this module will also be changed.
        """
        return self._backend

    @backend.setter
    def backend(self, value: str):
        self._backend = value
        self.qk_lif.backend = value
        self.attn_lif.backend = value
        self.proj_lif.backend = value

    @property
    def qka_type(self):
        """
        只读。构造时设置，随后不可修改。

        Read-only. Set when constructing, and cannot be modified afterwards.
        """
        return self._qka_type

    def _qka_forward_torch(self, qk):
        # qk.shape = [T, N, 2, NUM_HEADS, Cph, L]
        # q, k = [T, N, NUM_HEADS, Cph, L]
        q, k = qk.flatten(2, 3).chunk(2, dim=2)
        q = torch.sum(q, dim=self.sum_dim, keepdim=True)
        # [T, N, NUM_HEADS, 1, L] if qka_type == "token"
        # [T, N, NUM_HEADS, Cph, 1] if qka_type == "channel"
        attn = self.attn_lif(q)
        x_seq = attn * k
        return x_seq  # [T, N, NUM_HEADS, Cph, L]

    def forward(self, x_seq):
        """
        :param x_seq: ``shape=[T, N, C, L]``
        :type x_seq: torch.Tensor

        :return: ``shape=[T, N, C, L]``
        :rtype: torch.Tensor
        """
        if x_seq.ndim != 4:
            raise ValueError(
                f"expected 4D input with shape [T, N, C, L], "
                f"but got input with shape {x_seq.shape}"
            )
        T, N, C, L = x_seq.shape

        qk = self.qk_conv_bn(x_seq)
        qk = self.qk_lif(qk)  # [T, N, 2*C, L]
        qk = qk.reshape(T, N, 2, self.num_heads, C // self.num_heads, L)

        x_seq = self._qka_forward_torch(qk)
        x_seq = x_seq.flatten(2, 3)  # [T, N, C, L]

        x_seq = self.proj_conv_bn(x_seq)
        x_seq = self.proj_lif(x_seq)
        return x_seq

    def extra_repr(self):
        return (
            f"dim={self.dim}, num_heads={self.num_heads}, "
            f"qka_type={self.qka_type}, backend={self.backend}"
        )


class TokenQKAttention(QKAttention):
    def __init__(self, dim: int, num_heads: int = 8, backend: str = "torch"):
        """
        ``QKAttention(..., qka_type="token")`` . See :class:`QKAttention` .
        """
        super().__init__(dim, num_heads, qka_type="token", backend=backend)


class ChannelQKAttention(QKAttention):
    def __init__(self, dim: int, num_heads: int = 8, backend: str = "torch"):
        """
        ``QKAttention(..., qka_type="channel")`` . See :class:`QKAttention` .
        """
        super().__init__(dim, num_heads, qka_type="channel", backend=backend)
