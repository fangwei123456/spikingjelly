import torch
from torch import nn


class TimeAttention(nn.Module):
    def __init__(self, in_planes, ratio=16, fc=False):
        super(TimeAttention, self).__init__()
        if fc ==True:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)

            self.sharedMLP = nn.Sequential(
                nn.Conv1d(in_planes, in_planes // ratio, 1, bias=False),
                nn.ReLU(),
                nn.Conv1d(in_planes // ratio, in_planes, 1, bias=False),
            )
        else:
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
        #x = rearrange(x, "f b c h w -> b c f h w")
        x = x.permute(1, 2, 0, 3, 4)
        avgout = self.sharedMLP(self.avg_pool(x))
        maxout = self.sharedMLP(self.max_pool(x))
        out = self.sigmoid(avgout + maxout)
        #out = rearrange(out, "b c f h w -> f b c h w")
        out = out.permute(2, 0, 1, 3, 4)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), "kernel size must be 3 or 7"
        padding = 3 if kernel_size == 7 else 1

        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #x = rearrange(x, "f b c h w -> b (f c) h w")
        x = x.permute(1, 0, 2, 3, 4)
        x = x.flatten(1, 2)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avgout, maxout], dim=1)
        x = self.conv(x)
        x = x.unsqueeze(1)
        #x = rearrange(x, "b f c h w -> f b c h w")
        x = x.permute(1, 0, 2, 3, 4)
        return self.sigmoid(x)


class TCSA(nn.Module):

    def __init__(self, T, channels, c_ratio=16, t_ratio=16):
        """
                * :ref:`API in English <TCSA.__init__-en>`

                .. _TCSA.__init__-cn:

                :param T: 输入数据的时间步长

                :param channel: 输入数据的通道大小

                :param c_ratio: Channel Attention 的压缩比

                :param t_ratio: Temporal Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的TCSA层。这一模块融合了Temporal Attention、Channel Attention 与 Spatial Attention。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <TCSA.__init__-cn>`
                .. _TCSA.__init__-en:

                :param T: timewindows of the input

                :param channel: channel numbers of the input

                :param c_ratio: reduction ratio in channel dimension

                :param t_ratio: reduction ratio in temporal dimension

                The TCSA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The TCSA layer mix three attention mechanisms from Convolutional Neural Network: Temporal Attention, Channel Attention and Spatial Attention.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.


                """
        super(TCSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(T, t_ratio)
        self.sa = SpatialAttention()


    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TCA(nn.Module):
    def __init__(self, T, channels, c_ratio=16, t_ratio=16):
        """
                * :ref:`API in English <TCA.__init__-en>`

                .. _TCA.__init__-cn:

                :param T: 输入数据的时间步长

                :param channel: 输入数据的通道大小

                :param c_ratio: Channel Attention 的压缩比

                :param t_ratio: Temporal Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的TCA层。这一模块融合了Temporal Attention 与 Channel Attention。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <TCA.__init__-cn>`

                .. _TCA.__init__-en:

                :param T: timewindows of the input

                :param channel: channel numbers of the input

                :param c_ratio: reduction ratio in channel dimension

                :param t_ratio: reduction ratio in temporal dimension

                The TCA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The TCA layer mix two attention mechanisms from Convolutional Neural Network: Temporal Attention and Channel Attention.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.


                """
        super(TCA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.ta = TimeAttention(T, t_ratio)

    def forward(self, x):
        out = self.ta(x) * x
        out = self.ca(out) * out  # 广播机制
        # out = self.sa(x) * out  # 广播机制

        out = self.relu(out)
        return out


class CSA(nn.Module):
    def __init__(self, channels, c_ratio=16):
        """
                * :ref:`API in English <CSA.__init__-en>`

                .. _CSA.__init__-cn:

                :param channel: 输入数据的通道大小

                :param c_ratio: Channel Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的CSA层。这一模块融合了Spatial Attention 与 Channel Attention。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <CSA.__init__-cn>`

                .. _CSA.__init__-en:

                :param channel: channel numbers of the input

                :param c_ratio: reduction ratio in channel dimension

                The CSA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The CSA layer mix two attention mechanisms from Convolutional Neural Network: Spatial Attention and Channel Attention.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.


                """
        super(CSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.ca = ChannelAttention(channels, c_ratio)
        self.sa = SpatialAttention()


    def forward(self, x):
        # out = self.ta(x) * x
        out = self.ca(x) * x  # 广播机制
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TSA(nn.Module):
    def __init__(self, T, t_ratio=16):
        """
                * :ref:`API in English <TSA.__init__-en>`

                .. _TSA.__init__-cn:

                :param T: 输入数据的时间步长

                :param t_ratio: Temporal Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的TSA层。这一模块融合了Temporal Attention 与 Spatial Attention。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <TSA.__init__-cn>`

                .. _TSA.__init__-en:

                :param T: timewindows of the input

                :param t_ratio: reduction ratio in temporal dimension

                The TSA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The TSA layer mix two attention mechanisms from Convolutional Neural Network: Spatial Attention and Temporal Attention.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.
                """

        super(TSA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.ta = TimeAttention(T, t_ratio)
        self.sa = SpatialAttention()


    def forward(self, x):
        out = self.ta(x) * x
        out = self.sa(out) * out  # 广播机制

        out = self.relu(out)
        return out


class TA(nn.Module):
    def __init__(self, T, t_ratio=16, fc=False):
        """
                * :ref:`API in English <TA.__init__-en>`

                .. _TA.__init__-cn:

                :param T: 输入数据的时间步长

                :param t_ratio: Temporal Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的TA层。这一模块主要提供在Temporal维度上的Attention实现。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <TA.__init__-cn>`

                .. _TA.__init__-en:

                :param T: timewindows of the input

                :param t_ratio: reduction ratio in temporal dimension

                The TA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The TA module mainly employ the attention mechanism in temporal dimension.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.
                """
        super(TA, self).__init__()

        self.relu = nn.ReLU(inplace=True)
        self.ta = TimeAttention(T, t_ratio, fc)


    def forward(self, x):
        out = self.ta(x) * x
        out = self.relu(out)
        return out


class CA(nn.Module):
    def __init__(self, channels, c_ratio=16):
        """
                * :ref:`API in English <CA.__init__-en>`

                .. _CA.__init__-cn:

                :param channel: 输入数据的通道大小

                :param c_ratio: Channel Attention 的压缩比

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的CA层。这一模块主要提供在Channel维度上的Attention实现。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <CA.__init__-cn>`

                .. _CA.__init__-en:

                :param channel: channel numbers of the input

                :param c_ratio: reduction ratio in channel dimension

                The CA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The CA module mainly employ the attention mechanism in channel dimension.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.
                """
        super(CA, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.ca = ChannelAttention(channels, c_ratio)

    def forward(self, x):
        out = self.ca(x) * x  # 广播机制
        out = self.relu(out)
        return out


class SA(nn.Module):
    def __init__(self):
        """
                * :ref:`API in English <CA.__init__-en>`

                .. _CA.__init__-cn:

                `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_ 中提出
                的SA层。这一模块主要提供在Spatial维度上的Attention实现。
                输入维度为``[T, N, C, H, W]``，输出维度为``[T, N, C, H, W]``。

                .. Note::

                    这一模块仅能用于多步传播中。

                * :ref:`中文API <SA.__init__-cn>`

                .. _SA.__init__-en:

                The SA layer is proposed by `Attention Spiking Neural Networks <https://arxiv.org/abs/2209.13929>`_. The SA module mainly employ the attention mechanism in spatial dimension.
                The dimension of the input could be described by ``[T, N, C, H, W]``, where the output could be described by ``[T, N, C, H, W]``.

                .. Note::

                    This module only could be utilized in multi-step mode.
                """
        super(SA, self).__init__()
        self.relu = nn.ReLU(inplace=True)
        self.sa = SpatialAttention()


    def forward(self, x):
        out = self.sa(x) * x  # 广播机制

        out = self.relu(out)
        return out
