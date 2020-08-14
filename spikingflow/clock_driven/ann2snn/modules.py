import torch
import torch.nn as nn
import torch.nn.functional as F


class MaxPool2d(nn.Module):
    def __init__(self, kernel_size, stride=None, padding=0, dilation=1,
                 return_indices=False, ceil_mode=False, momentum=None):
        '''
        * :ref:`API in English <MaxPool2d.__init__-en>`

        .. _MaxPool2d.__init__-cn:

        :param kernel_size:  窗口取最大的大小
        :param stride: 窗口的步长. 默认值为 :attr:`kernel_size`
        :param padding: 隐式两侧填充零的大小
        :param dilation: 控制窗口中元素的步幅的参数
        :param return_indices: 当 ``True`` ，将返回最大序号并输出
        :param ceil_mode: 当 ``True`` ，将使用 `ceil` 而不是 `floor` 来计算输出形状
        :param momentum: 当在[0，1]中，将在门控函数中使用在线动量统计;
                        当为 ``None`` 时，将在门控函数中使用累计脉冲数
        :return: ``None``

        基于文献[1]中2.2.6章节设计MaxPool2d模块。为了兼容Pytorch的MaxPool2d模块，众多参数设定和Pytorch相同。详情请见 ``torch.nn.MaxPool2d`` 。
        基本想法是对输入脉冲进行统计，统计量可以控制门控函数确定以哪一路输入信号作为输出。
        根据 `momentum` 参数类型不同可以有不同的统计功能。 `momentum` 参数支持None值和[0,1]区间的浮点数数值作为输出。
        假定在t时刻，脉冲输入张量为 :math:`s_t` ，脉冲统计量为 :math:`p_t` 
        当 `momentum` 参数为 ``None`` 时，统计量为累计脉冲数
        
        .. math::
            p_t = p_{t-1} + s_t

        当 `momentum` 参数为[0,1]区间的浮点数时，统计量为在线的动量累积
        
        .. math::
            p_t = momentum * p_{t-1} + (1-momentum) * s_t

        * :ref:`中文API <MaxPool2d.__init__-cn>`

        .. _MaxPool2d.__init__-en:

        :param kernel_size:  the size of the window to take a max over
        :param stride: the stride of the window. Default value is :attr:`kernel_size`
        :param padding: implicit zero padding to be added on both sides
        :param dilation: a parameter that controls the stride of elements in the window
        :param return_indices: if ``True``, will return the max indices along with the outputs.
                        Useful for :class:`torch.nn.MaxUnpool2d` later
        :param ceil_mode: when ``True``, will use `ceil` instead of `floor` to compute the output shape
        :param momentum: when in [0,1], will use online momentum statistics in gate functions;
                        when ``None``, will use accumulated spike in gate functions
        :return: ``None``

        Design the MaxPool2d module based on section 2.2.6 in [1]. In order to be compatible with Pytorch's MaxPool2d module, many parameter settings are the same as Pytorch. See ``torch.nn.MaxPool2d`` for details.
        The basic idea is to accumulate the input spikes, which can control the gating function to determine which input spike is used as output.
        Depending on the type of `momentum` parameter, different statistical functions can be used.
        `momentum` supports the floating-point value in [0,1] or value ``None``
        Assume at time t, the spike input is :math:`s_t` and the spike statistic is :math:`p_t`.
        When `momentum` is ``None``, the statistic is sum of spikes over time.
        
        .. math::
            p_t = p_{t-1} + s_t

        When `momentum` is a floating point in [0,1], the statistic is online momentum of spikes.
        
        .. math::
            p_t = momentum * p_{t-1} + (1-momentum) * s_t

        [1] Rueckauer B, Lungu I-A, Hu Y, Pfeiffer M and Liu S-C (2017) Conversion of Continuous-Valued Deep Networks to
        Efficient Event-Driven Networks for Image Classification. Front. Neurosci. 11:682.
        '''

        super(MaxPool2d, self).__init__()
        self.kernel_size = kernel_size
        self.stride = stride or kernel_size
        self.padding = padding
        self.dilation = dilation
        self.return_indices = return_indices
        self.ceil_mode = ceil_mode

        assert(momentum is None or momentum <=1)
        self.momentum = momentum

        self.v = 0

    def forward(self, dv: torch.Tensor):
        if self.momentum is not None:
            self.v = self.v * self.momentum + (1 - self.momentum) * dv
        else:
            self.v += dv
        (dv_out, ind) = F.max_pool2d(self.v, self.kernel_size, self.stride,
                                     self.padding, self.dilation, self.ceil_mode, True)
        unpool_dv_out = F.max_unpool2d(dv_out, ind, self.kernel_size, self.stride, self.padding, self.v.size())
        max_gate = (unpool_dv_out != 0.0).float()
        gated_spk = dv * max_gate
        spk = F.max_pool2d(gated_spk, self.kernel_size, self.stride,
                                  self.padding)
        return spk

    def reset(self):
        '''
        :return: None

        重置神经元为初始状态
        '''
        self.v = 0