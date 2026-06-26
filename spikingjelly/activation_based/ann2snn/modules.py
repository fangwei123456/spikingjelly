import torch.nn as nn
import torch


__all__ = ["VoltageHook", "VoltageScaler"]


class VoltageHook(nn.Module):
    def __init__(self, scale=1.0, momentum=0.1, mode="Max"):
        r"""
        **API Language** - :ref:`中文 <VoltageHook.__init__-cn>` | :ref:`English <VoltageHook.__init__-en>`

        ----

        .. _VoltageHook.__init__-cn:

        * **中文**

        :class:`VoltageHook` 的构造函数。

        :param scale: 缩放初始值
        :type scale: float
        :param momentum: 动量值
        :type momentum: float
        :param mode: 模式。``"Max"`` 表示记录ANN激活最大值；``"99.9%"`` 表示记录99.9%分位点；
            0-1 的 float 表示记录激活最大值的对应倍数
        :type mode: str, float

        ----

        .. _VoltageHook.__init__-en:

        * **English**

        Constructor of :class:`VoltageHook`.

        :param scale: initial scaling value
        :type scale: float
        :param momentum: momentum value
        :type momentum: float
        :param mode: Mode. ``"Max"`` means recording the maximum value of ANN activation;
            ``"99.9%"`` means recording the 99.9% percentile; a float of 0-1 means
            recording the corresponding multiple of the maximum value
        :type mode: str, float
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))
        self.mode = mode
        self.num_batches_tracked = 0
        self.momentum = momentum

    def forward(self, x):
        r"""
        **API Language** - :ref:`中文 <VoltageHook.forward-cn>` | :ref:`English <VoltageHook.forward-en>`

        ----

        .. _VoltageHook.forward-cn:

        * **中文**

        前向传播函数。不对输入张量做任何处理，只是抓取ReLU的激活值用于确定ANN激活范围。

        :param x: 输入张量
        :type x: torch.Tensor
        :return: 原输入张量
        :rtype: torch.Tensor

        ----

        .. _VoltageHook.forward-en:

        * **English**

        Forward function. It doesn't process input tensors, but hooks the activation
        values of ReLU to determine ANN activation ranges.

        :param x: input tensor
        :type x: torch.Tensor
        :return: original input tensor
        :rtype: torch.Tensor
        """
        err_msg = "You have used a non-defined VoltageScale Method."
        if isinstance(self.mode, str):
            if not self.mode:
                raise NotImplementedError(err_msg)
            if self.mode[-1] == "%":
                try:
                    quantile = float(self.mode[:-1]) / 100.0
                    if not (0.0 <= quantile <= 1.0):
                        raise NotImplementedError(err_msg)
                    s_t = torch.quantile(x.detach(), quantile)
                except (ValueError, RuntimeError) as exc:
                    raise NotImplementedError(err_msg) from exc
            elif self.mode.lower() in ["max"]:
                s_t = x.max().detach()
            else:
                raise NotImplementedError(err_msg)
        elif (
            isinstance(self.mode, (int, float))
            and not isinstance(self.mode, bool)
            and self.mode <= 1
            and self.mode > 0
        ):
            s_t = x.max().detach() * self.mode
        else:
            raise NotImplementedError(err_msg)

        if self.num_batches_tracked == 0:
            self.scale = s_t
        else:
            self.scale = (1 - self.momentum) * self.scale + self.momentum * s_t
        self.num_batches_tracked += x.shape[0]
        return x


class VoltageScaler(nn.Module):
    def __init__(self, scale=1.0):
        r"""
        **API Language** - :ref:`中文 <VoltageScaler.__init__-cn>` | :ref:`English <VoltageScaler.__init__-en>`

        ----

        .. _VoltageScaler.__init__-cn:

        * **中文**

        :class:`VoltageScaler` 的构造函数。用于SNN推理中缩放电流。

        :param scale: 缩放值
        :type scale: float

        ----

        .. _VoltageScaler.__init__-en:

        * **English**

        Constructor of :class:`VoltageScaler`. Used for scaling current in SNN inference.

        :param scale: scaling value
        :type scale: float
        """
        super().__init__()
        self.register_buffer("scale", torch.tensor(scale))

    def forward(self, x):
        r"""
        **API Language** - :ref:`中文 <VoltageScaler.forward-cn>` | :ref:`English <VoltageScaler.forward-en>`

        ----

        .. _VoltageScaler.forward-cn:

        * **中文**

        前向传播函数。对输入电流进行缩放。

        :param x: 输入张量，亦即输入电流
        :type x: torch.Tensor
        :return: 缩放后的电流
        :rtype: torch.Tensor

        ----

        .. _VoltageScaler.forward-en:

        * **English**

        Forward function. Scales the input current.

        :param x: input tensor, or input current
        :type x: torch.Tensor
        :return: current after scaling
        :rtype: torch.Tensor
        """
        return x * self.scale

    def extra_repr(self):
        return "%f" % self.scale.item()
