import torch
import torch.nn as nn


__all__ = [
    "fused_conv2d_weight_of_convbn2d",
    "fused_conv2d_bias_of_convbn2d",
    "scale_fused_conv2d_weight_of_convbn2d",
    "scale_fused_conv2d_bias_of_convbn2d",
    "fuse_convbn2d",
]


def fused_conv2d_weight_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    **API Language:**
    :ref:`中文 <fused_conv2d_weight_of_convbn2d-cn>` | :ref:`English <fused_conv2d_weight_of_convbn2d-en>`

    ----

    .. _fused_conv2d_weight_of_convbn2d-cn:

    * **中文**

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数返回合并后的卷积的权重。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d

    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d

    :return: 合并后的卷积权重
    :rtype: torch.Tensor

    ----

    .. _fused_conv2d_weight_of_convbn2d-en:

    * **English**

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the weight of this fused module.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d

    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    :return: the weight of this fused module
    :rtype: torch.Tensor
    """
    assert conv2d.bias is None
    return (conv2d.weight.transpose(0, 3) * bn2d.weight / (
            bn2d.running_var + bn2d.eps).sqrt()).transpose(0, 3)


def fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    **API Language:**
    :ref:`中文 <fused_conv2d_bias_of_convbn2d-cn>` | :ref:`English <fused_conv2d_bias_of_convbn2d-en>`

    ----

    .. _fused_conv2d_bias_of_convbn2d-cn:

    * **中文**

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数返回合并后的卷积的偏置项。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d

    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d

    :return: 合并后的卷积偏置
    :rtype: torch.Tensor

    ----

    .. _fused_conv2d_bias_of_convbn2d-en:

    * **English**

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the bias of this fused module.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d

    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    :return: the bias of this fused module
    :rtype: torch.Tensor
    """
    assert conv2d.bias is None
    return bn2d.bias - bn2d.running_mean * bn2d.weight / (bn2d.running_var + bn2d.eps).sqrt()


@torch.no_grad()
def scale_fused_conv2d_weight_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    **API Language:**
    :ref:`中文 <scale_fused_conv2d_weight_of_convbn2d-cn>` | :ref:`English <scale_fused_conv2d_weight_of_convbn2d-en>`

    ----

    .. _scale_fused_conv2d_weight_of_convbn2d-cn:

    * **中文**

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对 ``{Conv2d-BatchNorm2d}`` 模块整体的等效权重进行 ``weight = k * weight + b`` 的线性变换。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d

    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d

    :return: None

    ----

    .. _scale_fused_conv2d_weight_of_convbn2d-en:

    * **English**

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function applies a linear transform ``weight = k * weight + b`` on the equivalent weight of the whole ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d

    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    :return: None
    """
    assert conv2d.bias is None
    if k is not None:
        conv2d.weight.data *= k
    if b is not None:
        conv2d.weight.data += b


@torch.no_grad()
def scale_fused_conv2d_bias_of_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d, k=None, b=None):
    """
    **API Language:**
    :ref:`中文 <scale_fused_conv2d_bias_of_convbn2d-cn>` | :ref:`English <scale_fused_conv2d_bias_of_convbn2d-en>`

    ----

    .. _scale_fused_conv2d_bias_of_convbn2d-cn:

    * **中文**

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对 ``{Conv2d-BatchNorm2d}`` 模块整体的等效偏置项进行 ``bias = k * bias + b`` 的线性变换。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d

    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d

    :return: None

    ----

    .. _scale_fused_conv2d_bias_of_convbn2d-en:

    * **English**

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function applies a linear transform ``bias = k * bias + b`` on the equivalent bias of the whole ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d

    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    :return: None
    """
    assert conv2d.bias is None
    if k is not None:
        bn2d.bias.data *= k
        bn2d.running_mean *= k
    if b is not None:
        bn2d.bias.data += b


@torch.no_grad()
def fuse_convbn2d(conv2d: nn.Conv2d, bn2d: nn.BatchNorm2d):
    """
    **API Language:**
    :ref:`中文 <detach_net-cn>` | :ref:`English <fuse_convbn2d-en>`

    ----

    .. _fuse_convbn2d-cn:

    * **中文**

    ``{Conv2d-BatchNorm2d}`` 模块可以合并为一个单个的 ``{Conv2d}``，其中``BatchNorm2d`` 的参数会被吸收进 ``Conv2d``。
    本函数对返回这个等效的合并后的 ``{Conv2d}``。

    .. note::

        这里按照 ``conv2d.bias`` 为 ``None`` 进行处理。原因参见 `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ 。

    :param conv2d: 一个2D卷积层
    :type conv2d: torch.nn.Conv2d

    :param bn2d: 一个2D的BN层
    :type bn2d: torch.nn.BatchNorm2d

    :return: 合并后的卷积层
    :rtype: torch.nn.Conv2d

    ----

    .. _fuse_convbn2d-en:

    * **English**

    A ``{Conv2d-BatchNorm2d}`` can be fused to a ``{Conv2d}`` module with ``BatchNorm2d`` 's parameters being absorbed into ``Conv2d``.
    This function returns the fused ``{Conv2d}`` merged by ``{Conv2d-BatchNorm2d}``.

    .. admonition:: Note
        :class: note

        We assert ``conv2d.bias`` is ``None``. See `Disable bias for convolutions directly followed by a batch norm <https://pytorch.org/tutorials/recipes/recipes/tuning_guide.html#disable-bias-for-convolutions-directly-followed-by-a-batch-norm>`_ for more details.

    :param conv2d: a Conv2d layer
    :type conv2d: torch.nn.Conv2d

    :param bn2d: a BatchNorm2d layer
    :type bn2d: torch.nn.BatchNorm2d

    :return: the fused ``Conv2d`` layer
    :rtype: torch.nn.Conv2d
    """
    fused_conv = nn.Conv2d(in_channels=conv2d.in_channels, out_channels=conv2d.out_channels,
                           kernel_size=conv2d.kernel_size,
                           stride=conv2d.stride, padding=conv2d.padding, dilation=conv2d.dilation,
                           groups=conv2d.groups, bias=True,
                           padding_mode=conv2d.padding_mode)
    fused_conv.weight.data = fused_conv2d_weight_of_convbn2d(conv2d, bn2d)
    fused_conv.bias.data = fused_conv2d_bias_of_convbn2d(conv2d, bn2d)
    return fused_conv

