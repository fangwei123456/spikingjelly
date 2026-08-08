from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import (
    _size_any_t,
    _size_1_t,
    _size_2_t,
    _size_3_t,
    _ratio_any_t,
)
import numpy as np

from .. import base, functional


__all__ = [
    "Conv1d",
    "Conv2d",
    "Conv3d",
    "Upsample",
    "ConvTranspose1d",
    "ConvTranspose2d",
    "ConvTranspose3d",
    "GroupNorm",
    "MaxPool1d",
    "MaxPool2d",
    "MaxPool3d",
    "MaxUnpool1d",
    "MaxUnpool2d",
    "MaxUnpool3d",
    "AvgPool1d",
    "AvgPool2d",
    "AvgPool3d",
    "AdaptiveAvgPool1d",
    "AdaptiveAvgPool2d",
    "AdaptiveAvgPool3d",
    "Linear",
    "Flatten",
    "WSConv2d",
    "WSLinear",
]


################################################################################
# nn.Module wrappers with ``step_mode``                                        #
################################################################################


class Conv1d(nn.Conv1d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Conv1d.__init__-cn>` | :ref:`English <Conv1d.__init__-en>`

        ----

        .. _Conv1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv1d`

        ----

        .. _Conv1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv1d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class Conv2d(nn.Conv2d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Conv2d.__init__-cn>` | :ref:`English <Conv2d.__init__-en>`

        ----

        .. _Conv2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv2d`

        ----

        .. _Conv2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv2d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class Conv3d(nn.Conv3d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: Union[str, _size_3_t] = 0,
        dilation: _size_3_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Conv3d.__init__-cn>` | :ref:`English <Conv3d.__init__-en>`

        ----

        .. _Conv3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv3d`

        ----

        .. _Conv3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv3d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class Upsample(nn.Upsample, base.StepModule):
    def __init__(
        self,
        size: Optional[_size_any_t] = None,
        scale_factor: Optional[_ratio_any_t] = None,
        mode: str = "nearest",
        align_corners: Optional[bool] = None,
        recompute_scale_factor: Optional[bool] = None,
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Upsample.__init__-cn>` | :ref:`English <Upsample.__init__-en>`

        ----

        .. _Upsample.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Upsample`

        ----

        .. _Upsample.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Upsample` for other parameters' API
        """
        super().__init__(
            size, scale_factor, mode, align_corners, recompute_scale_factor
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor) -> Tensor:
        if self.step_mode == "s":
            return super().forward(x)
        return functional.seq_to_ann_forward(x, (super().forward,))


class ConvTranspose1d(nn.ConvTranspose1d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: _size_1_t = 0,
        output_padding: _size_1_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_1_t = 1,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ConvTranspose1d.__init__-cn>` | :ref:`English <ConvTranspose1d.__init__-en>`

        ----

        .. _ConvTranspose1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose1d`

        ----

        .. _ConvTranspose1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose1d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class ConvTranspose2d(nn.ConvTranspose2d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: _size_2_t = 0,
        output_padding: _size_2_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: int = 1,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ConvTranspose2d.__init__-cn>` | :ref:`English <ConvTranspose2d.__init__-en>`

        ----

        .. _ConvTranspose2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose2d`

        ----

        .. _ConvTranspose2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose2d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class ConvTranspose3d(nn.ConvTranspose3d, base.StepModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_3_t,
        stride: _size_3_t = 1,
        padding: _size_3_t = 0,
        output_padding: _size_3_t = 0,
        groups: int = 1,
        bias: bool = True,
        dilation: _size_3_t = 1,
        padding_mode: str = "zeros",
        step_mode: str = "s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ConvTranspose3d.__init__-cn>` | :ref:`English <ConvTranspose3d.__init__-en>`

        ----

        .. _ConvTranspose3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose3d`

        ----

        .. _ConvTranspose3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose3d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            output_padding,
            groups,
            bias,
            dilation,
            padding_mode,
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class GroupNorm(nn.GroupNorm, base.StepModule):
    def __init__(
        self,
        num_groups: int,
        num_channels: int,
        eps: float = 1e-5,
        affine: bool = True,
        step_mode="s",
    ):
        r"""
        **API Language** - :ref:`中文 <GroupNorm.__init__-cn>` | :ref:`English <GroupNorm.__init__-en>`

        ----

        .. _GroupNorm.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.GroupNorm`

        ----

        .. _GroupNorm.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.GroupNorm` for other parameters' API
        """
        super().__init__(num_groups, num_channels, eps, affine)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        return functional.seq_to_ann_forward(x, (super().forward,))


class MaxPool1d(nn.MaxPool1d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        dilation: _size_1_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxPool1d.__init__-cn>` | :ref:`English <MaxPool1d.__init__-en>`

        ----

        .. _MaxPool1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool1d`

        ----

        .. _MaxPool1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool1d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class MaxPool2d(nn.MaxPool2d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        dilation: _size_2_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxPool2d.__init__-cn>` | :ref:`English <MaxPool2d.__init__-en>`

        ----

        .. _MaxPool2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool2d`

        ----

        .. _MaxPool2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool2d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class MaxPool3d(nn.MaxPool3d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        dilation: _size_3_t = 1,
        return_indices: bool = False,
        ceil_mode: bool = False,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxPool3d.__init__-cn>` | :ref:`English <MaxPool3d.__init__-en>`

        ----

        .. _MaxPool3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool3d`

        ----

        .. _MaxPool3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool3d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, dilation, return_indices, ceil_mode
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class MaxUnpool1d(nn.MaxUnpool1d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: Optional[_size_1_t] = None,
        padding: _size_1_t = 0,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxUnpool1d.__init__-cn>` | :ref:`English <MaxUnpool1d.__init__-en>`

        ----

        .. _MaxUnpool1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxUnpool1d`

        .. note::
            在多步模式 (``'m'``) 下，``forward`` 的输入 ``x`` 和 ``indices``
            的形状均为 ``[T, N, C, L]``；若给定 ``output_size``，会在展平
            ``[T, N]`` 后传递给底层 PyTorch 模块，可以使用空间尺寸 (例如
            ``(L,)``)，也可以使用底层模块支持的展平后完整输出尺寸。

        ----

        .. _MaxUnpool1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxUnpool1d` for other parameters' API

        .. admonition:: Note
            :class: note

            In multi-step mode (``'m'``), both ``x`` and ``indices`` of
            ``forward`` have ``shape=[T, N, C, L]``. If given, ``output_size``
            is passed to the underlying PyTorch module after ``[T, N]`` are
            flattened. It may be a spatial-only size (e.g., ``(L,)``) or a full
            output size accepted by the underlying module for the flattened input.
        """
        super().__init__(kernel_size, stride, padding)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(
        self, x: Tensor, indices: Tensor, output_size: Optional[Sequence[int]] = None
    ) -> Tensor:
        if self.step_mode == "s":
            return super().forward(x, indices, output_size)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        if indices.shape != x.shape:
            raise ValueError(
                f"expected indices with the same shape as x ({x.shape}), "
                f"but got indices with shape {indices.shape}!"
            )
        sup = super().forward
        return functional.seq_to_ann_forward(
            (x, indices), (lambda values, index: sup(values, index, output_size),)
        )


class MaxUnpool2d(nn.MaxUnpool2d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxUnpool2d.__init__-cn>` | :ref:`English <MaxUnpool2d.__init__-en>`

        ----

        .. _MaxUnpool2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxUnpool2d`

        .. note::
            在多步模式 (``'m'``) 下，``forward`` 的输入 ``x`` 和 ``indices``
            的形状均为 ``[T, N, C, H, W]``；若给定 ``output_size``，会在展平
            ``[T, N]`` 后传递给底层 PyTorch 模块，可以使用空间尺寸 (例如
            ``(H, W)``)，也可以使用底层模块支持的展平后完整输出尺寸。

        ----

        .. _MaxUnpool2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxUnpool2d` for other parameters' API

        .. admonition:: Note
            :class: note

            In multi-step mode (``'m'``), both ``x`` and ``indices`` of
            ``forward`` have ``shape=[T, N, C, H, W]``. If given, ``output_size``
            is passed to the underlying PyTorch module after ``[T, N]`` are
            flattened. It may be a spatial-only size (e.g., ``(H, W)``) or a full
            output size accepted by the underlying module for the flattened input.
        """
        super().__init__(kernel_size, stride, padding)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(
        self, x: Tensor, indices: Tensor, output_size: Optional[Sequence[int]] = None
    ) -> Tensor:
        if self.step_mode == "s":
            return super().forward(x, indices, output_size)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        if indices.shape != x.shape:
            raise ValueError(
                f"expected indices with the same shape as x ({x.shape}), "
                f"but got indices with shape {indices.shape}!"
            )
        sup = super().forward
        return functional.seq_to_ann_forward(
            (x, indices), (lambda values, index: sup(values, index, output_size),)
        )


class MaxUnpool3d(nn.MaxUnpool3d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <MaxUnpool3d.__init__-cn>` | :ref:`English <MaxUnpool3d.__init__-en>`

        ----

        .. _MaxUnpool3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxUnpool3d`

        .. note::
            在多步模式 (``'m'``) 下，``forward`` 的输入 ``x`` 和 ``indices``
            的形状均为 ``[T, N, C, D, H, W]``；若给定 ``output_size``，会在展平
            ``[T, N]`` 后传递给底层 PyTorch 模块，可以使用空间尺寸 (例如
            ``(D, H, W)``)，也可以使用底层模块支持的展平后完整输出尺寸。

        ----

        .. _MaxUnpool3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxUnpool3d` for other parameters' API

        .. admonition:: Note
            :class: note

            In multi-step mode (``'m'``), both ``x`` and ``indices`` of
            ``forward`` have ``shape=[T, N, C, D, H, W]``. If given, ``output_size``
            is passed to the underlying PyTorch module after ``[T, N]`` are
            flattened. It may be a spatial-only size (e.g., ``(D, H, W)``) or a full
            output size accepted by the underlying module for the flattened input.
        """
        super().__init__(kernel_size, stride, padding)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(
        self, x: Tensor, indices: Tensor, output_size: Optional[Sequence[int]] = None
    ) -> Tensor:
        if self.step_mode == "s":
            return super().forward(x, indices, output_size)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        if indices.shape != x.shape:
            raise ValueError(
                f"expected indices with the same shape as x ({x.shape}), "
                f"but got indices with shape {indices.shape}!"
            )
        sup = super().forward
        return functional.seq_to_ann_forward(
            (x, indices), (lambda values, index: sup(values, index, output_size),)
        )


class AvgPool1d(nn.AvgPool1d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_1_t,
        stride: _size_1_t = None,
        padding: _size_1_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <AvgPool1d.__init__-cn>` | :ref:`English <AvgPool1d.__init__-en>`

        ----

        .. _AvgPool1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool1d`

        ----

        .. _AvgPool1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool1d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class AvgPool2d(nn.AvgPool2d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_2_t,
        stride: Optional[_size_2_t] = None,
        padding: _size_2_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <AvgPool2d.__init__-cn>` | :ref:`English <AvgPool2d.__init__-en>`

        ----

        .. _AvgPool2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool2d`

        ----

        .. _AvgPool2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool2d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class AvgPool3d(nn.AvgPool3d, base.StepModule):
    def __init__(
        self,
        kernel_size: _size_3_t,
        stride: Optional[_size_3_t] = None,
        padding: _size_3_t = 0,
        ceil_mode: bool = False,
        count_include_pad: bool = True,
        divisor_override: Optional[int] = None,
        step_mode="s",
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <AvgPool3d.__init__-cn>` | :ref:`English <AvgPool3d.__init__-en>`

        ----

        .. _AvgPool3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool3d`

        ----

        .. _AvgPool3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool3d` for other parameters' API
        """
        super().__init__(
            kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override
        )
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, base.StepModule):
    def __init__(self, output_size, step_mode="s") -> None:
        r"""
        **API Language** - :ref:`中文 <AdaptiveAvgPool1d.__init__-cn>` | :ref:`English <AdaptiveAvgPool1d.__init__-en>`

        ----

        .. _AdaptiveAvgPool1d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool1d`

        ----

        .. _AdaptiveAvgPool1d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool1d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 4:
            raise ValueError(
                f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, base.StepModule):
    def __init__(self, output_size, step_mode="s") -> None:
        r"""
        **API Language** - :ref:`中文 <AdaptiveAvgPool2d.__init__-cn>` | :ref:`English <AdaptiveAvgPool2d.__init__-en>`

        ----

        .. _AdaptiveAvgPool2d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool2d`

        ----

        .. _AdaptiveAvgPool2d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool2d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, base.StepModule):
    def __init__(self, output_size, step_mode="s") -> None:
        r"""
        **API Language** - :ref:`中文 <AdaptiveAvgPool3d.__init__-cn>` | :ref:`English <AdaptiveAvgPool3d.__init__-en>`

        ----

        .. _AdaptiveAvgPool3d.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool3d`

        ----

        .. _AdaptiveAvgPool3d.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool3d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        if x.dim() != 6:
            raise ValueError(
                "expected x with shape [T, N, C, D, H, W], "
                f"but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, (super().forward,))


class Linear(nn.Linear, base.StepModule):
    def __init__(
        self, in_features: int, out_features: int, bias: bool = True, step_mode="s"
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <Linear.__init__-cn>` | :ref:`English <Linear.__init__-en>`

        ----

        .. _Linear.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Linear`

        ----

        .. _Linear.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Linear` for other parameters' API
        """
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode


class Flatten(nn.Flatten, base.StepModule):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, step_mode="s") -> None:
        r"""
        **API Language** - :ref:`中文 <Flatten.__init__-cn>` | :ref:`English <Flatten.__init__-en>`

        ----

        .. _Flatten.__init__-cn:

        * **中文**

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Flatten`

        ----

        .. _Flatten.__init__-en:

        * **English**

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Flatten` for other parameters' API
        """
        super().__init__(start_dim, end_dim)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f", step_mode={self.step_mode}"

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return super().forward(x)
        return functional.seq_to_ann_forward(x, (super().forward,))


################################################################################
# scaled weight standardization modules                                        #
################################################################################


class WSConv2d(Conv2d):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        step_mode: str = "s",
        gain: bool = True,
        eps: float = 1e-4,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <WSConv2d.__init__-cn>` | :ref:`English <WSConv2d.__init__-en>`

        ----

        .. _WSConv2d.__init__-cn:

        * **中文**

        :param gain: 是否对权重引入可学习的缩放系数
        :type gain: bool

        :param eps: 预防数值问题的小量
        :type eps: float

        其他的参数API参见 :class:`Conv2d`

        ----

        .. _WSConv2d.__init__-en:

        * **English**

        :param gain: whether introduce learnable scale factors for weights
        :type step_mode: bool

        :param eps: a small number to prevent numerical problems
        :type eps: float

        Refer to :class:`Conv2d` for other parameters' API
        """
        super().__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dilation,
            groups,
            bias,
            padding_mode,
            step_mode,
        )
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1, 1, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1, 2, 3], keepdims=True)
        var = torch.var(self.weight, axis=[1, 2, 3], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def _forward(self, x: Tensor):
        return F.conv2d(
            x,
            self.get_weight(),
            self.bias,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )

    def forward(self, x: Tensor):
        if self.step_mode == "s":
            return self._forward(x)
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!"
            )
        return functional.seq_to_ann_forward(x, self._forward)


class WSLinear(Linear):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        step_mode="s",
        gain=True,
        eps=1e-4,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <WSLinear.__init__-cn>` | :ref:`English <WSLinear.__init__-en>`

        ----

        .. _WSLinear.__init__-cn:

        * **中文**

        :param gain: 是否对权重引入可学习的缩放系数
        :type gain: bool

        :param eps: 预防数值问题的小量
        :type eps: float

        其他的参数API参见 :class:`Linear`

        ----

        .. _WSLinear.__init__-en:

        * **English**

        :param gain: whether introduce learnable scale factors for weights
        :type step_mode: bool

        :param eps: a small number to prevent numerical problems
        :type eps: float

        Refer to :class:`Linear` for other parameters' API
        """
        super().__init__(in_features, out_features, bias, step_mode)
        if gain:
            self.gain = nn.Parameter(torch.ones(self.out_channels, 1))
        else:
            self.gain = None
        self.eps = eps

    def get_weight(self):
        fan_in = np.prod(self.weight.shape[1:])
        mean = torch.mean(self.weight, axis=[1], keepdims=True)
        var = torch.var(self.weight, axis=[1], keepdims=True)
        weight = (self.weight - mean) / ((var * fan_in + self.eps) ** 0.5)
        if self.gain is not None:
            weight = weight * self.gain
        return weight

    def forward(self, x: Tensor):
        return F.linear(x, self.get_weight(), self.bias)
