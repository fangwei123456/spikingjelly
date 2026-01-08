"""
This module contains :class:`base.StepModule <spikingjelly.activation_based.base.StepModule>`
wrappers for commonly used :class:`torch.nn.Module`. See :doc:`../tutorials/en/basic_concept`
for more details.
"""
from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
import numpy as np

from .. import base, functional


__all__ = [
    "Conv1d", "Conv2d", "Conv3d",
    "Upsample", "ConvTranspose1d", "ConvTranspose2d", "ConvTranspose3d", 
    "GroupNorm", "MaxPool1d", "MaxPool2d", "MaxPool3d",
    "AvgPool1d", "AvgPool2d", "AvgPool3d",
    "AdaptiveAvgPool1d", "AdaptiveAvgPool2d", "AdaptiveAvgPool3d",
    "Linear", "Flatten", "WSConv2d", "WSLinear"
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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <Conv1d-en>`

        .. _Conv1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv1d`

        * :ref:`中文 API <Conv1d-cn>`

        .. _Conv1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv1d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, N, C, L], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <Conv2d-en>`

        .. _Conv2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv2d`

        * :ref:`中文 API <Conv2d-cn>`

        .. _Conv2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv2d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <Conv3d-en>`

        .. _Conv3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Conv3d`

        * :ref:`中文 API <Conv3d-cn>`

        .. _Conv3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Conv3d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class Upsample(nn.Upsample, base.StepModule):
    def __init__(self,
                 size: Optional[_size_any_t] = None,
                 scale_factor: Optional[_ratio_any_t] = None,
                 mode: str = 'nearest',
                 align_corners: Optional[bool] = None,
                 recompute_scale_factor: Optional[bool] = None,
                 step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <Upsample-en>`

        .. _Upsample-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Upsample`

        * :ref:`中文 API <Upsample-cn>`

        .. _Upsample-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Upsample` for other parameters' API
        """
        super().__init__(size, scale_factor, mode, align_corners, recompute_scale_factor)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor) -> Tensor:
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <ConvTranspose1d-en>`

        .. _ConvTranspose1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose1d`

        * :ref:`中文 API <ConvTranspose1d-cn>`

        .. _ConvTranspose1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose1d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, N, C, L], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <ConvTranspose2d-en>`

        .. _ConvTranspose2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose2d`

        * :ref:`中文 API <ConvTranspose2d-cn>`

        .. _ConvTranspose2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose2d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


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
            padding_mode: str = 'zeros',
            step_mode: str = 's'
    ) -> None:
        """
        * :ref:`API in English <ConvTranspose3d-en>`

        .. _ConvTranspose3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.ConvTranspose3d`

        * :ref:`中文 API <ConvTranspose3d-cn>`

        .. _ConvTranspose3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.ConvTranspose3d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, output_padding, groups, bias, dilation, padding_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class GroupNorm(nn.GroupNorm, base.StepModule):
    def __init__(
            self,
            num_groups: int, num_channels: int, eps: float = 1e-5, affine: bool = True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <GroupNorm-en>`

        .. _GroupNorm-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.GroupNorm`

        * :ref:`中文 API <GroupNorm-cn>`

        .. _GroupNorm-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.GroupNorm` for other parameters' API
        """
        super().__init__(num_groups, num_channels, eps, affine)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, super().forward)


class MaxPool1d(nn.MaxPool1d, base.StepModule):
    def __init__(self, kernel_size: _size_1_t, stride: Optional[_size_1_t] = None,
                 padding: _size_1_t = 0, dilation: _size_1_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False, step_mode='s') -> None:
        """
        * :ref:`API in English <MaxPool1d-en>`

        .. _MaxPool1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool1d`

        * :ref:`中文 API <MaxPool1d-cn>`

        .. _MaxPool1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool1d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, N, C, L], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class MaxPool2d(nn.MaxPool2d, base.StepModule):
    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None,
                 padding: _size_2_t = 0, dilation: _size_2_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False, step_mode='s') -> None:
        """
        * :ref:`API in English <MaxPool2d-en>`

        .. _MaxPool2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool2d`

        * :ref:`中文 API <MaxPool2d-cn>`

        .. _MaxPool2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool2d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class MaxPool3d(nn.MaxPool3d, base.StepModule):
    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = None,
                 padding: _size_3_t = 0, dilation: _size_3_t = 1,
                 return_indices: bool = False, ceil_mode: bool = False, step_mode='s') -> None:
        """
        * :ref:`API in English <MaxPool3d-en>`

        .. _MaxPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.MaxPool3d`

        * :ref:`中文 API <MaxPool3d-cn>`

        .. _MaxPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.MaxPool3d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, dilation, return_indices, ceil_mode)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AvgPool1d(nn.AvgPool1d, base.StepModule):
    def __init__(self, kernel_size: _size_1_t, stride: _size_1_t = None, padding: _size_1_t = 0, ceil_mode: bool = False,
                 count_include_pad: bool = True, step_mode='s') -> None:
        """
        * :ref:`API in English <AvgPool1d-en>`

        .. _AvgPool1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool1d`

        * :ref:`中文 API <AvgPool1d-cn>`

        .. _AvgPool1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool1d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, N, C, L], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AvgPool2d(nn.AvgPool2d, base.StepModule):
    def __init__(self, kernel_size: _size_2_t, stride: Optional[_size_2_t] = None, padding: _size_2_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, step_mode='s') -> None:
        """
        * :ref:`API in English <AvgPool2d-en>`

        .. _AvgPool2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool2d`

        * :ref:`中文 API <AvgPool2d-cn>`

        .. _AvgPool2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool2d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AvgPool3d(nn.AvgPool3d, base.StepModule):
    def __init__(self, kernel_size: _size_3_t, stride: Optional[_size_3_t] = None, padding: _size_3_t = 0,
                 ceil_mode: bool = False, count_include_pad: bool = True, divisor_override: Optional[int] = None, step_mode='s') -> None:
        """
        * :ref:`API in English <AvgPool3d-en>`

        .. _AvgPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AvgPool3d`

        * :ref:`中文 API <AvgPool3d-cn>`

        .. _AvgPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AvgPool3d` for other parameters' API
        """
        super().__init__(kernel_size, stride, padding, ceil_mode, count_include_pad, divisor_override)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AdaptiveAvgPool1d(nn.AdaptiveAvgPool1d, base.StepModule):
    def __init__(self, output_size, step_mode='s') -> None:
        """
        * :ref:`API in English <AdaptiveAvgPool1d-en>`

        .. _AdaptiveAvgPool1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool1d`

        * :ref:`中文 API <AdaptiveAvgPool1d-cn>`

        .. _AdaptiveAvgPool1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool1d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4:
                raise ValueError(f'expected x with shape [T, N, C, L], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AdaptiveAvgPool2d(nn.AdaptiveAvgPool2d, base.StepModule):
    def __init__(self, output_size, step_mode='s') -> None:
        """
        * :ref:`API in English <AdaptiveAvgPool2d-en>`

        .. _AdaptiveAvgPool2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool2d`

        * :ref:`中文 API <AdaptiveAvgPool2d-cn>`

        .. _AdaptiveAvgPool2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool2d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class AdaptiveAvgPool3d(nn.AdaptiveAvgPool3d, base.StepModule):
    def __init__(self, output_size, step_mode='s') -> None:
        """
        * :ref:`API in English <AdaptiveAvgPool3d-en>`

        .. _AdaptiveAvgPool3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.AdaptiveAvgPool3d`

        * :ref:`中文 API <AdaptiveAvgPool3d-cn>`

        .. _AdaptiveAvgPool3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.AdaptiveAvgPool3d` for other parameters' API
        """
        super().__init__(output_size)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, super().forward)

        return x


class Linear(nn.Linear, base.StepModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, step_mode='s') -> None:
        """
        * :ref:`API in English <Linear-en>`

        .. _Linear-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Linear`

        * :ref:`中文 API <Linear-cn>`

        .. _Linear-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Linear` for other parameters' API
        """
        super().__init__(in_features, out_features, bias)
        self.step_mode = step_mode


class Flatten(nn.Flatten, base.StepModule):
    def __init__(self, start_dim: int = 1, end_dim: int = -1, step_mode='s') -> None:
        """
        * :ref:`API in English <Flatten-en>`

        .. _Flatten-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.Flatten`

        * :ref:`中文 API <Flatten-cn>`

        .. _Flatten-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.Flatten` for other parameters' API
        """
        super().__init__(start_dim, end_dim)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = super().forward(x)

        elif self.step_mode == 'm':
            x = functional.seq_to_ann_forward(x, super().forward)
        return x



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
            padding_mode: str = 'zeros',
            step_mode: str = 's',
            gain: bool = True,
            eps: float = 1e-4
    ) -> None:
        """
        * :ref:`API in English <WSConv2d-en>`

        .. _WSConv2d-cn:

        :param gain: 是否对权重引入可学习的缩放系数
        :type gain: bool

        :param eps: 预防数值问题的小量
        :type eps: float

        其他的参数API参见 :class:`Conv2d`

        * :ref:`中文 API <WSConv2d-cn>`

        .. _WSConv2d-en:

        :param gain: whether introduce learnable scale factors for weights
        :type step_mode: bool

        :param eps: a small number to prevent numerical problems
        :type eps: float

        Refer to :class:`Conv2d` for other parameters' API
        """
        super().__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias, padding_mode, step_mode)
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
        return F.conv2d(x, self.get_weight(), self.bias, self.stride, self.padding, self.dilation, self.groups)

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            x = self._forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            x = functional.seq_to_ann_forward(x, self._forward)

        return x


class WSLinear(Linear):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, step_mode='s', gain=True, eps=1e-4) -> None:
        """
        * :ref:`API in English <WSLinear-en>`

        .. _WSLinear-cn:

        :param gain: 是否对权重引入可学习的缩放系数
        :type gain: bool

        :param eps: 预防数值问题的小量
        :type eps: float

        其他的参数API参见 :class:`Linear`

        * :ref:`中文 API <WSLinear-cn>`

        .. _WSLinear-en:

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
