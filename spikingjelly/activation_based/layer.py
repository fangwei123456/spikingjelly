import logging

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from . import base, functional
from torch import Tensor
from torch.nn.common_types import _size_any_t, _size_1_t, _size_2_t, _size_3_t, _ratio_any_t
from typing import Optional, List, Tuple, Union
from typing import Callable
from torch.nn.modules.batchnorm import _BatchNorm
import numpy as np


class MultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[T, batch_size, ...]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[T, batch_size, ...]``
        :rtype: Tensor
        """
        return functional.multi_step_forward(x_seq, super().forward)


class SeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!")

    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[T, batch_size, ...]
        :type x_seq: Tensor
        :return: y_seq, shape=[T, batch_size, ...]
        :rtype: Tensor
        """
        return functional.seq_to_ann_forward(x_seq, super().forward)



class TLastMultiStepContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by MultiStepContainer!")
    def forward(self, x_seq: Tensor):
        """
        :param x_seq: ``shape=[batch_size, ..., T]``
        :type x_seq: Tensor
        :return: y_seq with ``shape=[batch_size, ..., T]``
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward)
    
class TLastSeqToANNContainer(nn.Sequential, base.MultiStepModule):
    def __init__(self, *args):
        """
        Please refer to :class:`spikingjelly.activation_based.functional.t_last_seq_to_ann_forward` .
        """
        super().__init__(*args)
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by SeqToANNContainer!")


    def forward(self, x_seq: Tensor):
        """
        :param x_seq: shape=[batch_size, ..., T]
        :type x_seq: Tensor
        :return: y_seq, shape=[batch_size, ..., T]
        :rtype: Tensor
        """
        return functional.t_last_seq_to_ann_forward(x_seq, super().forward) 

class StepModeContainer(nn.Sequential, base.StepModule):
    def __init__(self, stateful: bool, *args):
        super().__init__(*args)
        self.stateful = stateful
        for m in self:
            assert not hasattr(m, 'step_mode') or m.step_mode == 's'
            if isinstance(m, base.StepModule):
                if 'm' in m.supported_step_mode():
                    logging.warning(f"{m} supports for step_mode == 's', which should not be contained by StepModeContainer!")
        self.step_mode = 's'


    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return super().forward(x)
        elif self.step_mode == 'm':
            if self.stateful:
                return functional.multi_step_forward(x, super().forward)
            else:
                return functional.seq_to_ann_forward(x, super().forward)


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


class BatchNorm1d(nn.BatchNorm1d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm1d-en>`

        .. _BatchNorm1d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm1d`

        * :ref:`中文 API <BatchNorm1d-cn>`

        .. _BatchNorm1d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm1d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 4 and x.dim() != 3:
                raise ValueError(f'expected x with shape [T, N, C, L] or [T, N, C], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)

class BatchNorm2d(nn.BatchNorm2d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm2d-en>`

        .. _BatchNorm2d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm2d`

        * :ref:`中文 API <BatchNorm2d-cn>`

        .. _BatchNorm2d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm2d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 5:
                raise ValueError(f'expected x with shape [T, N, C, H, W], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)

class BatchNorm3d(nn.BatchNorm3d, base.StepModule):
    def __init__(
            self,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <BatchNorm3d-en>`

        .. _BatchNorm3d-cn:

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        其他的参数API参见 :class:`torch.nn.BatchNorm3d`

        * :ref:`中文 API <BatchNorm3d-cn>`

        .. _BatchNorm3d-en:

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Refer to :class:`torch.nn.BatchNorm3d` for other parameters' API
        """
        super().__init__(num_features, eps, momentum, affine, track_running_stats)
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f', step_mode={self.step_mode}'

    def forward(self, x: Tensor):
        if self.step_mode == 's':
            return super().forward(x)

        elif self.step_mode == 'm':
            if x.dim() != 6:
                raise ValueError(f'expected x with shape [T, N, C, D, H, W], but got x with shape {x.shape}!')
            return functional.seq_to_ann_forward(x, super().forward)

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


class NeuNorm(base.MemoryModule):
    def __init__(self, in_channels, height, width, k=0.9, shared_across_channels=False, step_mode='s'):
        """
        * :ref:`API in English <NeuNorm.__init__-en>`

        .. _NeuNorm.__init__-cn:

        :param in_channels: 输入数据的通道数

        :param height: 输入数据的宽

        :param width: 输入数据的高

        :param k: 动量项系数

        :param shared_across_channels: 可学习的权重 ``w`` 是否在通道这一维度上共享。设置为 ``True`` 可以大幅度节省内存

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_ 中提出\\
        的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如：

        ``Conv2d -> LIF -> NeuNorm``

        要求输入的尺寸是 ``[batch_size, in_channels, height, width]``。

        ``in_channels`` 是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        ``k`` 是动量项系数，相当于论文中的 :math:`k_{\\tau 2}`。

        论文中的 :math:`\\frac{v}{F}` 会根据 :math:`k_{\\tau 2} + vF = 1` 自动算出。

        * :ref:`中文API <NeuNorm.__init__-cn>`

        .. _NeuNorm.__init__-en:

        :param in_channels: channels of input

        :param height: height of input

        :param width: height of width

        :param k: momentum factor

        :param shared_across_channels: whether the learnable parameter ``w`` is shared over channel dim. If set ``True``,
            the consumption of memory can decrease largely

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        The NeuNorm layer is proposed in `Direct Training for Spiking Neural Networks: Faster, Larger, Better <https://arxiv.org/abs/1809.05793>`_.

        It should be placed after spiking neurons behind convolution layer, e.g.,

        ``Conv2d -> LIF -> NeuNorm``

        The input should be a 4-D tensor with ``shape = [batch_size, in_channels, height, width]``.

        ``in_channels`` is the channels of input，which is :math:`F` in the paper.

        ``k`` is the momentum factor，which is :math:`k_{\\tau 2}` in the paper.

        :math:`\\frac{v}{F}` will be calculated by :math:`k_{\\tau 2} + vF = 1` autonomously.

        """
        super().__init__()
        self.step_mode = step_mode
        self.register_memory('x', 0.)
        self.k0 = k
        self.k1 = (1. - self.k0) / in_channels ** 2
        if shared_across_channels:
            self.w = nn.Parameter(Tensor(1, height, width))
        else:
            self.w = nn.Parameter(Tensor(in_channels, height, width))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))

    def single_step_forward(self, in_spikes: Tensor):
        self.x = self.k0 * self.x + self.k1 * in_spikes.sum(dim=1,
                                                            keepdim=True)  # x.shape = [batch_size, 1, height, width]
        return in_spikes - self.w * self.x

    def extra_repr(self) -> str:
        return f'shape={self.w.shape}'


class Dropout(base.MemoryModule):
    def __init__(self, p=0.5, step_mode='s'):
        """
        * :ref:`API in English <Dropout.__init__-en>`

        .. _Dropout.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        与 ``torch.nn.Dropout`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
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

        * :ref:`中文API <Dropout.__init__-cn>`

        .. _Dropout.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

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
        """
        super().__init__()
        self.step_mode = step_mode
        assert 0 <= p < 1
        self.register_memory('mask', None)
        self.p = p

    def extra_repr(self):
        return f'p={self.p}'

    def create_mask(self, x: Tensor):
        self.mask = F.dropout(torch.ones_like(x.data), self.p, training=True)

    def single_step_forward(self, x: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x)

            return x * self.mask
        else:
            return x

    def multi_step_forward(self, x_seq: Tensor):
        if self.training:
            if self.mask is None:
                self.create_mask(x_seq[0])

            return x_seq * self.mask
        else:
            return x_seq


class Dropout2d(Dropout):
    def __init__(self, p=0.2, step_mode='s'):
        """
        * :ref:`API in English <Dropout2d.__init__-en>`

        .. _Dropout2d.__init__-cn:

        :param p: 每个元素被设置为0的概率
        :type p: float
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        与 ``torch.nn.Dropout2d`` 的几乎相同。区别在于，在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        关于SNN中Dropout的更多信息，参见 :ref:`layer.Dropout <Dropout.__init__-cn>`。

        * :ref:`中文API <Dropout2d.__init__-cn>`

        .. _Dropout2d.__init__-en:

        :param p: probability of an element to be zeroed
        :type p: float
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        This layer is almost same with ``torch.nn.Dropout2d``. The difference is that elements have been zeroed at first
        step during a simulation will always be zero. The indexes of zeroed elements will be update only after ``reset()``
        has been called and a new simulation is started.

        For more information about Dropout in SNN, refer to :ref:`layer.Dropout <Dropout.__init__-en>`.
        """
        super().__init__(p, step_mode)

    def create_mask(self, x: Tensor):
        self.mask = F.dropout2d(torch.ones_like(x.data), self.p, training=True)


class SynapseFilter(base.MemoryModule):
    def __init__(self, tau=100.0, learnable=False, step_mode='s'):
        """
        * :ref:`API in English <LowPassSynapse.__init__-en>`

        .. _LowPassSynapse.__init__-cn:

        :param tau: time 突触上电流衰减的时间常数

        :param learnable: 时间常数在训练过程中是否是可学习的。若为 ``True``，则 ``tau`` 会被设定成时间常数的初始值

        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        具有滤波性质的突触。突触的输出电流满足，当没有脉冲输入时，输出电流指数衰减：

        .. math::
            \\tau \\frac{\\mathrm{d} I(t)}{\\mathrm{d} t} = - I(t)

        当有新脉冲输入时，输出电流自增1：

        .. math::
            I(t) = I(t) + 1

        记输入脉冲为 :math:`S(t)`，则离散化后，统一的电流更新方程为：

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \\frac{1}{\\tau} I(t-1) + S(t)

        这种突触能将输入脉冲进行平滑，简单的示例代码和输出结果：

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label='in spike')
            pyplot.xlabel('t')
            pyplot.ylabel('spike')
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label='out i')
            pyplot.xlabel('t')
            pyplot.ylabel('i')
            pyplot.legend()
            pyplot.show()

        .. image:: ../_static/API/activation_based/layer/SynapseFilter.png

        输出电流不仅取决于当前时刻的输入，还取决于之前的输入，使得该突触具有了一定的记忆能力。

        这种突触偶有使用，例如：

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        另一种视角是将其视为一种输入为脉冲，并输出其电压的LIF神经元。并且该神经元的发放阈值为 :math:`+\\infty` 。

        神经元最后累计的电压值一定程度上反映了该神经元在整个仿真过程中接收脉冲的数量，从而替代了传统的直接对输出脉冲计数（即发放频率）来表示神经元活跃程度的方法。因此通常用于最后一层，在以下文章中使用：

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        * :ref:`中文API <LowPassSynapse.__init__-cn>`

        .. _LowPassSynapse.__init__-en:

        :param tau: time constant that determines the decay rate of current in the synapse

        :param learnable: whether time constant is learnable during training. If ``True``, then ``tau`` will be the
            initial value of time constant

        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        The synapse filter that can filter input current. The output current will decay when there is no input spike:

        .. math::
            \\tau \\frac{\\mathrm{d} I(t)}{\\mathrm{d} t} = - I(t)

        The output current will increase 1 when there is a new input spike:

        .. math::
            I(t) = I(t) + 1

        Denote the input spike as :math:`S(t)`, then the discrete current update equation is as followed:

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \\frac{1}{\\tau} I(t-1) + S(t)

        This synapse can smooth input. Here is the example and output:

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label='in spike')
            pyplot.xlabel('t')
            pyplot.ylabel('spike')
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label='out i')
            pyplot.xlabel('t')
            pyplot.ylabel('i')
            pyplot.legend()
            pyplot.show()

        .. image:: ../_static/API/activation_based/layer/SynapseFilter.png

        The output current is not only determined by the present input but also by the previous input, which makes this
        synapse have memory.

        This synapse is sometimes used, e.g.:

        `Unsupervised learning of digit recognition using spike-timing-dependent plasticity <https://www.frontiersin.org/articles/10.3389/fncom.2015.00099/full>`_

        `Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network <https://arxiv.org/abs/2003.02944>`_

        Another view is regarding this synapse as a LIF neuron with a :math:`+\\infty` threshold voltage.

        The final output of this synapse (or the final voltage of this LIF neuron) represents the accumulation of input
        spikes, which substitute for traditional firing rate that indicates the excitatory level. So, it can be used in
        the last layer of the network, e.g.:

        `Enabling spike-based backpropagation for training deep neural network architectures <https://arxiv.org/abs/1903.06379>`_

        """
        super().__init__()
        self.step_mode = step_mode
        self.learnable = learnable
        assert tau > 1
        if learnable:
            init_w = - math.log(tau - 1)
            self.w = nn.Parameter(torch.as_tensor(init_w))
        else:
            self.tau = tau

        self.register_memory('out_i', 0.)

    def extra_repr(self):
        if self.learnable:
            with torch.no_grad():
                tau = 1. / self.w.sigmoid()
        else:
            tau = self.tau

        return f'tau={tau}, learnable={self.learnable}, step_mode={self.step_mode}'

    @staticmethod
    @torch.jit.script
    def js_single_step_forward_learnable(x: torch.Tensor, w: torch.Tensor, out_i: torch.Tensor):
        inv_tau = w.sigmoid()
        out_i = out_i - (1. - x) * out_i * inv_tau + x
        return out_i

    @staticmethod
    @torch.jit.script
    def js_single_step_forward(x: torch.Tensor, tau: float, out_i: torch.Tensor):
        inv_tau = 1. / tau
        out_i = out_i - (1. - x) * out_i * inv_tau + x
        return out_i

    def single_step_forward(self, x: Tensor):
        if isinstance(self.out_i, float):
            out_i_init = self.out_i
            self.out_i = torch.zeros_like(x.data)
            if out_i_init != 0.:
                torch.fill_(self.out_i, out_i_init)

        if self.learnable:
            self.out_i = self.js_single_step_forward_learnable(x, self.w, self.out_i)
        else:
            self.out_i = self.js_single_step_forward(x, self.tau, self.out_i)
        return self.out_i


class DropConnectLinear(base.MemoryModule):
    def __init__(self, in_features: int, out_features: int, bias: bool = True, p: float = 0.5, samples_num: int = 1024,
                 invariant: bool = False, activation: Optional[nn.Module] = nn.ReLU(), step_mode='s') -> None:
        """
        * :ref:`API in English <DropConnectLinear.__init__-en>`

        .. _DropConnectLinear.__init__-cn:

        :param in_features: 每个输入样本的特征数
        :type in_features: int
        :param out_features: 每个输出样本的特征数
        :type out_features: int
        :param bias: 若为 ``False``，则本层不会有可学习的偏置项。
            默认为 ``True``
        :type bias: bool
        :param p: 每个连接被断开的概率。默认为0.5
        :type p: float
        :param samples_num: 在推理时，从高斯分布中采样的数据数量。默认为1024
        :type samples_num: int
        :param invariant: 若为 ``True``，线性层会在第一次执行前向传播时被按概率断开，断开后的线性层会保持不变，直到 ``reset()`` 函数
            被调用，线性层恢复为完全连接的状态。完全连接的线性层，调用 ``reset()`` 函数后的第一次前向传播时被重新按概率断开。 若为
            ``False``，在每一次前向传播时线性层都会被重新完全连接再按概率断开。 阅读 :ref:`layer.Dropout <Dropout.__init__-cn>` 以
            获得更多关于此参数的信息。
            默认为 ``False``
        :type invariant: bool
        :param activation: 在线性层后的激活层
        :type activation: Optional[nn.Module]
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        DropConnect，由 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
        一文提出。DropConnect与Dropout非常类似，区别在于DropConnect是以概率 ``p`` 断开连接，而Dropout是将输入以概率置0。

        .. Note::

            在使用DropConnect进行推理时，输出的tensor中的每个元素，都是先从高斯分布中采样，通过激活层激活，再在采样数量上进行平均得到的。
            详细的流程可以在 `Regularization of Neural Networks using DropConnect <http://proceedings.mlr.press/v28/wan13.pdf>`_
            一文中的 `Algorithm 2` 找到。激活层 ``activation`` 在中间的步骤起作用，因此我们将其作为模块的成员。

        * :ref:`中文API <DropConnectLinear.__init__-cn>`

        .. _DropConnectLinear.__init__-en:

        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: If set to ``False``, the layer will not learn an additive bias.
            Default: ``True``
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
        :type invariant: bool
        :param activation: the activation layer after the linear layer
        :type activation: Optional[nn.Module]
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

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
        """
        super().__init__()
        self.step_mode = step_mode
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(Tensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

        self.p = p  # 置0的概率
        self.register_memory('dropped_w', None)
        if self.bias is not None:
            self.register_memory('dropped_b', None)

        self.samples_num = samples_num
        self.invariant = invariant
        self.activation = activation

    def reset_parameters(self) -> None:
        """
        * :ref:`API in English <DropConnectLinear.reset_parameters-en>`

        .. _DropConnectLinear.reset_parameters-cn:

        :return: None
        :rtype: None

        初始化模型中的可学习参数。

        * :ref:`中文API <DropConnectLinear.reset_parameters-cn>`

        .. _DropConnectLinear.reset_parameters-en:

        :return: None
        :rtype: None

        Initialize the learnable parameters of this module.
        """
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)

    def reset(self):
        """
        * :ref:`API in English <DropConnectLinear.reset-en>`

        .. _DropConnectLinear.reset-cn:

        :return: None
        :rtype: None

        将线性层重置为完全连接的状态，若 ``self.activation`` 也是一个有状态的层，则将其也重置。

        * :ref:`中文API <DropConnectLinear.reset-cn>`

        .. _DropConnectLinear.reset-en:

        :return: None
        :rtype: None

        Reset the linear layer to fully-connected status. If ``self.activation`` is also stateful, this function will
        also reset it.
        """
        super().reset()
        if hasattr(self.activation, 'reset'):
            self.activation.reset()

    def drop(self, batch_size: int):
        mask_w = (torch.rand_like(self.weight.unsqueeze(0).repeat([batch_size] + [1] * self.weight.dim())) > self.p)
        # self.dropped_w = mask_w.to(self.weight) * self.weight  # shape = [batch_size, out_features, in_features]
        self.dropped_w = self.weight * mask_w

        if self.bias is not None:
            mask_b = (torch.rand_like(self.bias.unsqueeze(0).repeat([batch_size] + [1] * self.bias.dim())) > self.p)
            # self.dropped_b = mask_b.to(self.bias) * self.bias
            self.dropped_b = self.bias * mask_b

    def single_step_forward(self, input: Tensor) -> Tensor:
        if self.training:
            if self.invariant:
                if self.dropped_w is None:
                    self.drop(input.shape[0])
            else:
                self.drop(input.shape[0])
            if self.bias is None:
                ret = torch.bmm(self.dropped_w, input.unsqueeze(-1)).squeeze(-1)
            else:
                ret = torch.bmm(self.dropped_w, input.unsqueeze(-1)).squeeze(-1) + self.dropped_b
            if self.activation is None:
                return ret
            else:
                return self.activation(ret)
        else:
            mu = (1 - self.p) * F.linear(input, self.weight, self.bias)  # shape = [batch_size, out_features]
            if self.bias is None:
                sigma2 = self.p * (1 - self.p) * F.linear(input.square(), self.weight.square())
            else:
                sigma2 = self.p * (1 - self.p) * F.linear(input.square(), self.weight.square(), self.bias.square())
            dis = torch.distributions.normal.Normal(mu, sigma2.sqrt())
            samples = dis.sample(torch.Size([self.samples_num]))

            if self.activation is None:
                ret = samples
            else:
                ret = self.activation(samples)
            return ret.mean(dim=0)

    def extra_repr(self) -> str:
        return f'in_features={self.in_features}, out_features={self.out_features}, bias={self.bias is not None}, p={self.p}, invariant={self.invariant}'



class PrintShapeModule(nn.Module):
    def __init__(self, ext_str='PrintShapeModule'):
        """
        * :ref:`API in English <PrintModule.__init__-en>`

        .. _PrintModule.__init__-cn:

        :param ext_str: 额外打印的字符串
        :type ext_str: str

        只打印 ``ext_str`` 和输入的 ``shape``，不进行任何操作的网络层，可以用于debug。

        * :ref:`中文API <PrintModule.__init__-cn>`

        .. _PrintModule.__init__-en:

        :param ext_str: extra strings for printing
        :type ext_str: str

        This layer will not do any operation but print ``ext_str`` and the shape of input, which can be used for debugging.

        """
        super().__init__()
        self.ext_str = ext_str

    def forward(self, x: Tensor):
        print(self.ext_str, x.shape)
        return x


class ElementWiseRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, element_wise_function: Callable, step_mode='s'):
        """
        * :ref:`API in English <ElementWiseRecurrentContainer-en>`

        .. _ElementWiseRecurrentContainer-cn:

        :param sub_module: 被包含的模块
        :type sub_module: torch.nn.Module
        :param element_wise_function: 用户自定义的逐元素函数，应该形如 ``z=f(x, y)``
        :type element_wise_function: Callable
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        使用逐元素运算的自连接包装器。记 ``sub_module`` 的输入输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入为 :math:`x[t]`，则

        .. math::

            i[t] = f(x[t], y[t-1])

        其中 :math:`f` 是用户自定义的逐元素函数。我们默认 :math:`y[-1] = 0`。


        .. Note::

            ``sub_module`` 输入和输出的尺寸需要相同。

        示例代码：

        .. code-block:: python

            T = 8
            net = ElementWiseRecurrentContainer(neuron.IFNode(v_reset=None), element_wise_function=lambda x, y: x + y)
            print(net)
            x = torch.zeros([T])
            x[0] = 1.5
            for t in range(T):
                print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

            functional.reset_net(net)


        * :ref:`中文 API <ElementWiseRecurrentContainer-cn>`

        .. _ElementWiseRecurrentContainer-en:

        :param sub_module: the contained module
        :type sub_module: torch.nn.Module
        :param element_wise_function: the user-defined element-wise function, which should have the format ``z=f(x, y)``
        :type element_wise_function: Callable
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        A container that use a element-wise recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = f(x[t], y[t-1])

        where :math:`f` is the user-defined element-wise function. We set :math:`y[-1] = 0`.

        .. admonition:: Note
            :class: note

            The shape of inputs and outputs of ``sub_module`` must be the same.

        Codes example:

        .. code-block:: python

            T = 8
            net = ElementWiseRecurrentContainer(neuron.IFNode(v_reset=None), element_wise_function=lambda x, y: x + y)
            print(net)
            x = torch.zeros([T])
            x[0] = 1.5
            for t in range(T):
                print(t, f'x[t]={x[t]}, s[t]={net(x[t])}')

            functional.reset_net(net)
        """
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module = sub_module
        self.element_wise_function = element_wise_function
        self.register_memory('y', None)

    def single_step_forward(self, x: Tensor):
        if self.y is None:
            self.y = torch.zeros_like(x.data)
        self.y = self.sub_module(self.element_wise_function(self.y, x))
        return self.y

    def extra_repr(self) -> str:
        return f'element-wise function={self.element_wise_function}, step_mode={self.step_mode}'


class LinearRecurrentContainer(base.MemoryModule):
    def __init__(self, sub_module: nn.Module, in_features: int, out_features: int, bias: bool = True,
                 step_mode='s') -> None:
        """
        * :ref:`API in English <LinearRecurrentContainer-en>`

        .. _LinearRecurrentContainer-cn:

        :param sub_module: 被包含的模块
        :type sub_module: torch.nn.Module
        :param in_features: 输入的特征数量
        :type in_features: int
        :param out_features: 输出的特征数量
        :type out_features: int
        :param bias: 若为 ``False``，则线性自连接不会带有可学习的偏执项
        :type bias: bool
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        使用线性层的自连接包装器。记 ``sub_module`` 的输入和输出为 :math:`i[t]` 和 :math:`y[t]` （注意 :math:`y[t]` 也是整个模块的输出），
        整个模块的输入记作 :math:`x[t]` ，则

        .. math::

            i[t] = \\begin{pmatrix} x[t] \\\\ y[t-1]\\end{pmatrix} W^{T} + b

        其中 :math:`W, b` 是线性层的权重和偏置项。默认 :math:`y[-1] = 0`。

        :math:`x[t]` 应该 ``shape = [N, *, in_features]``，:math:`y[t]` 则应该 ``shape = [N, *, out_features]``。

        .. Note::

            自连接是由 ``torch.nn.Linear(in_features + out_features, in_features, bias)`` 实现的。

        .. code-block:: python

            in_features = 4
            out_features = 2
            T = 8
            N = 2
            net = LinearRecurrentContainer(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    neuron.LIFNode(),
                ),
                in_features, out_features)
            print(net)
            x = torch.rand([T, N, in_features])
            for t in range(T):
                print(t, net(x[t]))

            functional.reset_net(net)

        * :ref:`中文 API <LinearRecurrentContainer-cn>`

        .. _LinearRecurrentContainer-en:

        :param sub_module: the contained module
        :type sub_module: torch.nn.Module
        :param in_features: size of each input sample
        :type in_features: int
        :param out_features: size of each output sample
        :type out_features: int
        :param bias: If set to ``False``, the linear recurrent layer will not learn an additive bias
        :type bias: bool
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        A container that use a linear recurrent connection. Denote the inputs and outputs of ``sub_module`` as :math:`i[t]`
        and :math:`y[t]` (Note that :math:`y[t]` is also the outputs of this module), and the inputs of this module as
        :math:`x[t]`, then

        .. math::

            i[t] = \\begin{pmatrix} x[t] \\\\ y[t-1]\\end{pmatrix} W^{T} + b

        where :math:`W, b` are the weight and bias of the linear connection. We set :math:`y[-1] = 0`.

        :math:`x[t]` should have the shape ``[N, *, in_features]``, and :math:`y[t]` has the shape ``[N, *, out_features]``.

        .. admonition:: Note
            :class: note

            The recurrent connection is implement by ``torch.nn.Linear(in_features + out_features, in_features, bias)``.

        .. code-block:: python

            in_features = 4
            out_features = 2
            T = 8
            N = 2
            net = LinearRecurrentContainer(
                nn.Sequential(
                    nn.Linear(in_features, out_features),
                    neuron.LIFNode(),
                ),
                in_features, out_features)
            print(net)
            x = torch.rand([T, N, in_features])
            for t in range(T):
                print(t, net(x[t]))

            functional.reset_net(net)

        """
        super().__init__()
        self.step_mode = step_mode
        assert not hasattr(sub_module, 'step_mode') or sub_module.step_mode == 's'
        self.sub_module_out_features = out_features
        self.rc = nn.Linear(in_features + out_features, in_features, bias)
        self.sub_module = sub_module
        self.register_memory('y', None)

    def single_step_forward(self, x: Tensor):
        if self.y is None:
            if x.ndim == 2:
                self.y = torch.zeros([x.shape[0], self.sub_module_out_features]).to(x)
            else:
                out_shape = [x.shape[0]]
                out_shape.extend(x.shape[1:-1])
                out_shape.append(self.sub_module_out_features)
                self.y = torch.zeros(out_shape).to(x)
        x = torch.cat((x, self.y), dim=-1)
        self.y = self.sub_module(self.rc(x))
        return self.y

    def extra_repr(self) -> str:
        return f', step_mode={self.step_mode}'

class _ThresholdDependentBatchNormBase(_BatchNorm, base.MultiStepModule):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.step_mode = 'm'
        self.alpha = alpha
        self.v_th = v_th
        assert self.affine, "ThresholdDependentBatchNorm needs to set `affine = True`!"
        torch.nn.init.constant_(self.weight, alpha * v_th)

    def forward(self, x_seq):
        assert self.step_mode == 'm', "ThresholdDependentBatchNormBase can only be used in the multi-step mode!"
        return functional.seq_to_ann_forward(x_seq, super().forward)


class ThresholdDependentBatchNorm1d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm1d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm1d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm1d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm1d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm1d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 4 - 1 or input.dim() == 3 - 1  # [T * N, C, L]



class ThresholdDependentBatchNorm2d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm2d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm2d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm2d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm2d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm2d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 5 - 1  # [T * N, C, H, W]

class ThresholdDependentBatchNorm3d(_ThresholdDependentBatchNormBase):
    def __init__(self, alpha: float, v_th: float, *args, **kwargs):
        """
        * :ref:`API in English <MultiStepThresholdDependentBatchNorm3d.__init__-en>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-cn:

        :param alpha: 由网络结构决定的超参数
        :type alpha: float
        :param v_th: 下一个脉冲神经元层的阈值
        :type v_th: float

        ``*args, **kwargs`` 中的参数与 :class:`torch.nn.BatchNorm3d` 的参数相同。

        `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_ 一文提出
        的Threshold-Dependent Batch Normalization (tdBN)。

        * :ref:`中文API <MultiStepThresholdDependentBatchNorm3d.__init__-cn>`

        .. _MultiStepThresholdDependentBatchNorm3d.__init__-en:

        :param alpha: the hyper-parameter depending on network structure
        :type alpha: float
        :param v_th: the threshold of next spiking neurons layer
        :type v_th: float

        Other parameters in ``*args, **kwargs`` are same with those of :class:`torch.nn.BatchNorm3d`.

        The Threshold-Dependent Batch Normalization (tdBN) proposed in `Going Deeper With Directly-Trained Larger Spiking Neural Networks <https://arxiv.org/abs/2011.05280>`_.
        """
        super().__init__(alpha, v_th, *args, **kwargs)

    def _check_input_dim(self, input):
        assert input.dim() == 6 - 1  # [T * N, C, H, W, D]


class TemporalWiseAttention(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, reduction: int = 16, dimension: int = 4):
        """
        * :ref:`API in English <MultiStepTemporalWiseAttention.__init__-en>`

        .. _MultiStepTemporalWiseAttention.__init__-cn:

        :param T: 输入数据的时间步长

        :param reduction: 压缩比

        :param dimension: 输入数据的维度。当输入数据为[T, N, C, H, W]时， dimension = 4；输入数据维度为[T, N, L]时，dimension = 2。

        `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_ 中提出
        的MultiStepTemporalWiseAttention层。MultiStepTemporalWiseAttention层必须放在二维卷积层之后脉冲神经元之前，例如：

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        输入的尺寸是 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` ，经过MultiStepTemporalWiseAttention层，输出为 ``[T, N, C, H, W]`` 或者 ``[T, N, L]`` 。

        ``reduction`` 是压缩比，相当于论文中的 :math:`r`。

        * :ref:`中文API <MultiStepTemporalWiseAttention.__init__-cn>`

        .. _MultiStepTemporalWiseAttention.__init__-en:

        :param T: timewindows of input

        :param reduction: reduction ratio

        :param dimension: Dimensions of input. If the input dimension is [T, N, C, H, W], dimension = 4; when the input dimension is [T, N, L], dimension = 2.

        The MultiStepTemporalWiseAttention layer is proposed in `Temporal-Wise Attention Spiking Neural Networks for Event Streams Classification <https://openaccess.thecvf.com/content/ICCV2021/html/Yao_Temporal-Wise_Attention_Spiking_Neural_Networks_for_Event_Streams_Classification_ICCV_2021_paper.html>`_.

        It should be placed after the convolution layer and before the spiking neurons, e.g.,

        ``Conv2d -> MultiStepTemporalWiseAttention -> LIF``

        The dimension of the input is ``[T, N, C, H, W]`` or  ``[T, N, L]`` , after the MultiStepTemporalWiseAttention layer, the output dimension is ``[T, N, C, H, W]`` or  ``[T, N, L]`` .

        ``reduction`` is the reduction ratio，which is :math:`r` in the paper.

        """
        super().__init__()
        self.step_mode = 'm'
        assert dimension == 4 or dimension == 2, 'dimension must be 4 or 2'

        self.dimension = dimension

        # Sequence
        if self.dimension == 2:
            self.avg_pool = nn.AdaptiveAvgPool1d(1)
            self.max_pool = nn.AdaptiveMaxPool1d(1)
        elif self.dimension == 4:
            self.avg_pool = nn.AdaptiveAvgPool3d(1)
            self.max_pool = nn.AdaptiveMaxPool3d(1)

        assert T >= reduction, 'reduction cannot be greater than T'

        # Excitation
        self.sharedMLP = nn.Sequential(
            nn.Linear(T, T // reduction, bias=False),
            nn.ReLU(),
            nn.Linear(T // reduction, T, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x_seq: torch.Tensor):
        assert x_seq.dim() == 3 or x_seq.dim() == 5, ValueError(
            f'expected 3D or 5D input with shape [T, N, M] or [T, N, C, H, W], but got input with shape {x_seq.shape}')
        x_seq = x_seq.transpose(0, 1)
        avgout = self.sharedMLP(self.avg_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]]))
        maxout = self.sharedMLP(self.max_pool(x_seq).view([x_seq.shape[0], x_seq.shape[1]]))
        scores = self.sigmoid(avgout + maxout)
        if self.dimension == 2:
            y_seq = x_seq * scores[:, :, None]
        elif self.dimension == 4:
            y_seq = x_seq * scores[:, :, None, None, None]
        y_seq = y_seq.transpose(0, 1)
        return y_seq

class MultiDimensionalAttention(nn.Module, base.MultiStepModule):
    def __init__(self, T: int, C: int, reduction_t: int = 16, reduction_c: int = 16, kernel_size=3):
        """
        * :ref:`API in English <MultiStepMultiDimensionalAttention.__init__-en>`

        .. _MultiStepMultiDimensionalAttention.__init__-cn:

        :param T: 输入数据的时间步长

        :param C: 输入数据的通道数

        :param reduction_t: 时间压缩比

        :param reduction_c: 通道压缩比

        :param kernel_size: 空间注意力机制的卷积核大小

        `Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_ 中提出
        的MA-SNN模型以及MultiStepMultiDimensionalAttention层。

        您可以从以下链接中找到MA-SNN的示例项目:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        输入的尺寸是 ``[T, N, C, H, W]`` ，经过MultiStepMultiDimensionalAttention层，输出为 ``[T, N, C, H, W]`` 。

        * :ref:`中文API <MultiStepMultiDimensionalAttention.__init__-cn>`

        .. _MultiStepMultiDimensionalAttention.__init__-en:

        :param T: timewindows of input

        :param C: channel number of input

        :param reduction_t: temporal reduction ratio

        :param reduction_c: channel reduction ratio

        :param kernel_size: convolution kernel size of SpatialAttention

        The MA-SNN model and MultiStepMultiDimensionalAttention layer are proposed in ``Attention Spiking Neural Networks <https://ieeexplore.ieee.org/document/10032591>`_.

        You can find the example projects of MA-SNN in the following links:
        - https://github.com/MA-SNN/MA-SNN
        - https://github.com/ridgerchu/SNN_Attention_VGG

        The dimension of the input is ``[T, N, C, H, W]`` , after the MultiStepMultiDimensionalAttention layer, the output dimension is ``[T, N, C, H, W]`` .


        """
        super().__init__()

        assert T >= reduction_t, 'reduction_t cannot be greater than T'
        assert C >= reduction_c, 'reduction_c cannot be greater than C'
        
        from einops import rearrange
        
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
            f'expected 5D input with shape [T, N, C, H, W], but got input with shape {x.shape}')
        x = x.transpose(0, 1)
        out = self.ta(x) * x
        out = self.ca(out) * out
        out = self.sa(out) * out
        out = self.relu(out)
        out = out.transpose(0, 1)
        return out

class VotingLayer(nn.Module, base.StepModule):
    def __init__(self, voting_size: int = 10, step_mode='s'):
        """
        * :ref:`API in English <VotingLayer-en>`

        .. _VotingLayer-cn:

        :param voting_size: 决定一个类别的投票数量
        :type voting_size: int
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        投票层，对 ``shape = [..., C * voting_size]`` 的输入在最后一维上做 ``kernel_size = voting_size, stride = voting_size`` 的平均池化

        * :ref:`中文 API <VotingLayer-cn>`

        .. _VotingLayer-en:

        :param voting_size: the voting numbers for determine a class
        :type voting_size: int
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        Applies average pooling with ``kernel_size = voting_size, stride = voting_size`` on the last dimension of the input with ``shape = [..., C * voting_size]``

        """
        super().__init__()
        self.voting_size = voting_size
        self.step_mode = step_mode

    def extra_repr(self):
        return super().extra_repr() + f'voting_size={self.voting_size}, step_mode={self.step_mode}'

    def single_step_forward(self, x: torch.Tensor):
        return F.avg_pool1d(x.unsqueeze(1), self.voting_size, self.voting_size).squeeze(1)

    def forward(self, x: torch.Tensor):
        if self.step_mode == 's':
            return self.single_step_forward(x)
        elif self.step_mode == 'm':
            return functional.seq_to_ann_forward(x, self.single_step_forward)

class Delay(base.MemoryModule):
    def __init__(self, delay_steps: int, step_mode='s'):
        """
        * :ref:`API in English <Delay.__init__-en>`

        .. _Delay.__init__-cn:

        :param delay_steps: 延迟的时间步数
        :type delay_steps: int
        :param step_mode: 步进模式，可以为 `'s'` (单步) 或 `'m'` (多步)
        :type step_mode: str

        延迟层，可以用来延迟输入，使得 ``y[t] = x[t - delay_steps]``。缺失的数据用0填充。

        代码示例：

        .. code-block:: python

            delay_layer = Delay(delay=1, step_mode='m')
            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay_layer(x)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

        输出为：

        .. code-block:: bash

            x=
            tensor([[0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])


        * :ref:`中文API <Delay.__init__-cn>`

        .. _Delay.__init__-en:

        :param delay_steps: the number of delayed time-steps
        :type delay_steps: int
        :param step_mode: the step mode, which can be `s` (single-step) or `m` (multi-step)
        :type step_mode: str

        A delay layer that can delay inputs and makes ``y[t] = x[t - delay_steps]``. The nonexistent data will be regarded as 0.

        Codes example:

        .. code-block:: python

            delay_layer = Delay(delay=1, step_mode='m')
            x = torch.rand([5, 2])
            x[3:].zero_()
            x.requires_grad = True
            y = delay_layer(x)
            print('x=')
            print(x)
            print('y=')
            print(y)
            y.sum().backward()
            print('x.grad=')
            print(x.grad)

        The outputs are:

        .. code-block:: bash

            x=
            tensor([[0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000],
                    [0.0000, 0.0000]], requires_grad=True)
            y=
            tensor([[0.0000, 0.0000],
                    [0.2510, 0.7246],
                    [0.5303, 0.3160],
                    [0.2531, 0.5961],
                    [0.0000, 0.0000]], grad_fn=<CatBackward0>)
            x.grad=
            tensor([[1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [1., 1.],
                    [0., 0.]])
        """
        super().__init__()
        assert delay_steps >= 0 and isinstance(delay_steps, int)
        self._delay_steps = delay_steps
        self.step_mode = step_mode

        self.register_memory('queue', [])
        # used for single step mode

    @property
    def delay_steps(self):
        return self._delay_steps

    def single_step_forward(self, x: torch.Tensor):
        self.queue.append(x)
        if self.queue.__len__() > self.delay_steps:
            return self.queue.pop(0)
        else:
            return torch.zeros_like(x)

    def multi_step_forward(self, x_seq: torch.Tensor):
        return functional.delay(x_seq, self.delay_steps)


class TemporalEffectiveBatchNormNd(base.MemoryModule):

    bn_instance = _BatchNorm

    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        super().__init__()

        self.bn = self.bn_instance(num_features, eps, momentum, affine, track_running_stats, step_mode)
        self.scale = nn.Parameter(torch.ones([T]))
        self.register_memory('t', 0)

    def single_step_forward(self, x: torch.Tensor):
        return self.bn(x) * self.scale[self.t]

class TemporalEffectiveBatchNorm1d(TemporalEffectiveBatchNormNd):
    bn_instance = BatchNorm1d
    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm1d-en>`

        .. _TemporalEffectiveBatchNorm1d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm1d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm1d-cn>`

        .. _TemporalEffectiveBatchNorm1d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm1d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, L]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1)


class TemporalEffectiveBatchNorm2d(TemporalEffectiveBatchNormNd):
    bn_instance = BatchNorm2d
    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm2d-en>`

        .. _TemporalEffectiveBatchNorm2d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm2d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm2d-cn>`

        .. _TemporalEffectiveBatchNorm2d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm2d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, H, W]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1)



class TemporalEffectiveBatchNorm3d(TemporalEffectiveBatchNormNd):
    bn_instance = BatchNorm3d
    def __init__(
            self,
            T: int,
            num_features,
            eps=1e-5,
            momentum=0.1,
            affine=True,
            track_running_stats=True,
            step_mode='s'
    ):
        """
        * :ref:`API in English <TemporalEffectiveBatchNorm3d-en>`

        .. _TemporalEffectiveBatchNorm3d-cn:

        :param T: 总时间步数
        :type T: int

        其他参数的API参见 :class:`BatchNorm3d`

        `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_ 一文提出的Temporal Effective Batch Normalization (TEBN)。

        TEBN给每个时刻的输出增加一个缩放。若普通的BN在 ``t`` 时刻的输出是 ``y[t]``，则TEBN的输出为 ``k[t] * y[t]``，其中 ``k[t]`` 是可
        学习的参数。

        * :ref:`中文 API <TemporalEffectiveBatchNorm3d-cn>`

        .. _TemporalEffectiveBatchNorm3d-en:

        :param T: the number of time-steps
        :type T: int

        Refer to :class:`BatchNorm3d` for other parameters' API

        Temporal Effective Batch Normalization (TEBN) proposed by `Temporal Effective Batch Normalization in Spiking Neural Networks <https://openreview.net/forum?id=fLIgyyQiJqz>`_.

        TEBN adds a scale on outputs of each time-step from the native BN. Denote the output at time-step ``t`` of the native BN as ``y[t]``, then the output of TEBN is ``k[t] * y[t]``, where ``k[t]`` is the learnable scale.

        """
        super().__init__(T, num_features, eps, momentum, affine, track_running_stats, step_mode)

    def multi_step_forward(self, x_seq: torch.Tensor):
        # x.shape = [T, N, C, H, W, D]
        return self.bn(x_seq) * self.scale.view(-1, 1, 1, 1, 1, 1)


# OTTT modules

class ReplaceforGrad(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, x_r):
        return x_r

    @staticmethod
    def backward(ctx, grad):
        return (grad, grad)


class GradwithTrace(nn.Module):
    def __init__(self, module):
        """
        * :ref:`API in English <GradwithTrace-en>`

        .. _GradwithTrace-cn:

        :param module: 需要包装的模块

        用于随时间在线训练时，根据神经元的迹计算梯度
        出处：'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _GradwithTrace-en:

        :param module: the module that requires wrapping

        Used for online training through time, calculate gradients by the traces of neurons
        Reference: 'Online Training Through Time for Spiking Neural Networks <https://openreview.net/forum?id=Siv3nHYHheI>'

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
        
        with torch.no_grad():
            out = self.module(spike).detach()

        in_for_grad = ReplaceforGrad.apply(spike, trace)
        out_for_grad = self.module(in_for_grad)

        x = ReplaceforGrad.apply(out_for_grad, out)

        return x


class SpikeTraceOp(nn.Module):
    def __init__(self, module):
        """
        * :ref:`API in English <SpikeTraceOp-en>`

        .. _SpikeTraceOp-cn:

        :param module: 需要包装的模块

        对脉冲和迹进行相同的运算，如Dropout，AvgPool等

        * :ref:`中文 API <GradwithTrace-cn>`

        .. _SpikeTraceOp-en:

        :param module: the module that requires wrapping

        perform the same operations for spike and trace, such as Dropout, Avgpool, etc.

        """
        super().__init__()
        self.module = module

    def forward(self, x: Tensor):
        # x: [spike, trace], defined in OTTTLIFNode in neuron.py
        spike, trace = x[0], x[1]
        
        spike = self.module(spike)
        with torch.no_grad():
            trace = self.module(trace)

        x = [spike, trace]

        return x


class OTTTSequential(nn.Sequential):
    def __init__(self, *args):
        super().__init__(*args)

    def forward(self, input):
        for module in self:
            if not isinstance(input, list):
                input = module(input)
            else:
                if len(list(module.parameters())) > 0: # e.g., Conv2d, Linear, etc.
                    module = GradwithTrace(module)
                else: # e.g., Dropout, AvgPool, etc.
                    module = SpikeTraceOp(module)
                input = module(input)
        return input


# weight standardization modules

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
