import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.cpp_extension import load_inline
from torch.cuda.amp import custom_fwd, custom_bwd
import logging
from . import tensor_cache

from torch import Tensor
from typing import Optional, Union
from torch.types import _int, _size
from torch.nn.modules.utils import _single, _pair, _triple

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.spike_op: {e}')
    cupy = None


try:
    logging.info('spikingjelly.clock_driven.spike_op: try to use `torch.utils.cpp_extension.load_inline` to load cudnn functions.')
    logging.info(f'If it is hanging, pleast try to delete torch_extensions cache directory. (In most cases, the directory is {torch.utils.cpp_extension._get_build_directory("", False)}.)')
    cpp_wrapper = load_inline(
            name='cpp_wrapper',
            cpp_sources='using namespace at;',
            functions=[
                'cudnn_convolution_backward',
                'cudnn_convolution_backward_input',
                'cudnn_convolution_backward_weight'
            ],
            with_cuda=True
    )
except BaseException as e:
    logging.info(f'spikingjelly.clock_driven.spike_op: {e}')
    cpp_wrapper = None

'''
aten/src/ATen/native/cudnn/ConvPlaceholders.cpp

at::Tensor cudnn_convolution(
    const at::Tensor& input, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation,
    int64_t groups, bool benchmark, bool deterministic, bool allow_tf32)

There are two overloaded C++ methods `cudnn_convolution`. So, we need to use an alternative syntax to cast the overloaded function.
Refer to https://pybind11.readthedocs.io/en/stable/classes.html#overloaded-methods and https://github.com/pytorch/pytorch/issues/39518 for more details.
    
aten/src/ATen/native/cudnn/ConvShared.cpp

Tensor cudnn_convolution_forward(
    CheckedFrom c,
    const TensorArg& input, const TensorArg& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)

aten/src/ATen/native/cudnn/ConvPlaceholders.cpp

std::tuple<at::Tensor,at::Tensor> cudnn_convolution_backward(
    const at::Tensor& input, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32, std::array<bool,2> output_mask)
  
aten/src/ATen/native/cudnn/ConvShared.cpp

at::Tensor cudnn_convolution_backward_input(
    IntArrayRef input_size, const at::Tensor& grad_output, const at::Tensor& weight,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
    
aten/src/ATen/native/cudnn/ConvShared.cpp

at::Tensor cudnn_convolution_backward_weight(
    IntArrayRef weight_size, const at::Tensor& grad_output, const at::Tensor& input,
    IntArrayRef padding, IntArrayRef stride, IntArrayRef dilation, int64_t groups,
    bool benchmark, bool deterministic, bool allow_tf32)
'''

class spikeConvolution(torch.autograd.Function):
    # Pytorch only provides cudnn_convolution without bias.
    # Refer to https://github.com/pytorch/pytorch/issues/3823 for more details.
    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias, stride, padding, dilation, groups):
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)

            if ctx.needs_input_grad[0]:
                ctx.save_for_backward(weight)

            ctx.padding = padding
            ctx.stride = stride
            ctx.dilation = dilation
            ctx.groups = groups
            ctx.weight_shape = weight.shape

        if spike.dim() == 3:
            return F.conv1d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 4:
            return F.conv2d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)
        elif spike.dim() == 5:
            return F.conv3d(input=spike, weight=weight, bias=bias, stride=stride, padding=padding, dilation=dilation, groups=groups)



    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        grad_spike = None
        grad_weight = None
        grad_bias = None
        if ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            weight = weight.to(grad_output.dtype)
            grad_spike, grad_weight = cpp_wrapper.cudnn_convolution_backward(spike, grad_output, weight, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32, (
                                                                               True,
                                                                               True))

        elif not ctx.needs_input_grad[0] and ctx.needs_input_grad[1]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)
            grad_weight = cpp_wrapper.cudnn_convolution_backward_weight(ctx.weight_shape, grad_output, spike, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32)

        elif ctx.needs_input_grad[0] and not ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
            weight = weight.to(grad_output.dtype)
            grad_spike = cpp_wrapper.cudnn_convolution_backward_input(ctx.spike_shape, grad_output, weight, ctx.padding,
                                                                               ctx.stride, ctx.dilation, ctx.groups,
                                                                               torch.backends.cudnn.benchmark,
                                                                               torch.backends.cudnn.deterministic,
                                                                               torch.backends.cudnn.allow_tf32)

        if ctx.needs_input_grad[2]:
            # grad_output.shape = [N, C, *]
            out_channels = grad_output.shape[1]
            grad_bias = grad_output.transpose(0, 1).reshape(out_channels, -1).sum(1)
        return grad_spike, grad_weight, grad_bias, None, None, None, None

class spikeLinear(torch.autograd.Function):
    @staticmethod
    @custom_fwd
    def forward(ctx, spike, weight, bias=None):
        # spike.shape = [N, *, in_features]
        # weight.shape = [out_features, in_features]
        # bias.shape = [out_features]
        if ctx.needs_input_grad[0] or ctx.needs_input_grad[1] or ctx.needs_input_grad[2]:
            if ctx.needs_input_grad[1]:
                ctx.s_shape = spike.shape
                ctx.s_tk = tensor_cache.BOOL_TENSOR_CACHE.store_bool(spike)
            if ctx.needs_input_grad[1]:
                ctx.save_for_backward(weight)
        return F.linear(spike, weight, bias)

    @staticmethod
    @custom_bwd
    def backward(ctx, grad_output):
        # grad_output.shape = [N, *, out_features]
        if ctx.needs_input_grad[1]:
            weight = ctx.saved_tensors[0]
        if ctx.needs_input_grad[0]:
            spike = tensor_cache.BOOL_TENSOR_CACHE.get_float(ctx.s_tk, ctx.s_shape)

        grad_spike = grad_weight = grad_bias = None

        if ctx.needs_input_grad[0]:
            grad_spike = F.linear(grad_output, weight.t(), bias=None)
        if ctx.needs_input_grad[1]:
            in_features = spike.shape[-1]
            out_features = grad_output.shape[-1]
            # grad_output.reshape(-1, out_features).t().shape = [out_features, N*]
            # spike.reshape(-1, in_features).shape = [N*, in_features]
            grad_weight = torch.mm(grad_output.reshape(-1, out_features).t(), spike.reshape(-1, in_features).to(grad_output.dtype))
        if ctx.needs_input_grad[2]:
            out_features = grad_output.shape[-1]
            grad_bias = grad_output.reshape(-1, out_features).sum(0)
        return grad_spike, grad_weight, grad_bias

def spike_linear(spike: Tensor, weight: Tensor, bias: Optional[Tensor] = None) -> Tensor:
    """
    * :ref:`API in English <spike_linear-en>`

    .. _spike_linear-cn:

    :class:`torch.nn.functional.linear` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.linear` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_linear-cn>`

    .. _spike_linear-en:

    A specific case of :class:`torch.nn.functional.linear` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.linear` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.linear(spike, weight, bias)
    else:
        return spikeLinear.apply(spike, weight, bias)

def spike_conv1d(spike: Tensor, weight: Tensor, bias: Tensor=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor:
    """
    * :ref:`API in English <spike_conv1d-en>`

    .. _spike_conv1d-cn:

    :class:`torch.nn.functional.conv1d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv1d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv1d-cn>`

    .. _spike_conv1d-en:

    A specific case of :class:`torch.nn.functional.conv1d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv1d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv1d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)

def spike_conv2d(spike: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor:
    """
    * :ref:`API in English <spike_conv2d-en>`

    .. _spike_conv2d-cn:

    :class:`torch.nn.functional.conv2d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv2d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv2d-cn>`

    .. _spike_conv2d-en:

    A specific case of :class:`torch.nn.functional.conv2d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv2d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv2d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)

def spike_conv3d(spike: Tensor, weight: Tensor, bias: Optional[Tensor]=None, stride: Union[_int, _size]=1, padding: str="valid", dilation: Union[_int, _size]=1, groups: _int=1) -> Tensor:
    """
    * :ref:`API in English <spike_conv3d-en>`

    .. _spike_conv3d-cn:

    :class:`torch.nn.functional.conv3d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上训练时拥有比 :class:`torch.nn.functional.conv3d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <spike_conv3d-cn>`

    .. _spike_conv3d-en:

    A specific case of :class:`torch.nn.functional.conv3d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.functional.conv3d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """
    if spike.get_device() < 0:
        return F.conv3d(spike, weight, bias, stride, padding, dilation, groups)
    else:
        return spikeConvolution.apply(spike, weight, bias, stride, padding, dilation, groups)


class SpikeLinear(nn.Linear):
    """
    * :ref:`API in English <SpikeLinear-en>`

    .. _SpikeLinear-cn:

    :class:`torch.nn.Linear` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Linear` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeLinear-cn>`

    .. _SpikeLinear-en:

    A specific case of :class:`torch.nn.Linear` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Linear` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def forward(self, spike: Tensor) -> Tensor:
        return spike_linear(spike, self.weight, self.bias)


class SpikeConv1d(nn.Conv1d):
    """
    * :ref:`API in English <SpikeConv1d-en>`

    .. _SpikeConv1d-cn:

    :class:`torch.nn.Conv1d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv1d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv1d-cn>`

    .. _SpikeConv1d-en:

    A specific case of :class:`torch.nn.Conv1d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv1d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return spike_conv1d(F.pad(spike, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                           weight, bias, self.stride,
                                           _single(0), self.dilation, self.groups)
        return spike_conv1d(spike, weight, bias, self.stride,
                                       self.padding, self.dilation, self.groups)


class SpikeConv2d(nn.Conv2d):
    """
    * :ref:`API in English <SpikeConv2d-en>`

    .. _SpikeConv2d-cn:

    :class:`torch.nn.Conv2d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv2d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv2d-cn>`

    .. _SpikeConv2d-en:

    A specific case of :class:`torch.nn.Conv2d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv2d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != 'zeros':
            return spike_conv2d(F.pad(spike, self._reversed_padding_repeated_twice, mode=self.padding_mode),
                                           weight, bias, self.stride,
                                           _pair(0), self.dilation, self.groups)
        return spike_conv2d(spike, weight, bias, self.stride,
                                       self.padding, self.dilation, self.groups)


class SpikeConv3d(nn.Conv3d):
    """
    * :ref:`API in English <SpikeConv3d-en>`

    .. _SpikeConv3d-cn:

    :class:`torch.nn.Conv3d` 在输入为脉冲时的特例。

    .. note::

        在CUDA设备上运行时拥有比 :class:`torch.nn.Conv3d` 更低的显存消耗。

    .. warning::

        `spike` 中的任何元素都必须为0或1。

    * :ref:`中文API <SpikeConv3d-cn>`

    .. _SpikeConv3d-en:

    A specific case of :class:`torch.nn.Conv3d` with inputs are spikes.

    .. admonition:: Note
        :class: note

        This function has less memory consumption than :class:`torch.nn.Conv3d` when training on CUDA devices.

    .. admonition:: Warning
        :class: warning

        Any element in `spike` must be 0 or 1.
    """

    def _conv_forward(self, spike: Tensor, weight: Tensor, bias: Optional[Tensor]):
        if self.padding_mode != "zeros":
            return spike_conv3d(
                F.pad(
                    spike, self._reversed_padding_repeated_twice, mode=self.padding_mode
                ),
                weight,
                bias,
                self.stride,
                _triple(0),
                self.dilation,
                self.groups,
            )
        return spike_conv3d(
            spike, weight, bias, self.stride, self.padding, self.dilation, self.groups
        )
