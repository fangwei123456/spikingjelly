from typing import Optional

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from spikingjelly import configure

from ..autocast import _register_autocast
from ..spike_compress import _pack, _unpack


@torch.library.custom_op(
    "sj_spike_conv::forward", mutates_args=(), device_types=("cpu", "cuda")
)
def _convolution(
    spike: torch.Tensor,
    weight: torch.Tensor,
    bias: Optional[torch.Tensor],
    stride: list[int],
    padding: list[int],
    dilation: list[int],
    groups: int,
) -> torch.Tensor:
    if spike.ndim == 3:
        return F.conv1d(spike, weight, bias, stride, padding, dilation, groups)
    if spike.ndim == 4:
        return F.conv2d(spike, weight, bias, stride, padding, dilation, groups)
    if spike.ndim == 5:
        return F.conv3d(spike, weight, bias, stride, padding, dilation, groups)
    raise ValueError("spike convolution expects 3D/4D/5D input")


@torch.library.register_fake("sj_spike_conv::forward")
def _fake(spike, weight, bias, stride, padding, dilation, groups):
    if spike.ndim == 3:
        return F.conv1d(spike, weight, bias, stride, padding, dilation, groups)
    if spike.ndim == 4:
        return F.conv2d(spike, weight, bias, stride, padding, dilation, groups)
    if spike.ndim == 5:
        return F.conv3d(spike, weight, bias, stride, padding, dilation, groups)
    raise ValueError("spike convolution expects 3D/4D/5D input")


def _setup(ctx, inputs, output):
    spike, weight, bias, stride, padding, dilation, groups = inputs
    ctx.shape = spike.shape
    ctx.input_dtype = spike.dtype
    ctx.weight_dtype = weight.dtype
    ctx.bias_dtype = None if bias is None else bias.dtype
    ctx.stride, ctx.padding, ctx.dilation, ctx.groups = (
        stride,
        padding,
        dilation,
        groups,
    )
    ctx.packed = configure.save_bool_spike_level == 1
    if configure.save_bool_spike_level not in (0, 1):
        raise ValueError("save_bool_spike_level must be 0 or 1")
    saved = _pack(spike) if ctx.packed else spike.bool()
    ctx.save_for_backward(saved, weight)


@once_differentiable
def _backward(ctx, grad_output):
    saved, weight = ctx.saved_tensors
    spike = (
        _unpack(saved, ctx.shape, grad_output.dtype)
        if ctx.packed
        else saved.to(grad_output.dtype)
    )
    gx, gw, _ = torch.ops.aten.convolution_backward.default(
        grad_output,
        spike,
        weight.to(grad_output.dtype),
        None,
        ctx.stride,
        ctx.padding,
        ctx.dilation,
        False,
        [0] * len(ctx.stride),
        ctx.groups,
        [True, True, False],
    )
    gb = (
        grad_output.sum((0, *range(2, grad_output.ndim))).to(ctx.bias_dtype)
        if ctx.bias_dtype is not None
        else None
    )
    return gx.to(ctx.input_dtype), gw.to(ctx.weight_dtype), gb, None, None, None, None


torch.library.register_autograd(
    "sj_spike_conv::forward", _backward, setup_context=_setup
)
_autocast_libraries = _register_autocast("sj_spike_conv::forward")
