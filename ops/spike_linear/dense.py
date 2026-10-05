import math
from typing import Optional

import torch
from torch.autograd.function import once_differentiable
from torch.nn import functional as F

from spikingjelly import configure

from ..autocast import _register_autocast
from ..spike_compress import _pack, _unpack


@torch.library.custom_op(
    "sj_spike_linear::dense", mutates_args=(), device_types=("cpu", "cuda")
)
def _linear(
    spike: torch.Tensor, weight: torch.Tensor, bias: Optional[torch.Tensor]
) -> torch.Tensor:
    return F.linear(spike, weight, bias)


@torch.library.register_fake("sj_spike_linear::dense")
def _fake(spike, weight, bias):
    return F.linear(spike, weight, bias)


def _setup(ctx, inputs, output):
    spike, weight, bias = inputs
    ctx.shape = spike.shape
    ctx.input_dtype = spike.dtype
    ctx.weight_dtype = weight.dtype
    ctx.bias_dtype = None if bias is None else bias.dtype
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
    grad_spike = F.linear(grad_output, weight.to(grad_output.dtype).t()).to(
        ctx.input_dtype
    )
    batch = math.prod(ctx.shape[:-1])
    grad_weight = (
        grad_output.reshape(batch, weight.shape[0]).t()
        @ spike.reshape(batch, weight.shape[1])
    ).to(ctx.weight_dtype)
    grad_bias = (
        grad_output.reshape(batch, weight.shape[0]).sum(0).to(ctx.bias_dtype)
        if ctx.bias_dtype is not None
        else None
    )
    return grad_spike, grad_weight, grad_bias


torch.library.register_autograd(
    "sj_spike_linear::dense", _backward, setup_context=_setup
)
_autocast_libraries = _register_autocast("sj_spike_linear::dense")
