"""
FLOP definition:

- 1 multiply = 1 FLOP
- 1 add = 1 FLOP
- element-wise ops are counted
"""
from collections import defaultdict
from typing import Any, Callable

import torch
aten = torch.ops.aten
import torch.nn as nn

from .base import BaseCounter


__all__ = ["FlopCounter"]

def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _flop_mm(args, kwargs, out):
    """out = x @ y"""
    x, y = args[:2]
    m, k = x.shape
    kk, n = y.shape
    if k != kk:
        raise AssertionError(f"mm: inner dimensions mismatch [{x.shape} and {y.shape}]")
    return m * n * (2 * k - 1)

def _flop_addmm(args, kwargs, out):
    """out = beta * b + alpha * (x @ y)"""
    b, x, y = args[:3]
    m, k = x.shape
    kk, n = y.shape
    if k != kk:
        raise AssertionError(f"addmm: inner dimensions mismatch [{x.shape} and {y.shape}]")

    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)

    flops = m * n * (2 * k - 1) # matmul; 2k-1 flops for each output element
    if alpha != 1:
        flops += m * n # scale by alpha
    if beta == 1:
        flops += m * n # add b to the m*n matrix
    elif beta != 0:
        flops += b.numel() + m * n # scale b by beta, and add it to the m*n matrix
    return flops

def _flop_bmm(args, kwargs, out):
    """Batch matrix multiply: out[b] = x[b] @ y[b]"""
    x, y = args[:2]
    b, m, l = x.shape
    bb, kk, n = y.shape
    if b != bb or l != kk:
        raise AssertionError(
            f"bmm: batch or inner dimensions mismatch [{x.shape} and {y.shape}]"
        )
    return b * m * n * (2 * l - 1)


def _flop_baddbmm(args, kwargs, out):
    """out[b] = beta * b[b] + alpha * (x[b] @ y[b])"""
    b, x, y = args[:3]
    b, m, k = x.shape
    bb, kk, n = y.shape
    if b != bb or k != kk:
        raise AssertionError(
            f"baddmm: batch or inner dimensions mismatch [{x.shape}, {y.shape}]"
        )

    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)

    flops = b * m * n * (2 * k - 1) # batched matmul
    if alpha != 1:
        flops += b * m * n # scale by alpha
    if beta == 1:
        flops += b * m * n # add b to the b*m*n matrix
    elif beta != 0:
        flops += b.numel() + b * m * n  # scale b, then add it to the b*m*n matrix
    return flops


def _flop_convolution(args, kwargs, out):
    """
    args[0]: input, shape [B, C_in, ...]
    args[1]: weight, shape [C_out, C_in, *kernel_shape]
    args[2]: bias or None
    """
    x, w, bias = args[:3]
    print(x.shape, w.shape, bias, out.shape)
    transposed = kwargs.get("transposed", False)

    b = x.shape[0]
    c_out, c_in, *kernel_shape = w.shape

    spatial_shape = x.shape[2:] if transposed else out.shape[2:]
    flops_per_position = 2 * c_in * _prod(kernel_shape)
    flops = flops_per_position * _prod(spatial_shape) * c_out * b
    flops -= out.numel() # for each output element, the first add can be avoided
    if bias is not None:
        flops += out.numel()
    return flops


def _flop_convolution_backward(args, kwargs, out):
    """
    Outputs (by output_mask):
        0: grad_input
        1: grad_weight
        2: grad_bias
    """
    (
        grad_out,
        x,
        w,
        bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
    ) = args
    flops = 0

    if output_mask[0]:
        grad_input = out[0]
        flops += _flop_convolution(
            [grad_out, w, bias], {"transposed": not transposed}, grad_input
        )

    if output_mask[1]:
        grad_weight = out[1]
        if transposed:
            pseudo_x = grad_out
            pseudo_w = x
        else:
            pseudo_x = x
            pseudo_w = grad_out
        pseudo_x = pseudo_x.transpose(0, 1)
        pseudo_w = pseudo_w.transpose(0, 1)

        flops += _flop_convolution(
            [pseudo_x, pseudo_w, None], {"transposed": False}, grad_weight
        )

    if output_mask[2] and bias is not None:
        B = grad_out.shape[0]
        C_out = grad_out.shape[1]
        spatial_shape = grad_out.shape[2:]
        flops += C_out * (B * _prod(spatial_shape) - 1)

    return flops


def _flop_add(args, kwargs, out):
    alpha = kwargs.get("alpha", 1.)
    if alpha == 1.:
        return out.numel()
    else:
        nb = args[1].numel() if torch.is_tensor(args[1]) else 1
        return nb + out.numel()


def _flop_element_wise(args, kwargs, out):
    return out.numel()


class FlopCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = []
    ):
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _flop_mm,
            aten.addmm.default: _flop_addmm,
            aten.bmm.default: _flop_bmm,
            aten.baddbmm.default: _flop_baddbmm,
            aten.convolution.default: _flop_convolution,
            aten.convolution_backward.default: _flop_convolution_backward,
            aten.add.Tensor: _flop_add,
            aten.add.Scalar: _flop_add,
            aten.sub.Tensor: _flop_add,
            aten.sub.Scalar: _flop_add,
            aten.neg.default: _flop_element_wise,
            aten.mul.Tensor: _flop_element_wise,
            aten.mul.Scalar: _flop_element_wise,
            aten.div.Tensor: _flop_element_wise,
            aten.div.Scalar: _flop_element_wise,
            aten.eq.Tensor: _flop_element_wise,
            aten.eq.Scalar: _flop_element_wise,
            aten.ne.Tensor: _flop_element_wise,
            aten.ne.Scalar: _flop_element_wise,
            aten.lt.Tensor: _flop_element_wise,
            aten.lt.Scalar: _flop_element_wise,
            aten.le.Tensor: _flop_element_wise,
            aten.le.Scalar: _flop_element_wise,
            aten.gt.Tensor: _flop_element_wise,
            aten.gt.Scalar: _flop_element_wise,
            aten.ge.Tensor: _flop_element_wise,
            aten.ge.Scalar: _flop_element_wise,
            aten.logical_and.default: _flop_element_wise,
            aten.logical_or.default: _flop_element_wise,
            aten.logical_xor.default: _flop_element_wise,
            aten.logical_not.default: _flop_element_wise,
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
