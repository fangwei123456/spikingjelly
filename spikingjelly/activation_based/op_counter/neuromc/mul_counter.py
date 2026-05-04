from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter
from .utils import _is_spike, _prod

aten = torch.ops.aten

__all__ = ["NeuroMCMulCounter"]


def _conv_mul_add(args, out):
    x, w, bias = args[:3]
    groups = args[8] if len(args) > 8 else 1
    c_in_per_group = x.shape[1] // groups
    kernel_prod = _prod(w.shape[2:])
    mul_per_out = c_in_per_group * kernel_prod
    out_numel = out.numel()
    mul = out_numel * mul_per_out
    add = out_numel * max(mul_per_out - 1, 0)
    if bias is not None:
        add += out_numel
    return int(mul), int(add)


def _mul_mm(args, kwargs, out):
    x, y = args[:2]
    if _is_spike(x) or _is_spike(y):
        return 0
    m, k = x.shape
    _, n = y.shape
    return int(m * n * k)


def _mul_addmm(args, kwargs, out):
    _, x, y = args[:3]
    mul = _mul_mm((x, y), kwargs, out)
    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)
    if alpha != 1:
        mul += out.numel()
    if beta not in (0, 1):
        bias = args[0]
        if torch.is_tensor(bias):
            mul += bias.numel()
    return int(mul)


def _mul_bmm(args, kwargs, out):
    x, y = args[:2]
    if _is_spike(x) or _is_spike(y):
        return 0
    b, m, k = x.shape
    _, _, n = y.shape
    return int(b * m * n * k)


def _mul_baddbmm(args, kwargs, out):
    _, x, y = args[:3]
    mul = _mul_bmm((x, y), kwargs, out)
    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)
    if alpha != 1:
        mul += out.numel()
    if beta not in (0, 1):
        bias = args[0]
        if torch.is_tensor(bias):
            mul += bias.numel()
    return int(mul)


def _mul_convolution(args, kwargs, out):
    x, w = args[:2]
    if _is_spike(x) or _is_spike(w):
        return 0
    mul, _ = _conv_mul_add(args, out)
    return mul


def _mul_convolution_backward(args, kwargs, out):
    grad_out, x, w = args[:3]
    transposed, groups, output_mask = args[7], args[9], args[10]
    mul = 0
    if output_mask[0]:
        grad_x = out[0]
        pseudo_args = [
            grad_out,
            w,
            None,
            None,
            None,
            None,
            (not transposed),
            None,
            groups,
        ]
        m, _ = _conv_mul_add(pseudo_args, grad_x)
        mul += m
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
        pseudo_args = [pseudo_x, pseudo_w, None, None, None, None, False, None, groups]
        m, _ = _conv_mul_add(pseudo_args, grad_weight)
        mul += m
    return int(mul)


def _mul_element_wise(args, kwargs, out):
    return int(out.numel())


def _mul_add_sub_tensor(args, kwargs, out):
    alpha = kwargs.get("alpha", 1)
    if alpha == 1:
        return 0
    return int(out.numel())


def _mul_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    n, c = x.numel(), x.shape[1]
    has_affine = args[1] is not None
    has_running_stats = args[3] is not None
    mul = 0
    if train:
        mul += n
        mul += c
        mul += n
        if has_affine:
            mul += n
        if has_running_stats:
            mul += 2 * c
    else:
        mul += n
        if has_affine:
            mul += n
    return int(mul)


def _mul_native_batch_norm_backward(args, kwargs, out):
    grad_output = args[0]
    n = grad_output.numel()
    train = args[-3]
    output_mask = args[-1]
    mul = 0
    if train:
        if output_mask[0]:
            mul += 5 * n
        if output_mask[1]:
            mul += n
    else:
        if output_mask[0]:
            mul += 2 * n
        if output_mask[1]:
            mul += n
    return int(mul)


class NeuroMCMulCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.mm.default: _mul_mm,
            aten.addmm.default: _mul_addmm,
            aten.bmm.default: _mul_bmm,
            aten.baddbmm.default: _mul_baddbmm,
            aten.convolution.default: _mul_convolution,
            aten.convolution_backward.default: _mul_convolution_backward,
            aten.native_batch_norm.default: _mul_native_batch_norm,
            aten.native_batch_norm_backward.default: _mul_native_batch_norm_backward,
            aten.mul.Tensor: _mul_element_wise,
            aten.mul_.Tensor: _mul_element_wise,
            aten.mul.Scalar: _mul_element_wise,
            aten.mul_.Scalar: _mul_element_wise,
            aten.div.Tensor: _mul_element_wise,
            aten.div_.Tensor: _mul_element_wise,
            aten.div.Scalar: _mul_element_wise,
            aten.div_.Scalar: _mul_element_wise,
            aten.add.Tensor: _mul_add_sub_tensor,
            aten.add_.Tensor: _mul_add_sub_tensor,
            aten.add.Scalar: _mul_add_sub_tensor,
            aten.add_.Scalar: _mul_add_sub_tensor,
            aten.sub.Tensor: _mul_add_sub_tensor,
            aten.sub_.Tensor: _mul_add_sub_tensor,
            aten.sub.Scalar: _mul_add_sub_tensor,
            aten.sub_.Scalar: _mul_add_sub_tensor,
            aten.rsub.Tensor: _mul_add_sub_tensor,
            aten.rsub.Scalar: _mul_add_sub_tensor,
        }
        self.rules.update(extra_rules)
