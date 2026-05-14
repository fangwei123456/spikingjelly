from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter
from .utils import _prod, _spike_nnz

aten = torch.ops.aten

__all__ = ["NeuroMCAddCounter"]


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


def _add_mm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                out_counts = torch.ops.aten.mm.default(x.double(), y.double())
                add = torch.clamp(out_counts - 1.0, min=0.0).sum()
        return int(add.item())
    if nnz_x is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                row_nnz = x.double().sum(dim=1)
                add = torch.clamp(row_nnz - 1.0, min=0.0).sum()
        return int(add.item() * y.shape[1])
    if nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                col_nnz = y.double().sum(dim=0)
                add = torch.clamp(col_nnz - 1.0, min=0.0).sum()
        return int(add.item() * x.shape[0])
    m, k = x.shape
    _, n = y.shape
    return int(m * n * max(k - 1, 0))


def _add_addmm(args, kwargs, out):
    _, x, y = args[:3]
    add = _add_mm((x, y), kwargs, out)
    beta = kwargs.get("beta", 1)
    if beta != 0:
        add += out.numel()
    return int(add)


def _add_bmm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                out_counts = torch.ops.aten.bmm.default(x.double(), y.double())
                add = torch.clamp(out_counts - 1.0, min=0.0).sum()
        return int(add.item())
    if nnz_x is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                row_nnz = x.double().sum(dim=2)
                add = torch.clamp(row_nnz - 1.0, min=0.0).sum()
        return int(add.item() * y.shape[2])
    if nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                col_nnz = y.double().sum(dim=1)
                add = torch.clamp(col_nnz - 1.0, min=0.0).sum()
        return int(add.item() * x.shape[1])
    b, m, k = x.shape
    _, _, n = y.shape
    return int(b * m * n * max(k - 1, 0))


def _add_baddbmm(args, kwargs, out):
    _, x, y = args[:3]
    add = _add_bmm((x, y), kwargs, out)
    beta = kwargs.get("beta", 1)
    if beta != 0:
        add += out.numel()
    return int(add)


def _add_convolution(args, kwargs, out):
    x, w, _, stride, padding, dilation, transposed, output_padding, groups = args[:9]
    nnz_x = _spike_nnz(x)
    nnz_w = _spike_nnz(w)
    if nnz_x is not None and nnz_w is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                result = torch.ops.aten.convolution.default(
                    x.double(),
                    w.double(),
                    None,
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups,
                )
        return int(torch.clamp(result - 1.0, min=0.0).sum().item())
    if nnz_x is not None:
        w_ones = torch.ones(w.shape, dtype=torch.float64, device=x.device)
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                result = torch.ops.aten.convolution.default(
                    x.double(),
                    w_ones,
                    None,
                    stride,
                    padding,
                    dilation,
                    transposed,
                    output_padding,
                    groups,
                )
        return int(torch.clamp(result - 1.0, min=0.0).sum().item())
    if nnz_w is not None:
        ref = x if transposed else out
        return int(nnz_w * ref.shape[0] * _prod(ref.shape[2:]))
    _, add = _conv_mul_add(args, out)
    return add


def _add_convolution_backward(args, kwargs, out):
    grad_out, x, w, _bias_sizes = args[:4]
    transposed, groups, output_mask = args[7], args[9], args[10]
    add = 0
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
        _, a = _conv_mul_add(pseudo_args, grad_x)
        add += a
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
        _, a = _conv_mul_add(pseudo_args, grad_weight)
        add += a
    if output_mask[2] and isinstance(out, (tuple, list)) and torch.is_tensor(out[2]):
        b = grad_out.shape[0]
        c_out = grad_out.shape[1]
        add += c_out * (b * _prod(grad_out.shape[2:]) - 1)
    return int(add)


def _add_element_wise(args, kwargs, out):
    return int(out.numel())


def _add_sum(args, kwargs, out):
    x = args[0]
    return int(max(x.numel() - out.numel(), 0))


def _add_mean(args, kwargs, out):
    x = args[0]
    return int(max(x.numel() - out.numel(), 0))


def _add_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    n, c = x.numel(), x.shape[1]
    has_running_stats = args[3] is not None
    add = 0
    if train:
        add += n - c
        add += n + c
        add += n
        add += c
        if has_running_stats:
            add += 2 * c
    else:
        add += n + c
    return int(add)


def _add_native_batch_norm_backward(args, kwargs, out):
    grad_output = args[0]
    gamma = args[2]
    n = grad_output.numel()
    c = int(gamma.numel()) if torch.is_tensor(gamma) else int(grad_output.shape[1])
    train = args[-3]
    output_mask = args[-1]
    add = 0
    if train:
        if output_mask[0]:
            add += (n - c) + (2 * n - c)
        if output_mask[1]:
            add += n - c
        if output_mask[2]:
            add += n - c
    else:
        if output_mask[1]:
            add += n - c
        if output_mask[2]:
            add += n - c
    return int(add)


class NeuroMCAddCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        if extra_rules is None:
            extra_rules = {}
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.mm.default: _add_mm,
            aten.addmm.default: _add_addmm,
            aten.bmm.default: _add_bmm,
            aten.baddbmm.default: _add_baddbmm,
            aten.convolution.default: _add_convolution,
            aten.convolution_backward.default: _add_convolution_backward,
            aten.native_batch_norm.default: _add_native_batch_norm,
            aten.native_batch_norm_backward.default: _add_native_batch_norm_backward,
            aten.add.Tensor: _add_element_wise,
            aten.add_.Tensor: _add_element_wise,
            aten.add.Scalar: _add_element_wise,
            aten.add_.Scalar: _add_element_wise,
            aten.sub.Tensor: _add_element_wise,
            aten.sub_.Tensor: _add_element_wise,
            aten.sub.Scalar: _add_element_wise,
            aten.sub_.Scalar: _add_element_wise,
            aten.rsub.Tensor: _add_element_wise,
            aten.rsub.Scalar: _add_element_wise,
            aten.sum.default: _add_sum,
            aten.sum.dim_IntList: _add_sum,
            aten.mean.dim: _add_mean,
        }
        self.rules.update(extra_rules)
