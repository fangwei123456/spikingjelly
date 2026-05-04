from __future__ import annotations

from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from .base import BaseCounter


aten = torch.ops.aten
__all__ = [
    "NeuroMCBaseCounter",
    "NeuroMCMulCounter",
    "NeuroMCAddCounter",
    "NeuroMCCmpCounter",
    "NeuroMCSqrtCounter",
    "NeuroMCMuxCounter",
    "NeuroMCMemoryTrafficCounter",
]


_OPTIMIZER_HINTS = (
    "add_.",
    "sub_.",
    "mul_.",
    "div_.",
    "addcmul",
    "addcdiv",
    "lerp_",
    "copy_",
)


def _prod(dims) -> int:
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _is_spike(x: torch.Tensor | None) -> bool:
    if x is None or (not torch.is_tensor(x)):
        return False
    if x.dtype == torch.bool:
        return True
    return bool(x.eq(0).logical_or_(x.eq(1)).all().item())


def _spike_nnz(x: torch.Tensor | None) -> int | None:
    if x is None or (not torch.is_tensor(x)):
        return None
    if x.dtype == torch.bool:
        return int(x.count_nonzero().item())
    is_binary = bool(x.eq(0).logical_or_(x.eq(1)).all().item())
    if not is_binary:
        return None
    return int(x.count_nonzero().item())


def _tensor_bits(x: Any) -> int:
    if not torch.is_tensor(x):
        return 0
    return x.numel() * x.element_size() * 8


def _collect_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _infer_stage(func, args, kwargs, out) -> str:
    op_name = resolve_name(func)
    if "backward" in op_name:
        return "backward"

    if torch.is_grad_enabled():
        return "forward"

    tensors = _collect_tensors((args, kwargs, out))
    has_grad_tensor = any(t.requires_grad for t in tensors)
    is_optimizer_like = any(hint in op_name for hint in _OPTIMIZER_HINTS)
    if has_grad_tensor and is_optimizer_like:
        return "optimizer"
    return "forward"


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


def _add_mm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        return int(out.sum().item())
    if nnz_x is not None:
        return nnz_x * y.shape[1]
    if nnz_y is not None:
        return nnz_y * x.shape[0]
    m, k = x.shape
    _, n = y.shape
    return int(m * n * max(k - 1, 0))


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


def _add_addmm(args, kwargs, out):
    _, x, y = args[:3]
    add = _add_mm((x, y), kwargs, out)
    beta = kwargs.get("beta", 1)
    if beta != 0:
        add += out.numel()
    return int(add)


def _mul_bmm(args, kwargs, out):
    x, y = args[:2]
    if _is_spike(x) or _is_spike(y):
        return 0
    b, m, k = x.shape
    _, _, n = y.shape
    return int(b * m * n * k)


def _add_bmm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        return int(out.sum().item())
    if nnz_x is not None:
        return nnz_x * y.shape[2]
    if nnz_y is not None:
        return nnz_y * x.shape[1]
    b, m, k = x.shape
    _, _, n = y.shape
    return int(b * m * n * max(k - 1, 0))


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


def _add_baddbmm(args, kwargs, out):
    _, x, y = args[:3]
    add = _add_bmm((x, y), kwargs, out)
    beta = kwargs.get("beta", 1)
    if beta != 0:
        add += out.numel()
    return int(add)


def _mul_convolution(args, kwargs, out):
    x, w = args[:2]
    if _is_spike(x) or _is_spike(w):
        return 0
    mul, _ = _conv_mul_add(args, out)
    return mul


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
        return int(result.sum().item())
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
        return int(result.sum().item())
    if nnz_w is not None:
        ref = x if transposed else out
        return int(nnz_w * ref.shape[0] * _prod(ref.shape[2:]))
    _, add = _conv_mul_add(args, out)
    return add


def _mul_convolution_backward(args, kwargs, out):
    grad_out, x, w = args[:3]
    transposed, groups, output_mask = args[7], args[9], args[10]
    mul = 0
    if output_mask[0]:
        grad_x = out[0]
        pseudo_args = [grad_out, w, None, None, None, None, (not transposed), None, groups]
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


def _add_convolution_backward(args, kwargs, out):
    grad_out, x, w, bias = args[:4]
    transposed, groups, output_mask = args[7], args[9], args[10]
    add = 0
    if output_mask[0]:
        grad_x = out[0]
        pseudo_args = [grad_out, w, None, None, None, None, (not transposed), None, groups]
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
    if output_mask[2] and bias is not None:
        b = grad_out.shape[0]
        c_out = grad_out.shape[1]
        add += c_out * (b * _prod(grad_out.shape[2:]) - 1)
    return int(add)


def _add_element_wise(args, kwargs, out):
    return int(out.numel())


def _add_sum(args, kwargs, out):
    x = args[0]
    return int(x.numel() - out.numel())


def _add_mean(args, kwargs, out):
    x = args[0]
    return int(max(x.numel() - out.numel(), 0))


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
        mul += n  # x^2 in variance path
        mul += c  # mean^2
        mul += n  # (x-mean)*invstd
        if has_affine:
            mul += n
        if has_running_stats:
            mul += 2 * c
    else:
        mul += n  # (x-mean)*invstd
        if has_affine:
            mul += n
    return int(mul)


def _add_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    n, c = x.numel(), x.shape[1]
    has_running_stats = args[3] is not None
    add = 0
    if train:
        add += n - c  # mean reduction
        add += n + c  # var = E[x^2] - E[x]^2
        add += n  # x - mean
        add += c  # var + eps
        if has_running_stats:
            add += 2 * c
    else:
        add += n + c  # x - mean, var + eps
    return int(add)


def _sqrt_native_batch_norm(args, kwargs, out):
    x = args[0]
    c = x.shape[1]
    return int(c)


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


def _cmp_element_wise(args, kwargs, out):
    return int(out.numel())


def _cmp_max_pool2d_with_indices(args, kwargs, out):
    kernel_size = args[1]
    y = out[0]
    return int(y.numel() * max(_prod(kernel_size) - 1, 0))


def _sqrt_op(args, kwargs, out):
    return int(out.numel())


def _mux_where(args, kwargs, out):
    return int(out.numel())


def _memory_binary(args, kwargs, out):
    x, y = args[:2]
    read_bits = _tensor_bits(x) + _tensor_bits(y)
    write_bits = _tensor_bits(out)
    return read_bits, write_bits


def _memory_unary(args, kwargs, out):
    x = args[0]
    return _tensor_bits(x), _tensor_bits(out)


def _memory_sum_or_mean(args, kwargs, out):
    x = args[0]
    return _tensor_bits(x), _tensor_bits(out)


def _memory_mm(args, kwargs, out):
    x, y = args[:2]
    return _tensor_bits(x) + _tensor_bits(y), _tensor_bits(out)


def _memory_addmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    read = _tensor_bits(x) + _tensor_bits(y)
    if beta != 0:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_bmm(args, kwargs, out):
    x, y = args[:2]
    return _tensor_bits(x) + _tensor_bits(y), _tensor_bits(out)


def _memory_baddbmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    read = _tensor_bits(x) + _tensor_bits(y)
    if beta != 0:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_convolution(args, kwargs, out):
    x, w, bias = args[:3]
    read = _tensor_bits(x) + _tensor_bits(w)
    if bias is not None:
        read += _tensor_bits(bias)
    return read, _tensor_bits(out)


def _memory_convolution_backward(args, kwargs, out):
    grad_out, x, w, _bias, *_rest, output_mask = args
    read = _tensor_bits(grad_out)
    write = 0
    if output_mask[0]:
        read += _tensor_bits(w)
        write += _tensor_bits(out[0])
    if output_mask[1]:
        read += _tensor_bits(x)
        write += _tensor_bits(out[1])
    if output_mask[2]:
        write += _tensor_bits(out[2])
    return read, write


def _memory_native_batch_norm(args, kwargs, out):
    x, gamma, beta, mean, var, train = args[:6]
    read = (
        _tensor_bits(x)
        + _tensor_bits(gamma)
        + _tensor_bits(beta)
        + _tensor_bits(mean)
        + _tensor_bits(var)
    )
    write = _tensor_bits(out[0])
    if train:
        write += _tensor_bits(out[1]) + _tensor_bits(out[2])
    return read, write


def _memory_native_batch_norm_backward(args, kwargs, out):
    read = sum(_tensor_bits(x) for x in args[:7])
    if isinstance(out, (tuple, list)):
        write = sum(_tensor_bits(x) for x in out if torch.is_tensor(x))
    else:
        write = _tensor_bits(out)
    return read, write


def _memory_where(args, kwargs, out):
    cond, x, y = args[:3]
    read = _tensor_bits(cond) + _tensor_bits(x) + _tensor_bits(y)
    write = _tensor_bits(out)
    return read, write


class NeuroMCBaseCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {}
        self.ignore_modules: list[nn.Module] = []
        self.ignore_modules.extend(extra_ignore_modules)
        self.stage_records: dict[str, int] = defaultdict(int)
        self.op_records: dict[str, int] = defaultdict(int)
        self.stage_op_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        op_name = resolve_name(func)
        value = int(self.rules[func](args, kwargs, out))
        stage = _infer_stage(func, args, kwargs, out)
        self.stage_records[stage] += value
        self.op_records[op_name] += value
        self.stage_op_records[stage][op_name] += value
        return value

    def get_stage_counts(self) -> dict[str, int]:
        return dict(self.stage_records)

    def get_op_counts(self) -> dict[str, int]:
        return dict(self.op_records)

    def get_stage_op_counts(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_op_records.items()}


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


class NeuroMCAddCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
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


class NeuroMCCmpCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.eq.Tensor: _cmp_element_wise,
            aten.eq.Scalar: _cmp_element_wise,
            aten.ne.Tensor: _cmp_element_wise,
            aten.ne.Scalar: _cmp_element_wise,
            aten.lt.Tensor: _cmp_element_wise,
            aten.lt.Scalar: _cmp_element_wise,
            aten.le.Tensor: _cmp_element_wise,
            aten.le.Scalar: _cmp_element_wise,
            aten.gt.Tensor: _cmp_element_wise,
            aten.gt.Scalar: _cmp_element_wise,
            aten.ge.Tensor: _cmp_element_wise,
            aten.ge.Scalar: _cmp_element_wise,
            aten.logical_and.default: _cmp_element_wise,
            aten.logical_or.default: _cmp_element_wise,
            aten.logical_xor.default: _cmp_element_wise,
            aten.logical_not.default: _cmp_element_wise,
            aten.max_pool2d_with_indices.default: _cmp_max_pool2d_with_indices,
        }
        self.rules.update(extra_rules)


class NeuroMCSqrtCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.sqrt.default: _sqrt_op,
            aten.sqrt_.default: _sqrt_op,
            aten.rsqrt.default: _sqrt_op,
            aten.native_batch_norm.default: _sqrt_native_batch_norm,
        }
        self.rules.update(extra_rules)


class NeuroMCMuxCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.where.self: _mux_where,
            aten.where.ScalarOther: _mux_where,
            aten.where.ScalarSelf: _mux_where,
        }
        self.rules.update(extra_rules)


class NeuroMCMemoryTrafficCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        level_weights: dict[str, float] | None = None,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.mm.default: _memory_mm,
            aten.addmm.default: _memory_addmm,
            aten.bmm.default: _memory_bmm,
            aten.baddbmm.default: _memory_baddbmm,
            aten.convolution.default: _memory_convolution,
            aten.convolution_backward.default: _memory_convolution_backward,
            aten.native_batch_norm.default: _memory_native_batch_norm,
            aten.native_batch_norm_backward.default: _memory_native_batch_norm_backward,
            aten.add.Tensor: _memory_binary,
            aten.add_.Tensor: _memory_binary,
            aten.add.Scalar: _memory_binary,
            aten.add_.Scalar: _memory_binary,
            aten.sub.Tensor: _memory_binary,
            aten.sub_.Tensor: _memory_binary,
            aten.sub.Scalar: _memory_binary,
            aten.sub_.Scalar: _memory_binary,
            aten.rsub.Tensor: _memory_binary,
            aten.rsub.Scalar: _memory_binary,
            aten.mul.Tensor: _memory_binary,
            aten.mul_.Tensor: _memory_binary,
            aten.mul.Scalar: _memory_binary,
            aten.mul_.Scalar: _memory_binary,
            aten.div.Tensor: _memory_binary,
            aten.div_.Tensor: _memory_binary,
            aten.div.Scalar: _memory_binary,
            aten.div_.Scalar: _memory_binary,
            aten.sqrt.default: _memory_unary,
            aten.sqrt_.default: _memory_unary,
            aten.rsqrt.default: _memory_unary,
            aten.sum.default: _memory_sum_or_mean,
            aten.sum.dim_IntList: _memory_sum_or_mean,
            aten.mean.dim: _memory_sum_or_mean,
            aten.where.self: _memory_where,
            aten.where.ScalarOther: _memory_where,
            aten.where.ScalarSelf: _memory_where,
        }
        self.rules.update(extra_rules)
        self.level_weights = level_weights or {"reg": 2.0, "sram": 1.0, "dram": 0.25}
        self.level_records: dict[str, int] = defaultdict(int)
        self.level_rw_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.stage_level_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.op_level_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def count(self, func, args: tuple, kwargs: dict, out) -> int:
        op_name = resolve_name(func)
        read_bits, write_bits = self.rules[func](args, kwargs, out)
        total_bits = int(read_bits + write_bits)
        stage = _infer_stage(func, args, kwargs, out)
        self.stage_records[stage] += total_bits
        self.op_records[op_name] += total_bits
        self.stage_op_records[stage][op_name] += total_bits

        for level, weight in self.level_weights.items():
            scaled_read = int(round(read_bits * weight))
            scaled_write = int(round(write_bits * weight))
            scaled_total = scaled_read + scaled_write
            self.level_records[level] += scaled_total
            self.level_rw_records[level]["read_bits"] += scaled_read
            self.level_rw_records[level]["write_bits"] += scaled_write
            self.level_rw_records[level]["total_bits"] += scaled_total
            self.stage_level_records[stage][level] += scaled_total
            self.op_level_records[op_name][level] += scaled_total
        return total_bits

    def get_level_bits(self) -> dict[str, int]:
        return dict(self.level_records)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.level_rw_records.items()}

    def get_stage_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_level_records.items()}

    def get_op_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.op_level_records.items()}
