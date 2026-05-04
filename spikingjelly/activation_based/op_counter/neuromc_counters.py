from __future__ import annotations

from collections import OrderedDict, defaultdict
from dataclasses import dataclass, field
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._pytree import tree_flatten

from .base import BaseCounter

aten = torch.ops.aten
__all__ = [
    "MemoryHierarchyConfig",
    "NeuroMCBaseCounter",
    "NeuroMCMulCounter",
    "NeuroMCAddCounter",
    "NeuroMCCmpCounter",
    "NeuroMCSqrtCounter",
    "NeuroMCMuxCounter",
    "NeuroMCMemoryTrafficCounter",
    "NeuroMCMemoryResidencyCounter",
]


_DEFAULT_MEMORY_COST_PJ_PER_BIT = {
    "reg": 0.00901,
    "sram": 92.68282828 / 256.0,
    "dram": 1300.0 / 64.0,
}

_DEFAULT_CAPACITY_BITS = {
    "reg": float(256 * 1024 * 8),
    "sram": float(38 * 1024 * 1024 * 8),
    "dram": float("inf"),
}


@dataclass
class MemoryHierarchyConfig:
    memory_model: str = "residency"
    level_order: tuple[str, str, str] = ("reg", "sram", "dram")
    capacity_bits: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_CAPACITY_BITS)
    )
    energy_pj_per_bit: dict[str, float] = field(
        default_factory=lambda: dict(_DEFAULT_MEMORY_COST_PJ_PER_BIT)
    )
    eviction_policy: str = "LRU"

    @classmethod
    def neuromc_like_v1(cls, memory_model: str = "residency"):
        return cls(
            memory_model=memory_model,
            level_order=("reg", "sram", "dram"),
            capacity_bits=dict(_DEFAULT_CAPACITY_BITS),
            energy_pj_per_bit=dict(_DEFAULT_MEMORY_COST_PJ_PER_BIT),
            eviction_policy="LRU",
        )

    def copy(self):
        return MemoryHierarchyConfig(
            memory_model=self.memory_model,
            level_order=tuple(self.level_order),
            capacity_bits=dict(self.capacity_bits),
            energy_pj_per_bit=dict(self.energy_pj_per_bit),
            eviction_policy=self.eviction_policy,
        )

    def validate(self):
        if self.memory_model not in ("residency", "weighted"):
            raise ValueError(
                f"memory_model must be 'residency' or 'weighted', got {self.memory_model}."
            )
        if self.level_order != ("reg", "sram", "dram"):
            raise ValueError("level_order must be ('reg', 'sram', 'dram').")
        if self.eviction_policy != "LRU":
            raise ValueError("Only LRU eviction_policy is supported in v1.")


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


def _add_convolution_backward(args, kwargs, out):
    grad_out, x, w, bias = args[:4]
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


def _extract_tensors(tree: Any) -> list[torch.Tensor]:
    flat, _ = tree_flatten(tree)
    return [x for x in flat if torch.is_tensor(x)]


def _access_mm(args, kwargs, out):
    x, y = args[:2]
    return [x, y], [out]


def _access_addmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    reads = [x, y]
    if beta != 0:
        reads.append(bias)
    return reads, [out]


def _access_bmm(args, kwargs, out):
    x, y = args[:2]
    return [x, y], [out]


def _access_baddbmm(args, kwargs, out):
    bias, x, y = args[:3]
    beta = kwargs.get("beta", 1.0)
    reads = [x, y]
    if beta != 0:
        reads.append(bias)
    return reads, [out]


def _access_convolution(args, kwargs, out):
    x, w, bias = args[:3]
    reads = [x, w]
    if bias is not None:
        reads.append(bias)
    return reads, [out]


def _access_convolution_backward(args, kwargs, out):
    grad_out, x, w, _bias, *_rest, output_mask = args
    reads = [grad_out]
    writes: list[torch.Tensor] = []

    if not isinstance(out, (tuple, list)) or len(out) < 3:
        return reads, _extract_tensors(out)

    if output_mask[0]:
        reads.append(w)
        if torch.is_tensor(out[0]):
            writes.append(out[0])
    if output_mask[1]:
        reads.append(x)
        if torch.is_tensor(out[1]):
            writes.append(out[1])
    if output_mask[2]:
        if torch.is_tensor(out[2]):
            writes.append(out[2])
    return reads, writes


def _access_native_batch_norm(args, kwargs, out):
    x, gamma, beta, mean, var, train = args[:6]
    reads = [x, gamma, beta, mean, var]
    writes = [out[0]]
    if train:
        writes.extend([out[1], out[2]])
    return reads, writes


def _access_native_batch_norm_backward(args, kwargs, out):
    reads = [x for x in args[:7] if torch.is_tensor(x)]
    writes = _extract_tensors(out)
    return reads, writes


def _access_binary(args, kwargs, out):
    reads = [x for x in args[:2] if torch.is_tensor(x)]
    return reads, [out]


def _access_unary(args, kwargs, out):
    x = args[0]
    return [x], [out]


def _access_sum_mean(args, kwargs, out):
    x = args[0]
    return [x], [out]


def _access_where(args, kwargs, out):
    cond, x, y = args[:3]
    return [cond, x, y], [out]


_RESIDENCY_ACCESS_RULES: dict[Any, Callable] = {
    aten.mm.default: _access_mm,
    aten.addmm.default: _access_addmm,
    aten.bmm.default: _access_bmm,
    aten.baddbmm.default: _access_baddbmm,
    aten.convolution.default: _access_convolution,
    aten.convolution_backward.default: _access_convolution_backward,
    aten.native_batch_norm.default: _access_native_batch_norm,
    aten.native_batch_norm_backward.default: _access_native_batch_norm_backward,
    aten.add.Tensor: _access_binary,
    aten.add_.Tensor: _access_binary,
    aten.add.Scalar: _access_binary,
    aten.add_.Scalar: _access_binary,
    aten.sub.Tensor: _access_binary,
    aten.sub_.Tensor: _access_binary,
    aten.sub.Scalar: _access_binary,
    aten.sub_.Scalar: _access_binary,
    aten.rsub.Tensor: _access_binary,
    aten.rsub.Scalar: _access_binary,
    aten.mul.Tensor: _access_binary,
    aten.mul_.Tensor: _access_binary,
    aten.mul.Scalar: _access_binary,
    aten.mul_.Scalar: _access_binary,
    aten.div.Tensor: _access_binary,
    aten.div_.Tensor: _access_binary,
    aten.div.Scalar: _access_binary,
    aten.div_.Scalar: _access_binary,
    aten.sqrt.default: _access_unary,
    aten.sqrt_.default: _access_unary,
    aten.rsqrt.default: _access_unary,
    aten.sum.default: _access_sum_mean,
    aten.sum.dim_IntList: _access_sum_mean,
    aten.mean.dim: _access_sum_mean,
    aten.where.self: _access_where,
    aten.where.ScalarOther: _access_where,
    aten.where.ScalarSelf: _access_where,
}


class MemoryResidencySimulator:
    def __init__(self, config: MemoryHierarchyConfig):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.__init__-cn>` | :ref:`English <MemoryResidencySimulator.__init__-en>`

        ----

        .. _MemoryResidencySimulator.__init__-cn:

        * **中文**

        NeuroMC runtime 访存驻留模拟器。
        该模拟器维护 ``reg/sram/dram`` 三级层级，在每次读写事件发生时更新缓存驻留状态，并记录：

        - 各层级读/写/总 bit 数；
        - 分算子的层级读/写/总 bit 数；
        - 跨层搬移边（如 ``dram->sram``）的 bit 数。

        当前版本采用 LRU 淘汰策略，且层级顺序固定为 ``("reg", "sram", "dram")``。

        :param config: 驻留模拟配置
        :type config: MemoryHierarchyConfig

        ----

        .. _MemoryResidencySimulator.__init__-en:

        * **English**

        Runtime memory residency simulator for NeuroMC-style energy profiling.
        It tracks a 3-level hierarchy (``reg/sram/dram``), updates residency
        state on each read/write event, and records:

        - read/write/total bits per level;
        - read/write/total bits per op and level;
        - inter-level transfer bits by edges (e.g., ``dram->sram``).

        The current version uses LRU eviction and a fixed level order
        ``("reg", "sram", "dram")``.

        :param config: residency simulation config
        :type config: MemoryHierarchyConfig
        """
        if config.memory_model != "residency":
            raise ValueError(
                f"MemoryResidencySimulator requires memory_model='residency', got {config.memory_model}."
            )
        if config.eviction_policy != "LRU":
            raise ValueError("Only LRU eviction_policy is supported.")
        if config.level_order != ("reg", "sram", "dram"):
            raise ValueError("level_order must be ('reg', 'sram', 'dram').")

        capacity = config.capacity_bits or {}
        self.capacity_bits = {
            "reg": float(capacity.get("reg", float("inf"))),
            "sram": float(capacity.get("sram", float("inf"))),
            "dram": float(capacity.get("dram", float("inf"))),
        }
        self.reg_cache: OrderedDict[str, int] = OrderedDict()
        self.sram_cache: OrderedDict[str, int] = OrderedDict()
        self.usage_bits = {"reg": 0, "sram": 0}
        self.level_rw_bits: dict[str, dict[str, int]] = defaultdict(
            lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
        )
        self.op_level_rw_bits: dict[str, dict[str, dict[str, int]]] = defaultdict(
            lambda: defaultdict(
                lambda: {"read_bits": 0, "write_bits": 0, "total_bits": 0}
            )
        )
        self.move_bits_by_edge: dict[str, int] = defaultdict(int)
        self.move_bits_by_op: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )

    def _record_level_rw(self, level: str, rw: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._record_level_rw-cn>` | :ref:`English <MemoryResidencySimulator._record_level_rw-en>`

        ----

        .. _MemoryResidencySimulator._record_level_rw-cn:

        * **中文**

        在内部统计表中累计一次层级读/写 bit 事件。

        :param level: 访存层级名，如 ``reg/sram/dram``
        :type level: str

        :param rw: 读写字段名，通常为 ``read_bits`` 或 ``write_bits``
        :type rw: str

        :param bits: 本次事件 bit 数
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator._record_level_rw-en:

        * **English**

        Accumulate one read/write-bit event into internal per-level and
        per-op-per-level counters.

        :param level: memory level name, e.g. ``reg/sram/dram``
        :type level: str

        :param rw: read/write key, usually ``read_bits`` or ``write_bits``
        :type rw: str

        :param bits: event size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str
        """
        if bits <= 0:
            return
        self.level_rw_bits[level][rw] += bits
        self.level_rw_bits[level]["total_bits"] += bits
        self.op_level_rw_bits[op_name][level][rw] += bits
        self.op_level_rw_bits[op_name][level]["total_bits"] += bits

    def _record_move(self, src: str, dst: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._record_move-cn>` | :ref:`English <MemoryResidencySimulator._record_move-en>`

        ----

        .. _MemoryResidencySimulator._record_move-cn:

        * **中文**

        记录一次跨层搬移事件（如 ``sram->reg``），并同步折算为源层读与目标层写。

        :param src: 源层级
        :type src: str

        :param dst: 目标层级
        :type dst: str

        :param bits: 搬移 bit 数
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator._record_move-en:

        * **English**

        Record one inter-level movement event (e.g., ``sram->reg``), and
        reflect it as a read on source level and a write on destination level.

        :param src: source level
        :type src: str

        :param dst: destination level
        :type dst: str

        :param bits: transfer size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str
        """
        if bits <= 0:
            return
        edge = f"{src}->{dst}"
        self.move_bits_by_edge[edge] += bits
        self.move_bits_by_op[op_name][edge] += bits
        self._record_level_rw(src, "read_bits", bits, op_name)
        self._record_level_rw(dst, "write_bits", bits, op_name)

    def _touch(self, cache: OrderedDict[str, int], key: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._touch-cn>` | :ref:`English <MemoryResidencySimulator._touch-en>`

        ----

        .. _MemoryResidencySimulator._touch-cn:

        * **中文**

        将 LRU 容器中 ``key`` 标记为最新使用（移动到末尾）。

        :param cache: 目标 LRU 容器
        :type cache: OrderedDict[str, int]

        :param key: 张量键
        :type key: str

        ----

        .. _MemoryResidencySimulator._touch-en:

        * **English**

        Mark ``key`` as most recently used in a LRU container.

        :param cache: target LRU container
        :type cache: OrderedDict[str, int]

        :param key: tensor key
        :type key: str
        """
        cache.move_to_end(key, last=True)

    def _touch_inclusive_levels(self, key: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._touch_inclusive_levels-cn>` | :ref:`English <MemoryResidencySimulator._touch_inclusive_levels-en>`

        ----

        .. _MemoryResidencySimulator._touch_inclusive_levels-cn:

        * **中文**

        对包含该 ``key`` 的层级做联合触摸：
        若 ``reg`` 命中，同时刷新 ``sram`` 的 LRU 顺序，避免热点数据在下层被过早淘汰。

        :param key: 张量键
        :type key: str

        ----

        .. _MemoryResidencySimulator._touch_inclusive_levels-en:

        * **English**

        Touch all inclusive levels containing ``key``.
        For a ``reg`` hit, this also refreshes the ``sram`` LRU order to avoid
        premature lower-level eviction of hot tensors.

        :param key: tensor key
        :type key: str
        """
        if key in self.reg_cache:
            self._touch(self.reg_cache, key)
        if key in self.sram_cache:
            self._touch(self.sram_cache, key)

    def _insert_sram(self, key: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._insert_sram-cn>` | :ref:`English <MemoryResidencySimulator._insert_sram-en>`

        ----

        .. _MemoryResidencySimulator._insert_sram-cn:

        * **中文**

        尝试将张量插入 SRAM 层，必要时按 LRU 淘汰旧条目，并记录 ``sram->dram`` 搬移。

        :param key: 张量键
        :type key: str

        :param bits: 张量大小（bit）
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        :return: 是否成功插入 SRAM
        :rtype: bool

        ----

        .. _MemoryResidencySimulator._insert_sram-en:

        * **English**

        Try to insert a tensor into SRAM. If needed, evict old entries using
        LRU and record ``sram->dram`` movement.

        :param key: tensor key
        :type key: str

        :param bits: tensor size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str

        :return: whether insertion into SRAM succeeds
        :rtype: bool
        """
        if bits > self.capacity_bits["sram"]:
            return False
        if key in self.sram_cache:
            old = self.sram_cache[key]
            if bits > old:
                self.usage_bits["sram"] += bits - old
                self.sram_cache[key] = bits
            self._touch(self.sram_cache, key)
            return True

        while self.sram_cache and (
            self.usage_bits["sram"] + bits > self.capacity_bits["sram"]
        ):
            evict_key, evict_bits = self.sram_cache.popitem(last=False)
            self.usage_bits["sram"] -= evict_bits
            self._record_move("sram", "dram", evict_bits, op_name)
            reg_bits = self.reg_cache.pop(evict_key, None)
            if reg_bits is not None:
                self.usage_bits["reg"] -= reg_bits

        if self.usage_bits["sram"] + bits > self.capacity_bits["sram"]:
            return False

        self.sram_cache[key] = bits
        self.usage_bits["sram"] += bits
        return True

    def _insert_reg(self, key: str, bits: int, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._insert_reg-cn>` | :ref:`English <MemoryResidencySimulator._insert_reg-en>`

        ----

        .. _MemoryResidencySimulator._insert_reg-cn:

        * **中文**

        尝试将张量插入寄存器层。容量不足时按 LRU 淘汰并下推到 SRAM，
        记录 ``reg->sram`` 搬移。

        :param key: 张量键
        :type key: str

        :param bits: 张量大小（bit）
        :type bits: int

        :param op_name: 对应算子名
        :type op_name: str

        :return: 是否成功插入 reg
        :rtype: bool

        ----

        .. _MemoryResidencySimulator._insert_reg-en:

        * **English**

        Try to insert a tensor into register level. On capacity pressure, evict
        by LRU, push evicted entries to SRAM, and record ``reg->sram`` movement.

        :param key: tensor key
        :type key: str

        :param bits: tensor size in bits
        :type bits: int

        :param op_name: operator name
        :type op_name: str

        :return: whether insertion into reg succeeds
        :rtype: bool
        """
        if bits > self.capacity_bits["reg"]:
            return False
        if key in self.reg_cache:
            old = self.reg_cache[key]
            if bits > old:
                self.usage_bits["reg"] += bits - old
                self.reg_cache[key] = bits
            self._touch(self.reg_cache, key)
            return True

        while self.reg_cache and (
            self.usage_bits["reg"] + bits > self.capacity_bits["reg"]
        ):
            evict_key, evict_bits = self.reg_cache.popitem(last=False)
            self.usage_bits["reg"] -= evict_bits
            self._record_move("reg", "sram", evict_bits, op_name)
            self._insert_sram(evict_key, evict_bits, op_name)

        if self.usage_bits["reg"] + bits > self.capacity_bits["reg"]:
            return False

        self.reg_cache[key] = bits
        self.usage_bits["reg"] += bits
        return True

    def _tensor_key(self, tensor: torch.Tensor) -> str:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator._tensor_key-cn>` | :ref:`English <MemoryResidencySimulator._tensor_key-en>`

        ----

        .. _MemoryResidencySimulator._tensor_key-cn:

        * **中文**

        为张量构造驻留模拟使用的键，包含设备、dtype、data_ptr、
        storage_offset 和 shape。

        :param tensor: 输入张量
        :type tensor: torch.Tensor

        :return: 张量键
        :rtype: str

        ----

        .. _MemoryResidencySimulator._tensor_key-en:

        * **English**

        Build a simulator key for tensor identity using device, dtype, data_ptr,
        storage_offset, and shape.

        :param tensor: input tensor
        :type tensor: torch.Tensor

        :return: tensor key
        :rtype: str
        """
        data_ptr = int(tensor.data_ptr())
        return (
            f"{tensor.device}:{tensor.dtype}:{data_ptr}:"
            f"{int(tensor.storage_offset())}:{tuple(tensor.shape)}"
        )

    def on_tensor_read(self, tensor: torch.Tensor, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.on_tensor_read-cn>` | :ref:`English <MemoryResidencySimulator.on_tensor_read-en>`

        ----

        .. _MemoryResidencySimulator.on_tensor_read-cn:

        * **中文**

        处理一次张量读事件：

        - 若命中 reg：记 reg 读；
        - 若命中 sram：记 sram 读，并尝试提升到 reg；
        - 若 miss：记 dram 读，并尝试 ``dram->sram->reg`` 或 ``dram->reg`` 分配路径。

        :param tensor: 被读取张量
        :type tensor: torch.Tensor

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator.on_tensor_read-en:

        * **English**

        Handle one tensor read event:

        - reg hit: record reg read;
        - sram hit: record sram read and try to promote to reg;
        - miss: record dram read, then try allocation path
          ``dram->sram->reg`` or ``dram->reg``.

        :param tensor: tensor being read
        :type tensor: torch.Tensor

        :param op_name: operator name
        :type op_name: str
        """
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        self._record_level_rw("dram", "read_bits", bits, op_name)
        sram_ok = self._insert_sram(key, bits, op_name)
        if sram_ok:
            self._record_move("dram", "sram", bits, op_name)
            self._record_level_rw("sram", "read_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "read_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_move("dram", "reg", bits, op_name)
            self._record_level_rw("reg", "read_bits", bits, op_name)

    def on_tensor_write(self, tensor: torch.Tensor, op_name: str):
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.on_tensor_write-cn>` | :ref:`English <MemoryResidencySimulator.on_tensor_write-en>`

        ----

        .. _MemoryResidencySimulator.on_tensor_write-cn:

        * **中文**

        处理一次张量写事件：

        - 若命中 reg：记 reg 写；
        - 若命中 sram：记 sram 写，并尝试提升到 reg；
        - 若 miss：优先分配到 reg，再尝试 sram，否则落到 dram 写。

        :param tensor: 被写入张量
        :type tensor: torch.Tensor

        :param op_name: 对应算子名
        :type op_name: str

        ----

        .. _MemoryResidencySimulator.on_tensor_write-en:

        * **English**

        Handle one tensor write event:

        - reg hit: record reg write;
        - sram hit: record sram write and try to promote to reg;
        - miss: allocate to reg first, then sram, otherwise record dram write.

        :param tensor: tensor being written
        :type tensor: torch.Tensor

        :param op_name: operator name
        :type op_name: str
        """
        bits = _tensor_bits(tensor)
        if bits <= 0:
            return
        key = self._tensor_key(tensor)

        if key in self.reg_cache:
            self._touch_inclusive_levels(key)
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if key in self.sram_cache:
            self._touch(self.sram_cache, key)
            self._record_level_rw("sram", "write_bits", bits, op_name)
            if self._insert_reg(key, bits, op_name):
                self._record_move("sram", "reg", bits, op_name)
                self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_reg(key, bits, op_name):
            self._record_level_rw("reg", "write_bits", bits, op_name)
            return

        if self._insert_sram(key, bits, op_name):
            self._record_level_rw("sram", "write_bits", bits, op_name)
            return

        self._record_level_rw("dram", "write_bits", bits, op_name)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_level_rw_bits-cn>` | :ref:`English <MemoryResidencySimulator.get_level_rw_bits-en>`

        ----

        .. _MemoryResidencySimulator.get_level_rw_bits-cn:

        * **中文**

        获取各层级读/写/总 bit 统计快照。

        :return: ``dict[level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, int]]

        ----

        .. _MemoryResidencySimulator.get_level_rw_bits-en:

        * **English**

        Get a snapshot of per-level read/write/total bit statistics.

        :return: ``dict[level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, int]]
        """
        return {k: dict(v) for k, v in self.level_rw_bits.items()}

    def get_op_level_rw_bits(self) -> dict[str, dict[str, dict[str, int]]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_op_level_rw_bits-cn>` | :ref:`English <MemoryResidencySimulator.get_op_level_rw_bits-en>`

        ----

        .. _MemoryResidencySimulator.get_op_level_rw_bits-cn:

        * **中文**

        获取分算子、分层级的读/写/总 bit 统计快照。

        :return: ``dict[op_name][level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, dict[str, int]]]

        ----

        .. _MemoryResidencySimulator.get_op_level_rw_bits-en:

        * **English**

        Get a snapshot of read/write/total bit statistics grouped by op and level.

        :return: ``dict[op_name][level][read_bits|write_bits|total_bits]``
        :rtype: dict[str, dict[str, dict[str, int]]]
        """
        return {
            op: {lv: dict(info) for lv, info in d.items()}
            for op, d in self.op_level_rw_bits.items()
        }

    def get_move_bits_by_edge(self) -> dict[str, int]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_move_bits_by_edge-cn>` | :ref:`English <MemoryResidencySimulator.get_move_bits_by_edge-en>`

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_edge-cn:

        * **中文**

        获取跨层搬移边统计（例如 ``dram->sram``）。

        :return: ``dict[edge] = moved_bits``
        :rtype: dict[str, int]

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_edge-en:

        * **English**

        Get inter-level transfer statistics by edge (e.g., ``dram->sram``).

        :return: ``dict[edge] = moved_bits``
        :rtype: dict[str, int]
        """
        return dict(self.move_bits_by_edge)

    def get_move_bits_by_op(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <MemoryResidencySimulator.get_move_bits_by_op-cn>` | :ref:`English <MemoryResidencySimulator.get_move_bits_by_op-en>`

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_op-cn:

        * **中文**

        获取分算子的跨层搬移统计。

        :return: ``dict[op_name][edge] = moved_bits``
        :rtype: dict[str, dict[str, int]]

        ----

        .. _MemoryResidencySimulator.get_move_bits_by_op-en:

        * **English**

        Get inter-level transfer statistics grouped by op name.

        :return: ``dict[op_name][edge] = moved_bits``
        :rtype: dict[str, dict[str, int]]
        """
        return {op: dict(d) for op, d in self.move_bits_by_op.items()}


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


class NeuroMCMemoryResidencyCounter(NeuroMCBaseCounter):
    def __init__(
        self,
        memory_config: MemoryHierarchyConfig | None = None,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = dict(_RESIDENCY_ACCESS_RULES)
        self.rules.update(extra_rules)
        if memory_config is None:
            memory_config = MemoryHierarchyConfig.neuromc_like_v1(
                memory_model="residency"
            )
        self.simulator = MemoryResidencySimulator(memory_config)
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
        stage = _infer_stage(func, args, kwargs, out)
        before_level_rw = self.simulator.get_level_rw_bits()
        read_tensors, write_tensors = self.rules[func](args, kwargs, out)

        total_bits = 0
        for tensor in read_tensors:
            bits = _tensor_bits(tensor)
            total_bits += bits
            self.simulator.on_tensor_read(tensor, op_name)
        for tensor in write_tensors:
            bits = _tensor_bits(tensor)
            total_bits += bits
            self.simulator.on_tensor_write(tensor, op_name)

        after_level_rw = self.simulator.get_level_rw_bits()
        all_levels = set(before_level_rw.keys()) | set(after_level_rw.keys())
        for level in all_levels:
            old_info = before_level_rw.get(level, {})
            new_info = after_level_rw.get(level, {})
            delta_read = int(
                new_info.get("read_bits", 0) - old_info.get("read_bits", 0)
            )
            delta_write = int(
                new_info.get("write_bits", 0) - old_info.get("write_bits", 0)
            )
            delta_total = int(
                new_info.get("total_bits", 0) - old_info.get("total_bits", 0)
            )
            if delta_total == 0 and delta_read == 0 and delta_write == 0:
                continue
            self.level_records[level] += delta_total
            self.level_rw_records[level]["read_bits"] += delta_read
            self.level_rw_records[level]["write_bits"] += delta_write
            self.level_rw_records[level]["total_bits"] += delta_total
            self.stage_level_records[stage][level] += delta_total
            self.op_level_records[op_name][level] += delta_total

        self.stage_records[stage] += total_bits
        self.op_records[op_name] += total_bits
        self.stage_op_records[stage][op_name] += total_bits
        return total_bits

    def get_level_bits(self) -> dict[str, int]:
        return dict(self.level_records)

    def get_level_rw_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.level_rw_records.items()}

    def get_op_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.op_level_records.items()}

    def get_stage_level_bits(self) -> dict[str, dict[str, int]]:
        return {k: dict(v) for k, v in self.stage_level_records.items()}

    def get_move_bits_by_edge(self) -> dict[str, int]:
        return self.simulator.get_move_bits_by_edge()

    def get_move_bits_by_op(self) -> dict[str, dict[str, int]]:
        return self.simulator.get_move_bits_by_op()
