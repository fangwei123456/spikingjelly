from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter

aten = torch.ops.aten
__all__ = ["ACCounter"]


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _spike_nnz(x: torch.Tensor) -> int | None:
    """Return the number of non-zero elements if *x* is a binary spike tensor, else None."""
    if x.dtype == torch.bool:
        return int(x.count_nonzero().item())
    is_binary = bool(x.eq(0).logical_or_(x.eq(1)).all().item())
    if not is_binary:
        return None
    return int(x.count_nonzero().item())


def _ac_element_wise(args, kwargs, out):
    x = args[0]
    y = args[1] if (len(args) > 1 and torch.is_tensor(args[1])) else None
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y) if y is not None else None

    if (nnz_x is not None) and (nnz_y is not None):
        return nnz_x + nnz_y  # two binary tensors
    else:
        return out.numel()


def _ac_mm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        return int(out.sum().item())
    elif nnz_x is not None:
        return nnz_x * y.shape[1]
    elif nnz_y is not None:
        return nnz_y * x.shape[0]
    else:
        return 0


def _ac_addmm(args, kwargs, out):
    _, x, y = args[:3]
    alpha = kwargs.get("alpha", 1)
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                result = torch.ops.aten.mm.default(x.double(), y.double())
        return int(result.sum().item())
    elif alpha != 1:
        return 0
    elif nnz_x is not None:
        return nnz_x * y.shape[1]
    elif nnz_y is not None:
        return nnz_y * x.shape[0]
    else:
        return 0


def _ac_bmm(args, kwargs, out):
    x, y = args[:2]
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        return int(out.sum().item())
    elif nnz_x is not None:
        return nnz_x * y.shape[2]
    elif nnz_y is not None:
        return nnz_y * x.shape[1]
    else:
        return 0


def _ac_baddbmm(args, kwargs, out):
    _, x, y = args[:3]
    alpha = kwargs.get("alpha", 1)
    nnz_x = _spike_nnz(x)
    nnz_y = _spike_nnz(y)
    if nnz_x is not None and nnz_y is not None:
        with torch.no_grad():
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                result = torch.ops.aten.bmm.default(x.double(), y.double())
        return int(result.sum().item())
    elif alpha != 1:
        return 0
    elif nnz_x is not None:
        return nnz_x * y.shape[2]
    elif nnz_y is not None:
        return nnz_y * x.shape[1]
    else:
        return 0


def _ac_convolution(args, _kwargs, out):
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
    elif nnz_x is not None:
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
    elif nnz_w is not None:
        ref = x if transposed else out
        return nnz_w * ref.shape[0] * _prod(ref.shape[2:])
    else:
        return 0


def _ac_avg_pool2d(args, kwargs, out):
    kernel_size = args[1]
    return out.numel() * (_prod(kernel_size) - 1)


def _ac_sum(args, kwargs, out):
    x = args[0]
    return x.numel() - out.numel()


def _ac_mean(args, kwargs, out):
    x = args[0]
    return x.numel() - out.numel()


def _ac_sigmoid(args, kwargs, out):
    return out.numel()


def _ac_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    n, c = x.numel(), x.shape[1]
    has_running_stats = args[3] is not None
    if train:
        ac = n - c  # mean: reduction sum per channel
        ac += n  # var: E[x^2] - E[x]^2
        ac += c  # var + eps
        ac += n  # x - mean
        if has_running_stats:
            ac += 2 * c  # old + m * (new - old) ; mean and var
    else:
        ac = c + n  # var + eps, x - mean
    return ac


class ACCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        r"""
        **API Language:**
        :ref:`中文 <ACCounter.__init__-cn>` | :ref:`English <ACCounter.__init__-en>`

        ----

        .. _ACCounter.__init__-cn:

        * **中文**

        硬件级累加（Accumulate，AC）操作计数器，从硬件视角统计网络中的纯加法次数。

        与 :class:`SynOpCounter <spikingjelly.activation_based.op_counter.synop.SynOpCounter>` 的区别：
        ``SynOpCounter`` 只关注脉冲驱动的矩阵乘法和卷积；
        ``ACCounter`` 还会统计 BN、add/sub 等算子内部的加法，范围更广但语义更宽泛。
        例如，SEW ResNet 中残差连接处的加法操作将被计入 AC。

        ``ACCounter`` 应与 :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` 搭配使用。

        .. warning::

            ``ACCounter`` 只能统计前向传播期间的 AC 数量。部分专用于反向传播的算子还未覆盖。

            目前，``ACCounter`` 支持的 aten 操作类型有限。查看源代码以获取操作列表。如需添加新操作，
            可以使用 ``extra_rules`` 参数；也欢迎提交 pull request 来完善默认的 :attr:`rules`！

        .. warning::

            ``ACCounter`` 会如实考虑 BN 内部的 AC 操作。如果想在推理时忽略 BN 内部的 AC，请将 BN
            融合到线性层中；或者使用 ``extra_ignore_modules`` 参数将 BN 模块加入忽略列表。

        :param extra_rules: 额外的操作规则，格式为 ``{aten_op: func}``，
            其中 ``func`` 是一个函数，接受 ``(args, kwargs, out)`` 并返回 AC 次数
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: 额外需要忽略的模块列表
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        .. _ACCounter.__init__-en:

        * **English**

        Hardware-level Accumulate (AC) operation counter that counts pure additions in a network
        from a hardware perspective.

        Compared with :class:`SynOpCounter <spikingjelly.activation_based.op_counter.synop.SynOpCounter>`:
        ``SynOpCounter`` only covers spike-driven matmul and conv;
        ``ACCounter`` also covers BN, add/sub, etc., thus is broader but more semantically general.

        ``ACCounter`` should be used with :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`.

        .. warning::

            ``ACCounter`` can only count ACs during the forward pass. Some operators
            dedicated to backward pass are not yet covered.

            Currently, ``ACCounter`` supports a limited number of aten operations.
            See the source code for the operation list.
            If you want to add new operations, use the ``extra_rules`` parameter.
            Welcome to submit a pull request to improve the default :attr:`rules`!

        .. warning::

            ``ACCounter`` counts AC operations inside BN. To ignore AC inside BN during inference,
            please fuse BN into linear/conv layers; or use the ``extra_ignore_modules`` parameter to add
            BN modules to the ignore list.

        :param extra_rules: additional operation rules, format as ``{aten_op: func}``,
            where ``func`` is a function that takes ``(args, kwargs, out)`` and returns the AC count
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: additional list of modules to ignore
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                ACCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
            spike = (torch.rand(32, 100) < 0.1).float()  # sparse binary input

            ac_counter = ACCounter()
            with DispatchCounterMode([ac_counter]):
                model(spike)

            print(f"Total ACs: {ac_counter.get_total()}")  # only the 1st layer counts
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _ac_mm,
            aten.addmm.default: _ac_addmm,
            aten.bmm.default: _ac_bmm,
            aten.baddbmm.default: _ac_baddbmm,
            aten.convolution.default: _ac_convolution,
            aten.native_batch_norm.default: _ac_native_batch_norm,
            aten.avg_pool2d.default: _ac_avg_pool2d,
            aten.sum.default: _ac_sum,
            aten.sum.dim_IntList: _ac_sum,
            aten.mean.dim: _ac_mean,
            aten.add.Tensor: _ac_element_wise,
            aten.add_.Tensor: _ac_element_wise,
            aten.add.Scalar: _ac_element_wise,
            aten.add_.Scalar: _ac_element_wise,
            aten.sub.Tensor: _ac_element_wise,
            aten.sub_.Tensor: _ac_element_wise,
            aten.sub.Scalar: _ac_element_wise,
            aten.sub_.Scalar: _ac_element_wise,
            aten.rsub.Tensor: _ac_element_wise,
            aten.rsub.Scalar: _ac_element_wise,
            aten.sigmoid_.default: _ac_sigmoid,
            # other aten ops do not involve AC operations
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
