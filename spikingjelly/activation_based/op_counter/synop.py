from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter

aten = torch.ops.aten
__all__ = ["SynOpCounter"]


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


def _synop_mm(args, kwargs, out):
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


def _synop_addmm(args, kwargs, out):
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


def _synop_bmm(args, kwargs, out):
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


def _synop_baddbmm(args, kwargs, out):
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


def _synop_convolution(args, _kwargs, out):
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


class SynOpCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        r"""
        **API Language:**
        :ref:`中文 <SynOpCounter.__init__-cn>` | :ref:`English <SynOpCounter.__init__-en>`

        ----

        .. _SynOpCounter.__init__-cn:

        * **中文**

        突触操作（Synaptic Operations，SynOps）计数器，用于统计 SNN 中由 spike 驱动的突触权重累加次数。

        与 :class:`ACCounter <spikingjelly.activation_based.op_counter.ac.ACCounter>` 的区别：
        ``ACCounter`` 除了权重层线性操作外，还会统计 BN、add/sub 等算子内部的加法；
        ``SynOpCounter`` 只关注脉冲驱动的矩阵乘法和卷积，范围更窄但语义更直接。
        例如，SEW ResNet 中残差连接处的加法操作将不会被计入 SynOps。

        ``SynOpCounter`` 应与 :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` 搭配使用。

        .. warning::

            ``SynOpCounter`` 只能统计前向传播期间的突触操作数量。部分专用于反向传播的算子还未覆盖。

            目前，``SynOpCounter`` 支持的 aten 操作类型有限（mm、addmm、bmm、baddbmm、convolution）。
            如需添加新操作，可以使用 ``extra_rules`` 参数；也欢迎提交 pull request 来完善默认的 :attr:`rules`！

        :param extra_rules: 额外的操作规则，格式为 ``{aten_op: func}``，
            其中 ``func`` 是一个函数，接受 ``(args, kwargs, out)`` 并返回 SynOps 次数
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: 额外需要忽略的模块列表，这些模块中的操作不会被计数
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        .. _SynOpCounter.__init__-en:

        * **English**

        Synaptic Operations (SynOps) counter that tracks spike-driven weight accumulations in SNNs.

        Compared with :class:`ACCounter <spikingjelly.activation_based.op_counter.ac.ACCounter>`:
        ``ACCounter`` also covers BN internals, add/sub, ...
        ``SynOpCounter`` is narrower: only spike-driven matmul and conv are considered.
        This makes it more directly aligned with the intuitive concept of "synaptic operations" in neuromorphic computing.

        ``SynOpCounter`` should be used with :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`.

        .. warning::

            ``SynOpCounter`` can only count SynOps during the forward pass. Some operators
            dedicated to backward pass are not yet covered.

            Currently, ``SynOpCounter`` supports mm, addmm, bmm, baddbmm, and convolution.
            If you want to add new operations, use the ``extra_rules`` parameter.
            Welcome to submit a pull request to improve the default :attr:`rules`!

        :param extra_rules: additional operation rules, format as ``{aten_op: func}``,
            where ``func`` is a function that takes ``(args, kwargs, out)`` and returns the SynOps count
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: additional list of modules to ignore.
            Operations within these modules will not be counted
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                SynOpCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn

            model = nn.Linear(10, 5, bias=False)
            spike = (torch.rand(4, 10) < 0.2).float()

            counter = SynOpCounter()
            with DispatchCounterMode([counter]):
                model(spike)

            print(f"SynOp count: {counter.get_total()}")
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _synop_mm,
            aten.addmm.default: _synop_addmm,
            aten.bmm.default: _synop_bmm,
            aten.baddbmm.default: _synop_baddbmm,
            aten.convolution.default: _synop_convolution,
            # other aten ops do not involve SynOp operations
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
