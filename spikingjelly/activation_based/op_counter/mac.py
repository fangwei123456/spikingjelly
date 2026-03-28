from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter

aten = torch.ops.aten
__all__ = ["MACCounter"]


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p


def _is_spike(x: torch.Tensor) -> bool:
    """Return True if *x* is a binary spike tensor (bool dtype or values in {0, 1})."""
    if x.dtype == torch.bool:
        return True
    return bool(x.eq(0).logical_or_(x.eq(1)).all().item())


def _mac_mm(args, kwargs, out):
    x, y = args[:2]
    if _is_spike(x) or _is_spike(y):
        return 0
    m, k = x.shape
    _, n = y.shape
    return m * n * k


def _mac_addmm(args, kwargs, out):
    _, x, y = args[:3]
    if _is_spike(x) or _is_spike(y):
        return 0
    m, k = x.shape
    _, n = y.shape
    return m * n * k


def _mac_bmm(args, kwargs, out):
    x, y = args[:2]
    if _is_spike(x) or _is_spike(y):
        return 0
    b, m, k = x.shape
    _, _, n = y.shape
    return b * m * n * k


def _mac_baddbmm(args, kwargs, out):
    _, x, y = args[:3]
    if _is_spike(x) or _is_spike(y):
        return 0
    b, m, k = x.shape
    _, _, n = y.shape
    return b * m * n * k


def _mac_convolution(args, _kwargs, out):
    x, w, transposed = args[0], args[1], args[6]
    if _is_spike(x) or _is_spike(w):
        return 0
    b = x.shape[0]
    spatial_shape = x.shape[2:] if transposed else out.shape[2:]
    c_out, c_in, *kernel_shape = w.shape

    mac_per_position = c_in * _prod(kernel_shape)
    return mac_per_position * _prod(spatial_shape) * c_out * b


def _mac_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    c = x.shape[1]
    has_affine = args[1] is not None
    has_running_stats = args[3] is not None
    mac = 0
    if has_affine:
        mac += x.numel()  # x_hat * gamma + beta
    if train and has_running_stats:
        mac += 2 * c  # old + m * (new - old) ; mean & var
    return mac


class MACCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        r"""
        **API Language:**
        :ref:`中文 <MACCounter.__init__-cn>` | :ref:`English <MACCounter.__init__-en>`

        ----

        .. _MACCounter.__init__-cn:

        * **中文**

        硬件级乘累加（Multiply-Accumulate，MAC）操作计数器，统计网络中所有 MAC 操作次数。

        MAC 乘法结果立即累加到累加器（如矩阵内积），而非写入新的内存位置。MAC与AC互斥：若一个计算步骤
        计入 MAC，则不会计入 AC；反之亦然。

        ``MACCounter`` 应与 :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` 搭配使用。

        .. warning::

            ``MACCounter`` 只能统计前向传播期间的 MAC 数量。部分专用于反向传播的算子还未覆盖。

            目前，``MACCounter`` 支持的 aten 操作类型有限。查看源代码以获取操作列表。如需添加新操作，
            可以使用 ``extra_rules`` 参数；也欢迎提交 pull request 来完善默认的 :attr:`rules`！

        .. warning::

            ``MACCounter`` 会如实考虑 BN 内部的 MAC 操作。如果想在推理时忽略 BN 内部的 MAC，请将 BN
            融合到线性层中；或者使用 ``extra_ignore_modules`` 参数将 BN 模块加入忽略列表。

        :param extra_rules: 额外的操作规则，格式为 ``{aten_op: func}``，
            其中 ``func`` 是一个函数，接受 ``(args, kwargs, out)`` 并返回 MAC 次数
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: 额外需要忽略的模块列表，这些模块中的操作不会被计数
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        .. _MACCounter.__init__-en:

        * **English**

        Hardware-level Multiply-Accumulate (MAC) operation counter that counts all MAC operations
        in a network.

        MAC's multiply result is immediately accumulated into a running accumulator (not
        written to a new memory location).
        ``MACCounter`` is mutually exclusive with :class:`ACCounter <spikingjelly.activation_based.op_counter.ac.ACCounter>` : if a computation step is counted as MAC, it will not be counted as AC, and vice versa.

        .. warning::

            ``MACCounter`` can only count MACs during the forward pass. Some operators
            dedicated to backward pass are not yet covered.

            Currently, ``MACCounter`` supports a limited number of aten operations.
            See the source code for the operation list.
            If you want to add new operations, use the ``extra_rules`` parameter.
            Welcome to submit a pull request to improve the default :attr:`rules`!

        :param extra_rules: additional operation rules, format as ``{aten_op: func}``,
            where ``func`` is a function that takes ``(args, kwargs, out)`` and returns the MAC count
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: additional list of modules to ignore.
            Operations within these modules will not be counted
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                MACCounter,
                ACCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
            x = (torch.randn(32, 100) < 0.1).float()  # sparse binary input

            mac_counter = MACCounter()
            with DispatchCounterMode([mac_counter]):
                output = model(x)

            print(f"Total MACs: {mac_counter.get_total()}")  # only the 2nd layer counts
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _mac_mm,
            aten.addmm.default: _mac_addmm,
            aten.bmm.default: _mac_bmm,
            aten.baddbmm.default: _mac_baddbmm,
            aten.convolution.default: _mac_convolution,
            aten.native_batch_norm.default: _mac_native_batch_norm,
            # other aten ops do not involve MAC operations
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
