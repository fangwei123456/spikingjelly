from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .ac import (
    _spike_addmm,
    _spike_baddbmm,
    _spike_bmm,
    _spike_convolution,
    _spike_mm,
)
from .base import BaseCounter

aten = torch.ops.aten
__all__ = ["SynOpCounter"]


class SynOpCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[type[nn.Module]] | None = None,
    ):
        r"""
        **API Language** - :ref:`中文 <SynOpCounter.__init__-cn>` | :ref:`English <SynOpCounter.__init__-en>`

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
            aten.mm.default: _spike_mm,
            aten.addmm.default: _spike_addmm,
            aten.bmm.default: _spike_bmm,
            aten.baddbmm.default: _spike_baddbmm,
            aten.convolution.default: _spike_convolution,
            # other aten ops do not involve SynOp operations
        }
        self.ignore_modules = []
        self.rules.update(extra_rules or {})
        self.ignore_modules.extend(extra_ignore_modules or [])
