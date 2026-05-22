from collections import defaultdict
from typing import Any, Callable

import torch
import torch.nn as nn

from .base import BaseCounter, is_binary_tensor

aten = torch.ops.aten
__all__ = ["LemaireAddressingCounter"]


def _prod(dims):
    p = 1
    for v in dims:
        p *= int(v)
    return p


def _address_linear(x: torch.Tensor, out: torch.Tensor) -> dict[str, int]:
    if is_binary_tensor(x):
        return {
            "acc_addr": int(x.count_nonzero().item()) * int(out.shape[-1]),
            "mac_addr": 0,
        }
    return {
        "acc_addr": int(x.numel()) + int(out.numel()),
        "mac_addr": 0,
    }


def _address_mm(args, kwargs, out):
    del kwargs
    x = args[0]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_addmm(args, kwargs, out):
    del kwargs
    x = args[1]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_bmm(args, kwargs, out):
    del kwargs
    x = args[0]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_baddbmm(args, kwargs, out):
    del kwargs
    x = args[1]
    if not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}
    return _address_linear(x, out)


def _address_convolution(args, kwargs, out):
    del kwargs
    x, w, _bias, _stride, _padding, _dilation, transposed, _output_padding, groups = (
        args[:9]
    )
    if transposed or not torch.is_tensor(x) or not torch.is_tensor(out):
        return {"acc_addr": 0, "mac_addr": 0}

    kernel_volume = _prod(w.shape[2:])
    out_channels = int(w.shape[0])
    if is_binary_tensor(x):
        spike_num_in = int(x.count_nonzero().item())
        out_channels_per_group = out_channels // int(groups)
        return {
            "acc_addr": spike_num_in * out_channels_per_group * kernel_volume,
            "mac_addr": spike_num_in * 2,
        }

    return {
        "acc_addr": int(x.numel()) + int(out.numel()) + out_channels * kernel_volume,
        "mac_addr": 0,
    }


class LemaireAddressingCounter(BaseCounter):
    def __init__(self):
        r"""
        **API Language:**
        :ref:`中文 <LemaireAddressingCounter.__init__-cn>` |
        :ref:`English <LemaireAddressingCounter.__init__-en>`

        ----

        .. _LemaireAddressingCounter.__init__-cn:
        * **中文**

        * **中文**

        Lemaire 风格的地址计数（Addressing）计数器，用于统计 SNN 推理中的
        累积（accumulation）和乘累加（multiply-accumulate）相关地址访问次数。

        该计数器跟踪 ``aten.mm``、``aten.addmm``、``aten.bmm``、
        ``aten.baddbmm`` 和 ``aten.convolution`` 操作的地址计数，
        并根据当前活跃模块的类型（如 :class:`nn.Linear` 或 :class:`nn.Conv2d`）
        自动选择合适的地址计算规则。

        对于脉冲输入（二元张量），该计数器会使用稀疏寻址规则以避免将
        非激活位置计入访问次数。

        该计数器的设计源自文献 [#lemaire2022]_ 中提出的分析方法，
        用于估计 SNN 推理过程中的 addressing 开销；当前实现适用于线性层与卷积层：

        ``LemaireAddressingCounter`` 应与
        :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`
        搭配使用。

        ----

        .. _LemaireAddressingCounter.__init__-en:
        * **English**

        * **English**

        Lemaire-style addressing counter that tracks accumulation and
        multiply-accumulate related address accesses during SNN inference.

        This counter tracks address counts for ``aten.mm``, ``aten.addmm``,
        ``aten.bmm``, ``aten.baddbmm``, and ``aten.convolution`` operations,
        and automatically selects the appropriate addressing rule based on
        the current active module type (e.g. :class:`nn.Linear` or
        :class:`nn.Conv2d`).

        For spike inputs (binary tensors), the counter applies sparse
        addressing rules so inactive positions are not counted as accesses.

        The design of this counter is derived from the analytical method
        proposed in [#lemaire2022]_ for estimating SNN inference addressing
        cost; the current implementation targets linear and convolution
        operators.

        ``LemaireAddressingCounter`` should be used with
        :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`.

        ---

        .. [#lemaire2022] Lemaire, Edgar, et al. "An Analytical Estimation of
            Spiking Neural Networks Energy Efficiency." International Conference
            on Neural Information Processing. 2022.
        :return: None
        :rtype: None
        """
        super().__init__()
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _address_mm,
            aten.addmm.default: _address_addmm,
            aten.bmm.default: _address_bmm,
            aten.baddbmm.default: _address_baddbmm,
            aten.convolution.default: _address_convolution,
        }
        self.ignore_modules = []
        self.metric_records: dict[str, dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self._pending_metrics: dict[str, int] | None = None

    def _active_linear(self, active_modules: set[nn.Module] | None) -> bool:
        if active_modules is None:
            return False
        return any(isinstance(module, nn.Linear) for module in active_modules)

    def _active_conv(self, active_modules: set[nn.Module] | None) -> bool:
        if active_modules is None:
            return False
        return any(
            isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d))
            for module in active_modules
        )

    def count(
        self,
        func,
        args: tuple,
        kwargs: dict,
        out,
        active_modules: set[nn.Module] | None = None,
        parent_names: set[str] | None = None,
    ) -> int:
        del parent_names
        if func in (
            aten.mm.default,
            aten.addmm.default,
            aten.bmm.default,
            aten.baddbmm.default,
        ):
            if not self._active_linear(active_modules):
                self._pending_metrics = None
                return 0
        elif func == aten.convolution.default:
            if not self._active_conv(active_modules):
                self._pending_metrics = None
                return 0
        else:
            self._pending_metrics = None
            return 0

        metrics = self.rules[func](args, kwargs, out)
        self._pending_metrics = {
            "acc_addr": int(metrics["acc_addr"]),
            "mac_addr": int(metrics["mac_addr"]),
        }
        return self._pending_metrics["acc_addr"] + self._pending_metrics["mac_addr"]

    def record(self, scope, func, value):
        super().record(scope, func, value)
        if self._pending_metrics is not None:
            for key, item in self._pending_metrics.items():
                self.metric_records[scope][key] += int(item)

    def finalize_record(self):
        self._pending_metrics = None

    def get_metric_counts(self) -> dict[str, dict[str, int]]:
        r"""
        **API Language:**
        :ref:`中文 <LemaireAddressingCounter.get_metric_counts-cn>` |
        :ref:`English <LemaireAddressingCounter.get_metric_counts-en>`

        ----

        .. _LemaireAddressingCounter.get_metric_counts-cn:

        * **中文**

        :return: 按 scope 聚合的地址计数，包含 ``acc_addr`` 和 ``mac_addr`` 两个指标
        :rtype: dict[str, dict[str, int]]

        ----

        .. _LemaireAddressingCounter.get_metric_counts-en:

        * **English**

        :return: addressing metrics aggregated by scope, containing ``acc_addr`` and ``mac_addr``
        :rtype: dict[str, dict[str, int]]
        """
        return {scope: dict(items) for scope, items in self.metric_records.items()}

    def reset(self):
        r"""
        **API Language:**
        :ref:`中文 <LemaireAddressingCounter.reset-cn>` |
        :ref:`English <LemaireAddressingCounter.reset-en>`

        ----

        .. _LemaireAddressingCounter.reset-cn:

        * **中文**

        重置计数器，清空所有已记录的计数和待处理指标。

        此方法继承自 :meth:`BaseCounter.reset`，额外清空了 :attr:`metric_records`
        和 :attr:`_pending_metrics`。

        :return: ``None``
        :rtype: None

        ----

        .. _LemaireAddressingCounter.reset-en:

        * **English**

        Reset the counter and clear all recorded counts and pending metrics.

        This method extends :meth:`BaseCounter.reset` by also clearing
        :attr:`metric_records` and :attr:`_pending_metrics`.

        :return: ``None``
        :rtype: None
        """
        super().reset()
        self.metric_records = defaultdict(lambda: defaultdict(int))
        self._pending_metrics = None
