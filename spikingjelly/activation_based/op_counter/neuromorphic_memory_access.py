from __future__ import annotations

from math import prod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..neuron.base_node import BaseNode
from .base import ModuleCounter, is_binary_tensor

__all__ = ["NeuromorphicMemoryAccessCounter"]


_MEMORY_METRICS = (
    "weight_read_bytes",
    "bias_read_bytes",
    "neuron_state_read_bytes",
    "neuron_state_write_bytes",
)
_SYNAPTIC_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _spike_conv_weight_uses_from_tensors(
    x: torch.Tensor,
    w: torch.Tensor,
    stride,
    padding,
    dilation,
    groups: int,
) -> int:
    group_kernel = torch.ones(
        (groups, w.shape[1], *w.shape[2:]),
        dtype=torch.float32,
        device=x.device,
    )
    with (
        torch.no_grad(),
        torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
        ),
    ):
        occupancy = torch.ops.aten.convolution.default(
            x.float(),
            group_kernel,
            None,
            stride,
            padding,
            dilation,
            False,
            tuple(0 for _ in stride),
            groups,
        )
    return int(occupancy.sum(dtype=torch.float64).item()) * (int(w.shape[0]) // groups)


def _spike_conv_weight_uses(module: nn.Module, x: torch.Tensor) -> int:
    if getattr(module, "step_mode", "s") == "m":
        x = x.flatten(0, 1)
    padding = module.padding
    if isinstance(padding, str) or module.padding_mode != "zeros":
        mode = "constant" if module.padding_mode == "zeros" else module.padding_mode
        x = F.pad(x, module._reversed_padding_repeated_twice, mode)
        padding = tuple(0 for _ in module.stride)
    return _spike_conv_weight_uses_from_tensors(
        x,
        module.weight,
        module.stride,
        padding,
        module.dilation,
        module.groups,
    )


class NeuromorphicMemoryAccessCounter(ModuleCounter):
    def __init__(
        self,
        *,
        extra_ignore_modules: list[type[nn.Module]] | None = None,
    ):
        r"""
        .. rubric:: API Language

        :ref:`中文 <NeuromorphicMemoryAccessCounter-cn>` |
        :ref:`English <NeuromorphicMemoryAccessCounter-en>`

        ----

        .. _NeuromorphicMemoryAccessCounter-cn:

        * **中文**

        神经形态推理逻辑访存计数器。它统计线性层和卷积层实际使用权重及
        bias 时产生的参数读取，以及 :class:`BaseNode` 注册的 tensor 状态在
        每个时间步的一次读取和一次写回。

        二元输入按实际 spike-triggered fanout 统计权重读取；非二元输入按
        稠密运算统计。计数使用运行时 dtype 换算字节，不包含输入电流、输出
        spike、FIFO、路由、寻址、中间张量或宿主设备 cache 流量。

        :param extra_ignore_modules: 不参与统计的额外模块类型
        :type extra_ignore_modules: Optional[list[type[torch.nn.Module]]]

        ----

        .. _NeuromorphicMemoryAccessCounter-en:

        * **English**

        Logical memory-access counter for neuromorphic inference. It counts
        parameter reads when linear and convolutional layers use weights and
        biases, plus one read and one write per timestep for tensor states
        registered by :class:`BaseNode`.

        Binary inputs use the actual spike-triggered fanout, while non-binary
        inputs use dense operation counts. Runtime dtypes determine byte counts.
        Input currents, output spikes, FIFOs, routing, addressing, intermediate
        tensors, and host-device cache traffic are outside this model.

        :param extra_ignore_modules: Additional module types excluded from counting
        :type extra_ignore_modules: Optional[list[type[torch.nn.Module]]]
        """
        super().__init__()
        self.ignore_modules.extend(extra_ignore_modules or [])
        self.rules = {
            **{
                ("forward", module_type): self._count_synaptic
                for module_type in _SYNAPTIC_MODULES
            },
            ("forward", BaseNode): self._count_neuron,
        }
        self._pending_metrics: dict[str, int] | None = None

    def _count_synaptic(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
        output: torch.Tensor,
    ) -> int:
        del kwargs
        x = inputs[0]
        if is_binary_tensor(x):
            if isinstance(module, nn.Linear):
                weight_uses = int(x.count_nonzero().item()) * module.out_features
            else:
                weight_uses = _spike_conv_weight_uses(module, x)
        elif isinstance(module, nn.Linear):
            weight_uses = int(output.numel()) * module.in_features
        else:
            weight_uses = int(output.numel()) * int(prod(module.weight.shape[1:]))

        self._pending_metrics = {
            "weight_read_bytes": weight_uses * int(module.weight.element_size()),
            "bias_read_bytes": (
                0
                if module.bias is None
                else int(output.numel()) * int(module.bias.element_size())
            ),
        }
        return sum(self._pending_metrics.values())

    def _count_neuron(
        self,
        module: BaseNode,
        inputs: tuple[torch.Tensor, ...],
        kwargs: dict[str, Any],
        output: Any,
    ) -> int:
        del kwargs, output
        time_steps = int(inputs[0].shape[0]) if module.step_mode == "m" else 1
        state_bytes = sum(
            int(state.numel()) * int(state.element_size())
            for state in module.memories()
            if torch.is_tensor(state)
        )
        state_bytes *= time_steps
        self._pending_metrics = {
            "neuron_state_read_bytes": state_bytes,
            "neuron_state_write_bytes": state_bytes,
        }
        return sum(self._pending_metrics.values())

    def record(self, scope: str, func: Any, value: int) -> None:
        del func, value
        if self._pending_metrics is None:
            return
        for name, metric_value in self._pending_metrics.items():
            self.records[scope][name] += metric_value

    def finalize_record(self) -> None:
        self._pending_metrics = None

    def reset(self) -> None:
        r"""重置全部计数。Reset all counts."""
        super().reset()
        self._pending_metrics = None

    def get_counts(self) -> dict[str, dict[str, int]]:
        r"""
        返回按模块作用域聚合的访存字节数。

        Return memory-access bytes aggregated by module scope.

        :return: 每个作用域的四类访存字节数 / Four byte-count metrics per scope
        :rtype: dict[str, dict[str, int]]
        """
        return {
            scope: {metric: values.get(metric, 0) for metric in _MEMORY_METRICS}
            for scope, values in self.records.items()
        }

    def get_total(self) -> int:
        r"""
        返回全局访存字节总数。Return the global total memory-access bytes.

        :return: 全局访存字节总数 / Global total bytes
        :rtype: int
        """
        return sum(self.records.get("Global", {}).values())
