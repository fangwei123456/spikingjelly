from __future__ import annotations

from collections import defaultdict
from math import prod
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..neuron.base_node import BaseNode
from .base import is_binary_tensor

__all__ = ["NeuromorphicMemoryAccessCounter"]


_MEMORY_METRICS = (
    "weight_read_bytes",
    "bias_read_bytes",
    "neuron_state_read_bytes",
    "neuron_state_write_bytes",
)
_SYNAPTIC_MODULES = (nn.Linear, nn.Conv1d, nn.Conv2d, nn.Conv3d)


def _spike_conv_weight_uses(module: nn.Module, x: torch.Tensor) -> int:
    if getattr(module, "step_mode", "s") == "m":
        x = x.flatten(0, 1)
    padding = module.padding
    with (
        torch.no_grad(),
        torch._C._ExcludeDispatchKeyGuard(
            torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
        ),
    ):
        x = x.double()
        if isinstance(padding, str) or module.padding_mode != "zeros":
            mode = (
                "constant" if module.padding_mode == "zeros" else module.padding_mode
            )
            x = F.pad(x, module._reversed_padding_repeated_twice, mode)
            padding = tuple(0 for _ in module.stride)
        out = torch.ops.aten.convolution.default(
            x,
            torch.ones_like(module.weight, dtype=torch.float64),
            None,
            module.stride,
            padding,
            module.dilation,
            False,
            tuple(0 for _ in module.stride),
            module.groups,
        )
    return int(out.sum().item())


class NeuromorphicMemoryAccessCounter:
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
        self.extra_ignore_modules = tuple(extra_ignore_modules or ())
        self.model: nn.Module | None = None
        self._module_names: dict[nn.Module, str] = {}
        self._handles: list[Any] = []
        self._records: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def bind_model(self, model: nn.Module) -> None:
        r"""
        绑定待统计模型。Bind the model to be counted.

        :param model: 待统计模型 / Model to profile
        :type model: torch.nn.Module
        """
        self.model = model
        self._module_names = {
            module: name or module.__class__.__name__
            for name, module in model.named_modules()
        }

    def _record(self, module: nn.Module, **metrics: int) -> None:
        scope = self._module_names[module]
        for name, value in metrics.items():
            self._records["Global"][name] += value
            self._records[scope][name] += value

    def _synaptic_hook(
        self,
        module: nn.Module,
        inputs: tuple[torch.Tensor, ...],
        output: torch.Tensor,
    ) -> None:
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

        self._record(
            module,
            weight_read_bytes=weight_uses * int(module.weight.element_size()),
            bias_read_bytes=(
                0
                if module.bias is None
                else int(output.numel()) * int(module.bias.element_size())
            ),
        )

    def _neuron_hook(
        self, module: BaseNode, inputs: tuple[torch.Tensor, ...], output: Any
    ) -> None:
        del output
        time_steps = int(inputs[0].shape[0]) if module.step_mode == "m" else 1
        state_bytes = sum(
            int(state.numel()) * int(state.element_size())
            for state in module.memories()
            if torch.is_tensor(state)
        )
        state_bytes *= time_steps
        self._record(
            module,
            neuron_state_read_bytes=state_bytes,
            neuron_state_write_bytes=state_bytes,
        )

    def __enter__(self) -> "NeuromorphicMemoryAccessCounter":
        if self.model is None:
            raise RuntimeError(
                "NeuromorphicMemoryAccessCounter.bind_model() must be called "
                "before entering the counter context."
            )
        self.reset()
        ignored_modules = {
            child
            for module in self.model.modules()
            if isinstance(module, self.extra_ignore_modules)
            for child in module.modules()
        }
        for module in self.model.modules():
            if module in ignored_modules:
                continue
            if isinstance(module, _SYNAPTIC_MODULES):
                self._handles.append(module.register_forward_hook(self._synaptic_hook))
            elif isinstance(module, BaseNode):
                self._handles.append(module.register_forward_hook(self._neuron_hook))
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        del exc_type, exc, tb
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def reset(self) -> None:
        r"""重置全部计数。Reset all counts."""
        self._records = defaultdict(lambda: defaultdict(int))

    def get_counts(self) -> dict[str, dict[str, int]]:
        r"""
        返回按模块作用域聚合的访存字节数。

        Return memory-access bytes aggregated by module scope.

        :return: 每个作用域的四类访存字节数 / Four byte-count metrics per scope
        :rtype: dict[str, dict[str, int]]
        """
        return {
            scope: {metric: values.get(metric, 0) for metric in _MEMORY_METRICS}
            for scope, values in self._records.items()
        }

    def get_total(self) -> int:
        r"""
        返回全局访存字节总数。Return the global total memory-access bytes.

        :return: 全局访存字节总数 / Global total bytes
        :rtype: int
        """
        return sum(self._records.get("Global", {}).values())
