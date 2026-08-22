from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from math import prod
from numbers import Real
from typing import Any, Callable

import torch
import torch.nn as nn

from ...neuron.base_node import BaseNode
from ..ac import ACCounter
from ..base import DispatchCounterMode, is_binary_tensor
from ..lemaire_addressing import LemaireAddressingCounter
from ..mac import MACCounter
from ..neuromorphic_memory_access import _spike_conv_weight_uses
from ..neuron_state import NeuronStateCounter
from ..synop import SynOpCounter

__all__ = [
    "LemaireEnergyConfig",
    "LemaireEnergyCostConfig",
    "LemaireEnergyProfiler",
    "LemaireEnergyReport",
    "estimate_lemaire_energy",
]

_LEMAIRE_ACCESS_WIDTH_BYTES = 4.0

_SUPPORTED_LEMAIRE_MEMORY_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)
_UNSUPPORTED_LEMAIRE_MEMORY_MODULES = (
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


@dataclass
class LemaireEnergyCostConfig:
    r"""
    .. rubric:: API Language

    :ref:`中文 <LemaireEnergyCostConfig-cn>` |
    :ref:`English <LemaireEnergyCostConfig-en>`

    ----

    .. _LemaireEnergyCostConfig-cn:

    * **中文**

    Lemaire 风格解析式能耗模型的成本配置。

    ----

    .. _LemaireEnergyCostConfig-en:

    * **English**

    Cost configuration for the Lemaire-style analytical energy model.
    """

    e_add_pj: float = 0.1
    e_mul_pj: float = 3.1
    memory_breakpoints: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (8.0 * 1024.0, 10.0 / _LEMAIRE_ACCESS_WIDTH_BYTES),
        (32.0 * 1024.0, 20.0 / _LEMAIRE_ACCESS_WIDTH_BYTES),
        (1024.0 * 1024.0, 100.0 / _LEMAIRE_ACCESS_WIDTH_BYTES),
    )

    def __post_init__(self):
        points = self.memory_breakpoints
        if len(points) != 4:
            raise ValueError("memory_breakpoints must contain exactly 4 (x, y) pairs.")
        prev_x = None
        for point in points:
            if not isinstance(point, tuple) or len(point) != 2:
                raise ValueError(
                    "memory_breakpoints must contain exactly 4 (x, y) pairs."
                )
            x, y = point
            if not isinstance(x, Real) or not isinstance(y, Real):
                raise ValueError(
                    "memory_breakpoints entries must be numeric (x, y) pairs."
                )
            if prev_x is not None and x <= prev_x:
                raise ValueError(
                    "memory_breakpoints x values must be strictly increasing."
                )
            prev_x = x

    def memory_cost_pj(self, memory: float) -> float:
        points = self.memory_breakpoints
        memory = max(points[0][0], min(memory, points[3][0]))
        if memory <= points[1][0]:
            (x0, y0), (x1, y1) = points[0], points[1]
            return y0 + (y1 - y0) / (x1 - x0) * (memory - x0)
        if memory <= points[2][0]:
            (x0, y0), (x1, y1) = points[1], points[2]
            return y0 + (y1 - y0) / (x1 - x0) * (memory - x0)
        (x0, y0), (x1, y1) = points[2], points[3]
        return y0 + (y1 - y0) / (x1 - x0) * (memory - x0)


@dataclass
class LemaireEnergyConfig:
    r"""
    .. rubric:: API Language

    :ref:`中文 <LemaireEnergyConfig-cn>` |
    :ref:`English <LemaireEnergyConfig-en>`

    ----

    .. _LemaireEnergyConfig-cn:

    * **中文**

    控制 inference-only、Lemaire 对齐能耗分析器的行为。

    :param snn_fifo_capacity_elements: 每层 SNN 输入/输出 FIFO 可容纳的消息数，
        默认 ``1000``，与论文实验设置一致
    :type snn_fifo_capacity_elements: int

    :raises ValueError: 当 ``snn_fifo_capacity_elements`` 非正数时抛出

    ----

    .. _LemaireEnergyConfig-en:

    * **English**

    Controls the inference-only, Lemaire-aligned energy profiler.

    :param snn_fifo_capacity_elements: Number of messages that each SNN input/output
        FIFO can hold. The default ``1000`` follows the paper's experiments
    :type snn_fifo_capacity_elements: int

    :raises ValueError: Raised when ``snn_fifo_capacity_elements`` is not positive
    """

    strict: bool = False
    cost_config: LemaireEnergyCostConfig = field(
        default_factory=LemaireEnergyCostConfig
    )
    extra_state_rules: dict[type[nn.Module], Callable] = field(default_factory=dict)
    sparse_zero_ratio_threshold: float = 0.5
    enable_sparse_memory_estimation: bool = True
    snn_fifo_capacity_elements: int = 1000

    def __post_init__(self):
        if self.snn_fifo_capacity_elements <= 0:
            raise ValueError("snn_fifo_capacity_elements must be positive.")


@dataclass
class LemaireEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <LemaireEnergyReport-cn>` |
    :ref:`English <LemaireEnergyReport-en>`

    ----

    .. _LemaireEnergyReport-cn:

    * **中文**

    单一 Lemaire 口径的前向推理能耗报告。

    ----

    .. _LemaireEnergyReport-en:

    * **English**

    Single-report, Lemaire-aligned forward inference energy report.
    """

    total_pj: float
    breakdown_pj: dict[str, float]
    counts: dict[str, int]
    buffer_sizes_bytes: dict[str, int]
    warnings: list[str]


class _LemaireForwardTracker:
    def __init__(self, *, strict: bool, fifo_capacity_elements: int):
        self.strict = strict
        self.fifo_capacity_elements = fifo_capacity_elements
        self.handles: list[Any] = []
        self.warnings: list[str] = []
        self._warned_module_types: set[type[nn.Module]] = set()
        self._accesses: list[tuple[str, int, int]] = []

    def _warn_or_raise_unsupported(self, module: nn.Module):
        module_type = type(module)
        if module_type in self._warned_module_types:
            return
        self._warned_module_types.add(module_type)
        message = (
            f"Lemaire memory formulas do not support {module_type.__name__}; "
            "its memory accesses are omitted."
        )
        if self.strict:
            raise ValueError(message)
        self.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def reset(self):
        self.warnings.clear()
        self._warned_module_types.clear()
        self._accesses.clear()

    def _record(self, name: str, access_bytes: int, capacity_bytes: int):
        self._accesses.append((name, access_bytes, capacity_bytes))

    @staticmethod
    def _time_steps(module: nn.Module, x: torch.Tensor) -> int:
        if getattr(module, "step_mode", "s") == "m":
            return int(x.shape[0])
        return 1

    def summarize(
        self, cost: LemaireEnergyCostConfig
    ) -> tuple[dict[str, int], dict[str, int], dict[str, float]]:
        counts = {
            "read_in_bytes": 0,
            "write_out_bytes": 0,
            "read_params_bytes": 0,
            "read_potential_bytes": 0,
            "write_potential_bytes": 0,
        }
        buffers = {
            "inout_buffer_bytes": 0,
            "params_buffer_bytes": 0,
            "potential_buffer_bytes": 0,
        }
        energy = {"inout_pj": 0.0, "params_pj": 0.0, "potential_pj": 0.0}
        for name, access_bytes, capacity_bytes in self._accesses:
            counts[name] += access_bytes
            if name in ("read_in_bytes", "write_out_bytes"):
                buffer_name, component = "inout_buffer_bytes", "inout_pj"
            elif name == "read_params_bytes":
                buffer_name, component = "params_buffer_bytes", "params_pj"
            else:
                buffer_name, component = "potential_buffer_bytes", "potential_pj"
            buffers[buffer_name] = max(buffers[buffer_name], capacity_bytes)
            energy[component] += access_bytes * cost.memory_cost_pj(capacity_bytes)
        return counts, buffers, energy

    def attach(self, model: nn.Module):
        self.remove()
        model_has_neurons = any(
            isinstance(module, BaseNode) for module in model.modules()
        )
        for module in model.modules():
            if isinstance(module, _UNSUPPORTED_LEMAIRE_MEMORY_MODULES):
                self._warn_or_raise_unsupported(module)

        def synaptic_hook(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            output: torch.Tensor,
        ):
            x = inputs[0]
            out = output

            input_bytes = int(x.numel()) * int(x.element_size())
            output_bytes = int(out.numel()) * int(out.element_size())
            params_capacity = sum(
                int(param.numel()) * int(param.element_size())
                for param in module.parameters(recurse=False)
            )
            is_spike_input = is_binary_tensor(x)
            active_inputs = int(x.count_nonzero().item()) if is_spike_input else 0
            if isinstance(module, nn.Linear):
                dense_weight_uses = int(out.numel()) * module.in_features
                event_fanout = active_inputs * module.out_features
            else:
                dense_weight_uses = int(out.numel()) * int(
                    prod(module.weight.shape[1:])
                )
                event_fanout = (
                    _spike_conv_weight_uses(module, x) if is_spike_input else 0
                )

            weight_uses = event_fanout if is_spike_input else dense_weight_uses
            bias_read_bytes = (
                0
                if module.bias is None
                else int(out.numel()) * int(module.bias.element_size())
            )
            if is_spike_input:
                read_in_bytes = active_inputs * int(x.element_size())
                input_capacity = self.fifo_capacity_elements * int(x.element_size())
            elif isinstance(module, nn.Linear):
                read_in_bytes = input_bytes
                input_capacity = input_bytes
            else:
                read_in_bytes = dense_weight_uses * int(x.element_size())
                input_capacity = input_bytes
            self._record(
                "read_in_bytes",
                read_in_bytes,
                input_capacity,
            )
            self._record(
                "read_params_bytes",
                weight_uses * int(module.weight.element_size()) + bias_read_bytes,
                params_capacity,
            )

            if is_spike_input:
                time_steps = self._time_steps(module, x)
                potential_capacity = output_bytes // max(time_steps, 1)
                potential_access_bytes = event_fanout * int(out.element_size())
                self._record(
                    "read_potential_bytes",
                    potential_access_bytes,
                    potential_capacity,
                )
                self._record(
                    "write_potential_bytes",
                    potential_access_bytes,
                    potential_capacity,
                )
            elif not model_has_neurons:
                self._record("write_out_bytes", output_bytes, output_bytes)

        def neuron_hook(module: BaseNode, inputs: tuple[Any, ...], output: Any):
            out = (
                output[0]
                if isinstance(output, (tuple, list)) and len(output) > 0
                else output
            )
            if not torch.is_tensor(out):
                return
            time_steps = self._time_steps(module, inputs[0])
            potential = module.v
            if torch.is_tensor(potential):
                potential_capacity = int(potential.numel()) * int(
                    potential.element_size()
                )
            else:
                potential_capacity = (
                    int(out.numel()) // max(time_steps, 1) * int(out.element_size())
                )
            potential_access_bytes = potential_capacity * time_steps
            self._record(
                "read_potential_bytes", potential_access_bytes, potential_capacity
            )
            self._record(
                "write_potential_bytes", potential_access_bytes, potential_capacity
            )
            self._record(
                "write_out_bytes",
                int(out.count_nonzero().item()) * int(out.element_size()),
                self.fifo_capacity_elements * int(out.element_size()),
            )

        for module in model.modules():
            if isinstance(module, _SUPPORTED_LEMAIRE_MEMORY_MODULES):
                self.handles.append(module.register_forward_hook(synaptic_hook))
            elif isinstance(module, BaseNode):
                self.handles.append(module.register_forward_hook(neuron_hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LemaireEnergyProfiler:
    def __init__(self, *, config: LemaireEnergyConfig | None = None):
        """
        .. rubric:: API Language

        :ref:`中文 <LemaireEnergyProfiler-cn>` |
        :ref:`English <LemaireEnergyProfiler-en>`

        ----

        .. _LemaireEnergyProfiler-cn:

        * **中文**

        基于多个 public counter 组装的、仅面向前向推理的 Lemaire 能耗分析器。

        :param config: 能耗配置，若为 ``None`` 则使用默认配置
        :type config: LemaireEnergyConfig | None

        ----

        .. _LemaireEnergyProfiler-en:

        * **English**

        Inference-only Lemaire energy profiler composed from public counters.

        :param config: Energy configuration. If ``None``, uses the default configuration
        :type config: LemaireEnergyConfig | None
        """
        self.config = copy.deepcopy(config or LemaireEnergyConfig())
        self.model: nn.Module | None = None
        ignore_neurons = [BaseNode]
        self.synop_counter = SynOpCounter()
        self.mac_counter = MACCounter(extra_ignore_modules=ignore_neurons)
        self.ac_counter = ACCounter(extra_ignore_modules=ignore_neurons)
        self.neuron_state_counter = NeuronStateCounter(
            strict=self.config.strict,
            extra_state_rules=self.config.extra_state_rules,
            zero_ratio_threshold=self.config.sparse_zero_ratio_threshold,
            enable_sparse_memory_estimation=self.config.enable_sparse_memory_estimation,
        )
        self.addressing_counter = LemaireAddressingCounter()
        self._dispatch_mode = DispatchCounterMode(
            [
                self.synop_counter,
                self.mac_counter,
                self.ac_counter,
                self.neuron_state_counter,
                self.addressing_counter,
            ],
            strict=self.config.strict,
        )
        self._warnings: list[str] = []
        self._lemaire_tracker = _LemaireForwardTracker(
            strict=self.config.strict,
            fifo_capacity_elements=self.config.snn_fifo_capacity_elements,
        )

    def bind_model(self, model: nn.Module):
        self.model = model
        warned = False
        for module in model.modules():
            if not isinstance(module, BaseNode):
                continue
            if module.backend == "torch":
                continue
            message = (
                "LemaireEnergyProfiler only supports torch backend for BaseNode modules, "
                f"got {module.backend!r} from {module.__class__.__name__}."
            )
            if self.config.strict:
                raise ValueError(message)
            if not warned:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                self._warnings.append(message)
                warned = True

    def __enter__(self):
        self.synop_counter.reset()
        self.mac_counter.reset()
        self.ac_counter.reset()
        self.neuron_state_counter.reset()
        self.addressing_counter.reset()
        self._lemaire_tracker.reset()
        if self.model is not None:
            self._lemaire_tracker.attach(self.model)
        try:
            self._dispatch_mode.__enter__()
        except BaseException:
            self._lemaire_tracker.remove()
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lemaire_tracker.remove()
        return self._dispatch_mode.__exit__(exc_type, exc, tb)

    def get_report(self) -> LemaireEnergyReport:
        cost = self.config.cost_config
        memory_counts, buffers, memory_breakdown = self._lemaire_tracker.summarize(cost)
        projection = self.neuron_state_counter.get_projection_counts().get("Global", {})
        addressing = self.addressing_counter.get_metric_counts().get("Global", {})
        counts = {
            "synop": int(self.synop_counter.get_total()),
            "mac": int(self.mac_counter.get_total()),
            "ac": int(self.ac_counter.get_total()),
            "state_mac_like": int(projection.get("state_mac_like", 0)),
            "state_acc_like": int(projection.get("state_acc_like", 0)),
            **memory_counts,
            "acc_addr": int(addressing.get("acc_addr", 0)),
            "mac_addr": int(addressing.get("mac_addr", 0)),
        }
        ops_pj = (counts["ac"] + counts["state_acc_like"]) * cost.e_add_pj + (
            counts["mac"] + counts["state_mac_like"]
        ) * (cost.e_mul_pj + cost.e_add_pj)
        addressing_pj = counts["acc_addr"] * cost.e_add_pj + counts["mac_addr"] * (
            cost.e_mul_pj + cost.e_add_pj
        )
        inout_pj = memory_breakdown["inout_pj"]
        params_pj = memory_breakdown["params_pj"]
        potential_pj = memory_breakdown["potential_pj"]
        memory_pj = inout_pj + params_pj + potential_pj
        total_pj = ops_pj + addressing_pj + memory_pj
        warnings_list = (
            list(self._warnings)
            + list(self.neuron_state_counter.warnings)
            + list(self._lemaire_tracker.warnings)
        )
        return LemaireEnergyReport(
            total_pj=total_pj,
            breakdown_pj={
                "ops_pj": ops_pj,
                "addressing_pj": addressing_pj,
                "memory_pj": memory_pj,
                "inout_pj": inout_pj,
                "params_pj": params_pj,
                "potential_pj": potential_pj,
            },
            counts=counts,
            buffer_sizes_bytes=buffers,
            warnings=warnings_list,
        )


def estimate_lemaire_energy(
    model: nn.Module,
    inputs,
    *,
    config: LemaireEnergyConfig | None = None,
) -> LemaireEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <estimate_lemaire_energy-cn>` |
    :ref:`English <estimate_lemaire_energy-en>`

    ----

    .. _estimate_lemaire_energy-cn:

    * **中文**

    对一次前向推理执行 Lemaire 对齐的解析式能耗估计。

    :param model: 待分析的 PyTorch 模型
    :type model: torch.nn.Module

    :param inputs: 模型输入。如果为 ``tuple`` 或 ``list`` 则解包后传入 ``model(*inputs)``
    :type inputs: Any

    :param config: Lemaire 能耗配置，若为 ``None`` 则使用默认配置
    :type config: Optional[LemaireEnergyConfig]

    :return: Lemaire 对齐的解析式能耗报告
    :rtype: LemaireEnergyReport

    ----

    .. _estimate_lemaire_energy-en:

    * **English**

    Run one forward inference pass and return a Lemaire-aligned analytical
    energy report.

    :param model: PyTorch model to profile
    :type model: torch.nn.Module

    :param inputs: input to the model. If it is a ``tuple`` or ``list``,
        it will be unpacked as ``model(*inputs)``
    :type inputs: Any

    :param config: Lemaire energy configuration. If ``None``, default config is used
    :type config: Optional[LemaireEnergyConfig]

    :return: Lemaire-aligned analytical energy report
    :rtype: LemaireEnergyReport
    """

    profiler = LemaireEnergyProfiler(config=config)
    profiler.bind_model(model)
    with profiler:
        if isinstance(inputs, (tuple, list)):
            model(*inputs)
        else:
            model(inputs)
    return profiler.get_report()
