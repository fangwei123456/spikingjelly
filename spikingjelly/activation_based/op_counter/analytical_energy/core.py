from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from numbers import Real
from typing import Any, Callable

import torch
import torch.nn as nn

from ...neuron.base_node import BaseNode
from .._sparse_memory import (
    active_element_count,
    dense_bytes,
    dense_bytes_tree,
    is_sparse_access_tensor,
    sparse_bytes,
)
from ..ac import ACCounter
from ..base import DispatchCounterMode
from ..lemaire_addressing import LemaireAddressingCounter
from ..mac import MACCounter
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
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


@dataclass
class LemaireEnergyCostConfig:
    r"""
    **API Language:**
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
    **API Language:**
    :ref:`中文 <LemaireEnergyConfig-cn>` |
    :ref:`English <LemaireEnergyConfig-en>`

    ----

    .. _LemaireEnergyConfig-cn:

    * **中文**

    控制 inference-only、Lemaire 对齐能耗分析器的行为。

    ----

    .. _LemaireEnergyConfig-en:

    * **English**

    Controls the inference-only, Lemaire-aligned energy profiler.
    """

    strict: bool = False
    cost_config: LemaireEnergyCostConfig = field(
        default_factory=LemaireEnergyCostConfig
    )
    extra_state_rules: dict[type[nn.Module], Callable] = field(default_factory=dict)
    sparse_zero_ratio_threshold: float = 0.5
    enable_sparse_memory_estimation: bool = True


@dataclass
class LemaireEnergyReport:
    r"""
    **API Language:**
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
    def __init__(self, *, zero_ratio_threshold: float, enable_sparse: bool):
        self.zero_ratio_threshold = zero_ratio_threshold
        self.enable_sparse = enable_sparse
        self.handles: list[Any] = []
        self.read_in = 0
        self.write_out = 0
        self.read_params = 0
        self.read_in_buffer_bytes = 0
        self.write_out_buffer_bytes = 0
        self.read_params_buffer_bytes = 0
        self.warnings: list[str] = []
        self._warned_module_types: set[type[nn.Module]] = set()

    def _warn_dense_fallback(self, module: nn.Module):
        module_type = type(module)
        if module_type in self._warned_module_types:
            return
        self._warned_module_types.add(module_type)
        message = (
            f"Lemaire forward tracker falls back to dense lower-bound memory for "
            f"{module_type.__name__}."
        )
        self.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=3)

    def reset(self):
        self.read_in = 0
        self.write_out = 0
        self.read_params = 0
        self.read_in_buffer_bytes = 0
        self.write_out_buffer_bytes = 0
        self.read_params_buffer_bytes = 0
        self.warnings.clear()
        self._warned_module_types.clear()

    def _track_bytes(self, read_in: int, write_out: int, read_params: int):
        self.read_in += int(read_in)
        self.write_out += int(write_out)
        self.read_params += int(read_params)
        self.read_in_buffer_bytes = max(self.read_in_buffer_bytes, int(read_in))
        self.write_out_buffer_bytes = max(self.write_out_buffer_bytes, int(write_out))
        self.read_params_buffer_bytes = max(
            self.read_params_buffer_bytes, int(read_params)
        )

    def attach(self, model: nn.Module):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if not inputs:
                return
            x = inputs[0]
            out = (
                output[0]
                if isinstance(output, (tuple, list)) and len(output) > 0
                else output
            )
            if not torch.is_tensor(x) or not torch.is_tensor(out):
                self._track_bytes(
                    dense_bytes_tree(inputs),
                    dense_bytes_tree(output),
                    sum(
                        dense_bytes(param) for param in module.parameters(recurse=False)
                    ),
                )
                return

            input_is_sparse = self.enable_sparse and is_sparse_access_tensor(
                x, zero_ratio_threshold=self.zero_ratio_threshold
            )
            output_is_sparse = self.enable_sparse and is_sparse_access_tensor(
                out, zero_ratio_threshold=self.zero_ratio_threshold
            )

            if isinstance(module, nn.Linear):
                if not input_is_sparse:
                    read_in = dense_bytes(x)
                    write_out = dense_bytes(out)
                    read_params = sum(
                        dense_bytes(param) for param in module.parameters(recurse=False)
                    )
                else:
                    active_inputs = active_element_count(x)
                    read_in = sparse_bytes(x)
                    read_params = (
                        active_inputs
                        * module.out_features
                        * int(module.weight.element_size())
                    )
                    if module.bias is not None:
                        output_active = (
                            active_element_count(out)
                            if output_is_sparse
                            else int(out.numel())
                        )
                        read_params += output_active * int(module.bias.element_size())
                    write_out = (
                        sparse_bytes(out) if output_is_sparse else dense_bytes(out)
                    )
                self._track_bytes(read_in, write_out, read_params)
                return

            if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
                if not input_is_sparse:
                    read_in = dense_bytes(x)
                    write_out = dense_bytes(out)
                    read_params = sum(
                        dense_bytes(param) for param in module.parameters(recurse=False)
                    )
                else:
                    active_inputs = active_element_count(x)
                    kernel_volume = 1
                    for dim in module.kernel_size:
                        kernel_volume *= int(dim)
                    out_channels_per_group = module.out_channels // module.groups
                    read_in = sparse_bytes(x)
                    read_params = (
                        active_inputs
                        * out_channels_per_group
                        * kernel_volume
                        * int(module.weight.element_size())
                    )
                    if module.bias is not None:
                        output_active = (
                            active_element_count(out)
                            if output_is_sparse
                            else int(out.numel())
                        )
                        read_params += output_active * int(module.bias.element_size())
                    write_out = (
                        sparse_bytes(out) if output_is_sparse else dense_bytes(out)
                    )
                self._track_bytes(read_in, write_out, read_params)
                return

            if input_is_sparse or output_is_sparse:
                self._warn_dense_fallback(module)
            self._track_bytes(
                dense_bytes_tree(inputs),
                dense_bytes_tree(output),
                sum(dense_bytes(param) for param in module.parameters(recurse=False)),
            )

        self.remove()
        for module in model.modules():
            if isinstance(module, _SUPPORTED_LEMAIRE_MEMORY_MODULES):
                self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class LemaireEnergyProfiler:
    r"""
    **API Language:**
    :ref:`中文 <LemaireEnergyProfiler-cn>` |
    :ref:`English <LemaireEnergyProfiler-en>`

    ----

    .. _LemaireEnergyProfiler-cn:

    * **中文**

    基于多个 public counter 组装的、仅面向前向推理的 Lemaire 能耗分析器。

    ----

    .. _LemaireEnergyProfiler-en:

    * **English**

    Inference-only Lemaire energy profiler composed from public counters.
    """

    def __init__(self, *, config: LemaireEnergyConfig | None = None):
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
            verbose=False,
        )
        self._warnings: list[str] = []
        self._lemaire_tracker = _LemaireForwardTracker(
            zero_ratio_threshold=self.config.sparse_zero_ratio_threshold,
            enable_sparse=self.config.enable_sparse_memory_estimation,
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
        self._dispatch_mode.__enter__()
        self._lemaire_tracker.reset()
        if self.model is not None:
            self._lemaire_tracker.attach(self.model)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._lemaire_tracker.remove()
        return self._dispatch_mode.__exit__(exc_type, exc, tb)

    def _snapshot_counts(self) -> tuple[dict[str, int], dict[str, int]]:
        projection = self.neuron_state_counter.get_projection_counts().get("Global", {})
        addressing = self.addressing_counter.get_metric_counts().get("Global", {})
        counts = {
            "synop": int(self.synop_counter.get_total()),
            "mac": int(self.mac_counter.get_total()),
            "ac": int(self.ac_counter.get_total()),
            "state_mac_like": int(projection.get("state_mac_like", 0)),
            "state_acc_like": int(projection.get("state_acc_like", 0)),
            "read_in_bytes": int(self._lemaire_tracker.read_in),
            "write_out_bytes": int(self._lemaire_tracker.write_out),
            "read_params_bytes": int(self._lemaire_tracker.read_params),
            "read_potential_bytes": int(projection.get("read_potential", 0)),
            "write_potential_bytes": int(projection.get("write_potential", 0)),
            "acc_addr": int(addressing.get("acc_addr", 0)),
            "mac_addr": int(addressing.get("mac_addr", 0)),
        }
        buffers = {
            "inout_buffer_bytes": max(
                int(self._lemaire_tracker.read_in_buffer_bytes),
                int(self._lemaire_tracker.write_out_buffer_bytes),
            ),
            "params_buffer_bytes": int(self._lemaire_tracker.read_params_buffer_bytes),
            "potential_buffer_bytes": int(projection.get("potential_buffer_bytes", 0)),
        }
        return counts, buffers

    def get_report(self) -> LemaireEnergyReport:
        counts, buffers = self._snapshot_counts()
        cost = self.config.cost_config
        ops_pj = (
            counts["synop"] + counts["ac"] + counts["state_acc_like"]
        ) * cost.e_add_pj + (counts["mac"] + counts["state_mac_like"]) * (
            cost.e_mul_pj + cost.e_add_pj
        )
        addressing_pj = counts["acc_addr"] * cost.e_add_pj + counts["mac_addr"] * (
            cost.e_mul_pj + cost.e_add_pj
        )
        inout_pj = counts["read_in_bytes"] * cost.memory_cost_pj(
            max(0, self._lemaire_tracker.read_in_buffer_bytes)
        )
        inout_pj += counts["write_out_bytes"] * cost.memory_cost_pj(
            max(0, self._lemaire_tracker.write_out_buffer_bytes)
        )
        params_pj = counts["read_params_bytes"] * cost.memory_cost_pj(
            max(0, buffers["params_buffer_bytes"])
        )
        potential_pj = counts["read_potential_bytes"] * cost.memory_cost_pj(
            max(0, buffers["potential_buffer_bytes"])
        )
        potential_pj += counts["write_potential_bytes"] * cost.memory_cost_pj(
            max(0, buffers["potential_buffer_bytes"])
        )
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
    **API Language:**
    :ref:`中文 <estimate_lemaire_energy-cn>` |
    :ref:`English <estimate_lemaire_energy-en>`

    ----

    .. _estimate_lemaire_energy-cn:

    * **中文**

    对一次前向推理执行 Lemaire 对齐的解析式能耗估计。

    ----

    .. _estimate_lemaire_energy-en:

    * **English**

    Run one forward inference pass and return a Lemaire-aligned analytical
    energy report.
    """

    profiler = LemaireEnergyProfiler(config=config)
    profiler.bind_model(model)
    with profiler:
        if isinstance(inputs, (tuple, list)):
            model(*inputs)
        else:
            model(inputs)
    return profiler.get_report()
