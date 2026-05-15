from __future__ import annotations

import copy
import warnings
from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Any, Callable
from numbers import Real

import torch
import torch.nn as nn

from ...neuron.base_node import BaseNode
from ..ac import ACCounter
from ..base import DispatchCounterMode, is_binary_tensor
from ..mac import MACCounter
from ..memory_access import MemoryAccessCounter
from ..memory_residency import MemoryResidencyCounter
from ..neuron_state import NeuronStateCounter
from ..synop import SynOpCounter

__all__ = [
    "AnalyticalEnergyConfig",
    "AnalyticalEnergyCostConfig",
    "AnalyticalEnergyProfiler",
    "AnalyticalEnergyReport",
    "InferenceOnlyLemaireCompatibleReport",
    "estimate_analytical_energy",
]


def _tensor_size_bytes(tree: Any) -> int:
    if torch.is_tensor(tree):
        return int(tree.numel() * tree.element_size())
    if isinstance(tree, (tuple, list)):
        return sum(_tensor_size_bytes(item) for item in tree)
    if isinstance(tree, dict):
        return sum(_tensor_size_bytes(item) for item in tree.values())
    return 0


_SUPPORTED_LEMAIRE_SYNAPTIC_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
)


def _add_nested(lhs: Any, rhs: Any) -> Any:
    if isinstance(lhs, dict) or isinstance(rhs, dict):
        lhs = lhs if isinstance(lhs, dict) else {}
        rhs = rhs if isinstance(rhs, dict) else {}
        keys = set(lhs.keys()) | set(rhs.keys())
        return {key: _add_nested(lhs.get(key, 0), rhs.get(key, 0)) for key in keys}
    return lhs + rhs


def _subtract_nested(after: Any, before: Any) -> Any:
    if isinstance(after, dict) or isinstance(before, dict):
        after = after if isinstance(after, dict) else {}
        before = before if isinstance(before, dict) else {}
        keys = set(after.keys()) | set(before.keys())
        return {
            key: _subtract_nested(after.get(key, 0), before.get(key, 0))
            for key in keys
        }
    return after - before


@dataclass
class AnalyticalEnergyCostConfig:
    r"""
    **API Language:**
    :ref:`中文 <AnalyticalEnergyCostConfig-cn>` |
    :ref:`English <AnalyticalEnergyCostConfig-en>`

    ----

    .. _AnalyticalEnergyCostConfig-cn:

    * **中文**

    解析式能耗模型的成本配置。默认采用 Lemaire 风格的加法、乘法与分段存储成本。

    ``memory_cost_pj`` 的输入语义是用于插值的 memory 标量，优先应传 buffer-size
    估计而不是累计访问流量。

    ----

    .. _AnalyticalEnergyCostConfig-en:

    * **English**

    Cost configuration for the analytical energy model. By default it uses a
    Lemaire-style add/mul cost table and a piecewise memory cost curve.

    ``memory_cost_pj`` expects the interpolation memory scalar, ideally a
    buffer-size estimate rather than cumulative access traffic.
    """
    e_add_pj: float = 0.1
    e_mul_pj: float = 3.1
    memory_breakpoints: tuple[tuple[float, float], ...] = (
        (0.0, 0.0),
        (8.0 * 1024.0, 10.0),
        (32.0 * 1024.0, 20.0),
        (1024.0 * 1024.0, 100.0),
    )

    def __post_init__(self):
        points = self.memory_breakpoints
        if len(points) != 4:
            raise ValueError(
                "memory_breakpoints must contain exactly 4 (x, y) pairs."
            )
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
        if memory <= points[3][0]:
            (x0, y0), (x1, y1) = points[2], points[3]
            return y0 + (y1 - y0) / (x1 - x0) * (memory - x0)
        (x0, y0), (x1, y1) = points[2], points[3]
        return y1 + (y1 - y0) / (x1 - x0) * (memory - x1)


@dataclass
class AnalyticalEnergyConfig:
    r"""
    **API Language:**
    :ref:`中文 <AnalyticalEnergyConfig-cn>` |
    :ref:`English <AnalyticalEnergyConfig-en>`

    ----

    .. _AnalyticalEnergyConfig-cn:

    * **中文**

    控制解析式能耗分析器的行为，包括 strict 模式、是否收集 residency 信息，以及
    是否生成 inference-only 的 Lemaire 兼容投影。

    ----

    .. _AnalyticalEnergyConfig-en:

    * **English**

    Controls the analytical energy profiler, including strict-mode behavior,
    optional residency collection, and whether to emit the inference-only
    Lemaire-compatible projection.
    """
    strict: bool = False
    collect_residency: bool = False
    cost_config: AnalyticalEnergyCostConfig = field(
        default_factory=AnalyticalEnergyCostConfig
    )
    enable_inference_only_lemaire_projection: bool = True
    extra_state_rules: dict[type[nn.Module], Callable] = field(default_factory=dict)


@dataclass
class InferenceOnlyLemaireCompatibleReport:
    r"""
    **API Language:**
    :ref:`中文 <InferenceOnlyLemaireCompatibleReport-cn>` |
    :ref:`English <InferenceOnlyLemaireCompatibleReport-en>`

    ----

    .. _InferenceOnlyLemaireCompatibleReport-cn:

    * **中文**

    仅针对前向推理阶段的 Lemaire 兼容分桶报告。

    ----

    .. _InferenceOnlyLemaireCompatibleReport-en:

    * **English**

    Lemaire-compatible bucketed report for forward inference only.
    """
    inference_only_E_op_pj: float = 0.0
    inference_only_E_addr_pj: float = 0.0
    inference_only_E_inout_pj: float = 0.0
    inference_only_E_params_pj: float = 0.0
    inference_only_E_potential_pj: float = 0.0
    inference_only_E_total_pj: float = 0.0
    available: bool = False
    unavailable_reason: str | None = None


@dataclass
class AnalyticalEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <AnalyticalEnergyReport-cn>` |
    :ref:`English <AnalyticalEnergyReport-en>`

    ----

    .. _AnalyticalEnergyReport-cn:

    * **中文**

    解析式能耗分析报告，包含按 stage、按 component 的能耗分解，以及 inference-only
    的 Lemaire 兼容字段。

    ----

    .. _AnalyticalEnergyReport-en:

    * **English**

    Report for the analytical energy profiler, including stage/component
    breakdowns and inference-only Lemaire-compatible fields.
    """
    energy_total_pj: float
    energy_by_stage: dict[str, float]
    energy_by_component: dict[str, Any]
    counter_totals: dict[str, Any]
    state_breakdown: dict[str, Any]
    warnings: list[str]
    inference_only_lemaire_compatible: InferenceOnlyLemaireCompatibleReport


class _LemaireForwardTracker:
    def __init__(self, stage_getter):
        self.stage_getter = stage_getter
        self.handles: list[Any] = []
        self.read_in = 0
        self.write_out = 0
        self.read_params = 0
        self.read_in_buffer_bytes = 0
        self.write_out_buffer_bytes = 0
        self.read_params_buffer_bytes = 0

    def attach(self, model: nn.Module):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if self.stage_getter() != "forward":
                return
            input_bytes = _tensor_size_bytes(inputs)
            output_bytes = _tensor_size_bytes(output)
            self.read_in += input_bytes
            self.write_out += output_bytes
            self.read_in_buffer_bytes = max(self.read_in_buffer_bytes, input_bytes)
            self.write_out_buffer_bytes = max(
                self.write_out_buffer_bytes, output_bytes
            )
            param_bytes = 0
            for param in module.parameters(recurse=False):
                param_bytes += int(param.numel() * param.element_size())
            self.read_params += param_bytes
            self.read_params_buffer_bytes = max(
                self.read_params_buffer_bytes, param_bytes
            )

        for module in model.modules():
            if isinstance(module, _SUPPORTED_LEMAIRE_SYNAPTIC_MODULES):
                self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class _LemaireAddressingEstimator:
    def __init__(self, stage_getter):
        self.stage_getter = stage_getter
        self.handles: list[Any] = []
        self.acc_addr = 0
        self.mac_addr = 0

    def attach(self, model: nn.Module):
        def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any):
            if self.stage_getter() != "forward":
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x):
                return
            out = output[0] if isinstance(output, (tuple, list)) else output
            if not torch.is_tensor(out):
                return
            if isinstance(module, nn.Linear):
                if is_binary_tensor(x):
                    self.acc_addr += int(x.count_nonzero().item()) * module.out_features
                else:
                    self.acc_addr += int(x.numel()) + int(out.numel())
                return

            kernel_size = module.kernel_size
            if isinstance(kernel_size, int):
                kernel_volume = kernel_size
            else:
                kernel_volume = 1
                for value in kernel_size:
                    kernel_volume *= int(value)

            if is_binary_tensor(x):
                spike_num_in = int(x.count_nonzero().item())
                out_channels_per_group = (
                    module.out_channels // module.groups
                    if module.groups > 0
                    else module.out_channels
                )
                self.mac_addr += spike_num_in * 2
                self.acc_addr += spike_num_in * out_channels_per_group * kernel_volume
            else:
                self.acc_addr += int(x.numel()) + int(out.numel())
                self.acc_addr += module.out_channels * kernel_volume

        for module in model.modules():
            if isinstance(module, _SUPPORTED_LEMAIRE_SYNAPTIC_MODULES):
                self.handles.append(module.register_forward_hook(hook))

    def remove(self):
        for handle in self.handles:
            handle.remove()
        self.handles.clear()


class AnalyticalEnergyProfiler:
    r"""
    **API Language:**
    :ref:`中文 <AnalyticalEnergyProfiler-cn>` |
    :ref:`English <AnalyticalEnergyProfiler-en>`

    ----

    .. _AnalyticalEnergyProfiler-cn:

    * **中文**

    组合多个 op_counter 的解析式能耗分析器。

    使用方式：

    - 先调用 ``bind_model`` 绑定模型
    - 再以 context manager 包住一次运行
    - 可通过 ``stage`` 分阶段记录 ``forward``、``backward``、``optimizer``
    - 结束后调用 ``get_report`` 获取完整报告

    ----

    .. _AnalyticalEnergyProfiler-en:

    * **English**

    Analytical energy profiler that combines multiple op counters.

    Usage:

    - bind a model with ``bind_model``
    - wrap a run in the profiler context
    - optionally use ``stage`` to tag ``forward``/``backward``/``optimizer``
    - call ``get_report`` afterwards to obtain the final report
    """
    def __init__(self, *, config: AnalyticalEnergyConfig | None = None):
        self.config = copy.deepcopy(config or AnalyticalEnergyConfig())
        self.model: nn.Module | None = None
        self._current_stage_name: str | None = None
        ignore_neurons = [BaseNode]
        self.synop_counter = SynOpCounter()
        self.mac_counter = MACCounter(extra_ignore_modules=ignore_neurons)
        self.ac_counter = ACCounter(extra_ignore_modules=ignore_neurons)
        self.memory_access_counter = MemoryAccessCounter(
            extra_ignore_modules=ignore_neurons
        )
        self.neuron_state_counter = NeuronStateCounter(
            strict=self.config.strict,
            extra_state_rules=self.config.extra_state_rules,
        )
        self.residency_counter = (
            MemoryResidencyCounter(extra_ignore_modules=ignore_neurons)
            if self.config.collect_residency
            else None
        )
        counters = [
            self.synop_counter,
            self.mac_counter,
            self.ac_counter,
            self.memory_access_counter,
            self.neuron_state_counter,
        ]
        if self.residency_counter is not None:
            counters.append(self.residency_counter)
        self._dispatch_mode = DispatchCounterMode(
            counters,
            strict=self.config.strict,
            verbose=False,
        )
        self._active = False
        self._stage_snapshots: dict[str, dict[str, Any]] = defaultdict(dict)
        self._warnings: list[str] = []
        self._training_run = False
        self._lemaire_tracker = _LemaireForwardTracker(lambda: self._current_stage_name)
        self._addr_estimator = _LemaireAddressingEstimator(
            lambda: self._current_stage_name
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
                "AnalyticalEnergyProfiler only supports torch backend for BaseNode modules, "
                f"got {module.backend!r} from {module.__class__.__name__}."
            )
            if self.config.strict:
                raise ValueError(message)
            if not warned:
                warnings.warn(message, RuntimeWarning, stacklevel=2)
                self._warnings.append(message)
                warned = True

    def __enter__(self):
        self._current_stage_name = "forward"
        self._dispatch_mode.__enter__()
        self._active = True
        if self.model is not None:
            self._lemaire_tracker.attach(self.model)
            self._addr_estimator.attach(self.model)
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        try:
            self._lemaire_tracker.remove()
            self._addr_estimator.remove()
        finally:
            self._current_stage_name = None
        return self._dispatch_mode.__exit__(exc_type, exc, tb)

    @contextmanager
    def suspend(self):
        if not self._active:
            yield
            return
        prev_stage = self._current_stage_name
        self._current_stage_name = None
        self._dispatch_mode.__exit__(None, None, None)
        try:
            yield
        finally:
            self._dispatch_mode.__enter__()
            self._current_stage_name = prev_stage

    @contextmanager
    def stage(self, name: str):
        before = self._snapshot_totals()
        prev = self._current_stage_name
        self._current_stage_name = name
        try:
            yield
        finally:
            after = self._snapshot_totals()
            delta = _subtract_nested(after, before)
            if name in self._stage_snapshots:
                self._stage_snapshots[name] = _add_nested(
                    self._stage_snapshots[name], delta
                )
            else:
                self._stage_snapshots[name] = delta
            if name in {"backward", "optimizer"}:
                self._training_run = True
            self._current_stage_name = prev

    def _snapshot_totals(self) -> dict[str, Any]:
        snapshot = {
            "synop": self.synop_counter.get_total(),
            "mac": self.mac_counter.get_total(),
            "ac": self.ac_counter.get_total(),
            "memory_access_bytes": self.memory_access_counter.get_total(),
            "memory_buffer_bytes": max(
                self._lemaire_tracker.read_in_buffer_bytes,
                self._lemaire_tracker.write_out_buffer_bytes,
                self._lemaire_tracker.read_params_buffer_bytes,
            ),
            "state": self.neuron_state_counter.get_metric_counts().get("Global", {}),
            "projection": self.neuron_state_counter.get_projection_counts().get(
                "Global", {}
            ),
        }
        if self.residency_counter is not None:
            snapshot["memory_residency_bits"] = self.residency_counter.get_total()
        return snapshot

    def _build_component_energy(self, snapshot: dict[str, Any]) -> dict[str, float]:
        cost = self.config.cost_config
        projection = snapshot.get("projection", {})
        synop = int(snapshot.get("synop", 0))
        mac = int(snapshot.get("mac", 0))
        ac = int(snapshot.get("ac", 0))
        state_mac_like = int(projection.get("state_mac_like", 0))
        state_acc_like = int(projection.get("state_acc_like", 0))
        memory_access = float(snapshot.get("memory_access_bytes", 0))
        memory_buffer_bytes = float(
            snapshot.get("memory_buffer_bytes", snapshot.get("memory_access_bytes", 0))
        )
        read_potential = float(projection.get("read_potential", 0))
        write_potential = float(projection.get("write_potential", 0))
        potential_buffer_bytes = float(
            projection.get(
                "potential_buffer_bytes",
                max(read_potential, write_potential),
            )
        )

        synop_compute = synop * cost.e_add_pj
        mac_compute = mac * (cost.e_mul_pj + cost.e_add_pj)
        ac_compute = ac * cost.e_add_pj
        state_compute = state_acc_like * cost.e_add_pj + state_mac_like * (
            cost.e_mul_pj + cost.e_add_pj
        )
        memory_access_pj = memory_access * cost.memory_cost_pj(memory_buffer_bytes)
        state_memory_pj = read_potential * cost.memory_cost_pj(potential_buffer_bytes)
        state_memory_pj += write_potential * cost.memory_cost_pj(
            potential_buffer_bytes
        )

        total_compute = synop_compute + mac_compute + ac_compute + state_compute
        total_memory = memory_access_pj + state_memory_pj
        return {
            "synop_compute_pj": synop_compute,
            "mac_compute_pj": mac_compute,
            "ac_compute_pj": ac_compute,
            "state_compute_pj": state_compute,
            "memory_access_pj": memory_access_pj,
            "state_memory_pj": state_memory_pj,
            "total_compute_pj": total_compute,
            "total_memory_pj": total_memory,
            "total_pj": total_compute + total_memory,
        }

    def _build_lemaire_projection(
        self, snapshot: dict[str, Any]
    ) -> InferenceOnlyLemaireCompatibleReport:
        report = InferenceOnlyLemaireCompatibleReport()
        if self._training_run:
            report.unavailable_reason = (
                "Lemaire-compatible projection is inference-only."
            )
            return report
        if not self.config.enable_inference_only_lemaire_projection:
            report.unavailable_reason = (
                "Lemaire-compatible projection is disabled by config."
            )
            return report

        cost = self.config.cost_config
        projection = snapshot.get("projection", {})
        synop = int(snapshot.get("synop", 0))
        mac = int(snapshot.get("mac", 0))
        ac = int(snapshot.get("ac", 0))
        state_mac_like = int(projection.get("state_mac_like", 0))
        state_acc_like = int(projection.get("state_acc_like", 0))
        read_potential = float(projection.get("read_potential", 0))
        write_potential = float(projection.get("write_potential", 0))
        read_in = float(self._lemaire_tracker.read_in)
        write_out = float(self._lemaire_tracker.write_out)
        read_params = float(self._lemaire_tracker.read_params)
        read_in_buffer_bytes = float(self._lemaire_tracker.read_in_buffer_bytes or read_in)
        write_out_buffer_bytes = float(
            self._lemaire_tracker.write_out_buffer_bytes or write_out
        )
        read_params_buffer_bytes = float(
            self._lemaire_tracker.read_params_buffer_bytes or read_params
        )
        potential_buffer_bytes = float(
            projection.get(
                "potential_buffer_bytes",
                max(read_potential, write_potential),
            )
        )
        acc_addr = float(self._addr_estimator.acc_addr)
        mac_addr = float(self._addr_estimator.mac_addr)

        report.inference_only_E_op_pj = (
            (synop + ac + state_acc_like) * cost.e_add_pj
            + (mac + state_mac_like) * (cost.e_mul_pj + cost.e_add_pj)
        )
        report.inference_only_E_addr_pj = acc_addr * cost.e_add_pj + mac_addr * (
            cost.e_mul_pj + cost.e_add_pj
        )
        report.inference_only_E_inout_pj = read_in * cost.memory_cost_pj(
            read_in_buffer_bytes
        )
        report.inference_only_E_inout_pj += write_out * cost.memory_cost_pj(
            write_out_buffer_bytes
        )
        report.inference_only_E_params_pj = read_params * cost.memory_cost_pj(
            read_params_buffer_bytes
        )
        report.inference_only_E_potential_pj = read_potential * cost.memory_cost_pj(
            potential_buffer_bytes
        )
        report.inference_only_E_potential_pj += write_potential * cost.memory_cost_pj(
            potential_buffer_bytes
        )
        report.inference_only_E_total_pj = (
            report.inference_only_E_op_pj
            + report.inference_only_E_addr_pj
            + report.inference_only_E_inout_pj
            + report.inference_only_E_params_pj
            + report.inference_only_E_potential_pj
        )
        report.available = True
        return report

    def get_report(self) -> AnalyticalEnergyReport:
        totals = self._snapshot_totals()
        if not self._stage_snapshots:
            self._stage_snapshots["forward"] = totals
        components_totals = self._build_component_energy(totals)
        components_by_stage = {
            stage: self._build_component_energy(snapshot)
            for stage, snapshot in self._stage_snapshots.items()
        }
        energy_by_stage = {
            stage: values["total_pj"] for stage, values in components_by_stage.items()
        }
        warnings_list = list(self._warnings) + list(self.neuron_state_counter.warnings)
        return AnalyticalEnergyReport(
            energy_total_pj=components_totals["total_pj"],
            energy_by_stage=energy_by_stage,
            energy_by_component={
                "totals": components_totals,
                "by_stage": components_by_stage,
            },
            counter_totals={
                "totals": totals,
                "by_stage": dict(self._stage_snapshots),
            },
            state_breakdown={
                "totals": totals.get("state", {}),
                "by_stage": {
                    stage: snapshot.get("state", {})
                    for stage, snapshot in self._stage_snapshots.items()
                },
            },
            warnings=warnings_list,
            inference_only_lemaire_compatible=self._build_lemaire_projection(totals),
        )
def estimate_analytical_energy(
    model: nn.Module,
    inputs,
    *,
    config: AnalyticalEnergyConfig | None = None,
    target=None,
    loss_fn=None,
    optimizer=None,
) -> AnalyticalEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <estimate_analytical_energy-cn>` |
    :ref:`English <estimate_analytical_energy-en>`

    ----

    .. _estimate_analytical_energy-cn:

    * **中文**

    对一次模型运行执行解析式能耗估计。

    当传入 ``optimizer`` 时，会先调用 ``optimizer.zero_grad(set_to_none=True)``；
    若未传入 ``optimizer`` 但传入了 ``target``/``loss_fn``，则会调用
    ``model.zero_grad(set_to_none=True)``。若只做推理，则不会额外清零梯度。

    :param model: 待分析的模型
    :type model: nn.Module
    :param inputs: 模型输入
    :type inputs: Any
    :param config: 解析式能耗配置
    :type config: Optional[AnalyticalEnergyConfig]
    :param target: 训练目标；需与 ``loss_fn`` 成对提供
    :type target: Any
    :param loss_fn: 损失函数；需与 ``target`` 成对提供
    :type loss_fn: Any
    :param optimizer: 优化器；若提供则会统计 optimizer stage
    :type optimizer: Any
    :return: 解析式能耗报告
    :rtype: AnalyticalEnergyReport

    ----

    .. _estimate_analytical_energy-en:

    * **English**

    Run one analytical energy estimation pass on a model execution.

    When ``optimizer`` is provided, ``optimizer.zero_grad(set_to_none=True)`` is
    called first. If ``optimizer`` is absent but ``target``/``loss_fn`` are
    provided, ``model.zero_grad(set_to_none=True)`` is called instead. Pure
    inference does not clear gradients.

    :param model: model to profile
    :type model: nn.Module
    :param inputs: model inputs
    :type inputs: Any
    :param config: analytical energy configuration
    :type config: Optional[AnalyticalEnergyConfig]
    :param target: training target; must be paired with ``loss_fn``
    :type target: Any
    :param loss_fn: loss function; must be paired with ``target``
    :type loss_fn: Any
    :param optimizer: optimizer; when provided, the optimizer stage is tracked
    :type optimizer: Any
    :return: analytical energy report
    :rtype: AnalyticalEnergyReport
    """
    if (target is None) ^ (loss_fn is None):
        raise ValueError("target and loss_fn must be provided together.")

    profiler = AnalyticalEnergyProfiler(config=config)
    profiler.bind_model(model)

    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
    elif target is not None:
        model.zero_grad(set_to_none=True)

    with profiler:
        with profiler.stage("forward"):
            if isinstance(inputs, (tuple, list)):
                outputs = model(*inputs)
            else:
                outputs = model(inputs)

        if target is not None and loss_fn is not None:
            with profiler.suspend():
                loss = loss_fn(outputs, target)
            with profiler.stage("backward"):
                loss.backward()
            if optimizer is not None:
                with profiler.stage("optimizer"):
                    optimizer.step()

    return profiler.get_report()
