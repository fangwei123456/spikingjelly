from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch
import torch.nn as nn

from ...neuron.base_node import BaseNode, SimpleBaseNode
from ...neuron.integrate_and_fire import IFNode, SimpleIFNode
from ...neuron.lif import LIFNode, SimpleLIFNode
from ..base import DispatchCounterMode
from .config import SpikeSimEnergyConfig
from .counter import SpikeSimCounter
from .formulas import (
    compute_spikesim_dense_energy_breakdown,
    compute_spikesim_event_energy_breakdown,
)

__all__ = [
    "SpikeSimEnergyConfig",
    "SpikeSimCounter",
    "SpikeSimEnergyReport",
    "SpikeSimEnergyProfiler",
    "SpikeSimEventEnergyReport",
    "SpikeSimEventEnergyProfiler",
    "estimate_spikesim_event_energy",
]


def _call_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


_SUPPORTED_SPIKESIM_NEURONS = (
    IFNode,
    LIFNode,
    SimpleIFNode,
    SimpleLIFNode,
)


def _unsupported_neuron_modules(model: nn.Module) -> list[tuple[str, str]]:
    unsupported: list[tuple[str, str]] = []
    for name, module in model.named_modules():
        if not isinstance(module, (BaseNode, SimpleBaseNode)):
            continue
        if isinstance(module, _SUPPORTED_SPIKESIM_NEURONS):
            continue
        unsupported.append(
            (name or module.__class__.__name__, module.__class__.__name__)
        )
    return unsupported


@dataclass
class SpikeSimEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <SpikeSimEnergyReport-cn>` |
    :ref:`English <SpikeSimEnergyReport-en>`

    ----

    .. _SpikeSimEnergyReport-cn:

    * **中文**

    SpikeSim 运行时能耗估计器的报告, 支持 ``dense`` 和 ``event`` 两种 activity
    mode。

    字段包括总能耗、stage 分解、统计量、stage 元数据和 warning。
    ``event_stats_by_stage`` 在两种 mode 下都会填充。

    ----

    .. _SpikeSimEnergyReport-en:

    * **English**

    Report for the SpikeSim runtime energy estimator, supporting both ``dense``
    and ``event`` activity modes.

    Fields include total energy, stage-wise energy breakdown, event stats,
    stage metadata, and warnings. ``event_stats_by_stage`` is populated
    regardless of the selected activity mode.
    """

    energy_total_pj: float
    energy_by_stage: dict[str, float]
    energy_by_component: dict[str, Any]
    event_stats_by_stage: dict[str, dict[str, Any]]
    stage_metadata: dict[str, dict[str, Any]]
    warnings: list[str]
    breakdown_pj: dict[str, float]
    counts: dict[str, int]


class SpikeSimEnergyProfiler:
    def __init__(
        self,
        *,
        config: SpikeSimEnergyConfig | None = None,
        strict: bool = False,
        verbose: bool = False,
    ):
        r"""
        .. rubric:: API Language

        :ref:`中文 <SpikeSimEnergyProfiler.__init__-cn>` |
        :ref:`English <SpikeSimEnergyProfiler.__init__-en>`

        ----

        .. _SpikeSimEnergyProfiler.__init__-cn:

        * **中文**

        Runtime SpikeSim-aligned energy profiler.

        使用方式：

        - 以 context manager 方式包住一次真实前向传播
        - 结束后调用 ``get_report()`` 获取能耗报告

        :param config: SpikeSim 能耗配置；默认使用 ``SpikeSimEnergyConfig()``
        :param strict: 是否在 unsupported 情况下直接抛异常
        :param verbose: 是否打印逐 stage 的运行时统计信息

        ----

        .. _SpikeSimEnergyProfiler.__init__-en:

        * **English**

        Runtime SpikeSim-aligned energy profiler.

        Usage:

        - wrap one real forward pass in the profiler context
        - call ``get_report()`` afterwards to build the energy report

        :param config: SpikeSim energy config; defaults to ``SpikeSimEnergyConfig()``
        :param strict: whether to raise immediately on unsupported behaviors
        :param verbose: whether to print per-stage runtime statistics
        """
        self.config = (config or SpikeSimEnergyConfig()).copy()
        self.config.validate()
        self.strict = strict
        self.verbose = verbose
        self._counter = SpikeSimCounter(
            config=self.config,
            strict=self.strict,
            verbose=self.verbose,
        )
        self._dispatch_mode = DispatchCounterMode(
            [self._counter],
            strict=False,
            verbose=self.verbose,
        )
        self._active = False
        self._warnings: list[str] = []

    def __enter__(self):
        self._dispatch_mode.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        return self._dispatch_mode.__exit__(exc_type, exc, tb)

    def add_warnings(self, messages: list[str]) -> None:
        self._warnings.extend(messages)

    def get_report(self) -> SpikeSimEnergyReport:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SpikeSimEnergyProfiler.get_report-cn>` |
        :ref:`English <SpikeSimEnergyProfiler.get_report-en>`

        ----

        .. _SpikeSimEnergyProfiler.get_report-cn:

        * **中文**

        生成并返回完整的 SpikeSim runtime 能耗报告。

        ----

        .. _SpikeSimEnergyProfiler.get_report-en:

        * **English**

        Build and return the full runtime SpikeSim energy report.
        """
        event_stats_by_stage = self._counter.get_stage_stats()
        stage_metadata = self._counter.get_stage_metadata()

        energy_by_stage: dict[str, float] = {}
        component_by_stage: dict[str, dict[str, float]] = {}
        component_totals: dict[str, float] = defaultdict(float)
        for stage, stats in event_stats_by_stage.items():
            metadata = stage_metadata[stage]
            if self.config.activity_mode == "event":
                breakdown = compute_spikesim_event_energy_breakdown(
                    stats=stats, metadata=metadata, config=self.config
                )
            else:
                breakdown = compute_spikesim_dense_energy_breakdown(
                    stats=stats, metadata=metadata, config=self.config
                )
            component_by_stage[stage] = breakdown
            energy_by_stage[stage] = breakdown["total_pj"]
            for key, value in breakdown.items():
                if key == "total_pj":
                    continue
                component_totals[key] += value

        warnings = list(self._counter.warnings)
        warnings = list(self._warnings) + warnings
        if not energy_by_stage:
            warnings.append(
                "No supported Conv2d forward inference stages were profiled by "
                "SpikeSim energy."
            )

        totals_dict = dict(component_totals)
        total_pj = sum(energy_by_stage.values())
        counts = {
            "dense_pe_cycle_count": int(self._counter.get_total()),
            "stage_count": len(stage_metadata),
        }
        if self.config.activity_mode == "event":
            warnings.append(
                "SpikeSim activity_mode='event' is an experimental sparse runtime "
                "extension; activity_mode='dense' is the original SpikeSim-aligned "
                "default."
            )
        return SpikeSimEnergyReport(
            energy_total_pj=total_pj,
            energy_by_stage=energy_by_stage,
            energy_by_component={
                "totals": {**totals_dict, "total_pj": total_pj},
                "by_stage": component_by_stage,
            },
            event_stats_by_stage=event_stats_by_stage,
            stage_metadata=stage_metadata,
            warnings=warnings,
            breakdown_pj=totals_dict,
            counts=counts,
        )

    def get_total(self) -> float:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SpikeSimEnergyProfiler.get_total-cn>` |
        :ref:`English <SpikeSimEnergyProfiler.get_total-en>`

        ----

        .. _SpikeSimEnergyProfiler.get_total-cn:

        * **中文**

        返回总能耗（pJ）。

        ----

        .. _SpikeSimEnergyProfiler.get_total-en:

        * **English**

        Return total energy in pJ.
        """
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SpikeSimEnergyProfiler.get_counts-cn>` |
        :ref:`English <SpikeSimEnergyProfiler.get_counts-en>`

        ----

        .. _SpikeSimEnergyProfiler.get_counts-cn:

        * **中文**

        返回统计量与 stage 元数据，便于和计数接口对齐。

        ----

        .. _SpikeSimEnergyProfiler.get_counts-en:

        * **English**

        Return event stats and stage metadata in an ``op_counter``-like
        count shape.
        """
        report = self.get_report()
        return {
            "event_stats_by_stage": report.event_stats_by_stage,
            "stage_metadata": report.stage_metadata,
        }


SpikeSimEventEnergyReport = SpikeSimEnergyReport
SpikeSimEventEnergyProfiler = SpikeSimEnergyProfiler


def estimate_spikesim_event_energy(
    model: nn.Module,
    inputs,
    *,
    config: SpikeSimEnergyConfig | None = None,
    strict: bool = False,
    verbose: bool = False,
) -> SpikeSimEventEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <estimate_spikesim_event_energy-cn>` |
    :ref:`English <estimate_spikesim_event_energy-en>`

    ----

    .. _estimate_spikesim_event_energy-cn:

    * **中文**

    SpikeSim runtime 能耗估计的便捷入口。

    该函数会执行一次真实前向传播并返回能耗报告。

    :param model: 待统计模型
    :param inputs: 模型输入；若为 tuple/list 则按 ``model(*inputs)`` 调用
    :param config: SpikeSim 能耗配置
    :param strict: 是否在 unsupported 情况下直接抛异常
    :param verbose: 是否打印逐 stage 的运行时统计信息

    ----

    .. _estimate_spikesim_event_energy-en:

    * **English**

    Convenience entry for runtime SpikeSim energy estimation.
    It runs one real forward pass and returns the energy report.

    :param model: model to profile
    :param inputs: model input; tuple/list will be passed as
        ``model(*inputs)``
    :param config: SpikeSim energy config
    :param strict: whether to raise immediately on unsupported behaviors
    :param verbose: whether to print per-stage runtime statistics
    """
    cfg = (config or SpikeSimEnergyConfig()).copy()
    cfg.validate()
    training_message = (
        "SpikeSim energy only covers forward inference; call model.eval() before "
        "profiling."
    )
    if model.training:
        if strict:
            raise ValueError(training_message)
    neuron_warnings: list[str] = []
    if cfg.require_if_lif_neurons:
        unsupported_neurons = _unsupported_neuron_modules(model)
        if unsupported_neurons:
            details = ", ".join(
                f"{name} ({class_name})" for name, class_name in unsupported_neurons
            )
            message = (
                "SpikeSim assumes IF/LIF neuronal modules. Unsupported neuron "
                f"modules found: {details}."
            )
            if strict:
                raise ValueError(message)
            neuron_warnings.append(message)
    with SpikeSimEnergyProfiler(config=cfg, strict=strict, verbose=verbose) as profiler:
        profiler.add_warnings(neuron_warnings)
        with torch.no_grad():
            _ = _call_model(model, inputs)
    report = profiler.get_report()
    if model.training:
        report.warnings.insert(0, training_message)
    return report
