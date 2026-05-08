from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from typing import Any

import torch.nn as nn

from ..base import ActiveModuleTracker
from .config import SpikeSimEnergyConfig
from .counter import SpikeSimEventCounter
from .formulas import compute_spikesim_event_energy_breakdown
from .trace import SpikeSimEventTraceMode

__all__ = [
    "SpikeSimEnergyConfig",
    "SpikeSimEventEnergyReport",
    "SpikeSimEventEnergyProfiler",
    "estimate_spikesim_event_energy",
]


def _call_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


@dataclass
class SpikeSimEventEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <SpikeSimEventEnergyReport-cn>` |
    :ref:`English <SpikeSimEventEnergyReport-en>`

    ----

    .. _SpikeSimEventEnergyReport-cn:

    * **中文**

    Report for the event-driven SpikeSim runtime energy estimator.

    字段包括总能耗、按 stage 的能耗分解、事件统计、stage 元数据和 warning。

    ----

    .. _SpikeSimEventEnergyReport-en:

    * **English**

    Report for the event-driven SpikeSim runtime energy estimator.

    Fields include total energy, stage-wise energy breakdown, event stats,
    stage metadata, and warnings.
    """

    energy_total_pj: float
    energy_by_stage: dict[str, float]
    energy_by_component: dict[str, Any]
    event_stats_by_stage: dict[str, dict[str, Any]]
    stage_metadata: dict[str, dict[str, Any]]
    warnings: list[str]


class SpikeSimEventEnergyProfiler:
    r"""
    **API Language:**
    :ref:`中文 <SpikeSimEventEnergyProfiler-cn>` |
    :ref:`English <SpikeSimEventEnergyProfiler-en>`

    ----

    .. _SpikeSimEventEnergyProfiler-cn:

    * **中文**

    Runtime event-driven energy profiler inspired by SpikeSim.

    使用方式：

    - 以 context manager 方式包住一次真实前向传播
    - 结束后调用 ``get_report()`` 获取能耗报告

    ----

    .. _SpikeSimEventEnergyProfiler-en:

    * **English**

    Runtime event-driven energy profiler inspired by SpikeSim.

    Usage:

    - wrap one real forward pass in the profiler context
    - call ``get_report()`` afterwards to build the energy report
    """

    def __init__(
        self,
        *,
        config: SpikeSimEnergyConfig | None = None,
        strict: bool = False,
        verbose: bool = False,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEventEnergyProfiler.__init__-cn>` |
        :ref:`English <SpikeSimEventEnergyProfiler.__init__-en>`

        ----

        .. _SpikeSimEventEnergyProfiler.__init__-cn:

        * **中文**

        :param config: SpikeSim 事件驱动能耗配置；默认使用 ``SpikeSimEnergyConfig()``
        :param strict: 是否在 unsupported 情况下直接抛异常
        :param verbose: 是否打印逐 stage 的运行时统计信息

        ----

        .. _SpikeSimEventEnergyProfiler.__init__-en:

        * **English**

        :param config: SpikeSim event-driven energy config; defaults to
            ``SpikeSimEnergyConfig()``
        :param strict: whether to raise immediately on unsupported behaviors
        :param verbose: whether to print per-stage runtime statistics
        """
        self.config = (config or SpikeSimEnergyConfig()).copy()
        self.config.validate()
        self.strict = strict
        self.verbose = verbose
        self._counter = SpikeSimEventCounter(
            config=self.config,
            strict=self.strict,
            verbose=self.verbose,
        )
        self._tracker = ActiveModuleTracker()
        self._trace_mode = SpikeSimEventTraceMode(
            self._tracker,
            counter=self._counter,
        )
        self._active = False

    def __enter__(self):
        self._tracker.__enter__()
        try:
            self._trace_mode.__enter__()
        except Exception:
            self._tracker.__exit__(None, None, None)
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        trace_ret = False
        tracker_ret = False
        try:
            trace_ret = self._trace_mode.__exit__(exc_type, exc, tb)
        finally:
            tracker_ret = self._tracker.__exit__(exc_type, exc, tb)
        return trace_ret or tracker_ret

    def get_report(self) -> SpikeSimEventEnergyReport:
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEventEnergyProfiler.get_report-cn>` |
        :ref:`English <SpikeSimEventEnergyProfiler.get_report-en>`

        ----

        .. _SpikeSimEventEnergyProfiler.get_report-cn:

        * **中文**

        生成并返回完整的 SpikeSim 事件驱动能耗报告。

        ----

        .. _SpikeSimEventEnergyProfiler.get_report-en:

        * **English**

        Build and return the full event-driven SpikeSim energy report.
        """
        event_stats_by_stage = self._counter.get_stage_stats()
        stage_metadata = self._counter.get_stage_metadata()

        energy_by_stage: dict[str, float] = {}
        component_by_stage: dict[str, dict[str, float]] = {}
        component_totals: dict[str, float] = defaultdict(float)
        for stage, stats in event_stats_by_stage.items():
            metadata = stage_metadata[stage]
            breakdown = compute_spikesim_event_energy_breakdown(
                stats=stats, metadata=metadata, config=self.config
            )
            component_by_stage[stage] = breakdown
            energy_by_stage[stage] = breakdown["total_pj"]
            for key, value in breakdown.items():
                component_totals[key] += value

        warnings = list(self._counter.warnings)
        if not energy_by_stage:
            warnings.append(
                "No supported Conv2d stages were profiled by SpikeSim event energy."
            )

        return SpikeSimEventEnergyReport(
            energy_total_pj=component_totals["total_pj"],
            energy_by_stage=energy_by_stage,
            energy_by_component={
                "totals": component_totals,
                "by_stage": component_by_stage,
            },
            event_stats_by_stage=event_stats_by_stage,
            stage_metadata=stage_metadata,
            warnings=warnings,
        )

    def get_total(self) -> float:
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEventEnergyProfiler.get_total-cn>` |
        :ref:`English <SpikeSimEventEnergyProfiler.get_total-en>`

        ----

        .. _SpikeSimEventEnergyProfiler.get_total-cn:

        * **中文**

        返回总能耗（pJ）。

        ----

        .. _SpikeSimEventEnergyProfiler.get_total-en:

        * **English**

        Return total energy in pJ.
        """
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
        r"""
        **API Language:**
        :ref:`中文 <SpikeSimEventEnergyProfiler.get_counts-cn>` |
        :ref:`English <SpikeSimEventEnergyProfiler.get_counts-en>`

        ----

        .. _SpikeSimEventEnergyProfiler.get_counts-cn:

        * **中文**

        返回事件统计与 stage 元数据，便于与 ``op_counter`` 的计数接口风格对齐。

        ----

        .. _SpikeSimEventEnergyProfiler.get_counts-en:

        * **English**

        Return event stats and stage metadata in an ``op_counter``-like
        count shape.
        """
        report = self.get_report()
        return {
            "event_stats_by_stage": report.event_stats_by_stage,
            "stage_metadata": report.stage_metadata,
        }


def estimate_spikesim_event_energy(
    model: nn.Module,
    inputs,
    *,
    config: SpikeSimEnergyConfig | None = None,
    strict: bool = False,
    verbose: bool = False,
) -> SpikeSimEventEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <estimate_spikesim_event_energy-cn>` |
    :ref:`English <estimate_spikesim_event_energy-en>`

    ----

    .. _estimate_spikesim_event_energy-cn:

    * **中文**

    SpikeSim 事件驱动能耗估计的便捷入口。该函数会执行一次真实前向传播并返回
    能耗报告。

    :param model: 待统计模型
    :param inputs: 模型输入；若为 tuple/list 则按 ``model(*inputs)`` 调用
    :param config: SpikeSim 能耗配置
    :param strict: 是否在 unsupported 情况下直接抛异常
    :param verbose: 是否打印逐 stage 的运行时统计信息

    ----

    .. _estimate_spikesim_event_energy-en:

    * **English**

    Convenience entry for SpikeSim event-driven energy estimation.
    It runs one real forward pass and returns the energy report.

    :param model: model to profile
    :param inputs: model input; tuple/list will be passed as
        ``model(*inputs)``
    :param config: SpikeSim energy config
    :param strict: whether to raise immediately on unsupported behaviors
    :param verbose: whether to print per-stage runtime statistics
    """
    with SpikeSimEventEnergyProfiler(
        config=config, strict=strict, verbose=verbose
    ) as profiler:
        _ = _call_model(model, inputs)
    return profiler.get_report()
