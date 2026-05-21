from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn

from .ac import ACCounter
from .base import DispatchCounterMode
from .flop import FlopCounter
from .mac import MACCounter
from .synop import SynOpCounter

__all__ = [
    "ComputeEnergyCostConfig",
    "ComputeEnergyConfig",
    "ComputeEnergyProfiler",
    "ComputeEnergyReport",
    "estimate_compute_energy",
]


def _call_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


@dataclass
class ComputeEnergyCostConfig:
    r"""
    **API Language:**
    :ref:`中文 <ComputeEnergyCostConfig-cn>` |
    :ref:`English <ComputeEnergyCostConfig-en>`

    ----

    .. _ComputeEnergyCostConfig-cn:

    * **中文**

    仅计算 MAC/AC 的 compute-only 能耗模型成本配置。

    默认值采用 SNN 文献中常见的 Horowitz 2014 口径：45nm、32-bit 浮点
    ``E_MAC = 4.6 pJ``，``E_AC = 0.9 pJ``。

    这是一个 cost-table-driven 的归一化模型。默认不会自动根据运行时
    ``dtype`` 推断成本；如需切换口径，请显式使用 ``fp32()``, ``fp16()``,
    ``int8()`` 等 preset。

    ----

    .. _ComputeEnergyCostConfig-en:

    * **English**

    Cost configuration for the compute-only MAC/AC energy model.

    Defaults follow the widely used Horowitz 2014 reference costs for 45nm,
    32-bit floating-point arithmetic: ``E_MAC = 4.6 pJ`` and
    ``E_AC = 0.9 pJ``.

    This is a normalized, cost-table-driven model. It does not automatically
    infer energy costs from runtime ``dtype``; use explicit presets such as
    ``fp32()``, ``fp16()``, or ``int8()`` when a different comparison regime is
    desired.
    """

    e_mac_pj: float = 4.6
    e_ac_pj: float = 0.9

    @classmethod
    def fp32(cls) -> "ComputeEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm FP32 preset.
        """
        return cls(e_mac_pj=4.6, e_ac_pj=0.9)

    @classmethod
    def fp16(cls) -> "ComputeEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm FP16 preset.

        Uses ``FMult16 = 1.1 pJ`` and ``FAdd16 = 0.4 pJ``, so
        ``E_MAC = 1.5 pJ`` and ``E_AC = 0.4 pJ``.
        """
        return cls(e_mac_pj=1.5, e_ac_pj=0.4)

    @classmethod
    def int8(cls) -> "ComputeEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm INT8 preset.

        Uses ``Mult8 = 0.2 pJ`` and ``Add8 = 0.03 pJ``, so
        ``E_MAC = 0.23 pJ`` and ``E_AC = 0.03 pJ``.
        """
        return cls(e_mac_pj=0.23, e_ac_pj=0.03)


@dataclass
class ComputeEnergyConfig:
    r"""
    **API Language:**
    :ref:`中文 <ComputeEnergyConfig-cn>` |
    :ref:`English <ComputeEnergyConfig-en>`

    ----

    .. _ComputeEnergyConfig-cn:

    * **中文**

    控制 compute-only MAC/AC 能耗分析器行为的配置。

    默认 ``cost_config`` 使用 ``ComputeEnergyCostConfig.fp32()`` 对应的口径。

    ----

    .. _ComputeEnergyConfig-en:

    * **English**

    Configuration for the compute-only MAC/AC energy profiler.

    The default ``cost_config`` matches ``ComputeEnergyCostConfig.fp32()``.
    """

    strict: bool = False
    verbose: bool = False
    cost_config: ComputeEnergyCostConfig = field(
        default_factory=ComputeEnergyCostConfig
    )
    extra_ignore_modules: list[nn.Module] | None = None


@dataclass
class ComputeEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <ComputeEnergyReport-cn>` |
    :ref:`English <ComputeEnergyReport-en>`

    ----

    .. _ComputeEnergyReport-cn:

    * **中文**

    compute-only MAC/AC 能耗报告。

    该模型只考虑计算能耗，不包含访存、寻址、状态驻留等开销。主结果
    ``energy_total_pj`` 由 ``MAC`` 和 ``AC`` 两部分组成。

    ``SynOps`` 与 ``FLOPs`` 作为辅助统计返回，便于与现有 SNN/ANN 文献对齐，
    但不参与主能耗计算。

    该估计器面向“统一比较口径”，而不是对真实 kernel、混合精度累加路径或
    特定硬件微架构做精确建模。

    ----

    .. _ComputeEnergyReport-en:

    * **English**

    Report for the compute-only MAC/AC energy model.

    This model only accounts for arithmetic compute energy, excluding memory,
    addressing, and state residency costs. The main result ``energy_total_pj``
    consists of ``MAC`` and ``AC`` contributions only.

    ``SynOps`` and ``FLOPs`` are returned as auxiliary counts for alignment
    with existing SNN/ANN literature, but they do not contribute to the primary
    energy estimate.

    The estimator is intended as a normalized comparison regime rather than an
    exact model of real kernels, mixed-precision accumulation paths, or a
    specific hardware microarchitecture.
    """

    energy_total_pj: float
    energy_mac_pj: float
    energy_ac_pj: float
    breakdown_pj: dict[str, float]
    counts: dict[str, int]
    warnings: list[str]


class ComputeEnergyProfiler:
    r"""
    **API Language:**
    :ref:`中文 <ComputeEnergyProfiler-cn>` |
    :ref:`English <ComputeEnergyProfiler-en>`

    ----

    .. _ComputeEnergyProfiler-cn:

    * **中文**

    基于 public counter 组装的 compute-only MAC/AC 能耗分析器。

    用法与其他能耗分析器一致：以 context manager 方式包住一次真实前向传播，
    然后调用 ``get_report()``。

    ----

    .. _ComputeEnergyProfiler-en:

    * **English**

    Compute-only MAC/AC energy profiler composed from public counters.

    Use it like the other energy profilers: wrap one real forward pass in the
    context manager and call ``get_report()`` afterwards.
    """

    def __init__(self, *, config: ComputeEnergyConfig | None = None):
        self.config = copy.deepcopy(config or ComputeEnergyConfig())
        ignore_modules = list(self.config.extra_ignore_modules or [])
        self.mac_counter = MACCounter(extra_ignore_modules=ignore_modules)
        self.ac_counter = ACCounter(extra_ignore_modules=ignore_modules)
        self.synop_counter = SynOpCounter(extra_ignore_modules=ignore_modules)
        self.flop_counter = FlopCounter(extra_ignore_modules=ignore_modules)
        self._dispatch_mode = DispatchCounterMode(
            [
                self.mac_counter,
                self.ac_counter,
                self.synop_counter,
                self.flop_counter,
            ],
            strict=self.config.strict,
            verbose=self.config.verbose,
        )

    def __enter__(self):
        self.mac_counter.reset()
        self.ac_counter.reset()
        self.synop_counter.reset()
        self.flop_counter.reset()
        self._dispatch_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._dispatch_mode.__exit__(exc_type, exc, tb)

    def get_report(self) -> ComputeEnergyReport:
        mac = int(self.mac_counter.get_total())
        ac = int(self.ac_counter.get_total())
        synop = int(self.synop_counter.get_total())
        flop = int(self.flop_counter.get_total())
        cost = self.config.cost_config

        energy_mac_pj = mac * cost.e_mac_pj
        energy_ac_pj = ac * cost.e_ac_pj
        total_pj = energy_mac_pj + energy_ac_pj

        return ComputeEnergyReport(
            energy_total_pj=total_pj,
            energy_mac_pj=energy_mac_pj,
            energy_ac_pj=energy_ac_pj,
            breakdown_pj={
                "mac_pj": energy_mac_pj,
                "ac_pj": energy_ac_pj,
            },
            counts={
                "mac": mac,
                "ac": ac,
                "synop": synop,
                "flop": flop,
            },
            warnings=[],
        )

    def get_total(self) -> float:
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
        report = self.get_report()
        return {"counts": report.counts}


def estimate_compute_energy(
    model: nn.Module,
    inputs,
    *,
    config: ComputeEnergyConfig | None = None,
) -> ComputeEnergyReport:
    r"""
    **API Language:**
    :ref:`中文 <estimate_compute_energy-cn>` |
    :ref:`English <estimate_compute_energy-en>`

    ----

    .. _estimate_compute_energy-cn:

    * **中文**

    compute-only MAC/AC 能耗估计的便捷入口。该函数执行一次真实前向传播，
    并返回总能耗与 MAC/AC 计数。

    默认使用 Horowitz 2014 的 FP32 成本口径；若需要 FP16 或 INT8 比较，
    请显式传入对应 preset。

    :param model: 待统计模型
    :param inputs: 模型输入；若为 tuple/list 则按 ``model(*inputs)`` 调用
    :param config: compute-only 能耗配置

    ----

    .. _estimate_compute_energy-en:

    * **English**

    Convenience entry for compute-only MAC/AC energy estimation.
    It runs one real forward pass and returns the energy report.

    The default comparison regime is Horowitz 2014 FP32. For FP16 or INT8
    comparisons, pass an explicit preset cost configuration.

    :param model: model to profile
    :param inputs: model input; tuple/list will be passed as ``model(*inputs)``
    :param config: compute-only energy configuration
    """
    profiler = ComputeEnergyProfiler(config=config)
    with profiler:
        _ = _call_model(model, inputs)
    return profiler.get_report()
