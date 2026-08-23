from __future__ import annotations

import copy
import math
from dataclasses import dataclass, field
from typing import Any

import torch.nn as nn
import torch

from spikingjelly.logger import logger

from .ac import ACCounter
from .base import DispatchCounterMode, EnergyModelInfo, ModuleCounterMode, call_model
from .mac import MACCounter
from .neuromorphic_memory_access import NeuromorphicMemoryAccessCounter
from .synop import SynOpCounter


__all__ = [
    "SimpleEnergyCostConfig",
    "SimpleEnergyConfig",
    "SimpleEnergyProfiler",
    "SimpleEnergyReport",
    "estimate_simple_energy",
]

_SIMPLE_MODEL_INFO = EnergyModelInfo(
    model_id="simple_horowitz_step_composite_v1",
    fidelity="spikingjelly-defined",
    source_urls=(
        "https://doi.org/10.1109/ISSCC.2014.6757323",
        "https://openreview.net/pdf?id=SzwU2XrXIS",
    ),
    technology_nm=45,
    precision="configurable comparison regime; runtime bytes use observed dtype",
    scope="runtime MAC/AC plus SpikingJelly logical parameter and neuron-state traffic",
)


@dataclass
class SimpleEnergyCostConfig:
    r"""
    .. rubric:: API Language

    :ref:`中文 <SimpleEnergyCostConfig-cn>` |
    :ref:`English <SimpleEnergyCostConfig-en>`

    ----

    .. _SimpleEnergyCostConfig-cn:

    * **中文**

    基于运行时 MAC、AC 和神经形态逻辑访存量的简单能耗模型成本配置。

    默认值采用 SNN 文献中常见的 Horowitz 2014 口径：45nm、32-bit 浮点
    ``E_MAC = 4.6 pJ``，``E_AC = 0.9 pJ``。默认访存成本为
    ``24.96 pJ/byte``，对应 STEP 所引用的 ``3.12 pJ/bit``。

    这是一个基于真实运行时计数的归一化模型。访存计数包括实际使用的权重、
    bias 读取，以及持久神经元状态每时间步的一次读取和一次写回；模型不会
    根据运行时 ``dtype`` 自动改变单位能耗，但字节数会随实际 dtype 改变。

    ----

    .. _SimpleEnergyCostConfig-en:

    * **English**

    Cost configuration for the simple runtime MAC/AC/neuromorphic-memory model.

    Defaults follow the widely used Horowitz 2014 reference costs for 45nm,
    32-bit floating-point arithmetic: ``E_MAC = 4.6 pJ`` and
    ``E_AC = 0.9 pJ``. The default memory cost is ``24.96 pJ/byte``,
    corresponding to the ``3.12 pJ/bit`` reference used by STEP.

    This is a normalized runtime model. Memory traffic consists of reads for
    weights and biases that are actually used, plus one read and one write per
    timestep for persistent neuron states. The cost per byte is not inferred
    from runtime ``dtype``; the measured byte count already reflects it.
    """

    e_mac_pj: float = 4.6
    e_ac_pj: float = 0.9
    e_memory_pj_per_byte: float = 24.96

    def __post_init__(self) -> None:
        for name in ("e_mac_pj", "e_ac_pj", "e_memory_pj_per_byte"):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")

    @classmethod
    def fp32(cls) -> "SimpleEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm FP32 preset.
        """
        return cls(e_mac_pj=4.6, e_ac_pj=0.9, e_memory_pj_per_byte=24.96)

    @classmethod
    def fp16(cls) -> "SimpleEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm FP16 preset.

        Uses ``FMult16 = 1.1 pJ`` and ``FAdd16 = 0.4 pJ``, so
        ``E_MAC = 1.5 pJ`` and ``E_AC = 0.4 pJ``.
        """
        return cls(e_mac_pj=1.5, e_ac_pj=0.4, e_memory_pj_per_byte=24.96)

    @classmethod
    def int8(cls) -> "SimpleEnergyCostConfig":
        r"""
        Return the Horowitz 2014 45nm INT8 preset.

        Uses ``Mult8 = 0.2 pJ`` and ``Add8 = 0.03 pJ``, so
        ``E_MAC = 0.23 pJ`` and ``E_AC = 0.03 pJ``.
        """
        return cls(e_mac_pj=0.23, e_ac_pj=0.03, e_memory_pj_per_byte=24.96)


@dataclass
class SimpleEnergyConfig:
    r"""
    .. rubric:: API Language

    :ref:`中文 <SimpleEnergyConfig-cn>` |
    :ref:`English <SimpleEnergyConfig-en>`

    ----

    .. _SimpleEnergyConfig-cn:

    * **中文**

    控制 Simple Energy 分析器行为的配置。

    默认 ``cost_config`` 使用 ``SimpleEnergyCostConfig.fp32()`` 对应的口径。

    ----

    .. _SimpleEnergyConfig-en:

    * **English**

    Configuration for the Simple Energy profiler.

    The default ``cost_config`` matches ``SimpleEnergyCostConfig.fp32()``.
    ``strict`` only applies to profiler-level validation added by this wrapper.
    The internal ``DispatchCounterMode`` is intentionally kept non-strict because
    it composes multiple specialized counters with non-identical rule coverage.
    """

    strict: bool = False
    cost_config: SimpleEnergyCostConfig = field(default_factory=SimpleEnergyCostConfig)
    extra_ignore_modules: list[type[nn.Module]] | None = None


@dataclass
class SimpleEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <SimpleEnergyReport-cn>` |
    :ref:`English <SimpleEnergyReport-en>`

    ----

    .. _SimpleEnergyReport-cn:

    * **中文**

    Simple Energy 运行时能耗报告。

    主结果 ``energy_total_pj`` 由 MAC、AC、参数读取和持久神经元状态读写组成。
    访存计数表示简单神经形态推理假设下的逻辑访问，不表示宿主 GPU 的
    cache load/store 次数。

    ``SynOps`` 作为 AC 中脉冲突触操作的辅助统计返回，但不参与主能耗计算。

    该估计器面向“统一比较口径”，而不是对真实 kernel、混合精度累加路径或
    特定硬件微架构做精确建模。

    ----

    .. _SimpleEnergyReport-en:

    * **English**

    Report for the Simple Energy runtime model.

    The primary result ``energy_total_pj`` consists of MAC energy, AC energy,
    parameter reads, and persistent neuron-state reads and writes. The memory
    counts describe logical neuromorphic accesses rather than host-GPU cache
    loads and stores.

    ``SynOps`` is returned as an auxiliary diagnostic for the AC count and does
    not contribute a second time to the primary energy estimate.

    The estimator is intended as a normalized comparison regime rather than an
    exact model of real kernels, mixed-precision accumulation paths, or a
    specific hardware microarchitecture.
    """

    energy_total_pj: float
    energy_compute_pj: float
    energy_mac_pj: float
    energy_ac_pj: float
    energy_memory_pj: float
    breakdown_pj: dict[str, float]
    counts: dict[str, int]
    warnings: list[str]
    model_info: EnergyModelInfo
    config: SimpleEnergyConfig


class SimpleEnergyProfiler:
    def __init__(self, *, config: SimpleEnergyConfig | None = None):
        """
        .. rubric:: API Language

        :ref:`中文 <SimpleEnergyProfiler-cn>` |
        :ref:`English <SimpleEnergyProfiler-en>`

        ----

        .. _SimpleEnergyProfiler-cn:

        * **中文**

        基于 public counter 组装的 Simple Energy 分析器。

        用法与其他能耗分析器一致：以 context manager 方式包住一次真实前向传播，
        然后调用 ``get_report()``。

        :param config: 能耗配置，若为 ``None`` 则使用默认配置
        :type config: SimpleEnergyConfig | None

        ----

        .. _SimpleEnergyProfiler-en:

        * **English**

        Simple Energy profiler composed from public counters.

        Use it like the other energy profilers: wrap one real forward pass in the
        context manager and call ``get_report()`` afterwards.

        :param config: Energy configuration. If ``None``, uses the default configuration
        :type config: SimpleEnergyConfig | None
        """
        self.config = copy.deepcopy(config or SimpleEnergyConfig())
        ignore_modules = list(self.config.extra_ignore_modules or [])
        self.mac_counter = MACCounter(extra_ignore_modules=ignore_modules)
        self.ac_counter = ACCounter(extra_ignore_modules=ignore_modules)
        self.synop_counter = SynOpCounter(extra_ignore_modules=ignore_modules)
        self.memory_counter = NeuromorphicMemoryAccessCounter(
            extra_ignore_modules=ignore_modules
        )
        self._dispatch_mode = DispatchCounterMode(
            [
                self.mac_counter,
                self.ac_counter,
                self.synop_counter,
            ],
            strict=False,
        )
        self._module_mode: ModuleCounterMode | None = None
        self._summary_logged = False

    def bind_model(self, model: nn.Module) -> None:
        r"""
        绑定待统计模型。手动使用 context manager 前必须调用本方法。

        Bind the model to profile. Manual context-manager usage must call this
        method first.

        :param model: 待统计模型 / Model to profile
        :type model: torch.nn.Module
        """
        self._module_mode = ModuleCounterMode(
            [self.memory_counter],
            model=model,
        )

    def __enter__(self):
        self.mac_counter.reset()
        self.ac_counter.reset()
        self.synop_counter.reset()
        self.memory_counter.reset()
        self._summary_logged = False
        if self._module_mode is None:
            raise RuntimeError(
                "SimpleEnergyProfiler.bind_model() must be called before entering."
            )
        self._module_mode.__enter__()
        try:
            self._dispatch_mode.__enter__()
        except BaseException:
            self._module_mode.__exit__(None, None, None)
            raise
        return self

    def __exit__(self, exc_type, exc, tb):
        try:
            return self._dispatch_mode.__exit__(exc_type, exc, tb)
        finally:
            self._module_mode.__exit__(exc_type, exc, tb)

    def get_report(self) -> SimpleEnergyReport:
        mac = self.mac_counter.get_total()
        ac = self.ac_counter.get_total()
        synop = self.synop_counter.get_total()
        memory_scopes = self.memory_counter.get_counts()
        memory_counts = memory_scopes.get("Global", {})
        weight_read_bytes = memory_counts.get("weight_read_bytes", 0)
        bias_read_bytes = memory_counts.get("bias_read_bytes", 0)
        neuron_state_read_bytes = memory_counts.get("neuron_state_read_bytes", 0)
        neuron_state_write_bytes = memory_counts.get("neuron_state_write_bytes", 0)
        parameter_read_bytes = weight_read_bytes + bias_read_bytes
        neuron_state_access_bytes = neuron_state_read_bytes + neuron_state_write_bytes
        memory_access_bytes = parameter_read_bytes + neuron_state_access_bytes
        cost = self.config.cost_config

        warnings_list: list[str] = []
        matched_counter_rules = len(
            set(self.mac_counter.get_counts().get("Global", {}))
            | set(self.ac_counter.get_counts().get("Global", {}))
            | set(self.synop_counter.get_counts().get("Global", {}))
        )
        if matched_counter_rules == 0 and not memory_scopes:
            message = (
                "SimpleEnergyProfiler did not match any supported operators. "
                "The model may not contain supported operators for this estimator."
            )
            if self.config.strict:
                raise RuntimeError(message)
            warnings_list.append(message)

        energy_mac_pj = mac * cost.e_mac_pj
        energy_ac_pj = ac * cost.e_ac_pj
        energy_compute_pj = energy_mac_pj + energy_ac_pj
        energy_parameter_memory_pj = parameter_read_bytes * cost.e_memory_pj_per_byte
        energy_neuron_state_memory_pj = (
            neuron_state_access_bytes * cost.e_memory_pj_per_byte
        )
        energy_memory_pj = energy_parameter_memory_pj + energy_neuron_state_memory_pj
        total_pj = energy_compute_pj + energy_memory_pj

        report = SimpleEnergyReport(
            energy_total_pj=total_pj,
            energy_compute_pj=energy_compute_pj,
            energy_mac_pj=energy_mac_pj,
            energy_ac_pj=energy_ac_pj,
            energy_memory_pj=energy_memory_pj,
            breakdown_pj={
                "mac_pj": energy_mac_pj,
                "ac_pj": energy_ac_pj,
                "parameter_memory_pj": energy_parameter_memory_pj,
                "neuron_state_memory_pj": energy_neuron_state_memory_pj,
                "memory_pj": energy_memory_pj,
            },
            counts={
                "mac": mac,
                "ac": ac,
                "synop": synop,
                "weight_read_bytes": weight_read_bytes,
                "bias_read_bytes": bias_read_bytes,
                "neuron_state_read_bytes": neuron_state_read_bytes,
                "neuron_state_write_bytes": neuron_state_write_bytes,
                "memory_access_bytes": memory_access_bytes,
            },
            warnings=warnings_list,
            model_info=_SIMPLE_MODEL_INFO,
            config=copy.deepcopy(self.config),
        )
        if not self._summary_logged:
            logger.info(
                "Counting completed: counter={} total_operations={} memory_access_bytes={} matched_counter_rules={} warnings={}",
                type(self).__name__,
                mac + ac,
                memory_access_bytes,
                matched_counter_rules,
                len(warnings_list),
            )
            self._summary_logged = True
        return report

    def get_total(self) -> float:
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, int]:
        return self.get_report().counts


def estimate_simple_energy(
    model: nn.Module,
    inputs: Any,
    *,
    config: SimpleEnergyConfig | None = None,
) -> SimpleEnergyReport:
    r"""
    .. rubric:: API Language

    :ref:`中文 <estimate_simple_energy-cn>` |
    :ref:`English <estimate_simple_energy-en>`

    ----

    .. _estimate_simple_energy-cn:

    * **中文**

    Simple Energy 能耗估计的便捷入口。该函数执行一次真实前向传播，
    并返回 MAC、AC、访存和总能耗计数。

    默认使用 Horowitz 2014 的 FP32 成本口径；若需要 FP16 或 INT8 比较，
    请显式传入对应 preset。

    :param model: 待统计模型
    :param inputs: 模型输入；若为 tuple/list 则按 ``model(*inputs)`` 调用
    :param config: simple 能耗配置

    ----

    .. _estimate_simple_energy-en:

    * **English**

    Convenience entry for Simple Energy estimation.
    It runs one real forward pass and returns the energy report.

    The default comparison regime is Horowitz 2014 FP32. For FP16 or INT8
    comparisons, pass an explicit preset cost configuration.

    :param model: model to profile
    :param inputs: model input; tuple/list will be passed as ``model(*inputs)``
    :param config: simple energy configuration
    """
    profiler = SimpleEnergyProfiler(config=config)
    profiler.bind_model(model)
    with profiler, torch.no_grad():
        _ = call_model(model, inputs)
    return profiler.get_report()
