from __future__ import annotations

import copy
import warnings
from dataclasses import dataclass, field
from math import ceil, isfinite, prod
from numbers import Real
from typing import Any

import torch
import torch.nn as nn

from ..neuron.base_node import BaseNode
from ..neuron.integrate_and_fire import IFNode
from ..neuron.lif import LIFNode
from .base import (
    EnergyModelInfo,
    ModuleCounter,
    ModuleCounterMode,
    call_model,
    is_binary_tensor,
)

__all__ = [
    "LemaireEnergyConfig",
    "LemaireEnergyCostConfig",
    "LemaireEnergyProfiler",
    "LemaireEnergyReport",
    "estimate_lemaire_energy",
]

_LEMAIRE_ACCESS_WIDTH_BYTES = 4.0

_LEMAIRE_MODEL_INFO = EnergyModelInfo(
    model_id="lemaire_2022_runtime_v1",
    fidelity="paper",
    source_urls=(
        "https://arxiv.org/abs/2210.13107",
        "https://doi.org/10.1007/978-3-031-30105-6_48",
    ),
    technology_nm=45,
    precision="32-bit integer operations and 32-bit SRAM accesses",
    scope="runtime Conv/Linear forward inference under the Lemaire analytical model",
)

_SUPPORTED_LEMAIRE_MEMORY_MODULES = (
    nn.Linear,
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
)
_SUPPORTED_LEMAIRE_NEURONS = (IFNode, LIFNode)
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

    :param e_add_pj: 单次 32-bit 整数加法能耗，单位为 pJ
    :type e_add_pj: float
    :param e_mul_pj: 单次 32-bit 整数乘法能耗，单位为 pJ
    :type e_mul_pj: float
    :param memory_breakpoints: 四个 ``(容量字节数, 每字节访问能耗 pJ)`` 插值点
    :type memory_breakpoints: tuple[tuple[float, float], ...]

    :raises ValueError: 当成本或插值点无效时抛出

    ----

    .. _LemaireEnergyCostConfig-en:

    * **English**

    Cost configuration for the Lemaire-style analytical energy model.

    :param e_add_pj: Energy of one 32-bit integer addition in pJ
    :type e_add_pj: float
    :param e_mul_pj: Energy of one 32-bit integer multiplication in pJ
    :type e_mul_pj: float
    :param memory_breakpoints: Four ``(capacity bytes, access pJ per byte)``
        interpolation points
    :type memory_breakpoints: tuple[tuple[float, float], ...]

    :raises ValueError: Raised for invalid costs or interpolation points
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
        for name in ("e_add_pj", "e_mul_pj"):
            value = getattr(self, name)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and nonnegative.")
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
            if not isfinite(x) or not isfinite(y) or x < 0 or y < 0:
                raise ValueError(
                    "memory_breakpoints entries must be finite and nonnegative."
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

    :param strict: 遇到论文范围外的 module 或 backend 时是否抛出异常
    :type strict: bool
    :param cost_config: 算术和 SRAM 访问成本
    :type cost_config: LemaireEnergyCostConfig

    :param snn_fifo_capacity_elements: 每层 SNN 输入/输出 FIFO 可容纳的消息数，
        默认 ``1000``，与论文实验设置一致
    :type snn_fifo_capacity_elements: int

    :raises ValueError: 当 ``snn_fifo_capacity_elements`` 非正数时抛出

    ----

    .. _LemaireEnergyConfig-en:

    * **English**

    Controls the inference-only, Lemaire-aligned energy profiler.

    :param strict: Whether modules or backends outside the paper scope raise
    :type strict: bool
    :param cost_config: Arithmetic and SRAM-access costs
    :type cost_config: LemaireEnergyCostConfig

    :param snn_fifo_capacity_elements: Number of messages that each SNN input/output
        FIFO can hold. The default ``1000`` follows the paper's experiments
    :type snn_fifo_capacity_elements: int

    :raises ValueError: Raised when ``snn_fifo_capacity_elements`` is not positive
    """

    strict: bool = True
    cost_config: LemaireEnergyCostConfig = field(
        default_factory=LemaireEnergyCostConfig
    )
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

    :param total_pj: 总能耗，单位为 pJ
    :type total_pj: float
    :param breakdown_pj: 按计算、寻址和访存分解的能耗
    :type breakdown_pj: dict[str, float]
    :param counts: 论文口径的操作和访问计数
    :type counts: dict[str, int]
    :param buffer_sizes_bytes: 各类本地存储的最大容量，单位为字节
    :type buffer_sizes_bytes: dict[str, int]
    :param warnings: 非严格模式下省略项的告警
    :type warnings: list[str]
    :param model_info: 模型来源与适用范围
    :type model_info: EnergyModelInfo
    :param config: 生成本报告的配置副本
    :type config: LemaireEnergyConfig

    ----

    .. _LemaireEnergyReport-en:

    * **English**

    Single-report, Lemaire-aligned forward inference energy report.

    :param total_pj: Total energy in pJ
    :type total_pj: float
    :param breakdown_pj: Energy split by compute, addressing, and memory
    :type breakdown_pj: dict[str, float]
    :param counts: Paper-aligned operation and access counts
    :type counts: dict[str, int]
    :param buffer_sizes_bytes: Maximum local-storage capacities in bytes
    :type buffer_sizes_bytes: dict[str, int]
    :param warnings: Omitted-scope warnings in non-strict mode
    :type warnings: list[str]
    :param model_info: Model provenance and applicability
    :type model_info: EnergyModelInfo
    :param config: Copy of the configuration used for this report
    :type config: LemaireEnergyConfig
    """

    total_pj: float
    breakdown_pj: dict[str, float]
    counts: dict[str, int]
    buffer_sizes_bytes: dict[str, int]
    warnings: list[str]
    model_info: EnergyModelInfo
    config: LemaireEnergyConfig


class _LemaireCounter(ModuleCounter):
    def __init__(self, *, strict: bool, fifo_capacity_elements: int):
        super().__init__()
        self.strict = strict
        self.fifo_capacity_elements = fifo_capacity_elements
        self.warnings: list[str] = []
        self._warned_module_types: set[type[nn.Module]] = set()
        self._accesses: list[tuple[str, int, int]] = []
        self.paper_ac = 0
        self.paper_mac = 0
        self.paper_synop = 0
        self.paper_acc_addr = 0
        self.paper_mac_addr = 0

    def _warn_or_raise_unsupported(self, module: nn.Module):
        module_type = type(module)
        if module_type in self._warned_module_types:
            return
        self._warned_module_types.add(module_type)
        message = (
            f"Lemaire formulas do not support {module_type.__name__}; "
            "its energy is omitted."
        )
        if self.strict:
            raise ValueError(message)
        self.warnings.append(message)
        warnings.warn(message, RuntimeWarning, stacklevel=2)

    def reset(self):
        super().reset()
        self.warnings.clear()
        self._warned_module_types.clear()
        self._accesses.clear()
        self.paper_ac = 0
        self.paper_mac = 0
        self.paper_synop = 0
        self.paper_acc_addr = 0
        self.paper_mac_addr = 0

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

    def bind_model(self, model: nn.Module) -> None:
        model_has_neurons = any(
            isinstance(module, BaseNode) for module in model.modules()
        )

        def synaptic_rule(
            module: nn.Module,
            inputs: tuple[torch.Tensor, ...],
            kwargs: dict[str, Any],
            output: torch.Tensor,
        ) -> int:
            x = (
                inputs[0]
                if inputs
                else next(value for value in kwargs.values() if torch.is_tensor(value))
            )
            out = output
            word_bytes = int(_LEMAIRE_ACCESS_WIDTH_BYTES)

            input_bytes = int(x.numel()) * word_bytes
            output_bytes = int(out.numel()) * word_bytes
            params_capacity = sum(
                int(param.numel()) * word_bytes
                for param in module.parameters(recurse=False)
            )
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                is_spike_input = is_binary_tensor(x)
                active_inputs = int(x.count_nonzero().item()) if is_spike_input else 0
            if isinstance(module, nn.Linear):
                dense_weight_uses = int(out.numel()) * module.in_features
                event_synops = active_inputs * module.out_features
                event_memory_fanout = event_synops
            else:
                out_channels_per_group = module.out_channels // module.groups
                dense_weight_uses = int(out.numel()) * int(
                    prod(module.weight.shape[1:])
                )
                event_synops = (
                    active_inputs
                    * out_channels_per_group
                    * prod(
                        ceil(kernel / stride)
                        for kernel, stride in zip(
                            module.kernel_size, module.stride, strict=True
                        )
                    )
                )
                event_memory_fanout = (
                    active_inputs * out_channels_per_group * prod(module.kernel_size)
                )

            weight_uses = event_memory_fanout if is_spike_input else dense_weight_uses
            bias_read_bytes = (
                0 if module.bias is None else int(out.numel()) * word_bytes
            )
            if is_spike_input:
                self.paper_ac += event_synops
                self.paper_synop += event_synops
                if isinstance(module, nn.Linear):
                    self.paper_acc_addr += active_inputs * module.out_features
                else:
                    self.paper_acc_addr += (
                        active_inputs
                        * (module.out_channels // module.groups)
                        * prod(module.kernel_size)
                    )
                    # Each spike needs two multiplies to locate its first output.
                    self.paper_mac_addr += active_inputs * 2
            else:
                self.paper_mac += dense_weight_uses
                if isinstance(module, nn.Linear):
                    self.paper_acc_addr += int(x.numel()) + int(out.numel())
                else:
                    self.paper_acc_addr += (
                        int(x.numel())
                        + int(out.numel())
                        + module.out_channels * prod(module.kernel_size)
                    )
            if module.bias is not None:
                self.paper_ac += int(out.numel())
            if is_spike_input:
                read_in_bytes = active_inputs * word_bytes
                input_capacity = self.fifo_capacity_elements * word_bytes
            elif isinstance(module, nn.Linear):
                read_in_bytes = input_bytes
                input_capacity = input_bytes
            else:
                read_in_bytes = dense_weight_uses * word_bytes
                input_capacity = input_bytes
            self._record(
                "read_in_bytes",
                read_in_bytes,
                input_capacity,
            )
            self._record(
                "read_params_bytes",
                weight_uses * word_bytes + bias_read_bytes,
                params_capacity,
            )

            if is_spike_input:
                time_steps = self._time_steps(module, x)
                potential_capacity = output_bytes // max(time_steps, 1)
                potential_access_bytes = event_memory_fanout * word_bytes
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
            return 0

        def neuron_rule(
            module: BaseNode,
            inputs: tuple[Any, ...],
            kwargs: dict[str, Any],
            output: Any,
        ) -> int:
            out = (
                output[0]
                if isinstance(output, (tuple, list)) and len(output) > 0
                else output
            )
            if not torch.is_tensor(out):
                return 0
            x = (
                inputs[0]
                if inputs
                else next(value for value in kwargs.values() if torch.is_tensor(value))
            )
            time_steps = self._time_steps(module, x)
            word_bytes = int(_LEMAIRE_ACCESS_WIDTH_BYTES)
            potential_capacity = int(out.numel()) // max(time_steps, 1) * word_bytes
            potential_access_bytes = potential_capacity * time_steps
            self._record(
                "read_potential_bytes", potential_access_bytes, potential_capacity
            )
            self._record(
                "write_potential_bytes", potential_access_bytes, potential_capacity
            )
            with torch._C._ExcludeDispatchKeyGuard(
                torch._C.DispatchKeySet(torch._C.DispatchKey.Python)
            ):
                output_spike_bytes = int(out.count_nonzero().item()) * word_bytes
            self.paper_ac += output_spike_bytes // word_bytes
            if isinstance(module, LIFNode):
                self.paper_mac += int(out.numel())
            self._record(
                "write_out_bytes",
                output_spike_bytes,
                self.fifo_capacity_elements * word_bytes,
            )
            return 0

        self.rules = {
            **{
                ("forward", module_type): synaptic_rule
                for module_type in _SUPPORTED_LEMAIRE_MEMORY_MODULES
            },
            **{
                ("forward", module_type): neuron_rule
                for module_type in _SUPPORTED_LEMAIRE_NEURONS
            },
        }

    def validate_model(self, model: nn.Module) -> None:
        neuron_internals = {
            child
            for module in model.modules()
            if isinstance(module, BaseNode)
            for child in module.modules()
            if child is not module
        }
        for module in model.modules():
            if module in neuron_internals:
                continue
            if isinstance(module, _UNSUPPORTED_LEMAIRE_MEMORY_MODULES):
                self._warn_or_raise_unsupported(module)
            elif isinstance(module, BaseNode):
                if not isinstance(module, _SUPPORTED_LEMAIRE_NEURONS):
                    self._warn_or_raise_unsupported(module)
            elif not any(module.children()) and not isinstance(
                module, _SUPPORTED_LEMAIRE_MEMORY_MODULES
            ):
                self._warn_or_raise_unsupported(module)

    def record(self, scope: str, func: Any, value: int) -> None:
        pass


class LemaireEnergyProfiler:
    def __init__(self, *, config: LemaireEnergyConfig | None = None):
        """
        .. rubric:: API Language

        :ref:`中文 <LemaireEnergyProfiler-cn>` |
        :ref:`English <LemaireEnergyProfiler-en>`

        ----

        .. _LemaireEnergyProfiler-cn:

        * **中文**

        基于 ``ModuleCounterMode`` 动态采集 module event 的 Lemaire 前向能耗分析器。

        :param config: 能耗配置，若为 ``None`` 则使用默认配置
        :type config: LemaireEnergyConfig | None

        ----

        .. _LemaireEnergyProfiler-en:

        * **English**

        Inference-only Lemaire profiler driven by runtime module events from
        ``ModuleCounterMode``.

        :param config: Energy configuration. If ``None``, uses the default configuration
        :type config: LemaireEnergyConfig | None
        """
        self.config = copy.deepcopy(config or LemaireEnergyConfig())
        self._warnings: list[str] = []
        self.lemaire_counter = _LemaireCounter(
            strict=self.config.strict,
            fifo_capacity_elements=self.config.snn_fifo_capacity_elements,
        )
        self._module_mode: ModuleCounterMode | None = None

    def bind_model(self, model: nn.Module) -> None:
        r"""
        **API Language** - :ref:`中文 <LemaireEnergyProfiler.bind_model-cn>` |
        :ref:`English <LemaireEnergyProfiler.bind_model-en>`

        ----

        .. _LemaireEnergyProfiler.bind_model-cn:

        * **中文**

        绑定模型并准备 Lemaire module 规则。

        :param model: 待分析模型
        :type model: torch.nn.Module
        :raises RuntimeError: 分析器处于活跃 context 时抛出
        :raises ValueError: 严格模式遇到不支持的神经元 backend 时抛出

        ----

        .. _LemaireEnergyProfiler.bind_model-en:

        * **English**

        Bind a model and prepare the Lemaire module rules.

        :param model: Model to profile
        :type model: torch.nn.Module
        :raises RuntimeError: Raised while the profiler context is active
        :raises ValueError: Raised for an unsupported neuron backend in strict mode
        """
        if self._module_mode is not None and self._module_mode._active:
            raise RuntimeError(
                "LemaireEnergyProfiler.bind_model() cannot run while profiling."
            )
        self._warnings.clear()
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
        self.lemaire_counter.bind_model(model)
        self._module_mode = ModuleCounterMode(
            [self.lemaire_counter],
            model=model,
        )

    def __enter__(self):
        self.lemaire_counter.reset()
        if self._module_mode is None:
            raise RuntimeError(
                "LemaireEnergyProfiler.bind_model() must be called before entering."
            )
        self.lemaire_counter.validate_model(self._module_mode.model)
        self._module_mode.__enter__()
        return self

    def __exit__(self, exc_type, exc, tb):
        return self._module_mode.__exit__(exc_type, exc, tb)

    def get_report(self) -> LemaireEnergyReport:
        r"""
        **API Language** - :ref:`中文 <LemaireEnergyProfiler.get_report-cn>` |
        :ref:`English <LemaireEnergyProfiler.get_report-en>`

        ----

        .. _LemaireEnergyProfiler.get_report-cn:

        * **中文**

        返回最近一次运行的 Lemaire 能耗报告。

        :return: 能耗、计数、存储容量、告警和来源信息
        :rtype: LemaireEnergyReport

        ----

        .. _LemaireEnergyProfiler.get_report-en:

        * **English**

        Return the Lemaire energy report for the latest run.

        :return: Energy, counts, storage capacities, warnings, and provenance
        :rtype: LemaireEnergyReport
        """
        cost = self.config.cost_config
        memory_counts, buffers, memory_breakdown = self.lemaire_counter.summarize(cost)
        counts = {
            "synop": self.lemaire_counter.paper_synop,
            "mac": self.lemaire_counter.paper_mac,
            "ac": self.lemaire_counter.paper_ac,
            **memory_counts,
            "acc_addr": self.lemaire_counter.paper_acc_addr,
            "mac_addr": self.lemaire_counter.paper_mac_addr,
        }
        ops_pj = counts["ac"] * cost.e_add_pj + counts["mac"] * (
            cost.e_mul_pj + cost.e_add_pj
        )
        addressing_pj = counts["acc_addr"] * cost.e_add_pj + counts["mac_addr"] * (
            cost.e_mul_pj + cost.e_add_pj
        )
        inout_pj = memory_breakdown["inout_pj"]
        params_pj = memory_breakdown["params_pj"]
        potential_pj = memory_breakdown["potential_pj"]
        memory_pj = inout_pj + params_pj + potential_pj
        total_pj = ops_pj + addressing_pj + memory_pj
        warnings_list = list(self._warnings) + list(self.lemaire_counter.warnings)
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
            model_info=_LEMAIRE_MODEL_INFO,
            config=copy.deepcopy(self.config),
        )


def estimate_lemaire_energy(
    model: nn.Module,
    inputs: Any,
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
    with profiler, torch.no_grad():
        call_model(model, inputs)
    return profiler.get_report()
