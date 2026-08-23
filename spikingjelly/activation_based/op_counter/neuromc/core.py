from __future__ import annotations

from collections import defaultdict, deque
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field, replace
from math import prod
from typing import Any, Callable, Optional

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from ..base import EnergyModelInfo, ModuleCounter, ModuleCounterMode, call_model
from .config import MemoryHierarchyConfig, MemoryInstanceSpec
from .utils import _is_spike

__all__ = [
    "MemoryHierarchyConfig",
    "NeuroMCEnergyProfiler",
    "NeuroMCRuntimeEnergyReport",
    "estimate_neuromc_runtime_energy",
]


_EXTRA_OP_COST_PJ = {
    "mux": 0.548 * (1.0 / 16.0),
    "add": 0.548,
    "mul": 0.812,
    "comp": 0.056,
    "sqrt": 0.514,
}

_MAC_COST_PJ = {
    "fp_soma": 0.548 + 0.548 * (1.0 / 16.0),
    "bp_grad": 0.548 + 0.812,
    "wg": 0.548 + 0.548 * (1.0 / 16.0),
    "ann_fe": 0.548 + 0.812,
    "ann_be": 0.548 + 0.812,
    "ann_we": 0.548 + 0.812,
}

_NEUROMC_MODEL_INFO = EnergyModelInfo(
    model_id="neuromc_712c66_runtime_v1",
    fidelity="source-aligned",
    source_urls=(
        "https://www.nature.com/articles/s41467-026-70586-x",
        "https://github.com/dayanhn/NeuroMC/commit/"
        "712c66f47cf76ae530a55f8bcad3858bd68788de",
    ),
    technology_nm=32,
    precision="1-bit spikes and 16-bit values under the NeuroMC v1 constants",
    scope="runtime fragment adapter for NeuroMC-style FE/BP/WG/BN/optimizer energy",
)

_EXTRA_RH2L_MULTIPLIER = {
    "fp_yi1": 1,
    "fp_u_l": 1,
    "fp_s_l": 1,
    "bp_conv_res": 1,
    "bp_u_l_pre": 1,
    "bp_s_l_pre": 1,
    "bp_smask_l_pre": 1,
    "bp_du_l_pre": 1,
    "fp_bn_mean_v": 2,
    "fp_bn_variance": 1,
    "fp_bn_n": 1,
    "fp_bn_sqrt": 1,
    "fp_bn_xi_": 1,
    "fp_bn_y": 1,
    "fp_bn_b": 1,
    "bp_bn_du_l_pre1": 2,
    "bp_bn_sqrt": 2,
    "bp_bn_y": 2,
    "bp_bn_m": 3,
    "bp_bn_n": 3,
    "bp_bn_sigma_m": 1,
    "bp_bn_sigma_n": 1,
    "bp_bn_sigma_mn": 2,
    "opt_y": 1,
    "opt_b": 1,
    "opt_w": 1,
    "opt_dy": 1,
    "opt_db": 1,
    "opt_dw": 1,
    "opt_v_y": 1,
    "opt_v_b": 1,
    "opt_v_w": 1,
    "opt_s_y": 1,
    "opt_s_b": 1,
    "opt_s_w": 1,
    "opt_vbc_y": 1,
    "opt_vbc_b": 1,
    "opt_vbc_w": 1,
    "opt_sbc_y": 1,
    "opt_sbc_b": 1,
    "opt_sbc_w": 1,
}

_EXTRA_WL2H = {
    "fp_yi1",
    "fp_u_l",
    "fp_s_l",
    "fp_smask_l",
    "fp_bn_mean_v",
    "fp_bn_variance",
    "fp_bn_n",
    "fp_bn_sqrt",
    "fp_bn_xi_",
    "bp_u_l_pre",
    "bp_du_l_pre",
    "bp_bn_m",
    "bp_bn_sigma_m",
    "bp_bn_sigma_n",
    "bp_bn_sigma_mn",
    "bp_bn_dy",
    "bp_bn_db",
    "bp_bn_du_l_pre2",
    "opt_v_y",
    "opt_v_b",
    "opt_v_w",
    "opt_s_y",
    "opt_s_b",
    "opt_s_w",
    "opt_vbc_y",
    "opt_vbc_b",
    "opt_vbc_w",
    "opt_sbc_y",
    "opt_sbc_b",
    "opt_sbc_w",
    "opt_y_updated",
    "opt_b_updated",
    "opt_w_updated",
}

_IGNORED_OP_PREFIXES = (
    "aten.detach",
    "aten.view",
    "aten.t.default",
    "aten.transpose",
    "aten.permute",
    "aten.expand",
    "aten.slice",
    "aten.select",
    "aten.alias",
    "aten._unsafe_view",
    "aten.as_strided",
    "aten.clone",
    "aten.copy_",
    "profiler.",
)

_AUXILIARY_ATEN_OPS = {
    "aten.all.default",
    "aten._local_scalar_dense.default",
    "aten.lift_fresh.default",
    "aten._to_copy.default",
    "aten.add.Tensor",
    "aten.add_.Tensor",
    "aten.add.Scalar",
    "aten.add_.Scalar",
    "aten.sub.Tensor",
    "aten.sub_.Tensor",
    "aten.sub.Scalar",
    "aten.sub_.Scalar",
    "aten.rsub.Tensor",
    "aten.rsub.Scalar",
    "aten.mul.Tensor",
    "aten.mul_.Tensor",
    "aten.mul.Scalar",
    "aten.mul_.Scalar",
    "aten.div.Tensor",
    "aten.div_.Tensor",
    "aten.div.Scalar",
    "aten.div_.Scalar",
    "aten.empty.memory_format",
    "aten.cat.default",
    "aten.stack.default",
    "aten.split.Tensor",
    "aten.index.Tensor",
    "aten.full_like.default",
    "aten.mse_loss.default",
    "aten.mse_loss_backward.default",
    "aten.where.self",
    "aten.where.ScalarOther",
    "aten.where.ScalarSelf",
    "aten.sqrt.default",
    "aten.sqrt_.default",
    "aten.rsqrt.default",
    "aten.sigmoid.default",
    "aten.sigmoid_.default",
    "aten.sum.default",
    "aten.sum.dim_IntList",
    "aten.mean.dim",
    "aten.eq.Tensor",
    "aten.eq.Scalar",
    "aten.ne.Tensor",
    "aten.ne.Scalar",
    "aten.lt.Tensor",
    "aten.lt.Scalar",
    "aten.le.Tensor",
    "aten.le.Scalar",
    "aten.gt.Tensor",
    "aten.gt.Scalar",
    "aten.ge.Tensor",
    "aten.ge.Scalar",
    "aten.logical_and.default",
    "aten.logical_or.default",
    "aten.logical_xor.default",
    "aten.logical_not.default",
    "aten.new_empty_strided.default",
    "aten.ones_like.default",
    "aten.zeros_like.default",
}


@dataclass
class NeuroMCRuntimeEnergyReport:
    r"""
    **API Language** - :ref:`中文 <NeuroMCRuntimeEnergyReport-cn>` | :ref:`English <NeuroMCRuntimeEnergyReport-en>`

    ----

    .. _NeuroMCRuntimeEnergyReport-cn:

    * **中文**

    NeuroMC 运行时能耗报告。标量字段记录总能耗及计算、内存细分；字典和列表
    字段记录阶段、算子、硬件映射和警告。传入的可变容器会浅复制，传入 ``None``
    等价于空容器。

    :param energy_total_pj: 总能耗，单位为 pJ。
    :type energy_total_pj: float
    :param energy_compute_pj: 总计算能耗，单位为 pJ。
    :type energy_compute_pj: float
    :param energy_memory_pj: 总内存能耗，单位为 pJ。
    :type energy_memory_pj: float
    :param energy_by_stage: 各执行阶段的能耗。
    :type energy_by_stage: Optional[dict[str, float]]
    :param energy_by_op: 各算子类型的能耗。
    :type energy_by_op: Optional[dict[str, float]]
    :param primitive_counts: 原始操作计数。
    :type primitive_counts: Optional[dict[str, Any]]
    :param memory_bits_by_level: 各内存层级的访问位数。
    :type memory_bits_by_level: Optional[dict[str, Any]]
    :param warnings: 分析期间产生的警告。
    :type warnings: Optional[list[str]]
    :param energy_mac_pj: MAC 操作能耗，单位为 pJ。
    :type energy_mac_pj: float
    :param energy_base_memory_pj: 基础内存能耗，单位为 pJ。
    :type energy_base_memory_pj: float
    :param energy_extra_memory_pj: 额外内存能耗，单位为 pJ。
    :type energy_extra_memory_pj: float
    :param energy_extra_compute_pj: 额外计算能耗，单位为 pJ。
    :type energy_extra_compute_pj: float
    :param energy_by_core_type: 各核心类型的能耗。
    :type energy_by_core_type: Optional[dict[str, float]]
    :param energy_by_process_key: 各处理路径的能耗。
    :type energy_by_process_key: Optional[dict[str, float]]
    :param energy_by_memory_level_dir: 各内存层级和方向的能耗。
    :type energy_by_memory_level_dir: Optional[dict[str, dict[str, float]]]
    :param counts_by_core_type: 各核心类型的操作计数。
    :type counts_by_core_type: Optional[dict[str, dict[str, int]]]
    :param counts_by_process_key: 各处理路径的操作计数。
    :type counts_by_process_key: Optional[dict[str, dict[str, int]]]
    :param mapping_summary: 硬件映射摘要。
    :type mapping_summary: Optional[list[dict[str, Any]]]

    ----

    .. _NeuroMCRuntimeEnergyReport-en:

    * **English**

    NeuroMC runtime energy report. Scalar fields store total, compute, and
    memory energy; dictionary and list fields store stage, operation, hardware
    mapping, and warning details. Mutable inputs are shallow-copied, and
    ``None`` is treated as an empty container.

    :param energy_total_pj: Total energy in pJ.
    :type energy_total_pj: float
    :param energy_compute_pj: Total compute energy in pJ.
    :type energy_compute_pj: float
    :param energy_memory_pj: Total memory energy in pJ.
    :type energy_memory_pj: float
    :param energy_by_stage: Energy by execution stage.
    :type energy_by_stage: Optional[dict[str, float]]
    :param energy_by_op: Energy by operation type.
    :type energy_by_op: Optional[dict[str, float]]
    :param primitive_counts: Raw primitive operation counts.
    :type primitive_counts: Optional[dict[str, Any]]
    :param memory_bits_by_level: Memory access bits by hierarchy level.
    :type memory_bits_by_level: Optional[dict[str, Any]]
    :param warnings: Warnings generated during profiling.
    :type warnings: Optional[list[str]]
    :param energy_mac_pj: MAC operation energy in pJ.
    :type energy_mac_pj: float
    :param energy_base_memory_pj: Base memory energy in pJ.
    :type energy_base_memory_pj: float
    :param energy_extra_memory_pj: Extra memory energy in pJ.
    :type energy_extra_memory_pj: float
    :param energy_extra_compute_pj: Extra compute energy in pJ.
    :type energy_extra_compute_pj: float
    :param energy_by_core_type: Energy by core type.
    :type energy_by_core_type: Optional[dict[str, float]]
    :param energy_by_process_key: Energy by processing path.
    :type energy_by_process_key: Optional[dict[str, float]]
    :param energy_by_memory_level_dir: Energy by memory level and direction.
    :type energy_by_memory_level_dir: Optional[dict[str, dict[str, float]]]
    :param counts_by_core_type: Operation counts by core type.
    :type counts_by_core_type: Optional[dict[str, dict[str, int]]]
    :param counts_by_process_key: Operation counts by processing path.
    :type counts_by_process_key: Optional[dict[str, dict[str, int]]]
    :param mapping_summary: Hardware mapping summary.
    :type mapping_summary: Optional[list[dict[str, Any]]]
    """

    energy_total_pj: float = 0.0
    energy_compute_pj: float = 0.0
    energy_memory_pj: float = 0.0
    energy_by_stage: Optional[dict[str, float]] = field(default_factory=dict)
    energy_by_op: Optional[dict[str, float]] = field(default_factory=dict)
    primitive_counts: Optional[dict[str, Any]] = field(default_factory=dict)
    memory_bits_by_level: Optional[dict[str, Any]] = field(default_factory=dict)
    warnings: Optional[list[str]] = field(default_factory=list)
    energy_mac_pj: float = 0.0
    energy_base_memory_pj: float = 0.0
    energy_extra_memory_pj: float = 0.0
    energy_extra_compute_pj: float = 0.0
    energy_by_core_type: Optional[dict[str, float]] = field(default_factory=dict)
    energy_by_process_key: Optional[dict[str, float]] = field(default_factory=dict)
    energy_by_memory_level_dir: Optional[dict[str, dict[str, float]]] = field(
        default_factory=dict
    )
    counts_by_core_type: Optional[dict[str, dict[str, int]]] = field(
        default_factory=dict
    )
    counts_by_process_key: Optional[dict[str, dict[str, int]]] = field(
        default_factory=dict
    )
    mapping_summary: Optional[list[dict[str, Any]]] = field(default_factory=list)
    model_info: Optional[EnergyModelInfo] = None
    memory_config: Optional[MemoryHierarchyConfig] = None

    def __post_init__(self) -> None:
        for name in (
            "energy_by_stage",
            "energy_by_op",
            "primitive_counts",
            "memory_bits_by_level",
            "energy_by_core_type",
            "energy_by_process_key",
            "energy_by_memory_level_dir",
            "counts_by_core_type",
            "counts_by_process_key",
        ):
            setattr(self, name, dict(getattr(self, name) or {}))
        self.warnings = list(self.warnings or [])
        self.mapping_summary = list(self.mapping_summary or [])


@dataclass
class _TraceTensor:
    shape: tuple[int, ...]
    dtype: torch.dtype
    requires_grad: bool
    is_spike: bool

    @property
    def ndim(self) -> int:
        return len(self.shape)


@dataclass
class _TraceEvent:
    op_name: str
    stage: str
    phase: str
    args: Any
    kwargs: Any
    out: Any


@dataclass
class _Fragment:
    stage: str
    phase: str
    op_name: str
    core_type: str
    process_key: str
    loop_dims: dict[str, int]
    input_precision_bits: int
    weight_precision_bits: int
    output_precision_bits: int
    input_numel: int
    weight_numel: int
    output_numel: int
    mac_count: int
    conv_type: str = "--"
    b_type: int = 0
    t_type: int = 0
    source: str = "trace"
    optimizer_has_momentum: bool = False
    optimizer_has_weight_decay: bool = False
    optimizer_has_momentum_buffer: bool = False


def _module_overlap_signature(fragment: _Fragment) -> tuple[Any, ...]:
    ld = fragment.loop_dims
    return (
        fragment.stage,
        fragment.phase,
        fragment.core_type,
        fragment.process_key,
        (
            ("BT", ld["B"] * ld["T"]),
            ("C", ld["C"]),
            ("K", ld["K"]),
            ("OY", ld["OY"]),
            ("OX", ld["OX"]),
            ("FY", ld["FY"]),
            ("FX", ld["FX"]),
        ),
        fragment.input_precision_bits,
        fragment.weight_precision_bits,
        fragment.output_precision_bits,
        fragment.input_numel,
        fragment.weight_numel,
        fragment.output_numel,
        fragment.mac_count,
        fragment.conv_type,
        fragment.b_type,
        fragment.t_type,
    )


def _tensor_numel(x: Any) -> int:
    if isinstance(x, _TraceTensor):
        return prod(x.shape)
    if not torch.is_tensor(x):
        return 0
    return int(x.numel())


def _tensor_layout(
    x: torch.Tensor | _TraceTensor, module: nn.Module | None = None
) -> tuple[int, int, int, tuple[int, ...]]:
    step_mode = getattr(module, "step_mode", "s") if module is not None else "s"
    if step_mode == "m":
        if x.ndim < 3:
            raise ValueError(
                f"Expected multi-step tensor with shape [T, N, C, ...], got {tuple(x.shape)}."
            )
        t = int(x.shape[0])
        b = int(x.shape[1])
        c = int(x.shape[2])
        spatial = tuple(int(v) for v in x.shape[3:])
    else:
        t = 1
        b = int(x.shape[0]) if x.ndim > 0 else 1
        if x.ndim >= 2:
            c = int(x.shape[1])
            spatial = tuple(int(v) for v in x.shape[2:])
        else:
            c = _tensor_numel(x)
            spatial = ()
    return t, b, c, spatial


def _snapshot_trace_value(value: Any) -> Any:
    if torch.is_tensor(value):
        return _TraceTensor(
            shape=tuple(int(v) for v in value.shape),
            dtype=value.dtype,
            requires_grad=bool(value.requires_grad),
            is_spike=_is_spike(value),
        )
    if isinstance(value, tuple):
        return tuple(_snapshot_trace_value(v) for v in value)
    if isinstance(value, list):
        return [_snapshot_trace_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _snapshot_trace_value(v) for k, v in value.items()}
    return value


def _is_spike_like(x: Any) -> bool:
    if isinstance(x, _TraceTensor):
        return x.is_spike
    return _is_spike(x)


def _shape_tuple(x: Any) -> tuple[int, ...]:
    if isinstance(x, (_TraceTensor, torch.Tensor)):
        return tuple(int(v) for v in x.shape)
    return ()


def _forward_core_type(is_spike_input: bool) -> str:
    return "fp_soma" if is_spike_input else "ann_fe"


def _grad_input_core_type(is_spike_input: bool) -> str:
    return "bp_grad" if is_spike_input else "ann_be"


def _weight_grad_core_type(is_spike_input: bool) -> str:
    return "wg" if is_spike_input else "ann_we"


def _optimizer_param_count(fragment: _Fragment) -> int:
    return fragment.loop_dims["K"] + (
        fragment.loop_dims["FX"] * fragment.loop_dims["FY"] * fragment.loop_dims["C"]
    )


def _clear_existing_grads(model: nn.Module, optimizer: torch.optim.Optimizer | None):
    if optimizer is not None:
        optimizer.zero_grad(set_to_none=True)
        return
    for p in model.parameters():
        p.grad = None


class _TraceMode(TorchDispatchMode):
    def __init__(self, profiler: "NeuroMCEnergyProfiler"):
        super().__init__()
        self.profiler = profiler
        self.op_counts: dict[str, int] = {}

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        kwargs = {} if kwargs is None else kwargs
        if self.profiler._suspended:
            return func(*args, **kwargs)
        op_name = resolve_name(func)
        self.op_counts[op_name] = self.op_counts.get(op_name, 0) + 1
        out = func(*args, **kwargs)
        self.profiler._maybe_record_trace_event(op_name, args, kwargs, out)
        return out


class NeuroMCEnergyProfiler(ModuleCounter):
    def __init__(
        self,
        *,
        memory_config: MemoryHierarchyConfig | None = None,
    ):
        r"""
        **API Language** - :ref:`中文 <NeuroMCEnergyProfiler-cn>` | :ref:`English <NeuroMCEnergyProfiler-en>`

        ----

        .. _NeuroMCEnergyProfiler-cn:

        * **中文**

        NeuroMC source-aligned 运行时适配器。它动态捕获真实执行 fragment，
        再应用固定 NeuroMC v1 成本表；它不包含完整 ZigZag mapping。

        :param memory_config: 内存层次配置；``None`` 使用 ``neuromc_like_v1``
        :type memory_config: MemoryHierarchyConfig | None

        ----

        .. _NeuroMCEnergyProfiler-en:

        * **English**

        Source-aligned NeuroMC runtime adapter. It captures executed fragments and
        applies the fixed NeuroMC v1 cost tables; it is not a complete ZigZag mapping.

        :param memory_config: Memory hierarchy configuration. ``None`` selects
            ``neuromc_like_v1``
        :type memory_config: MemoryHierarchyConfig | None
        """
        super().__init__()
        config = memory_config or MemoryHierarchyConfig()
        self.memory_config = replace(
            config, memory_instances=dict(config.memory_instances)
        )
        self.memory_config.validate()
        self._stage_stack: list[str] = []
        self._stage_options: dict[str, tuple[str, bool, bool]] = {}
        self._warnings: list[str] = []
        self._trace_mode = _TraceMode(self)
        self._trace_events: list[_TraceEvent] = []
        self._fragments: list[_Fragment] = []
        self._bound_model: nn.Module | None = None
        self._trainable_param_shapes: set[tuple[int, ...]] = set()
        self._active = False
        self._suspended = False
        self._optimizer: torch.optim.Optimizer | None = None
        self._module_mode: ModuleCounterMode | None = None

    def bind_model(self, model: nn.Module) -> None:
        if self._bound_model is not None:
            if self._bound_model is model:
                return
            raise RuntimeError(
                "NeuroMCEnergyProfiler is already bound to a different model."
            )
        from ...neuron.base_node import BaseNode

        supported = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Linear,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            BaseNode,
        )
        for module in model.modules():
            if isinstance(module, nn.Conv3d):
                raise ValueError("NeuroMC runtime does not support nn.Conv3d yet.")
        self._bound_model = model
        self._trainable_param_shapes = {
            tuple(parameter.shape)
            for parameter in model.parameters()
            if parameter.requires_grad
        }
        self.rules = {
            **{
                ("forward", module_type): self._count_forward_module
                for module_type in supported
            },
            **{
                ("backward", module_type): self._count_backward_module
                for module_type in supported
            },
        }
        self._module_mode = ModuleCounterMode([self], model=model)

    def bind_optimizer(self, optimizer: torch.optim.Optimizer | None) -> None:
        self._optimizer = optimizer

    def __enter__(self):
        self._stage_stack.clear()
        self._stage_options.clear()
        self._warnings.clear()
        self._trace_events.clear()
        self._fragments.clear()
        self._trace_mode.op_counts.clear()
        if self._module_mode is None:
            raise RuntimeError(
                "NeuroMCEnergyProfiler.bind_model() must be called before entering."
            )
        self._module_mode.__enter__()
        try:
            self._trace_mode.__enter__()
        except BaseException:
            self._module_mode.__exit__(None, None, None)
            raise
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        try:
            return self._trace_mode.__exit__(exc_type, exc, tb)
        finally:
            self._module_mode.__exit__(exc_type, exc, tb)
            if self._bound_model is not None:
                for module in self._bound_model.modules():
                    if hasattr(module, "_neuromc_last_input"):
                        delattr(module, "_neuromc_last_input")
            self._bound_model = None
            self._module_mode = None

    @contextmanager
    def stage(
        self,
        name: str,
        *,
        phase: str = "forward",
        reuse_weights: bool = False,
        batch_norm_backward: bool = True,
    ) -> Iterator["NeuroMCEnergyProfiler"]:
        r"""
        标记一个执行阶段及其 NeuroMC 映射语义。

        Label an execution stage and its NeuroMC mapping semantics.

        :param name: 报告中的阶段名称 / Stage name used in reports
        :type name: str
        :param phase: ``forward``、``backward`` 或 ``optimizer``
        :type phase: str
        :param reuse_weights: 是否复用已驻留权重 / Whether resident weights are reused
        :type reuse_weights: bool
        :param batch_norm_backward: 是否计入 BN backward / Whether BN backward is present
        :type batch_norm_backward: bool
        :raises ValueError: 当 ``phase`` 非法时抛出 / Raised for an invalid phase
        :raises RuntimeError: 当 profiler 未激活或 stage 嵌套时抛出 / Raised
            outside an active profiler or for nested stages
        """
        if not self._active:
            raise RuntimeError(
                "stage() can only be used inside active profiler context."
            )
        if self._stage_stack:
            raise RuntimeError("Nested stage() is not supported in NeuroMC v2.")
        if phase not in {"forward", "backward", "optimizer"}:
            raise ValueError(f"Unsupported NeuroMC phase={phase!r}.")
        self._stage_options[name] = (phase, reuse_weights, batch_norm_backward)
        self._stage_stack.append(name)
        try:
            yield self
        finally:
            self._stage_stack.pop()

    @contextmanager
    def suspend(self) -> Iterator["NeuroMCEnergyProfiler"]:
        old = self._suspended
        self._suspended = True
        try:
            yield self
        finally:
            self._suspended = old

    def _current_stage(self) -> str:
        if not self._stage_stack:
            return "unlabeled"
        return self._stage_stack[-1]

    def _bn_core_type(self, phase: str, is_spike_input: bool) -> str:
        if is_spike_input:
            return "fp_bn" if phase == "forward" else "bp_bn"
        return "ann_bn" if phase == "forward" else "ann_bn_bp"

    def _stage_phase(self, stage: str | None = None) -> str:
        name = stage or self._current_stage()
        return self._stage_options.get(name, ("forward", False, True))[0]

    def _stage_position(self, stage: str | None = None) -> tuple[int, int]:
        name = stage or self._current_stage()
        reuse_weights = self._stage_options.get(name, ("forward", False, True))[1]
        return 0, int(reuse_weights)

    def _stage_conv_type(self, stage: str | None = None) -> str:
        name = stage or self._current_stage()
        include_bn = self._stage_options.get(name, ("forward", False, True))[2]
        return "--" if include_bn else "without_bp_bn"

    def _maybe_record_trace_event(self, op_name: str, args, kwargs, out):
        if not self._active or self._suspended:
            return
        self._trace_events.append(
            _TraceEvent(
                op_name=op_name,
                stage=self._current_stage(),
                phase=self._stage_phase(),
                args=_snapshot_trace_value(args),
                kwargs=_snapshot_trace_value(kwargs),
                out=_snapshot_trace_value(out),
            )
        )

    def _count_forward_module(
        self,
        module: nn.Module,
        args: tuple[Any, ...],
        kwargs: dict[str, Any],
        out: Any,
    ) -> int:
        if not self._active or self._suspended:
            return 0
        stage = self._current_stage()
        from ...neuron.base_node import BaseNode

        x = (
            args[0]
            if args
            else next(value for value in kwargs.values() if torch.is_tensor(value))
        )
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            module._neuromc_last_input = x
            self._fragments.append(
                self._make_conv_forward_fragment(stage, module, x, out)
            )
        elif isinstance(module, nn.Linear):
            module._neuromc_last_input = x
            self._fragments.append(
                self._make_linear_forward_fragment(stage, module, x, out)
            )
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            module._neuromc_last_input = x
            self._fragments.append(
                self._make_bn_forward_fragment(stage, module, x, out)
            )
        elif isinstance(module, BaseNode):
            self._fragments.append(
                self._make_soma_forward_fragment(stage, module, x, out)
            )
        return 0

    def _count_backward_module(
        self,
        module: nn.Module,
        grad_input: tuple[Any, ...],
        kwargs: dict[str, Any],
        grad_output: tuple[Any, ...],
    ) -> int:
        if not self._active or self._suspended:
            return 0
        stage = self._current_stage()
        from ...neuron.base_node import BaseNode

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._fragments.extend(
                self._make_conv_backward_fragments(
                    stage, module, grad_input, grad_output
                )
            )
        elif isinstance(module, nn.Linear):
            self._fragments.extend(
                self._make_linear_backward_fragments(
                    stage, module, grad_input, grad_output
                )
            )
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self._fragments.append(
                self._make_bn_backward_fragment(stage, module, grad_input, grad_output)
            )
        elif isinstance(module, BaseNode):
            self._fragments.append(
                self._make_soma_backward_fragment(
                    stage, module, grad_input, grad_output
                )
            )
        return 0

    def record(self, scope: str, func: Any, value: int) -> None:
        del scope, func, value

    def _make_loop_dims(
        self,
        *,
        batch_size: int,
        time_steps: int = 1,
        channels_in: int,
        channels_out: int,
        oy: int,
        ox: int,
        fy: int,
        fx: int,
    ) -> dict[str, int]:
        return {
            "B": max(int(batch_size), 1),
            "T": max(int(time_steps), 1),
            "C": max(int(channels_in), 1),
            "K": max(int(channels_out), 1),
            "OY": max(int(oy), 1),
            "OX": max(int(ox), 1),
            "FY": max(int(fy), 1),
            "FX": max(int(fx), 1),
        }

    def _make_conv_forward_fragment(
        self, stage: str, module: nn.Module, x, out
    ) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        is_spike_input = _is_spike(x)
        t, b, _, _ = _tensor_layout(x, module)
        _, _, _, out_spatial = _tensor_layout(out, module)
        spatial = out_spatial if len(out_spatial) > 0 else (1, 1)
        if len(spatial) == 1:
            spatial = (spatial[0], 1)
        kernel = (
            tuple(module.kernel_size)
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size,)
        )
        if len(kernel) == 1:
            kernel = (kernel[0], 1)
        loop_dims = self._make_loop_dims(
            batch_size=b,
            time_steps=t,
            channels_in=int(module.in_channels),
            channels_out=int(module.out_channels),
            oy=int(spatial[0]),
            ox=int(spatial[1]),
            fy=int(kernel[0]),
            fx=int(kernel[1]),
        )
        return _Fragment(
            stage=stage,
            phase="forward",
            op_name="conv.forward",
            core_type=_forward_core_type(is_spike_input),
            process_key="with_nothing",
            loop_dims=loop_dims,
            input_precision_bits=1 if is_spike_input else 16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(x),
            weight_numel=_tensor_numel(module.weight),
            output_numel=_tensor_numel(out),
            mac_count=self._mac_count(loop_dims),
            b_type=b_type,
            t_type=t_type,
            source="module",
        )

    def _make_linear_forward_fragment(
        self, stage: str, module: nn.Linear, x, out
    ) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        is_spike_input = _is_spike(x)
        t, batch, _, _ = _tensor_layout(x, module)
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=int(module.in_features),
            channels_out=int(module.out_features),
            oy=1,
            ox=1,
            fy=1,
            fx=1,
        )
        return _Fragment(
            stage=stage,
            phase="forward",
            op_name="linear.forward",
            core_type=_forward_core_type(is_spike_input),
            process_key="with_nothing",
            loop_dims=loop_dims,
            input_precision_bits=1 if is_spike_input else 16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(x),
            weight_numel=_tensor_numel(module.weight),
            output_numel=_tensor_numel(out),
            mac_count=self._mac_count(loop_dims),
            b_type=b_type,
            t_type=t_type,
            source="module",
        )

    def _make_bn_forward_fragment(
        self, stage: str, module: nn.Module | None, x, out
    ) -> _Fragment:
        is_spike_input = _is_spike_like(x)
        t, batch, c, spatial = _tensor_layout(x, module)
        spatial_prod = max(prod(spatial), 1) if spatial else 1
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=c,
            channels_out=c,
            oy=spatial_prod,
            ox=1,
            fy=1,
            fx=1,
        )
        return _Fragment(
            stage=stage,
            phase="forward",
            op_name="bn.forward",
            core_type=self._bn_core_type("forward", is_spike_input),
            process_key="with_bn",
            loop_dims=loop_dims,
            input_precision_bits=16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(x),
            weight_numel=c * 2,
            output_numel=_tensor_numel(
                out[0] if isinstance(out, (tuple, list)) else out
            ),
            mac_count=0,
            source="module",
        )

    def _make_soma_forward_fragment(
        self, stage: str, module: nn.Module | None, x, out
    ) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        if torch.is_tensor(out_tensor):
            t, batch, c, spatial = _tensor_layout(out_tensor, module)
            spatial_prod = max(prod(spatial), 1) if spatial else 1
        else:
            t, batch, c, spatial_prod = 1, 1, 1, 1
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=c,
            channels_out=c,
            oy=spatial_prod,
            ox=1,
            fy=1,
            fx=1,
        )
        return _Fragment(
            stage=stage,
            phase="forward",
            op_name="soma.forward",
            core_type="fp_soma",
            process_key="with_sg",
            loop_dims=loop_dims,
            input_precision_bits=16,
            weight_precision_bits=16,
            output_precision_bits=1,
            input_numel=_tensor_numel(x),
            weight_numel=0,
            output_numel=_tensor_numel(out_tensor),
            mac_count=0,
            b_type=b_type,
            t_type=t_type,
            source="module",
        )

    def _make_conv_backward_fragments(self, stage, module, grad_input, grad_output):
        grad_out = (
            grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        )
        is_spike_input = _is_spike(getattr(module, "_neuromc_last_input", None))
        t, batch, _, spatial = _tensor_layout(grad_out, module)
        spatial = spatial if len(spatial) > 0 else (1, 1)
        if len(spatial) == 1:
            spatial = (spatial[0], 1)
        kernel = (
            tuple(module.kernel_size)
            if isinstance(module.kernel_size, tuple)
            else (module.kernel_size,)
        )
        if len(kernel) == 1:
            kernel = (kernel[0], 1)
        grad_in = (
            grad_input[0]
            if isinstance(grad_input, (tuple, list)) and grad_input
            else None
        )
        base_loop = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=int(module.in_channels),
            channels_out=int(module.out_channels),
            oy=int(spatial[0]),
            ox=int(spatial[1]),
            fy=int(kernel[0]),
            fx=int(kernel[1]),
        )
        fragments = []
        if torch.is_tensor(grad_in):
            fragments.append(
                _Fragment(
                    stage=stage,
                    phase="backward",
                    op_name="conv.backward.grad_input",
                    core_type=_grad_input_core_type(is_spike_input),
                    process_key="with_nothing",
                    loop_dims=base_loop,
                    input_precision_bits=16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(grad_out),
                    weight_numel=_tensor_numel(module.weight),
                    output_numel=_tensor_numel(grad_in),
                    mac_count=self._mac_count(base_loop),
                    source="module",
                )
            )
        if module.weight.requires_grad:
            fragments.append(
                _Fragment(
                    stage=stage,
                    phase="backward",
                    op_name="conv.backward.grad_weight",
                    core_type=_weight_grad_core_type(is_spike_input),
                    process_key="with_nothing",
                    loop_dims=base_loop,
                    input_precision_bits=1 if is_spike_input else 16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(
                        getattr(module, "_neuromc_last_input", None)
                    ),
                    weight_numel=_tensor_numel(grad_out),
                    output_numel=_tensor_numel(module.weight),
                    mac_count=self._mac_count(base_loop),
                    source="module",
                )
            )
        return fragments

    def _make_linear_backward_fragments(self, stage, module, grad_input, grad_output):
        grad_out = (
            grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        )
        is_spike_input = _is_spike(getattr(module, "_neuromc_last_input", None))
        t, batch, _, _ = _tensor_layout(grad_out, module)
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=int(module.in_features),
            channels_out=int(module.out_features),
            oy=1,
            ox=1,
            fy=1,
            fx=1,
        )
        grad_in = (
            grad_input[0]
            if isinstance(grad_input, (tuple, list)) and grad_input
            else None
        )
        fragments = []
        if torch.is_tensor(grad_in):
            fragments.append(
                _Fragment(
                    stage=stage,
                    phase="backward",
                    op_name="linear.backward.grad_input",
                    core_type=_grad_input_core_type(is_spike_input),
                    process_key="with_nothing",
                    loop_dims=loop_dims,
                    input_precision_bits=16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(grad_out),
                    weight_numel=_tensor_numel(module.weight),
                    output_numel=_tensor_numel(grad_in),
                    mac_count=self._mac_count(loop_dims),
                    source="module",
                )
            )
        if module.weight.requires_grad:
            fragments.append(
                _Fragment(
                    stage=stage,
                    phase="backward",
                    op_name="linear.backward.grad_weight",
                    core_type=_weight_grad_core_type(is_spike_input),
                    process_key="with_nothing",
                    loop_dims=loop_dims,
                    input_precision_bits=1 if is_spike_input else 16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(
                        getattr(module, "_neuromc_last_input", None)
                    ),
                    weight_numel=_tensor_numel(grad_out),
                    output_numel=_tensor_numel(module.weight),
                    mac_count=self._mac_count(loop_dims),
                    source="module",
                )
            )
        return fragments

    def _make_bn_backward_fragment(self, stage, module, grad_input, grad_output):
        is_spike_input = _is_spike(getattr(module, "_neuromc_last_input", None))
        grad_out = (
            grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        )
        grad_in = (
            grad_input[0]
            if isinstance(grad_input, (tuple, list)) and grad_input
            else grad_out
        )
        t, batch, c, spatial = _tensor_layout(grad_out, module)
        spatial_prod = max(prod(spatial), 1) if spatial else 1
        conv_type = self._stage_conv_type(stage)
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=c,
            channels_out=c,
            oy=spatial_prod,
            ox=1,
            fy=1,
            fx=1,
        )
        return _Fragment(
            stage=stage,
            phase="backward",
            op_name="bn.backward",
            core_type=self._bn_core_type("backward", is_spike_input),
            process_key="with_bn",
            loop_dims=loop_dims,
            input_precision_bits=16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(grad_out),
            weight_numel=c * 2,
            output_numel=_tensor_numel(grad_in),
            mac_count=0,
            conv_type=conv_type,
            source="module",
        )

    def _make_soma_backward_fragment(self, stage, module, grad_input, grad_output):
        grad_in = (
            grad_input[0]
            if isinstance(grad_input, (tuple, list)) and grad_input
            else None
        )
        grad_out = (
            grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        )
        tensor = grad_in if torch.is_tensor(grad_in) else grad_out
        t, batch, c, spatial = _tensor_layout(tensor, module)
        spatial_prod = max(prod(spatial), 1) if spatial else 1
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            time_steps=t,
            channels_in=c,
            channels_out=c,
            oy=spatial_prod,
            ox=1,
            fy=1,
            fx=1,
        )
        return _Fragment(
            stage=stage,
            phase="backward",
            op_name="soma.backward",
            core_type="bp_grad",
            process_key="with_sg",
            loop_dims=loop_dims,
            input_precision_bits=16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(grad_out),
            weight_numel=0,
            output_numel=_tensor_numel(tensor),
            mac_count=0,
            source="module",
        )

    def _mac_count(self, loop_dims: dict[str, int]) -> int:
        return int(
            loop_dims["B"]
            * loop_dims["T"]
            * loop_dims["C"]
            * loop_dims["K"]
            * loop_dims["OY"]
            * loop_dims["OX"]
            * loop_dims["FY"]
            * loop_dims["FX"]
        )

    def _matches_trainable_param_shape(self, out: Any) -> bool:
        if not (torch.is_tensor(out) or isinstance(out, _TraceTensor)):
            return False
        return tuple(out.shape) in self._trainable_param_shapes

    def _gemm_backward_fragment_kind(
        self, x: torch.Tensor | _TraceTensor, y: torch.Tensor | _TraceTensor, out: Any
    ) -> str | None:
        x_req = bool(x.requires_grad)
        y_req = bool(y.requires_grad)
        if x_req and (not y_req):
            return "wg"
        if (not x_req) and y_req:
            return "bp_grad"
        if self._matches_trainable_param_shape(out):
            return "wg"
        return None

    def _fallback_fragments_from_trace(
        self, *, supplemental_only: bool = False
    ) -> list[_Fragment]:
        fragments: list[_Fragment] = []
        gemm_forward_is_spike: dict[
            tuple[str, tuple[int, ...], tuple[int, ...]], deque[bool]
        ] = defaultdict(deque)
        supplemental_ops = {
            "aten.convolution.default",
            "aten.addmm.default",
            "aten.mm.default",
            "aten.bmm.default",
            "aten.native_batch_norm.default",
            "aten.native_batch_norm_backward.default",
        }
        for event in self._trace_events:
            op = event.op_name
            b_type, t_type = self._stage_position(event.stage)
            if supplemental_only and op not in supplemental_ops:
                continue
            if op in _AUXILIARY_ATEN_OPS or op.startswith(_IGNORED_OP_PREFIXES):
                continue
            if op == "aten.convolution.default" and event.phase == "forward":
                x, w = event.args[:2]
                out = event.out
                if x.ndim > 4 or out.ndim > 4:
                    raise ValueError(
                        "NeuroMC runtime does not support multi-step or 3D "
                        "Conv trace fallback yet."
                    )
                is_spike_input = _is_spike_like(x)
                spatial = tuple(out.shape[2:]) if out.ndim > 2 else (1, 1)
                if len(spatial) > 2:
                    raise ValueError(
                        "NeuroMC runtime does not support Conv3d fallback yet."
                    )
                if len(spatial) == 1:
                    spatial = (spatial[0], 1)
                kernel = tuple(w.shape[2:]) if w.ndim > 2 else (1, 1)
                if len(kernel) > 2:
                    raise ValueError(
                        "NeuroMC runtime does not support Conv3d fallback yet."
                    )
                if len(kernel) == 1:
                    kernel = (kernel[0], 1)
                loop_dims = self._make_loop_dims(
                    batch_size=int(x.shape[0]),
                    channels_in=int(x.shape[1]),
                    channels_out=int(out.shape[1]),
                    oy=int(spatial[0]),
                    ox=int(spatial[1]),
                    fy=int(kernel[0]),
                    fx=int(kernel[1]),
                )
                fragments.append(
                    _Fragment(
                        stage=event.stage,
                        phase=event.phase,
                        op_name=op,
                        core_type=_forward_core_type(is_spike_input),
                        process_key="with_nothing",
                        loop_dims=loop_dims,
                        input_precision_bits=1 if is_spike_input else 16,
                        weight_precision_bits=16,
                        output_precision_bits=16,
                        input_numel=_tensor_numel(x),
                        weight_numel=_tensor_numel(w),
                        output_numel=_tensor_numel(out),
                        mac_count=self._mac_count(loop_dims),
                        b_type=b_type,
                        t_type=t_type,
                    )
                )
            elif op in {"aten.addmm.default", "aten.mm.default", "aten.bmm.default"}:
                x = event.args[-2]
                y = event.args[-1]
                out = event.out
                if event.phase == "forward":
                    is_spike_input = _is_spike_like(x)
                    gemm_forward_is_spike[
                        (op, _shape_tuple(x), _shape_tuple(out))
                    ].append(is_spike_input)
                    batch = (
                        int(x.shape[0]) if x.ndim == 2 else int(x.shape[0] * x.shape[1])
                    )
                    k = int(x.shape[-1])
                    n = int(y.shape[-1])
                    loop_dims = self._make_loop_dims(
                        batch_size=batch,
                        channels_in=k,
                        channels_out=n,
                        oy=1,
                        ox=1,
                        fy=1,
                        fx=1,
                    )
                    fragments.append(
                        _Fragment(
                            stage=event.stage,
                            phase=event.phase,
                            op_name=op,
                            core_type=_forward_core_type(is_spike_input),
                            process_key="with_nothing",
                            loop_dims=loop_dims,
                            input_precision_bits=1 if is_spike_input else 16,
                            weight_precision_bits=16,
                            output_precision_bits=16,
                            input_numel=_tensor_numel(x),
                            weight_numel=_tensor_numel(y),
                            output_numel=_tensor_numel(out),
                            mac_count=self._mac_count(loop_dims),
                            b_type=b_type,
                            t_type=t_type,
                        )
                    )
                else:
                    kind = self._gemm_backward_fragment_kind(x, y, out)
                    if kind == "wg":
                        is_spike_input = _is_spike_like(x)
                        if x.ndim == 2:
                            batch = int(x.shape[-1])
                            c = int(x.shape[0])
                            k = int(y.shape[-1])
                        else:
                            batch = int(x.shape[0] * x.shape[-1])
                            c = int(x.shape[1])
                            k = int(y.shape[-1])
                        loop_dims = self._make_loop_dims(
                            batch_size=batch,
                            channels_in=c,
                            channels_out=k,
                            oy=1,
                            ox=1,
                            fy=1,
                            fx=1,
                        )
                        fragments.append(
                            _Fragment(
                                stage=event.stage,
                                phase=event.phase,
                                op_name=op,
                                core_type=_weight_grad_core_type(is_spike_input),
                                process_key="with_nothing",
                                loop_dims=loop_dims,
                                input_precision_bits=1 if is_spike_input else 16,
                                weight_precision_bits=16,
                                output_precision_bits=16,
                                input_numel=_tensor_numel(x),
                                weight_numel=_tensor_numel(y),
                                output_numel=_tensor_numel(out),
                                mac_count=self._mac_count(loop_dims),
                                b_type=b_type,
                                t_type=t_type,
                            )
                        )
                    else:
                        forward_key = (op, _shape_tuple(out), _shape_tuple(x))
                        is_spike_input = False
                        forward_queue = gemm_forward_is_spike.get(forward_key)
                        if forward_queue:
                            is_spike_input = forward_queue.pop()
                        elif op == "aten.addmm.default":
                            alt_key = (
                                "aten.mm.default",
                                _shape_tuple(out),
                                _shape_tuple(x),
                            )
                            alt_queue = gemm_forward_is_spike.get(alt_key)
                            if alt_queue:
                                is_spike_input = alt_queue.pop()
                        elif op == "aten.mm.default":
                            alt_key = (
                                "aten.addmm.default",
                                _shape_tuple(out),
                                _shape_tuple(x),
                            )
                            alt_queue = gemm_forward_is_spike.get(alt_key)
                            if alt_queue:
                                is_spike_input = alt_queue.pop()
                        # Missing forward provenance should bias toward the denser,
                        # more expensive path instead of undercounting energy.
                        batch = (
                            int(x.shape[0])
                            if x.ndim == 2
                            else int(x.shape[0] * x.shape[1])
                        )
                        k = int(x.shape[-1])
                        n = int(y.shape[-1])
                        loop_dims = self._make_loop_dims(
                            batch_size=batch,
                            channels_in=k,
                            channels_out=n,
                            oy=1,
                            ox=1,
                            fy=1,
                            fx=1,
                        )
                        fragments.append(
                            _Fragment(
                                stage=event.stage,
                                phase=event.phase,
                                op_name=op,
                                core_type=_grad_input_core_type(is_spike_input),
                                process_key="with_nothing",
                                loop_dims=loop_dims,
                                input_precision_bits=16,
                                weight_precision_bits=16,
                                output_precision_bits=16,
                                input_numel=_tensor_numel(x),
                                weight_numel=_tensor_numel(y),
                                output_numel=_tensor_numel(out),
                                mac_count=self._mac_count(loop_dims),
                                b_type=b_type,
                                t_type=t_type,
                            )
                        )
            elif op == "aten.native_batch_norm.default" and event.phase == "forward":
                x = event.args[0]
                if x.ndim > 4:
                    raise ValueError(
                        "NeuroMC runtime does not support multi-step or 3D "
                        "BatchNorm trace fallback yet."
                    )
                out = (
                    event.out[0] if isinstance(event.out, (tuple, list)) else event.out
                )
                fragments.append(
                    self._make_bn_forward_fragment(event.stage, None, x, out)
                )
            elif op == "aten.native_batch_norm_backward.default":
                fragments.append(
                    self._make_bn_backward_fragment(
                        event.stage, None, event.args, event.out
                    )
                )
        return fragments

    def _unsupported_ops(self) -> list[str]:
        unsupported = []
        for op_name, count in self._trace_mode.op_counts.items():
            if op_name.startswith(_IGNORED_OP_PREFIXES):
                continue
            if op_name in _AUXILIARY_ATEN_OPS:
                continue
            if op_name in {
                "aten.convolution.default",
                "aten.addmm.default",
                "aten.mm.default",
                "aten.bmm.default",
                "aten.native_batch_norm.default",
                "aten.native_batch_norm_backward.default",
            }:
                continue
            unsupported.append(f"{op_name} (calls={count})")
        return sorted(unsupported)

    def _extra_counts(self, fragment: _Fragment) -> dict[str, int]:
        counts = {"mux": 0, "add": 0, "mul": 0, "comp": 0, "sqrt": 0}
        ld = fragment.loop_dims
        oyoxkbt = ld["OY"] * ld["OX"] * ld["K"] * ld["B"] * ld["T"]
        oyoxcbt = ld["OY"] * ld["OX"] * ld["C"] * ld["B"] * ld["T"]
        ct = ld["C"] * ld["T"]
        if fragment.process_key == "with_opt":
            kt = ld["K"]
            fyfxkc = ld["FX"] * ld["FY"] * ld["C"]
        else:
            kt = ld["K"] * ld["T"]
            fyfxkc = ld["FY"] * ld["FX"] * ld["K"] * ld["C"]

        if fragment.process_key == "with_sg":
            if fragment.phase == "forward":
                counts["mux"] += oyoxkbt
                counts["add"] += oyoxkbt
                counts["mul"] += oyoxkbt
                counts["comp"] += oyoxkbt * 3
            else:
                counts["mux"] += oyoxcbt
                counts["add"] += oyoxcbt * 2
                counts["mul"] += oyoxcbt * 4
        elif fragment.process_key == "with_bn":
            if fragment.phase == "forward":
                counts["add"] += oyoxkbt * 3 + kt * 2
                counts["mul"] += oyoxkbt * 3 + kt * 4
                counts["sqrt"] += kt
            else:
                counts["add"] += oyoxcbt * 7
                counts["mul"] += oyoxcbt * 3 + ct * 22
                if fragment.conv_type == "without_bp_bn":
                    counts["add"] = 0
                    counts["mul"] = 0
        elif fragment.process_key == "with_opt":
            if fragment.op_name in {"adam", "adamw"}:
                counts["add"] += 8 * kt + 4 * fyfxkc
                counts["mul"] += 22 * kt + 11 * fyfxkc
                counts["sqrt"] += 2 * kt + fyfxkc
            elif fragment.op_name == "sgd":
                total_params = _optimizer_param_count(fragment)
                counts["add"] += total_params
                counts["mul"] += total_params
                if fragment.optimizer_has_weight_decay:
                    counts["add"] += total_params
                    counts["mul"] += total_params
                if (
                    fragment.optimizer_has_momentum
                    and fragment.optimizer_has_momentum_buffer
                ):
                    counts["add"] += total_params
                    counts["mul"] += total_params
        return counts

    def _memory_energy_per_element(
        self, spec: MemoryInstanceSpec, precision_bits: int, read: bool
    ) -> float:
        bw = spec.r_bw if read else spec.w_bw
        cost = spec.r_cost if read else spec.w_cost
        if precision_bits <= 0 or bw <= 0:
            return 0.0
        return cost / (bw / precision_bits)

    def _accumulate_memory(
        self,
        totals: dict[str, dict[str, int]],
        energy: dict[str, dict[str, float]],
        level: str,
        direction: str,
        bits: int,
        spec: MemoryInstanceSpec,
        precision_bits: int,
        read: bool,
    ):
        if bits <= 0:
            return
        if level == "dram" and self.memory_config.zero_dram_in_paper_energy:
            return
        if level == "noc" and self.memory_config.zero_noc_in_paper_energy:
            return
        if (
            level == "sram"
            and self.memory_config.zero_sram_high_directions
            and direction in {"rl2h", "wh2l"}
        ):
            return
        totals[level][direction] += bits
        energy[level][direction] += (
            bits / precision_bits
        ) * self._memory_energy_per_element(spec, precision_bits, read)

    def _base_memory_for_fragment(self, fragment: _Fragment):
        totals = defaultdict(lambda: defaultdict(int))
        energy = defaultdict(lambda: defaultdict(float))
        if (
            fragment.core_type
            not in {"fp_soma", "bp_grad", "wg", "ann_fe", "ann_be", "ann_we"}
            or fragment.mac_count == 0
        ):
            return totals, energy

        cfg = self.memory_config.memory_instances
        if fragment.core_type == "fp_soma":
            reg_i1, reg_i2, reg_o = cfg["reg_1b"], cfg["reg_16b"], cfg["reg_16b"]
            sram_i1, sram_i2, sram_o = (
                cfg["sram_fp_conv_in_s"],
                cfg["sram_fp_conv_in_w"],
                cfg["sram_fp_conv_out_xi"],
            )
        elif fragment.core_type == "ann_fe":
            reg_i1, reg_i2, reg_o = cfg["reg_16b"], cfg["reg_16b"], cfg["reg_16b"]
            sram_i1, sram_i2, sram_o = (
                cfg["sram_fp_conv_in_w"],
                cfg["sram_fp_conv_in_w"],
                cfg["sram_fp_conv_out_xi"],
            )
        elif fragment.core_type in {"bp_grad", "ann_be"}:
            reg_i1, reg_i2, reg_o = cfg["reg_16b"], cfg["reg_16b"], cfg["reg_16b"]
            sram_i1, sram_i2, sram_o = (
                cfg["sram_bp_conv_in_du"],
                cfg["sram_bp_conv_in_w"],
                cfg["sram_bp_conv_out_res"],
            )
        else:
            if fragment.core_type == "ann_we":
                reg_i1, reg_i2, reg_o = (
                    cfg["reg_16b"],
                    cfg["reg_16b"],
                    cfg["reg_16b"],
                )
                sram_i1, sram_i2, sram_o = (
                    cfg["sram_wg_conv_in_du"],
                    cfg["sram_wg_conv_in_du"],
                    cfg["sram_wg_conv_out_dw"],
                )
            else:
                reg_i1, reg_i2, reg_o = cfg["reg_1b"], cfg["reg_16b"], cfg["reg_16b"]
                sram_i1, sram_i2, sram_o = (
                    cfg["sram_wg_conv_in_s"],
                    cfg["sram_wg_conv_in_du"],
                    cfg["sram_wg_conv_out_dw"],
                )

        i1_bits = fragment.input_numel * fragment.input_precision_bits
        i2_bits = fragment.weight_numel * fragment.weight_precision_bits
        o_bits = fragment.output_numel * fragment.output_precision_bits
        reuse_weight = (
            fragment.b_type > 0 or fragment.t_type > 0
        ) and i2_bits <= sram_i2.size_bits

        self._accumulate_memory(
            totals,
            energy,
            "reg",
            "rh2l",
            i1_bits,
            reg_i1,
            fragment.input_precision_bits,
            True,
        )
        self._accumulate_memory(
            totals,
            energy,
            "sram",
            "rh2l",
            i1_bits,
            sram_i1,
            fragment.input_precision_bits,
            True,
        )
        if not reuse_weight:
            self._accumulate_memory(
                totals,
                energy,
                "reg",
                "rh2l",
                i2_bits,
                reg_i2,
                fragment.weight_precision_bits,
                True,
            )
            self._accumulate_memory(
                totals,
                energy,
                "sram",
                "rh2l",
                i2_bits,
                sram_i2,
                fragment.weight_precision_bits,
                True,
            )
        self._accumulate_memory(
            totals,
            energy,
            "reg",
            "wl2h",
            o_bits,
            reg_o,
            fragment.output_precision_bits,
            False,
        )
        self._accumulate_memory(
            totals,
            energy,
            "sram",
            "wl2h",
            o_bits,
            sram_o,
            fragment.output_precision_bits,
            False,
        )
        return totals, energy

    def _extra_memory_for_fragment(self, fragment: _Fragment):
        totals = defaultdict(lambda: defaultdict(int))
        energy = defaultdict(lambda: defaultdict(float))
        if fragment.process_key == "with_nothing":
            return totals, energy

        cfg = self.memory_config.memory_instances
        ld = fragment.loop_dims
        if fragment.process_key == "with_opt" and fragment.op_name == "sgd":
            total_params = _optimizer_param_count(fragment)
            total_bits = total_params * 32
            sram_spec = cfg["sram_6MB"]
            self._accumulate_memory(
                totals,
                energy,
                "sram",
                "rh2l",
                total_bits,
                sram_spec,
                32,
                True,
            )
            self._accumulate_memory(
                totals,
                energy,
                "sram",
                "rh2l",
                total_bits,
                sram_spec,
                32,
                True,
            )
            self._accumulate_memory(
                totals,
                energy,
                "sram",
                "wl2h",
                total_bits,
                sram_spec,
                32,
                False,
            )
            if (
                fragment.optimizer_has_momentum
                and fragment.optimizer_has_momentum_buffer
            ):
                self._accumulate_memory(
                    totals,
                    energy,
                    "sram",
                    "rh2l",
                    total_bits,
                    sram_spec,
                    32,
                    True,
                )
            if fragment.optimizer_has_momentum:
                self._accumulate_memory(
                    totals,
                    energy,
                    "sram",
                    "wl2h",
                    total_bits,
                    sram_spec,
                    32,
                    False,
                )
            return totals, energy

        scalar_counts = {
            "OYOXKBT": ld["OY"] * ld["OX"] * ld["K"] * ld["B"] * ld["T"],
            "KT": ld["K"] if fragment.process_key == "with_opt" else ld["K"] * ld["T"],
            "OYOXCBT": ld["OY"] * ld["OX"] * ld["C"] * ld["B"] * ld["T"],
            "CT": ld["C"] * ld["T"],
            "FYFXKC": (
                ld["FY"] * ld["FX"] * ld["C"]
                if fragment.process_key == "with_opt"
                else ld["FY"] * ld["FX"] * ld["K"] * ld["C"]
            ),
        }
        variables: dict[str, tuple[str, int, str, str]] = {}
        if fragment.process_key == "with_sg" and fragment.phase == "forward":
            variables = {
                "fp_yi1": ("OYOXKBT", 16, "reg_16b", "sram_fp_conv_out_xi"),
                "fp_u_l": ("OYOXKBT", 16, "reg_16b", "sram_fp_soma_u"),
                "fp_s_l": ("OYOXKBT", 1, "reg_1b", "sram_fp_soma_s"),
                "fp_smask_l": ("OYOXKBT", 1, "reg_1b", "sram_fp_soma_smask"),
            }
        elif fragment.process_key == "with_sg" and fragment.phase == "backward":
            variables = {
                "bp_conv_res": ("OYOXCBT", 16, "reg_16b", "sram_bp_conv_out_res"),
                "bp_u_l_pre": ("OYOXCBT", 16, "reg_16b", "sram_bp_grad_in_u"),
                "bp_s_l_pre": ("OYOXCBT", 1, "reg_1b", "sram_bp_grad_in_s"),
                "bp_smask_l_pre": ("OYOXCBT", 1, "reg_1b", "sram_bp_grad_in_smask"),
                "bp_du_l_pre": ("OYOXCBT", 16, "reg_16b", "sram_bp_grad_out_du"),
            }
        elif fragment.process_key == "with_bn" and fragment.phase == "forward":
            variables = {
                "fp_bn_mean_v": ("KT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_variance": ("KT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_n": ("OYOXKBT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_sqrt": ("KT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_xi_": ("OYOXKBT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_y": ("KT", 16, "reg_16b", "sram_2MB"),
                "fp_bn_b": ("KT", 16, "reg_16b", "sram_2MB"),
            }
        elif fragment.process_key == "with_bn" and fragment.phase == "backward":
            variables = {
                "bp_bn_du_l_pre1": ("OYOXCBT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_sqrt": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_y": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_m": ("OYOXCBT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_n": ("OYOXCBT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_sigma_m": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_sigma_n": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_sigma_mn": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_dy": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_db": ("CT", 16, "reg_16b", "sram_2MB"),
                "bp_bn_du_l_pre2": ("OYOXCBT", 16, "reg_16b", "sram_2MB"),
            }
        elif fragment.process_key == "with_opt":
            if fragment.op_name in {"adam", "adamw"}:
                variables = {
                    "opt_y": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_b": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_w": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_dy": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_db": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_dw": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_v_y": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_v_b": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_v_w": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_s_y": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_s_b": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_s_w": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_vbc_y": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_vbc_b": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_vbc_w": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_sbc_y": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_sbc_b": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_sbc_w": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                    "opt_y_updated": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_b_updated": ("KT", 32, "reg_32b", "sram_6MB"),
                    "opt_w_updated": ("FYFXKC", 32, "reg_32b", "sram_6MB"),
                }

        for variable_name, (
            count_key,
            bits_per_elem,
            reg_name,
            sram_name,
        ) in variables.items():
            total_bits = scalar_counts[count_key] * bits_per_elem
            read_multiplier = _EXTRA_RH2L_MULTIPLIER.get(variable_name, 0)
            for level, spec_name in (("reg", reg_name), ("sram", sram_name)):
                spec = cfg[spec_name]
                self._accumulate_memory(
                    totals,
                    energy,
                    level,
                    "rh2l",
                    total_bits * read_multiplier,
                    spec,
                    bits_per_elem,
                    True,
                )
                write_variable = variable_name
                if fragment.process_key == "with_sg" and fragment.phase == "backward":
                    if level == "reg" and variable_name == "bp_du_l_pre":
                        write_variable = "bp_du_l_pre"
                    elif level == "sram" and variable_name == "bp_u_l_pre":
                        write_variable = "bp_u_l_pre"
                    else:
                        write_variable = ""
                self._accumulate_memory(
                    totals,
                    energy,
                    level,
                    "wl2h",
                    total_bits if write_variable in _EXTRA_WL2H else 0,
                    spec,
                    bits_per_elem,
                    False,
                )
        return totals, energy

    def _validate_sgd_group(self, group: dict[str, Any]) -> None:
        if group.get("nesterov", False):
            raise ValueError(
                "NeuroMC runtime SGD modeling currently supports only nesterov=False."
            )
        if float(group.get("dampening", 0.0)) != 0.0:
            raise ValueError(
                "NeuroMC runtime SGD modeling currently supports only dampening=0."
            )
        if group.get("maximize", False):
            raise ValueError(
                "NeuroMC runtime SGD modeling currently supports only maximize=False."
            )

    def _make_optimizer_fragment(
        self,
        *,
        stage: str,
        op_name: str,
        kt: int,
        fyfxkc: int,
        optimizer_has_momentum: bool = False,
        optimizer_has_weight_decay: bool = False,
        optimizer_has_momentum_buffer: bool = False,
    ) -> _Fragment:
        loop_dims = self._make_loop_dims(
            batch_size=1,
            channels_in=1,
            channels_out=max(kt, 1),
            oy=1,
            ox=1,
            fy=1,
            fx=max(fyfxkc, 1),
        )
        loop_dims["K"] = kt
        loop_dims["C"] = 1
        loop_dims["FY"] = 1
        loop_dims["FX"] = fyfxkc
        return _Fragment(
            stage=stage,
            phase="optimizer",
            op_name=op_name,
            core_type="bp_grad_opt",
            process_key="with_opt",
            loop_dims=loop_dims,
            input_precision_bits=32,
            weight_precision_bits=32,
            output_precision_bits=32,
            input_numel=kt,
            weight_numel=fyfxkc,
            output_numel=kt + fyfxkc,
            mac_count=0,
            source="optimizer",
            optimizer_has_momentum=optimizer_has_momentum,
            optimizer_has_weight_decay=optimizer_has_weight_decay,
            optimizer_has_momentum_buffer=optimizer_has_momentum_buffer,
        )

    def _optimizer_fragments(self, stage: str) -> list[_Fragment]:
        if self._optimizer is None:
            raise RuntimeError("Optimizer stage requires a bound optimizer.")
        if isinstance(self._optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            kt = 0
            fyfxkc = 0
            for group in self._optimizer.param_groups:
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    if p.ndim <= 1:
                        kt += int(p.numel())
                    else:
                        fyfxkc += int(p.numel())
            return [
                self._make_optimizer_fragment(
                    stage=stage,
                    op_name=type(self._optimizer).__name__.lower(),
                    kt=kt,
                    fyfxkc=fyfxkc,
                )
            ]
        if isinstance(self._optimizer, torch.optim.SGD):
            buckets: dict[tuple[bool, bool, bool, str], int] = defaultdict(int)
            for group in self._optimizer.param_groups:
                self._validate_sgd_group(group)
                has_momentum = float(group.get("momentum", 0.0)) > 0.0
                has_weight_decay = float(group.get("weight_decay", 0.0)) > 0.0
                for p in group["params"]:
                    if p.grad is None:
                        continue
                    bucket_name = "kt" if p.ndim <= 1 else "fyfxkc"
                    has_momentum_buffer = (
                        has_momentum and "momentum_buffer" in self._optimizer.state[p]
                    )
                    buckets[
                        (
                            has_momentum,
                            has_weight_decay,
                            has_momentum_buffer,
                            bucket_name,
                        )
                    ] += int(p.numel())
            fragments = []
            for (
                has_momentum,
                has_weight_decay,
                has_momentum_buffer,
                bucket_name,
            ), count in buckets.items():
                if count <= 0:
                    continue
                fragments.append(
                    self._make_optimizer_fragment(
                        stage=stage,
                        op_name="sgd",
                        kt=count if bucket_name == "kt" else 0,
                        fyfxkc=count if bucket_name == "fyfxkc" else 0,
                        optimizer_has_momentum=has_momentum,
                        optimizer_has_weight_decay=has_weight_decay,
                        optimizer_has_momentum_buffer=has_momentum_buffer,
                    )
                )
            return fragments
        raise ValueError(
            "NeuroMC runtime optimizer modeling only supports Adam/AdamW and "
            "common SGD (nesterov=False, dampening=0, maximize=False); "
            f"got {type(self._optimizer).__name__}."
        )

    def get_report(self) -> NeuroMCRuntimeEnergyReport:
        fragments = list(self._fragments)
        trace_fragments = self._fallback_fragments_from_trace(
            supplemental_only=bool(fragments)
        )
        if fragments:
            existing_signatures = {
                _module_overlap_signature(fragment)
                for fragment in fragments
                if fragment.source == "module"
            }
            fragments.extend(
                fragment
                for fragment in trace_fragments
                if _module_overlap_signature(fragment) not in existing_signatures
            )
        else:
            fragments = trace_fragments

        unsupported = self._unsupported_ops()
        if unsupported:
            raise ValueError(
                "NeuroMC runtime does not support these aten ops: "
                + ", ".join(unsupported[:30])
            )
        if not fragments:
            raise ValueError("No NeuroMC-supported runtime fragments were recognized.")

        energy_by_stage = defaultdict(float)
        energy_by_op = defaultdict(float)
        energy_by_core_type = defaultdict(float)
        energy_by_process_key = defaultdict(float)
        energy_by_memory_level_dir = defaultdict(lambda: defaultdict(float))
        counts_by_core_type = defaultdict(lambda: defaultdict(int))
        counts_by_process_key = defaultdict(lambda: defaultdict(int))
        primitive_by_stage = defaultdict(lambda: defaultdict(int))
        primitive_by_op = defaultdict(lambda: defaultdict(int))
        memory_bits_by_level = defaultdict(lambda: defaultdict(int))
        mapping_summary = []
        warnings = list(self._warnings)

        energy_mac = 0.0
        energy_base_memory = 0.0
        energy_extra_memory = 0.0
        energy_extra_compute = 0.0

        for fragment in fragments:
            extra_counts = self._extra_counts(fragment)
            base_bits, base_energy = self._base_memory_for_fragment(fragment)
            extra_bits, extra_mem_energy = self._extra_memory_for_fragment(fragment)
            mac_energy = fragment.mac_count * _MAC_COST_PJ.get(fragment.core_type, 0.0)
            extra_compute_energy = (
                extra_counts["mux"] * _EXTRA_OP_COST_PJ["mux"]
                + extra_counts["add"] * _EXTRA_OP_COST_PJ["add"]
                + extra_counts["mul"] * _EXTRA_OP_COST_PJ["mul"]
                + extra_counts["comp"] * _EXTRA_OP_COST_PJ["comp"]
                + extra_counts["sqrt"] * _EXTRA_OP_COST_PJ["sqrt"]
            )
            base_energy_total = sum(
                sum(direction_map.values()) for direction_map in base_energy.values()
            )
            extra_memory_total = sum(
                sum(direction_map.values())
                for direction_map in extra_mem_energy.values()
            )
            total = (
                mac_energy
                + extra_compute_energy
                + base_energy_total
                + extra_memory_total
            )

            energy_mac += mac_energy
            energy_base_memory += base_energy_total
            energy_extra_memory += extra_memory_total
            energy_extra_compute += extra_compute_energy
            energy_by_stage[fragment.stage] += total
            energy_by_op[fragment.op_name] += total
            energy_by_core_type[fragment.core_type] += total
            energy_by_process_key[fragment.process_key] += total
            counts_by_core_type[fragment.core_type]["fragments"] += 1
            counts_by_core_type[fragment.core_type]["mac"] += fragment.mac_count
            counts_by_process_key[fragment.process_key]["fragments"] += 1
            counts_by_process_key[fragment.process_key]["mac"] += fragment.mac_count

            for primitive, count in extra_counts.items():
                primitive_by_stage[fragment.stage][primitive] += count
                primitive_by_op[fragment.op_name][primitive] += count
                counts_by_core_type[fragment.core_type][primitive] += count
                counts_by_process_key[fragment.process_key][primitive] += count

            for level, directions in base_bits.items():
                for direction, bits in directions.items():
                    memory_bits_by_level[level][direction] += bits
            for level, directions in extra_bits.items():
                for direction, bits in directions.items():
                    memory_bits_by_level[level][direction] += bits
            for level, directions in base_energy.items():
                for direction, value in directions.items():
                    energy_by_memory_level_dir[level][direction] += value
            for level, directions in extra_mem_energy.items():
                for direction, value in directions.items():
                    energy_by_memory_level_dir[level][direction] += value

            mapping_summary.append(
                {
                    "stage": fragment.stage,
                    "phase": fragment.phase,
                    "op_name": fragment.op_name,
                    "core_type": fragment.core_type,
                    "process_key": fragment.process_key,
                    "input_precision_bits": fragment.input_precision_bits,
                    "weight_precision_bits": fragment.weight_precision_bits,
                    "output_precision_bits": fragment.output_precision_bits,
                    "loop_dims": dict(fragment.loop_dims),
                    "b_type": fragment.b_type,
                    "t_type": fragment.t_type,
                    "mac_count": fragment.mac_count,
                    "source": fragment.source,
                    "optimizer_has_momentum": fragment.optimizer_has_momentum,
                    "optimizer_has_weight_decay": fragment.optimizer_has_weight_decay,
                    "optimizer_has_momentum_buffer": fragment.optimizer_has_momentum_buffer,
                }
            )

        primitive_totals = {
            primitive: int(
                sum(
                    stage_counts.get(primitive, 0)
                    for stage_counts in primitive_by_stage.values()
                )
            )
            for primitive in ("mux", "add", "mul", "comp", "sqrt")
        }
        primitive_totals["mac"] = int(sum(fragment.mac_count for fragment in fragments))

        energy_compute_pj = energy_mac + energy_extra_compute
        energy_memory_pj = energy_base_memory + energy_extra_memory
        energy_total_pj = energy_compute_pj + energy_memory_pj

        primitive_counts = {
            "totals": primitive_totals,
            "by_stage": {k: dict(v) for k, v in primitive_by_stage.items()},
            "by_op": {k: dict(v) for k, v in primitive_by_op.items()},
        }
        memory_report = {
            "preset_name": self.memory_config.preset_name,
            "totals": {
                level: int(sum(directions.values()))
                for level, directions in memory_bits_by_level.items()
            },
            "by_level_dir": {k: dict(v) for k, v in memory_bits_by_level.items()},
        }

        return NeuroMCRuntimeEnergyReport(
            energy_total_pj=energy_total_pj,
            energy_compute_pj=energy_compute_pj,
            energy_memory_pj=energy_memory_pj,
            energy_by_stage=dict(energy_by_stage),
            energy_by_op=dict(energy_by_op),
            primitive_counts=primitive_counts,
            memory_bits_by_level=memory_report,
            warnings=warnings,
            energy_mac_pj=energy_mac,
            energy_base_memory_pj=energy_base_memory,
            energy_extra_memory_pj=energy_extra_memory,
            energy_extra_compute_pj=energy_extra_compute,
            energy_by_core_type=dict(energy_by_core_type),
            energy_by_process_key=dict(energy_by_process_key),
            energy_by_memory_level_dir={
                level: dict(direction_map)
                for level, direction_map in energy_by_memory_level_dir.items()
            },
            counts_by_core_type={k: dict(v) for k, v in counts_by_core_type.items()},
            counts_by_process_key={
                k: dict(v) for k, v in counts_by_process_key.items()
            },
            mapping_summary=mapping_summary,
            model_info=_NEUROMC_MODEL_INFO,
            memory_config=replace(
                self.memory_config,
                memory_instances=dict(self.memory_config.memory_instances),
            ),
        )

    def get_total(self) -> float:
        return self.get_report().energy_total_pj

    def get_counts(self) -> dict[str, Any]:
        report = self.get_report()
        return {
            "primitive_counts": report.primitive_counts,
            "memory_bits_by_level": report.memory_bits_by_level,
            "counts_by_core_type": report.counts_by_core_type,
            "counts_by_process_key": report.counts_by_process_key,
        }

    def record_optimizer_step(self, stage: str = "optimizer") -> None:
        self._fragments.extend(self._optimizer_fragments(stage))


def estimate_neuromc_runtime_energy(
    model: nn.Module,
    inputs: Any,
    *,
    target: torch.Tensor | None = None,
    loss_fn: Callable | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    memory_config: MemoryHierarchyConfig | None = None,
) -> NeuroMCRuntimeEnergyReport:
    r"""
    **API Language** - :ref:`中文 <estimate_neuromc_runtime_energy-cn>` | :ref:`English <estimate_neuromc_runtime_energy-en>`

    ----

    .. _estimate_neuromc_runtime_energy-cn:

    * **中文**

    运行一次真实 forward；提供 loss 时继续捕获 backward，并按 optimizer 配置
    估算更新阶段。该函数不会执行 ``optimizer.step()``，因此不会修改参数。

    :param model: The PyTorch model to profile
    :type model: ``nn.Module``
    :param inputs: Input tensors for the forward pass
    :type inputs: Any
    :param target: Target tensors for loss computation
    :type target: torch.Tensor | None
    :param loss_fn: Loss function for the backward pass
    :type loss_fn: Callable | None
    :param optimizer: Optimizer for training-stage profiling
    :type optimizer: torch.optim.Optimizer | None
    :param memory_config: Memory hierarchy configuration. If ``None``, uses the default config
    :type memory_config: MemoryHierarchyConfig | None
    :return: Energy profiling report
    :rtype: NeuroMCRuntimeEnergyReport

    This is a convenience function that creates a :class:`NeuroMCEnergyProfiler`,
    binds the model and optional optimizer, runs the full profile, and returns
    the energy report.

    ----

    .. _estimate_neuromc_runtime_energy-en:

    * **English**

    Run one real forward pass, optionally capture backward, and estimate the
    configured optimizer stage. The function does not call ``optimizer.step()``
    and therefore does not update model parameters.

    :param model: The PyTorch model to profile
    :param inputs: Input tensors for the forward pass
    :param target: Target tensors for loss computation
    :param loss_fn: Loss function for the backward pass
    :param optimizer: Optimizer for training-stage profiling
    :param memory_config: Memory hierarchy configuration. If ``None``, uses the default config
    :type model: ``nn.Module``
    :type inputs: Any
    :type target: torch.Tensor | None
    :type loss_fn: Callable | None
    :type optimizer: torch.optim.Optimizer | None
    :type memory_config: MemoryHierarchyConfig | None
    :return: Energy profiling report
    :rtype: NeuroMCRuntimeEnergyReport
    """
    profiler = NeuroMCEnergyProfiler(
        memory_config=memory_config,
    )
    profiler.bind_model(model)
    profiler.bind_optimizer(optimizer)
    _clear_existing_grads(model, optimizer)

    with profiler:
        with profiler.stage("forward", phase="forward"):
            output = call_model(model, inputs)
        loss = None
        if loss_fn is not None:
            with profiler.suspend():
                if target is None:
                    raise ValueError("target is required when loss_fn is provided")
                loss = loss_fn(output, target)
        if loss is not None:
            with profiler.stage("backward", phase="backward"):
                loss.backward()
            if optimizer is not None:
                with profiler.stage("optimizer", phase="optimizer"):
                    profiler.record_optimizer_step("optimizer")
                    with profiler.suspend():
                        optimizer.zero_grad(set_to_none=True)
        elif optimizer is not None:
            raise ValueError(
                "NeuroMC runtime optimizer modeling requires loss_fn and target; "
                "optimizer.step() without backward is unsupported."
            )
    return profiler.get_report()
