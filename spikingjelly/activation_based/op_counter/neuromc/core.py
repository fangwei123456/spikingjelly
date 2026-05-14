from __future__ import annotations

from collections import defaultdict
from contextlib import contextmanager
from dataclasses import dataclass
import re
from typing import Any, Callable

import torch
import torch.nn as nn
from torch.overrides import resolve_name
from torch.utils._python_dispatch import TorchDispatchMode

from .config import MemoryHierarchyConfig, MemoryInstanceSpec

__all__ = [
    "MemoryHierarchyConfig",
    "NeuroMCEnergyProfiler",
    "NeuroMCRuntimeEnergyReport",
    "estimate_neuromc_runtime_energy",
]


_ALLOWED_CORE_TYPES = {
    "fp_soma",
    "fp_bn",
    "bp_grad",
    "bp_bn",
    "bp_grad_opt",
    "wg",
}

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
    energy_total_pj: float
    energy_compute_pj: float
    energy_memory_pj: float
    energy_by_stage: dict[str, float]
    energy_by_op: dict[str, float]
    primitive_counts: dict[str, Any]
    memory_bits_by_level: dict[str, Any]
    warnings: list[str]
    energy_mac_pj: float
    energy_base_memory_pj: float
    energy_extra_memory_pj: float
    energy_extra_compute_pj: float
    energy_by_core_type: dict[str, float]
    energy_by_process_key: dict[str, float]
    energy_by_memory_level_dir: dict[str, dict[str, float]]
    counts_by_core_type: dict[str, dict[str, int]]
    counts_by_process_key: dict[str, dict[str, int]]
    mapping_summary: list[dict[str, Any]]


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


def _prod(values) -> int:
    p = 1
    for v in values:
        p *= int(v)
    return p


def _is_spike_tensor(x: Any) -> bool:
    if not torch.is_tensor(x):
        return False
    if x.dtype == torch.bool:
        return True
    if x.numel() == 0:
        return False
    return bool(x.eq(0).logical_or(x.eq(1)).all().item())


def _tensor_numel(x: Any) -> int:
    if not torch.is_tensor(x):
        return 0
    return int(x.numel())


def _module_channel_count(x: torch.Tensor) -> int:
    if x.ndim >= 2:
        return int(x.shape[1])
    return int(x.numel())


def _resolve_loss_fn(loss_fn: Callable | None):
    if loss_fn is None:
        return None
    if isinstance(loss_fn, nn.Module):
        return loss_fn
    return loss_fn


def _call_model(model: nn.Module, inputs):
    if isinstance(inputs, (tuple, list)):
        return model(*inputs)
    return model(inputs)


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


class NeuroMCEnergyProfiler:
    def __init__(
        self,
        *,
        core_type: str = "fp_soma",
        memory_config: MemoryHierarchyConfig | None = None,
        strict: bool = False,
        verbose: bool = False,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        if core_type not in _ALLOWED_CORE_TYPES:
            raise ValueError(
                f"Unsupported NeuroMC core_type={core_type}. "
                f"Supported: {sorted(_ALLOWED_CORE_TYPES)}."
            )
        self.core_type = core_type
        self.memory_config = memory_config or MemoryHierarchyConfig.neuromc_like_v1()
        self.memory_config.validate()
        self.strict = strict
        self.verbose = verbose
        self.extra_ignore_modules = list(extra_ignore_modules or [])

        self._stage_stack: list[str] = []
        self._warnings: list[str] = []
        self._trace_mode = _TraceMode(self)
        self._trace_events: list[_TraceEvent] = []
        self._fragments: list[_Fragment] = []
        self._bound_model: nn.Module | None = None
        self._active = False
        self._suspended = False
        self._model_bound = False
        self._optimizer: torch.optim.Optimizer | None = None
        self._hook_handles = []

    def bind_model(self, model: nn.Module):
        if self._model_bound:
            return
        self._bound_model = model
        from ...neuron.base_node import BaseNode

        supported = (
            nn.Conv1d,
            nn.Conv2d,
            nn.Conv3d,
            nn.Linear,
            nn.BatchNorm1d,
            nn.BatchNorm2d,
            nn.BatchNorm3d,
            BaseNode,
        )
        for module in model.modules():
            if module in self.extra_ignore_modules:
                continue
            if isinstance(module, supported):
                self._hook_handles.append(module.register_forward_hook(self._forward_hook))
                self._hook_handles.append(
                    module.register_full_backward_hook(self._backward_hook)
                )
        self._model_bound = True

    def bind_optimizer(self, optimizer: torch.optim.Optimizer | None):
        self._optimizer = optimizer

    def __enter__(self):
        self._trace_mode.__enter__()
        self._active = True
        return self

    def __exit__(self, exc_type, exc, tb):
        self._active = False
        for handle in self._hook_handles:
            handle.remove()
        self._hook_handles.clear()
        return self._trace_mode.__exit__(exc_type, exc, tb)

    @contextmanager
    def stage(self, name: str):
        if not self._active:
            raise RuntimeError("stage() can only be used inside active profiler context.")
        if self._stage_stack:
            raise RuntimeError("Nested stage() is not supported in NeuroMC v2.")
        self._stage_stack.append(name)
        try:
            yield self
        finally:
            self._stage_stack.pop()

    @contextmanager
    def suspend(self):
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

    def _stage_phase(self, stage: str | None = None) -> str:
        name = stage or self._current_stage()
        lowered = name.lower()
        if "backward" in lowered:
            return "backward"
        if "optimizer" in lowered or "update" in lowered:
            return "optimizer"
        return "forward"

    def _stage_position(self, stage: str | None = None) -> tuple[int, int]:
        name = stage or self._current_stage()
        lowered = name.lower()
        b_match = re.search(r"(?:^|[^a-z0-9])b(\d+)(?:[^a-z0-9]|$)", lowered)
        t_match = re.search(r"(?:^|[^a-z0-9])t(\d+)(?:[^a-z0-9]|$)", lowered)
        b_type = int(b_match.group(1)) if b_match is not None else 0
        t_type = int(t_match.group(1)) if t_match is not None else 0
        return b_type, t_type

    def _stage_conv_type(self, stage: str | None = None) -> str:
        name = (stage or self._current_stage()).lower()
        if "without_bp_bn" in name:
            return "without_bp_bn"
        return "--"

    def _maybe_record_trace_event(self, op_name: str, args, kwargs, out):
        if not self._active or self._suspended:
            return
        self._trace_events.append(
            _TraceEvent(
                op_name=op_name,
                stage=self._current_stage(),
                phase=self._stage_phase(),
                args=args,
                kwargs=kwargs,
                out=out,
            )
        )

    def _forward_hook(self, module: nn.Module, args, out):
        if not self._active or self._suspended:
            return
        stage = self._current_stage()
        from ...neuron.base_node import BaseNode

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            x = args[0]
            module._neuromc_last_input = x
            self._fragments.append(self._make_conv_forward_fragment(stage, module, x, out))
        elif isinstance(module, nn.Linear):
            x = args[0]
            module._neuromc_last_input = x
            self._fragments.append(self._make_linear_forward_fragment(stage, module, x, out))
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            x = args[0]
            self._fragments.append(self._make_bn_forward_fragment(stage, x, out))
        elif isinstance(module, BaseNode):
            x = args[0]
            self._fragments.append(self._make_soma_forward_fragment(stage, x, out))

    def _backward_hook(self, module: nn.Module, grad_input, grad_output):
        if not self._active or self._suspended:
            return
        stage = self._current_stage()
        from ...neuron.base_node import BaseNode

        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            self._fragments.extend(
                self._make_conv_backward_fragments(stage, module, grad_input, grad_output)
            )
        elif isinstance(module, nn.Linear):
            self._fragments.extend(
                self._make_linear_backward_fragments(stage, module, grad_input, grad_output)
            )
        elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
            self._fragments.append(self._make_bn_backward_fragment(stage, grad_input, grad_output))
        elif isinstance(module, BaseNode):
            self._fragments.append(
                self._make_soma_backward_fragment(stage, grad_input, grad_output)
            )

    def _make_loop_dims(
        self,
        *,
        batch_size: int,
        channels_in: int,
        channels_out: int,
        oy: int,
        ox: int,
        fy: int,
        fx: int,
    ) -> dict[str, int]:
        return {
            "B": max(int(batch_size), 1),
            "T": 1,
            "C": max(int(channels_in), 1),
            "K": max(int(channels_out), 1),
            "OY": max(int(oy), 1),
            "OX": max(int(ox), 1),
            "FY": max(int(fy), 1),
            "FX": max(int(fx), 1),
        }

    def _make_conv_forward_fragment(self, stage: str, module: nn.Module, x, out) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        spatial = tuple(out.shape[2:]) if out.ndim > 2 else (1, 1)
        if len(spatial) == 1:
            spatial = (spatial[0], 1)
        kernel = tuple(module.kernel_size) if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
        if len(kernel) == 1:
            kernel = (kernel[0], 1)
        loop_dims = self._make_loop_dims(
            batch_size=int(x.shape[0]),
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
            core_type="fp_soma",
            process_key="with_nothing",
            loop_dims=loop_dims,
            input_precision_bits=1 if _is_spike_tensor(x) else 16,
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

    def _make_linear_forward_fragment(self, stage: str, module: nn.Linear, x, out) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        batch = int(x.shape[0]) if x.ndim > 1 else 1
        loop_dims = self._make_loop_dims(
            batch_size=batch,
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
            core_type="fp_soma",
            process_key="with_nothing",
            loop_dims=loop_dims,
            input_precision_bits=1 if _is_spike_tensor(x) else 16,
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

    def _make_bn_forward_fragment(self, stage: str, x, out) -> _Fragment:
        c = _module_channel_count(x)
        spatial_prod = max(int(x.numel() // (max(int(x.shape[0]), 1) * c)), 1)
        loop_dims = self._make_loop_dims(
            batch_size=int(x.shape[0]) if x.ndim > 1 else 1,
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
            core_type="fp_bn",
            process_key="with_bn",
            loop_dims=loop_dims,
            input_precision_bits=16,
            weight_precision_bits=16,
            output_precision_bits=16,
            input_numel=_tensor_numel(x),
            weight_numel=c * 2,
            output_numel=_tensor_numel(out[0] if isinstance(out, (tuple, list)) else out),
            mac_count=0,
            source="module",
        )

    def _make_soma_forward_fragment(self, stage: str, x, out) -> _Fragment:
        b_type, t_type = self._stage_position(stage)
        out_tensor = out[0] if isinstance(out, (tuple, list)) else out
        c = _module_channel_count(out_tensor) if torch.is_tensor(out_tensor) else 1
        spatial_prod = max(int(_tensor_numel(out_tensor) // max(c, 1)), 1)
        loop_dims = self._make_loop_dims(
            batch_size=1,
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
        grad_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        spatial = tuple(grad_out.shape[2:]) if grad_out.ndim > 2 else (1, 1)
        if len(spatial) == 1:
            spatial = (spatial[0], 1)
        kernel = tuple(module.kernel_size) if isinstance(module.kernel_size, tuple) else (module.kernel_size,)
        if len(kernel) == 1:
            kernel = (kernel[0], 1)
        batch = int(grad_out.shape[0])
        grad_in = grad_input[0] if isinstance(grad_input, (tuple, list)) and grad_input else None
        base_loop = self._make_loop_dims(
            batch_size=batch,
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
                    core_type="bp_grad",
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
                    core_type="wg",
                    process_key="with_nothing",
                    loop_dims=base_loop,
                    input_precision_bits=1
                    if _is_spike_tensor(getattr(module, "_neuromc_last_input", None))
                    else 16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(getattr(module, "_neuromc_last_input", None)),
                    weight_numel=_tensor_numel(grad_out),
                    output_numel=_tensor_numel(module.weight),
                    mac_count=self._mac_count(base_loop),
                    source="module",
                )
            )
        return fragments

    def _make_linear_backward_fragments(self, stage, module, grad_input, grad_output):
        grad_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        batch = int(grad_out.shape[0]) if grad_out.ndim > 1 else 1
        loop_dims = self._make_loop_dims(
            batch_size=batch,
            channels_in=int(module.in_features),
            channels_out=int(module.out_features),
            oy=1,
            ox=1,
            fy=1,
            fx=1,
        )
        grad_in = grad_input[0] if isinstance(grad_input, (tuple, list)) and grad_input else None
        fragments = []
        if torch.is_tensor(grad_in):
            fragments.append(
                _Fragment(
                    stage=stage,
                    phase="backward",
                    op_name="linear.backward.grad_input",
                    core_type="bp_grad",
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
                    core_type="wg",
                    process_key="with_nothing",
                    loop_dims=loop_dims,
                    input_precision_bits=1
                    if _is_spike_tensor(getattr(module, "_neuromc_last_input", None))
                    else 16,
                    weight_precision_bits=16,
                    output_precision_bits=16,
                    input_numel=_tensor_numel(getattr(module, "_neuromc_last_input", None)),
                    weight_numel=_tensor_numel(grad_out),
                    output_numel=_tensor_numel(module.weight),
                    mac_count=self._mac_count(loop_dims),
                    source="module",
                )
            )
        return fragments

    def _make_bn_backward_fragment(self, stage, grad_input, grad_output):
        grad_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        grad_in = grad_input[0] if isinstance(grad_input, (tuple, list)) and grad_input else grad_out
        batch = int(grad_out.shape[0]) if grad_out.ndim > 1 else 1
        c = _module_channel_count(grad_out)
        spatial_prod = max(int(_tensor_numel(grad_out) // max(batch * c, 1)), 1)
        conv_type = self._stage_conv_type(stage)
        loop_dims = self._make_loop_dims(
            batch_size=batch,
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
            core_type="bp_bn",
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

    def _make_soma_backward_fragment(self, stage, grad_input, grad_output):
        grad_in = grad_input[0] if isinstance(grad_input, (tuple, list)) and grad_input else None
        grad_out = grad_output[0] if isinstance(grad_output, (tuple, list)) else grad_output
        tensor = grad_in if torch.is_tensor(grad_in) else grad_out
        c = _module_channel_count(tensor)
        spatial_prod = max(int(_tensor_numel(tensor) // max(c, 1)), 1)
        loop_dims = self._make_loop_dims(
            batch_size=1,
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
        if self._bound_model is None or (not torch.is_tensor(out)):
            return False
        out_shape = tuple(out.shape)
        for p in self._bound_model.parameters():
            if p.requires_grad and tuple(p.shape) == out_shape:
                return True
        return False

    def _gemm_backward_fragment_kind(
        self, x: torch.Tensor, y: torch.Tensor, out: Any
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

    def _fallback_fragments_from_trace(self) -> list[_Fragment]:
        fragments: list[_Fragment] = []
        for event in self._trace_events:
            op = event.op_name
            if op in _AUXILIARY_ATEN_OPS or op.startswith(_IGNORED_OP_PREFIXES):
                continue
            if op == "aten.convolution.default" and event.phase == "forward":
                x, w = event.args[:2]
                out = event.out
                spatial = tuple(out.shape[2:]) if out.ndim > 2 else (1, 1)
                if len(spatial) == 1:
                    spatial = (spatial[0], 1)
                kernel = tuple(w.shape[2:]) if w.ndim > 2 else (1, 1)
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
                        core_type="fp_soma",
                        process_key="with_nothing",
                        loop_dims=loop_dims,
                        input_precision_bits=1 if _is_spike_tensor(x) else 16,
                        weight_precision_bits=16,
                        output_precision_bits=16,
                        input_numel=_tensor_numel(x),
                        weight_numel=_tensor_numel(w),
                        output_numel=_tensor_numel(out),
                        mac_count=self._mac_count(loop_dims),
                    )
                )
            elif op in {"aten.addmm.default", "aten.mm.default", "aten.bmm.default"}:
                x = event.args[-2]
                y = event.args[-1]
                out = event.out
                if event.phase == "forward":
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
                            core_type="fp_soma",
                            process_key="with_nothing",
                            loop_dims=loop_dims,
                            input_precision_bits=1 if _is_spike_tensor(x) else 16,
                            weight_precision_bits=16,
                            output_precision_bits=16,
                            input_numel=_tensor_numel(x),
                            weight_numel=_tensor_numel(y),
                            output_numel=_tensor_numel(out),
                            mac_count=self._mac_count(loop_dims),
                            b_type=0,
                            t_type=0,
                        )
                    )
                else:
                    kind = self._gemm_backward_fragment_kind(x, y, out)
                    if kind == "wg":
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
                                core_type="wg",
                                process_key="with_nothing",
                                loop_dims=loop_dims,
                                input_precision_bits=1 if _is_spike_tensor(x) else 16,
                                weight_precision_bits=16,
                                output_precision_bits=16,
                                input_numel=_tensor_numel(x),
                                weight_numel=_tensor_numel(y),
                                output_numel=_tensor_numel(out),
                                mac_count=self._mac_count(loop_dims),
                                b_type=0,
                                t_type=0,
                            )
                        )
                    else:
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
                                core_type="bp_grad",
                                process_key="with_nothing",
                                loop_dims=loop_dims,
                                input_precision_bits=16,
                                weight_precision_bits=16,
                                output_precision_bits=16,
                                input_numel=_tensor_numel(x),
                                weight_numel=_tensor_numel(y),
                                output_numel=_tensor_numel(out),
                                mac_count=self._mac_count(loop_dims),
                                b_type=0,
                                t_type=0,
                            )
                        )
            elif op == "aten.native_batch_norm.default" and event.phase == "forward":
                x = event.args[0]
                out = event.out[0] if isinstance(event.out, (tuple, list)) else event.out
                fragments.append(self._make_bn_forward_fragment(event.stage, x, out))
            elif op == "aten.native_batch_norm_backward.default":
                fragments.append(self._make_bn_backward_fragment(event.stage, event.args, event.out))
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
        kt = ld["K"] * ld["T"]
        oyoxcbt = ld["OY"] * ld["OX"] * ld["C"] * ld["B"] * ld["T"]
        ct = ld["C"] * ld["T"]
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
            counts["add"] += 8 * kt + 4 * fyfxkc
            counts["mul"] += 22 * kt + 11 * fyfxkc
            counts["sqrt"] += 2 * kt + fyfxkc
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
        energy[level][direction] += (bits / precision_bits) * self._memory_energy_per_element(
            spec, precision_bits, read
        )

    def _base_memory_for_fragment(self, fragment: _Fragment):
        totals = defaultdict(lambda: defaultdict(int))
        energy = defaultdict(lambda: defaultdict(float))
        if fragment.core_type not in {"fp_soma", "bp_grad", "wg"} or fragment.mac_count == 0:
            return totals, energy

        cfg = self.memory_config.memory_instances
        if fragment.core_type == "fp_soma":
            reg_i1, reg_i2, reg_o = cfg["reg_1b"], cfg["reg_16b"], cfg["reg_16b"]
            sram_i1, sram_i2, sram_o = (
                cfg["sram_fp_conv_in_s"],
                cfg["sram_fp_conv_in_w"],
                cfg["sram_fp_conv_out_xi"],
            )
        elif fragment.core_type == "bp_grad":
            reg_i1, reg_i2, reg_o = cfg["reg_16b"], cfg["reg_16b"], cfg["reg_16b"]
            sram_i1, sram_i2, sram_o = (
                cfg["sram_bp_conv_in_du"],
                cfg["sram_bp_conv_in_w"],
                cfg["sram_bp_conv_out_res"],
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
            (fragment.b_type > 0 or fragment.t_type > 0)
            and i2_bits <= sram_i2.size_bits
        )

        self._accumulate_memory(
            totals, energy, "reg", "rh2l", i1_bits, reg_i1, fragment.input_precision_bits, True
        )
        self._accumulate_memory(
            totals, energy, "sram", "rh2l", i1_bits, sram_i1, fragment.input_precision_bits, True
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
            totals, energy, "reg", "wl2h", o_bits, reg_o, fragment.output_precision_bits, False
        )
        self._accumulate_memory(
            totals, energy, "sram", "wl2h", o_bits, sram_o, fragment.output_precision_bits, False
        )
        return totals, energy

    def _extra_memory_for_fragment(self, fragment: _Fragment):
        totals = defaultdict(lambda: defaultdict(int))
        energy = defaultdict(lambda: defaultdict(float))
        if fragment.process_key == "with_nothing":
            return totals, energy

        cfg = self.memory_config.memory_instances
        ld = fragment.loop_dims
        scalar_counts = {
            "OYOXKBT": ld["OY"] * ld["OX"] * ld["K"] * ld["B"] * ld["T"],
            "KT": ld["K"] * ld["T"],
            "OYOXCBT": ld["OY"] * ld["OX"] * ld["C"] * ld["B"] * ld["T"],
            "CT": ld["C"] * ld["T"],
            "FYFXKC": ld["FY"] * ld["FX"] * ld["K"] * ld["C"],
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

        for _, (count_key, bits_per_elem, reg_name, sram_name) in variables.items():
            total_bits = scalar_counts[count_key] * bits_per_elem
            sram_spec = cfg[sram_name]
            self._accumulate_memory(
                totals, energy, "sram", "rh2l", total_bits, sram_spec, bits_per_elem, True
            )
            self._accumulate_memory(
                totals, energy, "sram", "wl2h", total_bits, sram_spec, bits_per_elem, False
            )
        return totals, energy

    def _optimizer_fragment(self, stage: str) -> _Fragment:
        if self._optimizer is None:
            raise RuntimeError("Optimizer stage requires a bound optimizer.")
        if not isinstance(self._optimizer, (torch.optim.Adam, torch.optim.AdamW)):
            raise ValueError(
                "Exact NeuroMC optimizer modeling only supports Adam/AdamW; "
                f"got {type(self._optimizer).__name__}."
            )
        kt = 0
        fyfxkc = 0
        for group in self._optimizer.param_groups:
            for p in group["params"]:
                if p.ndim <= 1:
                    kt += int(p.numel())
                else:
                    fyfxkc += int(p.numel())
        loop_dims = self._make_loop_dims(
            batch_size=1,
            channels_in=1,
            channels_out=max(kt, 1),
            oy=1,
            ox=1,
            fy=1,
            fx=max(fyfxkc, 1),
        )
        loop_dims["K"] = max(kt, 1)
        loop_dims["C"] = 1
        loop_dims["FY"] = 1
        loop_dims["FX"] = max(fyfxkc, 1)
        return _Fragment(
            stage=stage,
            phase="optimizer",
            op_name=type(self._optimizer).__name__.lower(),
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
        )

    def get_report(self) -> NeuroMCRuntimeEnergyReport:
        fragments = list(self._fragments)
        if not fragments:
            fragments = self._fallback_fragments_from_trace()

        unsupported = self._unsupported_ops()
        if unsupported:
            raise ValueError(
                "Exact NeuroMC runtime does not support these aten ops: "
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
                sum(direction_map.values()) for direction_map in extra_mem_energy.values()
            )
            total = mac_energy + extra_compute_energy + base_energy_total + extra_memory_total

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
                    "loop_dims": dict(fragment.loop_dims),
                    "b_type": fragment.b_type,
                    "t_type": fragment.t_type,
                    "mac_count": fragment.mac_count,
                    "source": fragment.source,
                }
            )

        primitive_totals = {
            primitive: int(
                sum(stage_counts.get(primitive, 0) for stage_counts in primitive_by_stage.values())
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
            "core_type": self.core_type,
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
            counts_by_process_key={k: dict(v) for k, v in counts_by_process_key.items()},
            mapping_summary=mapping_summary,
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


def estimate_neuromc_runtime_energy(
    model: nn.Module,
    inputs,
    *,
    target: torch.Tensor | None = None,
    loss_fn: Callable | None = None,
    optimizer: torch.optim.Optimizer | None = None,
    core_type: str = "fp_soma",
    memory_config: MemoryHierarchyConfig | None = None,
    strict: bool = False,
    verbose: bool = False,
    extra_ignore_modules: list[nn.Module] | None = None,
) -> NeuroMCRuntimeEnergyReport:
    profiler = NeuroMCEnergyProfiler(
        core_type=core_type,
        memory_config=(memory_config or MemoryHierarchyConfig.neuromc_like_v1()).copy(),
        strict=strict,
        verbose=verbose,
        extra_ignore_modules=extra_ignore_modules,
    )
    profiler.bind_model(model)
    profiler.bind_optimizer(optimizer)
    resolved_loss_fn = _resolve_loss_fn(loss_fn)
    _clear_existing_grads(model, optimizer)

    with profiler:
        with profiler.stage("forward"):
            output = _call_model(model, inputs)
        loss = None
        if resolved_loss_fn is not None:
            with profiler.suspend():
                if target is None:
                    raise ValueError("target is required when loss_fn is provided")
                loss = resolved_loss_fn(output, target)
        if loss is not None:
            with profiler.stage("backward"):
                loss.backward()
            if optimizer is not None:
                with profiler.stage("optimizer"):
                    opt_fragment = profiler._optimizer_fragment("optimizer")
                    profiler._fragments.append(opt_fragment)
                    with profiler.suspend():
                        optimizer.step()
                        optimizer.zero_grad(set_to_none=True)
        elif optimizer is not None:
            raise ValueError(
                "Exact NeuroMC optimizer modeling requires loss_fn and target; "
                "optimizer.step() without backward is unsupported."
            )
    return profiler.get_report()
