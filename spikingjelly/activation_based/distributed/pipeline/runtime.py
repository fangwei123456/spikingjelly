from dataclasses import dataclass
from typing import Any, Mapping, Optional, Sequence, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based.distributed.pipeline.partition import (
    resolve_pipeline_schedule_kind,
)

try:
    from torch.distributed._tensor import DTensor
except ImportError:
    DTensor = None

try:
    from torch.distributed.pipelining import (
        PipelineStage,
        Schedule1F1B,
        ScheduleGPipe,
        ScheduleInterleaved1F1B,
        ScheduleInterleavedZeroBubble,
    )

    PIPELINING_AVAILABLE = True
except ImportError:
    PipelineStage = None
    Schedule1F1B = None
    ScheduleGPipe = None
    ScheduleInterleaved1F1B = None
    ScheduleInterleavedZeroBubble = None
    PIPELINING_AVAILABLE = False


@dataclass
class SNNPipelineRuntime:
    r"""
    **API Language** - :ref:`中文 <SNNPipelineRuntime-cn>` | :ref:`English <SNNPipelineRuntime-en>`

    ----

    .. _SNNPipelineRuntime-cn:

    * **中文**

    SNN 流水线并行运行时。管理多 GPU 流水线调度与执行。

    ----

    .. _SNNPipelineRuntime-en:

    * **English**

    SNN pipeline parallel runtime.
    """

    schedule: Any
    stage_module: nn.Module
    stage_modules: Tuple[nn.Module, ...]
    local_stage_indices: Tuple[int, ...]
    stage_index: int
    num_stages: int
    device: torch.device
    n_microbatches: int
    model_family: str
    split_points: Tuple[str, ...]
    group: Optional[Any] = None
    stage_costs: Tuple[float, ...] = ()
    stage_input_example: Optional[Any] = None
    stage_input_examples: Tuple[Any, ...] = ()
    memopt_selected_stage_indices: Tuple[int, ...] = ()
    schedule_kind: str = "gpipe"
    virtual_pipeline_size: int = 1
    pp_layout: Optional[Tuple[int, ...]] = None
    delayed_wgrad: bool = False

    @property
    def is_first(self) -> bool:
        return bool(self.local_stage_indices) and self.local_stage_indices[0] == 0

    @property
    def is_last(self) -> bool:
        return (
            bool(self.local_stage_indices)
            and self.local_stage_indices[-1] == self.num_stages - 1
        )


def _collect_resettable_modules(module: nn.Module) -> Tuple[nn.Module, ...]:
    return tuple(child for child in module.modules() if hasattr(child, "reset"))


def _reset_collected_modules(modules: Sequence[nn.Module]):
    for module in modules:
        module.reset()


def _tensor_tree_numel(value: Any) -> int:
    if isinstance(value, torch.Tensor):
        return int(value.numel())
    if isinstance(value, (tuple, list)):
        return sum(_tensor_tree_numel(item) for item in value)
    if isinstance(value, Mapping):
        return sum(_tensor_tree_numel(item) for item in value.values())
    return 0


def _tensor_tree_sum(value: Any) -> Optional[torch.Tensor]:
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            return value.sum()
        return None
    if isinstance(value, (tuple, list)):
        accum = None
        for item in value:
            item_sum = _tensor_tree_sum(item)
            if item_sum is None:
                continue
            accum = item_sum if accum is None else accum + item_sum
        return accum
    if isinstance(value, Mapping):
        accum = None
        for item in value.values():
            item_sum = _tensor_tree_sum(item)
            if item_sum is None:
                continue
            accum = item_sum if accum is None else accum + item_sum
        return accum
    return None


def _make_pipeline_outputs_contiguous(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        # Zero-bubble pipeline backward currently calls in-place ``detach_`` on
        # cached stage outputs. Contiguous views still fail that path, so ensure
        # stage outputs become standalone tensors when necessary.
        if value._is_view():
            return value.clone(memory_format=torch.contiguous_format)
        return value.contiguous()
    if isinstance(value, tuple):
        return tuple(_make_pipeline_outputs_contiguous(item) for item in value)
    if isinstance(value, list):
        return [_make_pipeline_outputs_contiguous(item) for item in value]
    if isinstance(value, Mapping):
        return type(value)(
            (k, _make_pipeline_outputs_contiguous(v)) for k, v in value.items()
        )
    return value


def _infer_tensor_tree_device(value: Any) -> Optional[torch.device]:
    if isinstance(value, torch.Tensor):
        return value.device
    if isinstance(value, (tuple, list)):
        for item in value:
            device = _infer_tensor_tree_device(item)
            if device is not None:
                return device
    if isinstance(value, Mapping):
        for item in value.values():
            device = _infer_tensor_tree_device(item)
            if device is not None:
                return device
    return None


def _clone_tensor_tree_for_autograd(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        cloned = value.detach().clone()
        if cloned.is_floating_point() or cloned.is_complex():
            cloned.requires_grad_(True)
        return cloned
    if isinstance(value, tuple):
        return tuple(_clone_tensor_tree_for_autograd(item) for item in value)
    if isinstance(value, list):
        return [_clone_tensor_tree_for_autograd(item) for item in value]
    if isinstance(value, Mapping):
        return type(value)(
            (k, _clone_tensor_tree_for_autograd(v)) for k, v in value.items()
        )
    return value


def _measure_module_cost(module: nn.Module, input_value: Any) -> Tuple[Any, float]:
    import time

    reset_modules = _collect_resettable_modules(module)
    device = _infer_tensor_tree_device(input_value)
    with torch.no_grad():
        _reset_collected_modules(reset_modules)
        if device is not None and device.type == "cuda":
            torch.cuda.synchronize(device)
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            output_value = module(input_value)
            end_event.record()
            torch.cuda.synchronize(device)
            elapsed_ms = float(start_event.elapsed_time(end_event))
        else:
            start_time = time.perf_counter()
            output_value = module(input_value)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
        _reset_collected_modules(reset_modules)
    signal = _tensor_tree_numel(output_value)
    backward_ms = 0.0

    autograd_input = _clone_tensor_tree_for_autograd(input_value)
    _reset_collected_modules(reset_modules)
    module.zero_grad(set_to_none=True)
    if device is not None and device.type == "cuda":
        torch.cuda.synchronize(device)
        output_autograd = module(autograd_input)
        loss = _tensor_tree_sum(output_autograd)
        if loss is not None and loss.requires_grad:
            start_event = torch.cuda.Event(enable_timing=True)
            end_event = torch.cuda.Event(enable_timing=True)
            start_event.record()
            loss.backward()
            end_event.record()
            torch.cuda.synchronize(device)
            backward_ms = float(start_event.elapsed_time(end_event))
    else:
        output_autograd = module(autograd_input)
        loss = _tensor_tree_sum(output_autograd)
        if loss is not None and loss.requires_grad:
            start_time = time.perf_counter()
            loss.backward()
            backward_ms = (time.perf_counter() - start_time) * 1000.0
    module.zero_grad(set_to_none=True)
    _reset_collected_modules(reset_modules)

    param_numel = sum(p.numel() for p in module.parameters(recurse=True))
    activation_cost = signal / 1_000_000.0
    parameter_cost = param_numel / 10_000_000.0
    total_cost = max(elapsed_ms, 1e-6) + backward_ms + activation_cost + parameter_cost
    return output_value, total_cost


def _example_microbatch_args(
    example_input: torch.Tensor,
    n_microbatches: int,
) -> Tuple[torch.Tensor]:
    if n_microbatches <= 0:
        raise ValueError(f"n_microbatches must be positive, but got {n_microbatches}.")
    batch_size = example_input.shape[0]
    if batch_size <= 0:
        raise ValueError("example_input must contain at least one sample on dim 0.")
    if batch_size < n_microbatches:
        raise ValueError(
            f"example_input batch size ({batch_size}) must be >= n_microbatches ({n_microbatches})."
        )
    microbatch_size = max(1, batch_size // n_microbatches)
    return (example_input[:microbatch_size].contiguous(),)


def snn_sequence_cross_entropy(
    outputs: torch.Tensor, target: torch.Tensor
) -> torch.Tensor:
    if DTensor is not None and isinstance(outputs, DTensor):
        outputs = outputs.full_tensor()
    if outputs.ndim >= 3:
        outputs = outputs.mean(dim=0)
    if target.ndim > 1:
        target = target.argmax(dim=1)
    return F.cross_entropy(outputs, target)


class _PipelineSequentialModule(nn.Module):
    def __init__(self, stages: Sequence[nn.Module]):
        super().__init__()
        self.stages = nn.ModuleList(list(stages))

    def forward(self, x):
        for stage in self.stages:
            x = stage(x)
        return x


def _reset_module_states(module: nn.Module):
    _reset_collected_modules(_collect_resettable_modules(module))


class _MicrobatchResetStage(nn.Module):
    def __init__(self, inner: nn.Module):
        super().__init__()
        self.inner = inner
        self._reset_modules = _collect_resettable_modules(inner)

    def refresh_reset_modules(self):
        self._reset_modules = _collect_resettable_modules(self.inner)

    def forward(self, *args, **kwargs):
        _reset_collected_modules(self._reset_modules)
        return _make_pipeline_outputs_contiguous(self.inner(*args, **kwargs))


def _build_snn_pipeline_runtime(
    pipeline_module: nn.Module,
    example_input: torch.Tensor,
    device: torch.device,
    n_microbatches: int,
    stage_index: Optional[int],
    model_family: str,
    schedule_kind: str = "auto",
    virtual_pipeline_size: int = 1,
    delayed_wgrad: bool = False,
    pp_layout: Optional[Tuple[int, ...]] = None,
    group=None,
) -> SNNPipelineRuntime:
    if not PIPELINING_AVAILABLE:
        raise RuntimeError(
            "torch.distributed.pipelining is unavailable. Please use a PyTorch build with "
            "pipeline parallel support."
        )
    if not dist.is_initialized():
        raise RuntimeError(
            "Pipeline parallel requires torch.distributed to be initialized."
        )
    physical_num_stages = dist.get_world_size(group)
    if virtual_pipeline_size <= 0:
        raise ValueError(
            f"virtual_pipeline_size must be positive, but got {virtual_pipeline_size}."
        )
    schedule_kind = resolve_pipeline_schedule_kind(
        schedule_kind=schedule_kind,
        virtual_pipeline_size=virtual_pipeline_size,
        delayed_wgrad=delayed_wgrad,
    )
    num_stages = physical_num_stages * virtual_pipeline_size
    if n_microbatches < num_stages:
        raise ValueError(
            f"n_microbatches ({n_microbatches}) must be >= number of stages ({num_stages})."
        )
    if stage_index is None:
        stage_index = dist.get_rank(group)
    if not hasattr(pipeline_module, "stages"):
        raise TypeError("pipeline_module must expose a 'stages' attribute.")
    if stage_index < 0 or stage_index >= physical_num_stages:
        raise ValueError(
            f"stage_index={stage_index} is out of range for num_stages={physical_num_stages}."
        )
    if len(pipeline_module.stages) != num_stages:
        raise ValueError(
            f"pipeline_module exposes {len(pipeline_module.stages)} logical stages, but "
            f"{num_stages} are required for physical_num_stages={physical_num_stages} and "
            f"virtual_pipeline_size={virtual_pipeline_size}."
        )

    local_stage_indices = tuple(
        stage_index + physical_num_stages * offset
        for offset in range(virtual_pipeline_size)
    )
    stage_modules = tuple(
        _MicrobatchResetStage(pipeline_module.stages[logical_idx])
        for logical_idx in local_stage_indices
    )
    stage_module: nn.Module
    if len(stage_modules) == 1:
        stage_module = stage_modules[0]
    else:
        stage_module = nn.ModuleList(stage_modules)
    microbatch_input = _example_microbatch_args(example_input, n_microbatches)[0]
    stage_inputs: list[Any] = []
    stage_outputs: list[Any] = []
    pipeline_reset_modules = _collect_resettable_modules(pipeline_module)
    with torch.no_grad():
        current = microbatch_input
        _reset_collected_modules(pipeline_reset_modules)
        for stage_submodule in pipeline_module.stages:
            stage_inputs.append(current)
            current = stage_submodule(current)
            stage_outputs.append(current)
        _reset_collected_modules(pipeline_reset_modules)
    stages = [
        PipelineStage(
            stage_modules[local_idx],
            stage_index=logical_idx,
            num_stages=num_stages,
            device=device,
            input_args=stage_inputs[logical_idx],
            output_args=stage_outputs[logical_idx],
            group=group,
        )
        for local_idx, logical_idx in enumerate(local_stage_indices)
    ]
    if schedule_kind == "gpipe":
        schedule = ScheduleGPipe(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=snn_sequence_cross_entropy,
        )
    elif schedule_kind == "1f1b":
        schedule = Schedule1F1B(
            stages[0],
            n_microbatches=n_microbatches,
            loss_fn=snn_sequence_cross_entropy,
        )
    elif schedule_kind == "interleaved":
        schedule = ScheduleInterleaved1F1B(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=snn_sequence_cross_entropy,
        )
    elif schedule_kind == "zero_bubble":
        schedule = ScheduleInterleavedZeroBubble(
            stages,
            n_microbatches=n_microbatches,
            loss_fn=snn_sequence_cross_entropy,
        )
    else:
        raise ValueError(f"Unsupported pipeline schedule kind '{schedule_kind}'.")
    return SNNPipelineRuntime(
        schedule=schedule,
        stage_module=stage_module,
        stage_modules=stage_modules,
        local_stage_indices=local_stage_indices,
        stage_index=stage_index,
        num_stages=num_stages,
        device=device,
        n_microbatches=n_microbatches,
        model_family=model_family,
        split_points=tuple(f"stages.{idx}" for idx in range(1, num_stages)),
        group=group,
        stage_costs=tuple(
            float(cost) for cost in getattr(pipeline_module, "stage_costs", ())
        ),
        stage_input_example=stage_inputs[local_stage_indices[0]],
        stage_input_examples=tuple(stage_inputs[idx] for idx in local_stage_indices),
        memopt_selected_stage_indices=(),
        schedule_kind=schedule_kind,
        virtual_pipeline_size=virtual_pipeline_size,
        pp_layout=pp_layout,
        delayed_wgrad=delayed_wgrad,
    )
