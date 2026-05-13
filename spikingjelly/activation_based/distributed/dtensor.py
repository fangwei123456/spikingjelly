import os
import inspect
import copy
import time
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel

from spikingjelly.activation_based import base, layer

try:
    from torch.distributed._tensor import DeviceMesh, init_device_mesh

    try:
        from torch.distributed._tensor import DTensor
    except ImportError:
        DTensor = None

    DTENSOR_AVAILABLE = True
except ImportError:
    DeviceMesh = None
    init_device_mesh = None
    DTensor = None
    DTENSOR_AVAILABLE = False

try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        parallelize_module,
    )

    try:
        from torch.distributed.tensor.parallel import make_output_tensor
    except ImportError:
        make_output_tensor = None

    TENSOR_PARALLEL_AVAILABLE = True
except ImportError:
    ColwiseParallel = None
    ParallelStyle = object
    RowwiseParallel = None
    make_output_tensor = None
    parallelize_module = None
    TENSOR_PARALLEL_AVAILABLE = False

try:
    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    FSDP2_AVAILABLE = True
except ImportError:
    MixedPrecisionPolicy = None
    fully_shard = None
    FSDP2_AVAILABLE = False

try:
    from torch.distributed.optim import ZeroRedundancyOptimizer

    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = True
except ImportError:
    ZeroRedundancyOptimizer = None
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE = False

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


LinearLike = (nn.Linear, layer.Linear)
Conv1dLike = (nn.Conv1d,)
Conv2dLike = (nn.Conv2d, layer.Conv2d)
BatchNorm1dLike = (nn.BatchNorm1d,)
BatchNorm2dLike = (nn.BatchNorm2d, layer.BatchNorm2d)


class TensorShardMemoryModule(base.MemoryModule):
    def __init__(
        self,
        source: base.MemoryModule,
        shard_dim: int,
        logical_dim_size: Optional[int] = None,
        process_group=None,
    ):
        super().__init__()
        self.inner = copy.deepcopy(source)
        self.shard_dim = shard_dim
        self.logical_dim_size = logical_dim_size
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.expected_local_dim_size = None
        if logical_dim_size is not None:
            _require_even_shard(logical_dim_size, self.world_size, "logical_dim_size")
            start, end = _shard_range(logical_dim_size, self.rank, self.world_size)
            self.expected_local_dim_size = end - start
        self.step_mode = getattr(self.inner, "step_mode", "s")
        if hasattr(self.inner, "backend"):
            self.backend = self.inner.backend

    @property
    def supported_backends(self):
        supported = getattr(self.inner, "supported_backends", None)
        if supported is None:
            return ("torch",)
        return supported

    @property
    def store_v_seq(self):
        return getattr(self.inner, "store_v_seq", False)

    def reset(self):
        if hasattr(self.inner, "reset"):
            self.inner.reset()

    def extra_repr(self) -> str:
        return (
            f"shard_dim={self.shard_dim}, logical_dim_size={self.logical_dim_size}, "
            f"world_size={self.world_size}"
        )

    def forward(self, x: torch.Tensor):
        shard_dim = self.shard_dim if self.shard_dim >= 0 else x.dim() + self.shard_dim
        if shard_dim < 0 or shard_dim >= x.dim():
            raise ValueError(
                f"shard_dim={self.shard_dim} is invalid for input with shape {tuple(x.shape)}."
            )
        if (
            self.expected_local_dim_size is not None
            and x.shape[shard_dim] != self.expected_local_dim_size
        ):
            raise ValueError(
                f"Expected local shard size {self.expected_local_dim_size} on dim {shard_dim}, "
                f"but got input shape {tuple(x.shape)}."
            )
        return self.inner(x)


@dataclass
class SNNDistributedAnalysis:
    memory_module_names: Tuple[str, ...]
    tensor_parallel_candidate_names: Tuple[str, ...]
    unsupported_tensor_parallel_names: Tuple[str, ...]
    notes: Tuple[str, ...]


@dataclass
class SNNDistributedConfig:
    device_type: str = "cuda"
    mesh_shape: Optional[Tuple[int, ...]] = None
    device_mesh: Optional["DeviceMesh"] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    enable_data_parallel: bool = False
    enable_fsdp2: bool = False
    tensor_parallel_roots: Optional[Sequence[str]] = None
    tensor_parallel_plan: Optional[Mapping[str, Union[str, "ParallelStyle"]]] = None
    auto_tensor_parallel: bool = True
    experimental_conv_tensor_parallel: bool = False
    conv_tensor_parallel_roots: Optional[Sequence[str]] = None
    experimental_spikformer_tensor_parallel: bool = False
    spikformer_tensor_parallel_roots: Optional[Sequence[str]] = None
    experimental_spikformer_patch_stem_tensor_parallel: bool = False
    spikformer_patch_stem_tensor_parallel_roots: Optional[Sequence[str]] = None
    broadcast_buffers: bool = False
    find_unused_parameters: bool = False
    static_graph: bool = False
    fsdp_shard_roots: Optional[Sequence[str]] = None
    fsdp_shard_module_root: bool = True
    fsdp_root_reshard_after_forward: Optional[bool] = False
    fsdp_param_dtype: Optional[torch.dtype] = None
    fsdp_reduce_dtype: Optional[torch.dtype] = None
    fsdp_output_dtype: Optional[torch.dtype] = None


@dataclass
class SNNPipelineRuntime:
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


SNN_DISTRIBUTED_PREFERENCES = ("speed", "memory", "capacity")


@dataclass(frozen=True)
class SNNDistributedRecommendation:
    prefer: str
    model: str
    world_size: int
    mode: str
    optimizer_sharding: str = "none"
    memopt_level: int = 0
    mesh_shape: Optional[Tuple[int, ...]] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    pp_microbatches: Optional[int] = None
    pp_memopt_stage_budget_ratio: float = 0.5
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 1
    pp_layout: Optional[Tuple[int, ...]] = None
    pp_delay_wgrad: bool = False
    rationale: Tuple[str, ...] = ()


def ensure_distributed_initialized(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> bool:
    if dist.is_available() and dist.is_initialized():
        return False

    if not dist.is_available():
        raise RuntimeError(
            "torch.distributed is not available in the current PyTorch build."
        )

    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    kwargs = {}
    if init_method is not None:
        kwargs["init_method"] = init_method
    if rank is not None:
        kwargs["rank"] = rank
    if world_size is not None:
        kwargs["world_size"] = world_size
    if (
        backend == "nccl"
        and torch.cuda.is_available()
        and "device_id" in inspect.signature(dist.init_process_group).parameters
    ):
        kwargs["device_id"] = torch.device("cuda", torch.cuda.current_device())

    dist.init_process_group(backend=backend, **kwargs)
    return True


def build_device_mesh(
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    mesh_dim_names: Optional[Tuple[str, ...]] = None,
) -> "DeviceMesh":
    if not DTENSOR_AVAILABLE:
        raise RuntimeError(
            "DTensor DeviceMesh is unavailable. Please install a PyTorch build with "
            "torch.distributed._tensor support."
        )

    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed is not initialized. Call ensure_distributed_initialized() first."
        )

    if mesh_shape is None:
        mesh_shape = (dist.get_world_size(),)

    mesh_volume = 1
    for size in mesh_shape:
        mesh_volume *= size

    world_size = dist.get_world_size()
    if mesh_volume != world_size:
        raise ValueError(
            f"mesh_shape={mesh_shape} uses {mesh_volume} ranks, but world_size={world_size}."
        )

    return init_device_mesh(device_type, mesh_shape, mesh_dim_names=mesh_dim_names)


def _make_colwise_parallel(local_output: bool) -> "ParallelStyle":
    if not TENSOR_PARALLEL_AVAILABLE:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable.")
    signature = inspect.signature(ColwiseParallel)
    if "use_local_output" in signature.parameters:
        return ColwiseParallel(use_local_output=local_output)
    if local_output and make_output_tensor is not None:
        return ColwiseParallel(_prepare_output=make_output_tensor)
    return ColwiseParallel()


def _normalize_parallel_style(style: Union[str, "ParallelStyle"]) -> "ParallelStyle":
    if not TENSOR_PARALLEL_AVAILABLE:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable.")

    if not isinstance(style, str):
        return style

    lowered = style.lower()
    if lowered in ("colwise", "colwise_shard"):
        return _make_colwise_parallel(local_output=False)
    if lowered in ("colwise_local_output", "colwise_local"):
        return _make_colwise_parallel(local_output=True)
    if lowered == "rowwise":
        return RowwiseParallel()

    raise ValueError(
        f"Unsupported tensor parallel style '{style}'. "
        "Expected one of: colwise, colwise_local_output, rowwise."
    )


def _is_colwise_local_style(style: Union[str, "ParallelStyle"]) -> bool:
    if isinstance(style, str):
        return style.lower() in ("colwise_local_output", "colwise_local")
    if ColwiseParallel is not None and isinstance(style, ColwiseParallel):
        if hasattr(style, "use_local_output"):
            return bool(style.use_local_output)
        if (
            make_output_tensor is not None
            and getattr(style, "_prepare_output", None) is make_output_tensor
        ):
            return True
        return False
    return False


def _iter_named_modules_under_roots(
    module: nn.Module,
    roots: Optional[Sequence[str]] = None,
) -> Iterable[Tuple[str, nn.Module]]:
    if not roots:
        for name, child in module.named_modules():
            if name:
                yield name, child
        return

    named_children = dict(module.named_modules())
    for root in roots:
        if root not in named_children:
            raise KeyError(
                f"tensor_parallel_roots contains unknown module path '{root}'."
            )

        root_module = named_children[root]
        for sub_name, child in root_module.named_modules():
            full_name = root if not sub_name else f"{root}.{sub_name}"
            if full_name:
                yield full_name, child


def _replace_module_by_name(module: nn.Module, module_name: str, new_module: nn.Module):
    parent_name, _, child_name = module_name.rpartition(".")
    parent = module if not parent_name else dict(module.named_modules())[parent_name]
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def _even_partition_sizes(num_items: int, num_parts: int) -> List[int]:
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, but got {num_parts}.")
    base = num_items // num_parts
    rem = num_items % num_parts
    return [base + (1 if idx < rem else 0) for idx in range(num_parts)]


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


def _partition_costs_contiguously(costs: Sequence[float], num_parts: int) -> List[int]:
    if num_parts <= 0:
        raise ValueError(f"num_parts must be positive, but got {num_parts}.")
    num_items = len(costs)
    if num_items == 0:
        return [0 for _ in range(num_parts)]
    if num_items < num_parts:
        return _even_partition_sizes(num_items, num_parts)

    lo = max(float(cost) for cost in costs)
    hi = sum(float(cost) for cost in costs)

    def _fits(limit: float) -> bool:
        parts = 1
        acc = 0.0
        for cost in costs:
            cost = float(cost)
            if acc == 0.0 or acc + cost <= limit:
                acc += cost
            else:
                parts += 1
                acc = cost
        return parts <= num_parts

    for _ in range(48):
        mid = (lo + hi) * 0.5
        if _fits(mid):
            hi = mid
        else:
            lo = mid
    limit = hi

    sizes_reversed: List[int] = []
    acc = 0.0
    count = 0
    parts_remaining = num_parts
    for idx in range(num_items - 1, -1, -1):
        cost = float(costs[idx])
        remaining_items = idx + 1
        if count > 0 and (acc + cost > limit or remaining_items < parts_remaining):
            sizes_reversed.append(count)
            parts_remaining -= 1
            acc = 0.0
            count = 0
        acc += cost
        count += 1
    sizes_reversed.append(count)
    sizes = list(reversed(sizes_reversed))
    if len(sizes) < num_parts:
        sizes = [0] * (num_parts - len(sizes)) + sizes
    return sizes


def parse_pipeline_layout(
    layout: Optional[Union[str, Sequence[int]]],
    num_logical_stages: int,
    total_units: int,
) -> Optional[Tuple[int, ...]]:
    if layout is None:
        return None
    if isinstance(layout, str):
        raw_tokens = layout.replace(",", "|").split("|")
        counts = tuple(int(token.strip()) for token in raw_tokens if token.strip())
    else:
        counts = tuple(int(item) for item in layout)
    if len(counts) != num_logical_stages:
        raise ValueError(
            f"Pipeline layout must provide {num_logical_stages} stage counts, "
            f"but got {len(counts)} from {layout!r}."
        )
    if any(count < 0 for count in counts):
        raise ValueError(
            f"Pipeline layout counts must be non-negative, but got {counts}."
        )
    if sum(counts) != total_units:
        raise ValueError(
            f"Pipeline layout {counts} covers {sum(counts)} units, but the model requires "
            f"{total_units} units."
        )
    return counts


def resolve_pipeline_schedule_kind(
    schedule_kind: str,
    virtual_pipeline_size: int,
    delayed_wgrad: bool,
) -> str:
    normalized = schedule_kind.lower()
    if normalized not in ("auto", "gpipe", "1f1b", "interleaved", "zero_bubble"):
        raise ValueError(
            f"Unsupported pp schedule '{schedule_kind}'. "
            "Expected one of: auto, gpipe, 1f1b, interleaved, zero_bubble."
        )
    if normalized == "auto":
        if delayed_wgrad:
            normalized = "zero_bubble"
        elif virtual_pipeline_size > 1:
            normalized = "interleaved"
        else:
            normalized = "1f1b"
    if normalized == "gpipe" and delayed_wgrad:
        raise ValueError("pp_delay_wgrad is incompatible with pp_schedule='gpipe'.")
    if normalized in ("gpipe", "1f1b") and virtual_pipeline_size != 1:
        raise ValueError(
            f"pp_schedule='{normalized}' does not support pp_virtual_stages={virtual_pipeline_size}. "
            "Use pp_schedule='interleaved' or 'zero_bubble' when pp_virtual_stages > 1."
        )
    if normalized in ("interleaved", "zero_bubble") and virtual_pipeline_size < 2:
        raise ValueError(
            f"pp_schedule='{normalized}' requires pp_virtual_stages >= 2, "
            f"but got {virtual_pipeline_size}."
        )
    return normalized


def recommended_pipeline_microbatches(batch_size: int, num_stages: int) -> int:
    if batch_size <= 0:
        raise ValueError(f"batch_size must be positive, but got {batch_size}.")
    if num_stages <= 0:
        raise ValueError(f"num_stages must be positive, but got {num_stages}.")
    if batch_size < num_stages:
        raise ValueError(
            f"batch_size ({batch_size}) must be >= num_stages ({num_stages}) for pipeline "
            "parallelism with the current microbatch splitting implementation."
        )

    target = min(batch_size, num_stages * 4)
    for candidate in range(target, num_stages - 1, -1):
        if batch_size % candidate == 0:
            return candidate
    return num_stages


def _recommended_fsdp2_tp_mesh_shape(world_size: int) -> Optional[Tuple[int, int]]:
    if world_size < 4 or world_size % 2 != 0:
        return None
    return (world_size // 2, 2)


def recommend_snn_distributed_strategy(
    model: str,
    world_size: int,
    prefer: str,
    batch_size: int,
    backend: str = "inductor",
    zero_redundancy_optimizer_available: Optional[bool] = None,
    pipelining_available: Optional[bool] = None,
    fsdp2_available: Optional[bool] = None,
    tensor_parallel_available: Optional[bool] = None,
) -> SNNDistributedRecommendation:
    prefer = prefer.lower()
    if prefer not in SNN_DISTRIBUTED_PREFERENCES:
        raise ValueError(
            f"Unsupported prefer='{prefer}'. Expected one of {SNN_DISTRIBUTED_PREFERENCES}."
        )

    zero_available = (
        ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE
        if zero_redundancy_optimizer_available is None
        else zero_redundancy_optimizer_available
    )
    pipeline_available = (
        PIPELINING_AVAILABLE if pipelining_available is None else pipelining_available
    )
    fsdp_available = FSDP2_AVAILABLE if fsdp2_available is None else fsdp2_available
    tp_available = (
        TENSOR_PARALLEL_AVAILABLE
        if tensor_parallel_available is None
        else tensor_parallel_available
    )

    model_family = "spikformer" if model.startswith("spikformer") else model
    rationale: List[str] = [
        f"prefer='{prefer}' with model='{model_family}', world_size={world_size}, backend='{backend}'."
    ]

    if world_size <= 1:
        if prefer == "speed":
            rationale.append(
                "Single-rank run keeps the simplest local path with no distributed overhead."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="none",
                rationale=tuple(rationale),
            )
        rationale.append(
            "Single-rank run falls back to local training and uses memopt for memory savings."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="none",
            memopt_level=1,
            rationale=tuple(rationale),
        )

    if prefer == "speed":
        if model_family == "cifar10dvs_vgg" and fsdp_available and tp_available:
            mesh_shape = _recommended_fsdp2_tp_mesh_shape(world_size)
            if mesh_shape is not None:
                rationale.append(
                    "Use fsdp2_tp on multi-GPU CIFAR10DVSVGG because current inductor benchmarks show the best global throughput there."
                )
                return SNNDistributedRecommendation(
                    prefer=prefer,
                    model=model,
                    world_size=world_size,
                    mode="fsdp2_tp",
                    mesh_shape=mesh_shape,
                    tp_mesh_dim=1,
                    dp_mesh_dim=0,
                    rationale=tuple(rationale),
                )
        rationale.append(
            "Use data parallel training for the simplest throughput-oriented path, enabling ZeRO optimizer state sharding when available."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="dp",
            optimizer_sharding="zero" if zero_available else "none",
            dp_mesh_dim=0,
            rationale=tuple(rationale),
        )

    if prefer == "memory":
        mesh_shape = _recommended_fsdp2_tp_mesh_shape(world_size)
        if fsdp_available and tp_available and mesh_shape is not None:
            rationale.append(
                "Combine FSDP2 and TP to shard both parameters and activations, and enable memopt level 1 for the strongest memory reduction."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="fsdp2_tp",
                memopt_level=1,
                mesh_shape=mesh_shape,
                tp_mesh_dim=1,
                dp_mesh_dim=0,
                rationale=tuple(rationale),
            )
        if tp_available:
            rationale.append(
                "Use tensor parallel with memopt level 1 when two-dimensional FSDP2+TP is unavailable."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="tp",
                memopt_level=1,
                mesh_shape=(world_size,),
                rationale=tuple(rationale),
            )
        if fsdp_available:
            rationale.append(
                "Fall back to FSDP2 with memopt level 1 when TP is unavailable."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="fsdp2",
                memopt_level=1,
                dp_mesh_dim=0,
                rationale=tuple(rationale),
            )
        rationale.append(
            "Fall back to DP + memopt level 1 because TP/FSDP2 are unavailable."
        )
        return SNNDistributedRecommendation(
            prefer=prefer,
            model=model,
            world_size=world_size,
            mode="dp",
            optimizer_sharding="zero" if zero_available else "none",
            memopt_level=1,
            dp_mesh_dim=0,
            rationale=tuple(rationale),
        )

    if pipeline_available:
        if batch_size >= world_size * 2 and world_size >= 2:
            pp_virtual_stages = 2
        elif batch_size >= world_size:
            pp_virtual_stages = 1
        else:
            pp_virtual_stages = 0
        if pp_virtual_stages == 0:
            rationale.append(
                "Pipeline parallelism is skipped because the global batch is smaller than the number of physical stages."
            )
        else:
            logical_stages = world_size * pp_virtual_stages
            pp_schedule = "interleaved" if pp_virtual_stages > 1 else "1f1b"
            pp_delay_wgrad = False
            rationale.append(
                "Use pipeline parallelism with memopt level 1 when capacity is the priority; prefer the more stable interleaved schedule by default when multiple virtual stages are available."
            )
            return SNNDistributedRecommendation(
                prefer=prefer,
                model=model,
                world_size=world_size,
                mode="pp",
                memopt_level=1,
                pp_microbatches=recommended_pipeline_microbatches(
                    batch_size, logical_stages
                ),
                pp_memopt_stage_budget_ratio=0.5,
                pp_schedule=pp_schedule,
                pp_virtual_stages=pp_virtual_stages,
                pp_delay_wgrad=pp_delay_wgrad,
                rationale=tuple(rationale),
            )

    rationale.append(
        "Pipeline APIs are unavailable, so capacity preference falls back to the strongest memory-oriented strategy."
    )
    fallback = recommend_snn_distributed_strategy(
        model=model,
        world_size=world_size,
        prefer="memory",
        batch_size=batch_size,
        backend=backend,
        zero_redundancy_optimizer_available=zero_available,
        pipelining_available=False,
        fsdp2_available=fsdp_available,
        tensor_parallel_available=tp_available,
    )
    return SNNDistributedRecommendation(
        prefer=prefer,
        model=model,
        world_size=world_size,
        mode=fallback.mode,
        optimizer_sharding=fallback.optimizer_sharding,
        memopt_level=fallback.memopt_level,
        mesh_shape=fallback.mesh_shape,
        tp_mesh_dim=fallback.tp_mesh_dim,
        dp_mesh_dim=fallback.dp_mesh_dim,
        pp_microbatches=fallback.pp_microbatches,
        pp_memopt_stage_budget_ratio=fallback.pp_memopt_stage_budget_ratio,
        pp_schedule=fallback.pp_schedule,
        pp_virtual_stages=fallback.pp_virtual_stages,
        pp_layout=fallback.pp_layout,
        pp_delay_wgrad=fallback.pp_delay_wgrad,
        rationale=tuple(rationale + list(fallback.rationale)),
    )


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


class _CIFAR10DVSVGGPipelineStage(nn.Module):
    def __init__(
        self,
        feature_modules: Sequence[nn.Module],
        classifier: Optional[nn.Module] = None,
        transpose_input: bool = False,
    ):
        super().__init__()
        self.transpose_input = transpose_input
        self.features = nn.Sequential(*list(feature_modules))
        self.classifier = classifier

    def forward(self, x: torch.Tensor):
        if self.transpose_input:
            if x.ndim != 5:
                raise ValueError(
                    f"expected 5D input with shape [N, T, C, H, W], but got {tuple(x.shape)}"
                )
            x = x.transpose(0, 1).contiguous()
        x = self.features(x)
        if self.classifier is not None:
            x = torch.flatten(x, 2)
            x = self.classifier(x)
        return x


class _SpikformerPipelineStage(nn.Module):
    def __init__(
        self,
        *,
        patch_embed: Optional[nn.Module] = None,
        blocks: Sequence[nn.Module] = (),
        head: Optional[nn.Module] = None,
        T: Optional[int] = None,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(list(blocks))
        self.head = head
        self.T = T

    def forward(self, x: torch.Tensor):
        if self.patch_embed is not None:
            if x.ndim == 4:
                if self.T is None:
                    raise RuntimeError(
                        "Spikformer pipeline stage requires T for 4D inputs."
                    )
                x = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            elif x.ndim != 5:
                raise ValueError(
                    f"expected 4D [N, C, H, W] or 5D [T, N, C, H, W] input, but got {tuple(x.shape)}"
                )
            x = self.patch_embed(x)

        for block in self.blocks:
            x = block(x)

        if self.head is not None:
            x = x.flatten(3).mean(dim=-1)
            x = self.head(x)
        return x


def _build_cifar10dvs_vgg_pipeline_module(
    module: nn.Module,
    num_logical_stages: int,
    example_input: torch.Tensor,
    layout_counts: Optional[Sequence[int]] = None,
) -> _PipelineSequentialModule:
    if num_logical_stages < 2:
        raise ValueError("CIFAR10DVSVGG pipeline parallel requires at least 2 stages.")
    if not (hasattr(module, "features") and hasattr(module, "classifier")):
        raise TypeError(
            "Expected a CIFAR10DVSVGG-like module with features and classifier."
        )

    feature_modules = list(module.features.children())
    current = example_input.transpose(0, 1).contiguous()
    feature_costs: List[float] = []
    for feature_module in feature_modules:
        current, cost = _measure_module_cost(feature_module, current)
        feature_costs.append(cost)
    classifier_input = torch.flatten(current, 2)
    _, classifier_cost = _measure_module_cost(module.classifier, classifier_input)
    unit_costs = feature_costs + [classifier_cost]
    feature_counts = (
        list(layout_counts)
        if layout_counts is not None
        else _partition_costs_contiguously(unit_costs, num_logical_stages)
    )
    first_active_stage_idx = next(
        (idx for idx, count in enumerate(feature_counts) if count > 0), None
    )
    stages: List[nn.Module] = []
    cursor = 0
    total_feature_modules = len(feature_modules)
    classifier_assigned = False
    stage_costs: List[float] = []
    for stage_idx, count in enumerate(feature_counts):
        feature_end = min(cursor + count, total_feature_modules)
        stage_features = feature_modules[cursor:feature_end]
        classifier = None
        if not classifier_assigned and cursor + count > total_feature_modules:
            classifier = module.classifier
            classifier_assigned = True
        start_unit = cursor
        end_unit = cursor + count
        cursor = feature_end
        stages.append(
            _CIFAR10DVSVGGPipelineStage(
                feature_modules=stage_features,
                classifier=classifier,
                transpose_input=stage_idx == first_active_stage_idx,
            )
        )
        stage_costs.append(sum(float(cost) for cost in unit_costs[start_unit:end_unit]))
    pipeline_module = _PipelineSequentialModule(stages)
    pipeline_module.stage_costs = tuple(stage_costs)
    return pipeline_module


def _build_spikformer_pipeline_module(
    module: nn.Module,
    num_logical_stages: int,
    example_input: torch.Tensor,
    layout_counts: Optional[Sequence[int]] = None,
) -> _PipelineSequentialModule:
    if num_logical_stages < 2:
        raise ValueError("Spikformer pipeline parallel requires at least 2 stages.")
    if not (
        hasattr(module, "patch_embed")
        and hasattr(module, "blocks")
        and hasattr(module, "head")
    ):
        raise TypeError(
            "Expected a Spikformer-like module with patch_embed, blocks, and head."
        )

    blocks = list(module.blocks)
    current = example_input
    if current.ndim == 4:
        current = current.unsqueeze(0).repeat(getattr(module, "T", 1), 1, 1, 1, 1)
    unit_costs: List[float] = []
    current, patch_cost = _measure_module_cost(module.patch_embed, current)
    unit_costs.append(patch_cost)
    for block in blocks:
        current, block_cost = _measure_module_cost(block, current)
        unit_costs.append(block_cost)
    head_input = current.flatten(3).mean(dim=-1)
    _, head_cost = _measure_module_cost(module.head, head_input)
    unit_costs.append(head_cost)
    block_counts = (
        list(layout_counts)
        if layout_counts is not None
        else _partition_costs_contiguously(unit_costs, num_logical_stages)
    )
    first_active_stage_idx = next(
        (idx for idx, count in enumerate(block_counts) if count > 0), None
    )
    stages: List[nn.Module] = []
    cursor = 0
    unit_cursor = 0
    stage_costs: List[float] = []
    for stage_idx, count in enumerate(block_counts):
        patch_embed = (
            module.patch_embed
            if stage_idx == first_active_stage_idx and count > 0
            else None
        )
        units_remaining = count
        if patch_embed is not None:
            units_remaining -= 1
        block_take = min(max(units_remaining, 0), len(blocks) - cursor)
        stage_blocks = blocks[cursor : cursor + block_take]
        cursor += block_take
        units_remaining -= block_take
        head = module.head if units_remaining > 0 else None
        stages.append(
            _SpikformerPipelineStage(
                patch_embed=patch_embed,
                blocks=stage_blocks,
                head=head,
                T=getattr(module, "T", None),
            )
        )
        stage_costs.append(
            sum(float(cost) for cost in unit_costs[unit_cursor : unit_cursor + count])
        )
        unit_cursor += count
    pipeline_module = _PipelineSequentialModule(stages)
    pipeline_module.stage_costs = tuple(stage_costs)
    return pipeline_module


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
    stage_inputs: List[Any] = []
    stage_outputs: List[Any] = []
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


def recommend_pipeline_memopt_stages(
    stage_costs: Sequence[float],
    stage_budget_ratio: float = 0.5,
) -> Tuple[int, ...]:
    if not stage_costs:
        return ()
    if stage_budget_ratio <= 0.0 or stage_budget_ratio > 1.0:
        raise ValueError(
            f"stage_budget_ratio must be in (0, 1], but got {stage_budget_ratio}."
        )
    num_stages = len(stage_costs)
    target_count = max(1, min(num_stages, int(round(num_stages * stage_budget_ratio))))
    ranked = sorted(
        range(num_stages),
        key=lambda idx: (float(stage_costs[idx]), -idx),
        reverse=True,
    )
    selected = tuple(sorted(ranked[:target_count]))
    return selected


def apply_pipeline_stage_memopt(
    runtime: SNNPipelineRuntime,
    *,
    memopt_level: int,
    compress_x: bool = False,
    stage_budget_ratio: float = 0.5,
    use_plan_cache: bool = True,
) -> Tuple[SNNPipelineRuntime, float, bool]:
    if memopt_level <= 0:
        runtime.memopt_selected_stage_indices = ()
        return runtime, 0.0, False

    if runtime.model_family == "cifar10dvs_vgg":
        from spikingjelly.activation_based.examples.memopt.models import VGGBlock

        target_types = (VGGBlock,)
    elif runtime.model_family == "spikformer":
        from spikingjelly.activation_based.layer.attention import SpikingSelfAttention
        from spikingjelly.activation_based.model.spikformer import (
            SpikformerConv2dBNLIF,
            SpikformerMLP,
        )

        target_types = (SpikformerConv2dBNLIF, SpikingSelfAttention, SpikformerMLP)
    else:
        raise ValueError(
            f"Unsupported pipeline model_family='{runtime.model_family}' for memopt."
        )

    selected = recommend_pipeline_memopt_stages(
        runtime.stage_costs,
        stage_budget_ratio=stage_budget_ratio,
    )
    runtime.memopt_selected_stage_indices = selected
    local_selected_pairs = [
        (
            logical_idx,
            runtime.stage_modules[local_idx],
            runtime.stage_input_examples[local_idx],
        )
        for local_idx, logical_idx in enumerate(runtime.local_stage_indices)
        if logical_idx in selected
    ]
    if not local_selected_pairs:
        return runtime, 0.0, False

    from spikingjelly.activation_based.memopt import memory_optimization

    supports_plan_cache = (
        "use_plan_cache" in inspect.signature(memory_optimization).parameters
    )

    start = time.time()
    for logical_idx, stage_wrapper, stage_input_example in local_selected_pairs:
        if stage_input_example is None:
            raise RuntimeError(
                f"Pipeline memopt requires a stage_input_example for logical stage {logical_idx}."
            )
        optimize_kwargs = dict(
            dummy_input=(stage_input_example,),
            compress_x=compress_x,
            level=memopt_level,
            verbose=False,
        )
        if supports_plan_cache:
            optimize_kwargs["use_plan_cache"] = use_plan_cache
        optimized = memory_optimization(
            stage_wrapper.inner,
            target_types,
            **optimize_kwargs,
        )
        stage_wrapper.inner = optimized.to(runtime.device)
        stage_wrapper.refresh_reset_modules()
    return runtime, (time.time() - start) * 1000.0, True


def analyze_snn_distributed_capability(
    module: nn.Module,
    tensor_parallel_roots: Optional[Sequence[str]] = None,
) -> SNNDistributedAnalysis:
    memory_modules: List[str] = []
    tensor_parallel_candidates: List[str] = []
    unsupported_tp: List[str] = []
    notes: List[str] = []

    for name, child in module.named_modules():
        if not name:
            continue
        if isinstance(child, base.MemoryModule):
            memory_modules.append(name)

    for name, child in _iter_named_modules_under_roots(module, tensor_parallel_roots):
        if isinstance(child, LinearLike):
            tensor_parallel_candidates.append(name)
        elif isinstance(
            child,
            (nn.Conv1d, nn.Conv2d, nn.Conv3d, layer.Conv1d, layer.Conv2d, layer.Conv3d),
        ):
            unsupported_tp.append(name)

    if memory_modules:
        notes.append(
            "Stateful neuron modules remain local/replicated in this first DTensor-ready layer."
        )
    if unsupported_tp:
        notes.append(
            "Conv tensor parallel is not enabled in this first implementation; only Linear-like modules "
            "are auto-parallelized."
        )
    if not tensor_parallel_candidates:
        notes.append(
            "No Linear-like tensor-parallel candidates were found under the selected roots."
        )

    return SNNDistributedAnalysis(
        memory_module_names=tuple(memory_modules),
        tensor_parallel_candidate_names=tuple(tensor_parallel_candidates),
        unsupported_tensor_parallel_names=tuple(unsupported_tp),
        notes=tuple(notes),
    )


def auto_build_tensor_parallel_plan(
    module: nn.Module,
    tensor_parallel_roots: Optional[Sequence[str]] = None,
) -> Dict[str, "ParallelStyle"]:
    analysis = analyze_snn_distributed_capability(module, tensor_parallel_roots)
    candidate_names = list(analysis.tensor_parallel_candidate_names)

    if not candidate_names:
        raise ValueError("No Linear-like modules were found for tensor parallelism.")

    plan: Dict[str, ParallelStyle] = {}
    if len(candidate_names) == 1:
        plan[candidate_names[0]] = _make_colwise_parallel(local_output=False)
        return plan

    for name in candidate_names[:-1]:
        plan[name] = _make_colwise_parallel(local_output=True)
    plan[candidate_names[-1]] = RowwiseParallel()
    return plan


def wrap_tp_memory_modules(
    module: nn.Module,
    tensor_parallel_plan: Mapping[str, Union[str, "ParallelStyle"]],
    process_group,
):
    named_modules = dict(module.named_modules())
    wrapped: set[str] = set()
    for module_name, style in tensor_parallel_plan.items():
        if not _is_colwise_local_style(style):
            continue
        if module_name not in named_modules:
            continue
        source = named_modules[module_name]
        if isinstance(source, LinearLike):
            parent_name, _, child_name = module_name.rpartition(".")
            parent = module if not parent_name else named_modules[parent_name]
            if not isinstance(parent, (nn.Sequential, nn.ModuleList)):
                continue
            if not child_name.isdigit():
                continue
            child_index = int(child_name)
            next_index = child_index + 1
            if next_index >= len(parent):
                continue
            next_module = parent[next_index]
            next_name = (
                f"{parent_name}.{next_index}" if parent_name else str(next_index)
            )
            if next_name in wrapped:
                continue
            if isinstance(next_module, base.MemoryModule):
                parent[next_index] = TensorShardMemoryModule(
                    next_module,
                    shard_dim=-1,
                    logical_dim_size=source.out_features,
                    process_group=process_group,
                )
                wrapped.add(next_name)
    return module


def parallelize_snn_module(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    tensor_parallel_plan: Mapping[str, Union[str, "ParallelStyle"]],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    if not TENSOR_PARALLEL_AVAILABLE:
        raise RuntimeError(
            "torch.distributed.tensor.parallel is unavailable in the current PyTorch build."
        )

    normalized_plan = {
        module_name: _normalize_parallel_style(style)
        for module_name, style in tensor_parallel_plan.items()
    }
    signature = inspect.signature(parallelize_module)
    if "tp_mesh_dim" in signature.parameters:
        return parallelize_module(
            module=module,
            device_mesh=device_mesh,
            parallelize_plan=normalized_plan,
            tp_mesh_dim=tp_mesh_dim,
        )
    if getattr(device_mesh, "ndim", 1) > 1:
        if getattr(device_mesh, "mesh_dim_names", None):
            mesh_name = device_mesh.mesh_dim_names[tp_mesh_dim]
            device_mesh = device_mesh[mesh_name]
        else:
            raise ValueError(
                "This PyTorch version requires a 1D tensor-parallel mesh when parallelize_module "
                "does not accept tp_mesh_dim. Please build the mesh with mesh_dim_names."
            )
    return parallelize_module(
        module=module,
        device_mesh=device_mesh,
        parallelize_plan=normalized_plan,
    )


def prepare_snn_data_parallel(
    module: nn.Module,
    process_group=None,
    device_ids: Optional[Sequence[int]] = None,
    broadcast_buffers: bool = False,
    find_unused_parameters: bool = False,
    static_graph: bool = False,
) -> DistributedDataParallel:
    return DistributedDataParallel(
        module,
        device_ids=list(device_ids) if device_ids is not None else None,
        process_group=process_group,
        broadcast_buffers=broadcast_buffers,
        find_unused_parameters=find_unused_parameters,
        static_graph=static_graph,
    )


def unwrap_parallel_module(module: nn.Module) -> nn.Module:
    if isinstance(module, DistributedDataParallel):
        return module.module
    return module


def materialize_dtensor_output(output):
    if DTensor is not None and isinstance(output, DTensor):
        return output.full_tensor()
    full_tensor = getattr(output, "full_tensor", None)
    if callable(full_tensor):
        return full_tensor()
    return output


def _resolve_mesh_submesh(device_mesh: "DeviceMesh", mesh_dim: int) -> "DeviceMesh":
    if getattr(device_mesh, "ndim", 1) == 1:
        return device_mesh
    if getattr(device_mesh, "mesh_dim_names", None):
        return device_mesh[device_mesh.mesh_dim_names[mesh_dim]]
    raise ValueError(
        "A multi-dimensional DeviceMesh requires mesh_dim_names to derive a 1D submesh."
    )


def resolve_data_parallel_partition(
    device_mesh: Optional["DeviceMesh"],
    dp_mesh_dim: Optional[int],
    sharded_by_data_parallel: bool,
) -> Tuple[int, int]:
    if not sharded_by_data_parallel or device_mesh is None:
        return 1, 0

    mesh_tensor = getattr(device_mesh, "mesh", None)
    if mesh_tensor is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return world_size, rank

    mesh_shape = tuple(int(v) for v in mesh_tensor.shape)
    coordinate = (
        tuple(int(v) for v in device_mesh.get_coordinate())
        if hasattr(device_mesh, "get_coordinate")
        else None
    )
    if dp_mesh_dim is None:
        if len(mesh_shape) != 1:
            raise ValueError(
                "dp_mesh_dim must be specified for data-parallel sharding on a multi-dimensional mesh."
            )
        rank = (
            coordinate[0]
            if coordinate is not None
            else (dist.get_rank() if dist.is_initialized() else 0)
        )
        return mesh_shape[0], rank

    if dp_mesh_dim < 0 or dp_mesh_dim >= len(mesh_shape):
        raise ValueError(
            f"dp_mesh_dim={dp_mesh_dim} is out of range for a mesh with shape {mesh_shape}."
        )
    if coordinate is None:
        raise ValueError(
            "DeviceMesh does not expose coordinates for data partitioning."
        )
    return mesh_shape[dp_mesh_dim], coordinate[dp_mesh_dim]


def resolve_tensor_parallel_group_size(
    device_mesh: Optional["DeviceMesh"],
    tp_mesh_dim: int,
    tensor_parallel_enabled: bool,
) -> int:
    if not tensor_parallel_enabled or device_mesh is None:
        return 1
    mesh_tensor = getattr(device_mesh, "mesh", None)
    if mesh_tensor is None:
        return dist.get_world_size() if dist.is_initialized() else 1
    mesh_shape = tuple(int(v) for v in mesh_tensor.shape)
    if tp_mesh_dim < 0 or tp_mesh_dim >= len(mesh_shape):
        raise ValueError(
            f"tp_mesh_dim={tp_mesh_dim} is out of range for a mesh with shape {mesh_shape}."
        )
    return mesh_shape[tp_mesh_dim]


def build_snn_optimizer(
    module: nn.Module,
    mode: str,
    lr: float,
    weight_decay: float = 0.0,
    optimizer_sharding: str = "none",
    foreach: Optional[bool] = None,
    optimizer_cls=torch.optim.Adam,
    **optimizer_kwargs,
):
    if optimizer_sharding not in ("none", "zero"):
        raise ValueError(
            f"Unsupported optimizer_sharding='{optimizer_sharding}'. Expected 'none' or 'zero'."
        )

    if optimizer_sharding == "zero":
        if mode != "dp":
            raise ValueError(
                "optimizer_sharding='zero' is currently supported for pure 'dp' mode only."
            )
        if not dist.is_initialized():
            raise RuntimeError(
                "optimizer_sharding='zero' requires an initialized torch.distributed process group."
            )
        if not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE:
            raise RuntimeError(
                "torch.distributed.optim.ZeroRedundancyOptimizer is unavailable in the current PyTorch build."
            )
        return ZeroRedundancyOptimizer(
            module.parameters(),
            optimizer_class=optimizer_cls,
            lr=lr,
            weight_decay=weight_decay,
            foreach=foreach,
            **optimizer_kwargs,
        )

    return optimizer_cls(
        module.parameters(),
        lr=lr,
        weight_decay=weight_decay,
        foreach=foreach,
        **optimizer_kwargs,
    )


def _build_fsdp_mp_policy(config: SNNDistributedConfig):
    if MixedPrecisionPolicy is None:
        return None
    if (
        config.fsdp_param_dtype is None
        and config.fsdp_reduce_dtype is None
        and config.fsdp_output_dtype is None
    ):
        return None
    return MixedPrecisionPolicy(
        param_dtype=config.fsdp_param_dtype,
        reduce_dtype=config.fsdp_reduce_dtype,
        output_dtype=config.fsdp_output_dtype,
    )


def fully_shard_snn_module(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    shard_roots: Optional[Sequence[str]] = None,
    shard_module_root: bool = True,
    root_reshard_after_forward: Optional[bool] = False,
    mp_policy=None,
) -> nn.Module:
    if not FSDP2_AVAILABLE:
        raise RuntimeError(
            "FSDP2 fully_shard is unavailable in the current PyTorch build."
        )

    named_modules = dict(module.named_modules())
    shard_roots = list(shard_roots or [])
    for name in shard_roots:
        if name not in named_modules:
            raise KeyError(f"Unknown FSDP shard root '{name}'.")
        submodule = named_modules[name]
        if mp_policy is None:
            fully_shard(submodule, mesh=device_mesh)
        else:
            fully_shard(submodule, mesh=device_mesh, mp_policy=mp_policy)

    if shard_module_root:
        if mp_policy is None:
            fully_shard(
                module,
                mesh=device_mesh,
                reshard_after_forward=root_reshard_after_forward,
            )
        else:
            fully_shard(
                module,
                mesh=device_mesh,
                reshard_after_forward=root_reshard_after_forward,
                mp_policy=mp_policy,
            )
    return module


def apply_snn_fsdp2(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    dp_mesh_dim: Optional[int] = None,
    shard_roots: Optional[Sequence[str]] = None,
    shard_module_root: bool = True,
    root_reshard_after_forward: Optional[bool] = False,
    param_dtype: Optional[torch.dtype] = None,
    reduce_dtype: Optional[torch.dtype] = None,
    output_dtype: Optional[torch.dtype] = None,
) -> nn.Module:
    config = SNNDistributedConfig(
        enable_fsdp2=True,
        dp_mesh_dim=dp_mesh_dim,
        fsdp_shard_roots=shard_roots,
        fsdp_shard_module_root=shard_module_root,
        fsdp_root_reshard_after_forward=root_reshard_after_forward,
        fsdp_param_dtype=param_dtype,
        fsdp_reduce_dtype=reduce_dtype,
        fsdp_output_dtype=output_dtype,
    )
    fsdp_mesh_dim = dp_mesh_dim if dp_mesh_dim is not None else 0
    fsdp_mesh = _resolve_mesh_submesh(device_mesh, fsdp_mesh_dim)
    mp_policy = _build_fsdp_mp_policy(config)
    return fully_shard_snn_module(
        module=module,
        device_mesh=fsdp_mesh,
        shard_roots=shard_roots,
        shard_module_root=shard_module_root,
        root_reshard_after_forward=root_reshard_after_forward,
        mp_policy=mp_policy,
    )


def _resolve_dp_group_from_mesh(device_mesh: "DeviceMesh", dp_mesh_dim: Optional[int]):
    if dp_mesh_dim is None:
        return None

    if hasattr(device_mesh, "get_dim_groups"):
        dim_groups = device_mesh.get_dim_groups()
    elif hasattr(device_mesh, "get_all_groups"):
        dim_groups = device_mesh.get_all_groups()
    else:
        raise AttributeError(
            "DeviceMesh does not expose get_dim_groups() or get_all_groups()."
        )

    if dp_mesh_dim < 0 or dp_mesh_dim >= len(dim_groups):
        raise ValueError(
            f"dp_mesh_dim={dp_mesh_dim} is out of range for a mesh with {len(dim_groups)} dimensions."
        )
    return dim_groups[dp_mesh_dim]


def _resolve_mesh_dim_group(device_mesh: "DeviceMesh", mesh_dim: int):
    if hasattr(device_mesh, "get_dim_groups"):
        dim_groups = device_mesh.get_dim_groups()
    elif hasattr(device_mesh, "get_all_groups"):
        dim_groups = device_mesh.get_all_groups()
    else:
        raise AttributeError(
            "DeviceMesh does not expose get_dim_groups() or get_all_groups()."
        )

    if mesh_dim < 0 or mesh_dim >= len(dim_groups):
        raise ValueError(
            f"mesh_dim={mesh_dim} is out of range for a mesh with {len(dim_groups)} dimensions."
        )
    return dim_groups[mesh_dim]


def _require_even_shard(total: int, world_size: int, name: str):
    if total % world_size != 0:
        raise ValueError(
            f"{name}={total} must be divisible by tensor-parallel world_size={world_size}."
        )


def _shard_range(total: int, rank: int, world_size: int) -> Tuple[int, int]:
    shard = total // world_size
    start = shard * rank
    return start, start + shard


class ChannelShardConv2d(nn.Module):
    def __init__(self, source: nn.Module, process_group, mode: str):
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError("ChannelShardConv2d only supports groups=1.")

        self.mode = mode
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode

        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size

        if mode == "colwise":
            _require_even_shard(source.out_channels, self.world_size, "out_channels")
            start, end = _shard_range(source.out_channels, self.rank, self.world_size)
            weight = source.weight.detach()[start:end].clone()
            bias = (
                source.bias.detach()[start:end].clone()
                if source.bias is not None
                else None
            )
            self.local_out_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        elif mode == "rowwise":
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, self.rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.local_in_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        else:
            raise ValueError(f"Unsupported ChannelShardConv2d mode '{mode}'.")

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, step_mode={self.step_mode}, "
            f"in_channels={self.in_channels}, out_channels={self.out_channels}"
        )

    def _conv2d(self, x: torch.Tensor) -> torch.Tensor:
        if self.mode == "colwise":
            return F.conv2d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        y = F.conv2d(
            x,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.world_size > 1:
            dist.all_reduce(y, group=self.process_group)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1, 1)
        return y

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._conv2d(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got {x.shape}!"
            )

        y_shape = [x.shape[0], x.shape[1]]
        y = self._conv2d(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ChannelShardConv1d(nn.Module):
    def __init__(self, source: nn.Module, process_group, mode: str):
        super().__init__()
        if source.groups != 1:
            raise NotImplementedError("ChannelShardConv1d only supports groups=1.")

        self.mode = mode
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.stride = source.stride
        self.padding = source.padding
        self.dilation = source.dilation
        self.groups = source.groups
        self.padding_mode = source.padding_mode
        self.in_channels = source.in_channels
        self.out_channels = source.out_channels
        self.kernel_size = source.kernel_size

        if mode == "colwise":
            _require_even_shard(source.out_channels, self.world_size, "out_channels")
            start, end = _shard_range(source.out_channels, self.rank, self.world_size)
            weight = source.weight.detach()[start:end].clone()
            bias = (
                source.bias.detach()[start:end].clone()
                if source.bias is not None
                else None
            )
            self.local_out_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        elif mode == "rowwise":
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, self.rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.local_in_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter(
                "bias", nn.Parameter(bias) if bias is not None else None
            )
        else:
            raise ValueError(f"Unsupported ChannelShardConv1d mode '{mode}'.")

    def extra_repr(self) -> str:
        return (
            f"mode={self.mode}, in_channels={self.in_channels}, "
            f"out_channels={self.out_channels}"
        )

    def forward(self, x: torch.Tensor):
        if self.mode == "colwise":
            return F.conv1d(
                x,
                self.weight,
                self.bias,
                self.stride,
                self.padding,
                self.dilation,
                self.groups,
            )

        y = F.conv1d(
            x,
            self.weight,
            None,
            self.stride,
            self.padding,
            self.dilation,
            self.groups,
        )
        if self.world_size > 1:
            dist.all_reduce(y, group=self.process_group)
        if self.bias is not None:
            y = y + self.bias.view(1, -1, 1)
        return y


class ChannelShardBatchNorm2d(nn.Module):
    def __init__(self, source: nn.Module, process_group):
        super().__init__()
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.step_mode = getattr(source, "step_mode", "s")
        self.eps = source.eps
        self.momentum = source.momentum
        self.affine = source.affine
        self.track_running_stats = source.track_running_stats
        self.num_features = source.num_features

        _require_even_shard(source.num_features, self.world_size, "num_features")
        start, end = _shard_range(source.num_features, self.rank, self.world_size)

        if self.affine:
            self.register_parameter(
                "weight", nn.Parameter(source.weight.detach()[start:end].clone())
            )
            self.register_parameter(
                "bias", nn.Parameter(source.bias.detach()[start:end].clone())
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", source.running_mean.detach()[start:end].clone()
            )
            self.register_buffer(
                "running_var", source.running_var.detach()[start:end].clone()
            )
            num_batches_tracked = getattr(source, "num_batches_tracked", None)
            if num_batches_tracked is not None:
                self.register_buffer(
                    "num_batches_tracked", num_batches_tracked.detach().clone()
                )
            else:
                self.num_batches_tracked = None
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.num_batches_tracked = None

        self.training = source.training

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}, num_features={self.num_features}"

    def _batch_norm(self, x: torch.Tensor) -> torch.Tensor:
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked.add_(1)
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._batch_norm(x)

        if self.step_mode != "m":
            raise ValueError(f"Unsupported step_mode='{self.step_mode}'.")
        if x.dim() != 5:
            raise ValueError(
                f"expected x with shape [T, N, C, H, W], but got {x.shape}!"
            )
        y_shape = [x.shape[0], x.shape[1]]
        y = self._batch_norm(x.flatten(0, 1))
        y_shape.extend(y.shape[1:])
        return y.view(y_shape)


class ChannelShardBatchNorm1d(nn.Module):
    def __init__(self, source: nn.Module, process_group):
        super().__init__()
        self.process_group = process_group
        self.rank = dist.get_rank(process_group) if process_group is not None else 0
        self.world_size = (
            dist.get_world_size(process_group) if process_group is not None else 1
        )
        self.eps = source.eps
        self.momentum = source.momentum
        self.affine = source.affine
        self.track_running_stats = source.track_running_stats
        self.num_features = source.num_features

        _require_even_shard(source.num_features, self.world_size, "num_features")
        start, end = _shard_range(source.num_features, self.rank, self.world_size)

        if self.affine:
            self.register_parameter(
                "weight", nn.Parameter(source.weight.detach()[start:end].clone())
            )
            self.register_parameter(
                "bias", nn.Parameter(source.bias.detach()[start:end].clone())
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer(
                "running_mean", source.running_mean.detach()[start:end].clone()
            )
            self.register_buffer(
                "running_var", source.running_var.detach()[start:end].clone()
            )
            num_batches_tracked = getattr(source, "num_batches_tracked", None)
            if num_batches_tracked is not None:
                self.register_buffer(
                    "num_batches_tracked", num_batches_tracked.detach().clone()
                )
            else:
                self.num_batches_tracked = None
        else:
            self.register_buffer("running_mean", None)
            self.register_buffer("running_var", None)
            self.num_batches_tracked = None

        self.training = source.training

    def extra_repr(self) -> str:
        return f"num_features={self.num_features}"

    def forward(self, x: torch.Tensor):
        if (
            self.training
            and self.track_running_stats
            and self.num_batches_tracked is not None
        ):
            self.num_batches_tracked.add_(1)
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            self.training or not self.track_running_stats,
            self.momentum,
            self.eps,
        )


def _try_convert_vgg_like_block(
    block: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not (hasattr(block, "proj_bn") and hasattr(block, "neuron")):
        return None
    if not isinstance(block.proj_bn, nn.Sequential):
        return None

    modules = list(block.proj_bn.children())
    if len(modules) < 2:
        return None
    conv = modules[-2]
    bn = modules[-1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None

    converted = []
    converted.extend(modules[:-2])
    converted.append(ChannelShardConv2d(conv, process_group, mode=mode))
    if mode == "colwise":
        converted.append(ChannelShardBatchNorm2d(bn, process_group))
        if isinstance(block.neuron, base.MemoryModule):
            block.neuron = TensorShardMemoryModule(
                block.neuron,
                shard_dim=2,
                logical_dim_size=conv.out_channels,
                process_group=process_group,
            )
    else:
        converted.append(bn)
    block.proj_bn = nn.Sequential(*converted)
    return block


def _replace_child_module(parent: nn.Module, child_name: str, new_child: nn.Module):
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_child
    else:
        setattr(parent, child_name, new_child)


def _overwrite_sequential_children(target: nn.Module, source: nn.Module):
    target_children = list(target.named_children())
    source_children = list(source.named_children())
    if len(target_children) != len(source_children):
        raise ValueError(
            f"Cannot overwrite {type(target)} with {type(source)} because child counts differ: "
            f"{len(target_children)} vs {len(source_children)}."
        )
    for (target_name, _), (_, source_child) in zip(target_children, source_children):
        _replace_child_module(target, target_name, source_child)


def _wrap_tensor_shard_memory_module(
    module: nn.Module,
    process_group,
    shard_dim: int,
    logical_dim_size: Optional[int] = None,
) -> Optional[nn.Module]:
    if isinstance(module, TensorShardMemoryModule):
        return module
    if isinstance(module, base.MemoryModule):
        return TensorShardMemoryModule(
            module,
            shard_dim=shard_dim,
            logical_dim_size=logical_dim_size,
            process_group=process_group,
        )
    return None


def _convert_trailing_conv2d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, nn.Sequential):
        return None
    modules = list(container.children())
    if len(modules) < 2:
        return None
    conv = modules[-2]
    bn = modules[-1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None
    converted = list(modules[:-2])
    converted.append(ChannelShardConv2d(conv, process_group, mode=mode))
    converted.append(
        ChannelShardBatchNorm2d(bn, process_group) if mode == "colwise" else bn
    )
    return nn.Sequential(*converted)


def _convert_leading_conv2d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if len(modules) < 2:
        return None
    conv = modules[0]
    bn = modules[1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None
    converted = [ChannelShardConv2d(conv, process_group, mode=mode)]
    converted.append(
        ChannelShardBatchNorm2d(bn, process_group) if mode == "colwise" else bn
    )
    converted.extend(modules[2:])
    return layer.SeqToANNContainer(*converted)


def _convert_vgg_like_tree(
    module: nn.Module, process_group, mode: str, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {"projection_converted": False, "memory_wrapped": mode != "colwise"}

    converted = _try_convert_vgg_like_block(module, process_group, mode)
    if converted is not None:
        state["projection_converted"] = True
        state["memory_wrapped"] = True
        return True

    if not state["projection_converted"]:
        converted_container = _convert_trailing_conv2d_bn(module, process_group, mode)
        if converted_container is not None:
            if converted_container is not module:
                _overwrite_sequential_children(module, converted_container)
            state["projection_converted"] = True
            return True

    if state["projection_converted"] and not state["memory_wrapped"]:
        wrapped = _wrap_tensor_shard_memory_module(module, process_group, shard_dim=2)
        if wrapped is not None and wrapped is not module:
            state["memory_wrapped"] = True
            return True

    changed = False
    for child_name, child in list(module.named_children()):
        replacement = child
        if state["projection_converted"] and not state["memory_wrapped"]:
            wrapped = _wrap_tensor_shard_memory_module(
                child, process_group, shard_dim=2
            )
            if wrapped is not None and wrapped is not child:
                replacement = wrapped
                state["memory_wrapped"] = True
                changed = True
        if _convert_vgg_like_tree(replacement, process_group, mode, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def _convert_seq_to_ann_conv1d_bn(
    container: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not isinstance(container, layer.SeqToANNContainer):
        return None
    modules = list(container.children())
    if len(modules) != 2:
        return None
    conv, bn = modules
    if not isinstance(conv, Conv1dLike) or not isinstance(bn, BatchNorm1dLike):
        return None
    if mode == "colwise":
        return layer.SeqToANNContainer(
            ChannelShardConv1d(conv, process_group, mode=mode),
            ChannelShardBatchNorm1d(bn, process_group),
        )
    return layer.SeqToANNContainer(
        ChannelShardConv1d(conv, process_group, mode=mode),
        bn,
    )


def _try_convert_spiking_self_attention(
    attn: nn.Module, process_group
) -> Optional[nn.Module]:
    if not hasattr(attn, "qkv_conv_bn"):
        return None

    converted = _convert_seq_to_ann_conv1d_bn(
        attn.qkv_conv_bn, process_group, mode="colwise"
    )
    if converted is not None:
        attn.qkv_conv_bn = converted
        if isinstance(attn.qkv_lif, base.MemoryModule):
            attn.qkv_lif = TensorShardMemoryModule(
                attn.qkv_lif,
                shard_dim=2,
                logical_dim_size=getattr(attn, "dim", None) * 3
                if getattr(attn, "dim", None) is not None
                else None,
                process_group=process_group,
            )
        if isinstance(attn.attn_lif, base.MemoryModule):
            attn.attn_lif = TensorShardMemoryModule(
                attn.attn_lif,
                shard_dim=2,
                logical_dim_size=None,
                process_group=process_group,
            )
    converted = _convert_seq_to_ann_conv1d_bn(
        attn.proj_conv_bn, process_group, mode="rowwise"
    )
    if converted is not None:
        attn.proj_conv_bn = converted
    return attn


def _try_convert_spikformer_mlp(mlp: nn.Module, process_group) -> Optional[nn.Module]:
    if not (hasattr(mlp, "fc1") or hasattr(mlp, "fc2")):
        return None
    if hasattr(mlp, "fc1"):
        converted = _convert_seq_to_ann_conv1d_bn(
            mlp.fc1, process_group, mode="colwise"
        )
        if converted is not None:
            mlp.fc1 = converted
            if isinstance(mlp.neuron1, base.MemoryModule):
                logical_dim = None
                conv = list(mlp.fc1.children())[0]
                if hasattr(conv, "out_channels"):
                    logical_dim = (
                        conv.out_channels
                        if not hasattr(conv, "local_out_channels")
                        else conv.out_channels
                    )
                mlp.neuron1 = TensorShardMemoryModule(
                    mlp.neuron1,
                    shard_dim=2,
                    logical_dim_size=logical_dim,
                    process_group=process_group,
                )
    if hasattr(mlp, "fc2"):
        converted = _convert_seq_to_ann_conv1d_bn(
            mlp.fc2, process_group, mode="rowwise"
        )
        if converted is not None:
            mlp.fc2 = converted
    return mlp


def _convert_spiking_self_attention_tree(module: nn.Module, process_group) -> bool:
    converted = _try_convert_spiking_self_attention(module, process_group)
    if converted is not None:
        return True
    changed = False
    for child_name, child in list(module.named_children()):
        if _convert_spiking_self_attention_tree(child, process_group):
            changed = True
            _replace_child_module(module, child_name, child)
    return changed


def _convert_spikformer_mlp_tree(
    module: nn.Module, process_group, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {
            "fc1_converted": False,
            "neuron1_wrapped": False,
            "fc2_converted": False,
        }
    converted = _try_convert_spikformer_mlp(module, process_group)
    if converted is not None:
        state["fc1_converted"] = True
        state["neuron1_wrapped"] = True
        state["fc2_converted"] = True
        return True

    if not state["fc1_converted"]:
        converted_fc1 = _convert_seq_to_ann_conv1d_bn(
            module, process_group, mode="colwise"
        )
        if converted_fc1 is not None:
            if converted_fc1 is not module and isinstance(
                module, layer.SeqToANNContainer
            ):
                _overwrite_sequential_children(module, converted_fc1)
            state["fc1_converted"] = True
            return True

    if state["fc1_converted"] and not state["neuron1_wrapped"]:
        wrapped = _wrap_tensor_shard_memory_module(module, process_group, shard_dim=2)
        if wrapped is not None and wrapped is not module:
            state["neuron1_wrapped"] = True
            return True

    if state["neuron1_wrapped"] and not state["fc2_converted"]:
        converted_fc2 = _convert_seq_to_ann_conv1d_bn(
            module, process_group, mode="rowwise"
        )
        if converted_fc2 is not None:
            if converted_fc2 is not module and isinstance(
                module, layer.SeqToANNContainer
            ):
                _overwrite_sequential_children(module, converted_fc2)
            state["fc2_converted"] = True
            return True

    changed = False
    for child_name, child in list(module.named_children()):
        replacement = child
        if state["fc1_converted"] and not state["neuron1_wrapped"]:
            wrapped = _wrap_tensor_shard_memory_module(
                child, process_group, shard_dim=2
            )
            if wrapped is not None and wrapped is not child:
                replacement = wrapped
                state["neuron1_wrapped"] = True
                changed = True
        if _convert_spikformer_mlp_tree(replacement, process_group, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def _try_convert_spikformer_block(
    block: nn.Module, process_group
) -> Optional[nn.Module]:
    if not (hasattr(block, "attn") and hasattr(block, "mlp")):
        return None

    _convert_spiking_self_attention_tree(block.attn, process_group=process_group)
    _convert_spikformer_mlp_tree(block.mlp, process_group=process_group)
    return block


def _try_convert_spikformer_stem_block(
    block: nn.Module, process_group, mode: str
) -> Optional[nn.Module]:
    if not (hasattr(block, "conv_bn") and hasattr(block, "neuron")):
        return None
    conv_bn = getattr(block, "conv_bn")
    if not hasattr(conv_bn, "block"):
        return None
    modules = list(conv_bn.block.children())
    if len(modules) < 2:
        return None
    conv = modules[0]
    bn = modules[1]
    if not isinstance(conv, Conv2dLike) or not isinstance(bn, BatchNorm2dLike):
        return None

    converted = [ChannelShardConv2d(conv, process_group, mode=mode)]
    if mode == "colwise":
        converted.append(ChannelShardBatchNorm2d(bn, process_group))
        if isinstance(block.neuron, base.MemoryModule):
            block.neuron = TensorShardMemoryModule(
                block.neuron,
                shard_dim=2,
                logical_dim_size=conv.out_channels,
                process_group=process_group,
            )
    else:
        converted.append(bn)
    converted.extend(modules[2:])
    conv_bn.block = layer.SeqToANNContainer(*converted)
    return block


def _convert_spikformer_stem_tree(
    module: nn.Module, process_group, mode: str, state: Optional[dict] = None
) -> bool:
    if state is None:
        state = {"projection_converted": False, "memory_wrapped": mode != "colwise"}

    converted = _try_convert_spikformer_stem_block(module, process_group, mode)
    if converted is not None:
        state["projection_converted"] = True
        state["memory_wrapped"] = True
        return True

    if not state["projection_converted"]:
        if hasattr(module, "block"):
            converted_container = _convert_leading_conv2d_bn(
                module.block, process_group, mode
            )
            if converted_container is not None:
                module.block = converted_container
                state["projection_converted"] = True
                return True
        converted_container = _convert_leading_conv2d_bn(module, process_group, mode)
        if converted_container is not None:
            return True

    if state["projection_converted"] and not state["memory_wrapped"]:
        wrapped = _wrap_tensor_shard_memory_module(module, process_group, shard_dim=2)
        if wrapped is not None and wrapped is not module:
            state["memory_wrapped"] = True
            return True

    changed = False
    for child_name, child in list(module.named_children()):
        replacement = child
        if state["projection_converted"] and not state["memory_wrapped"]:
            wrapped = _wrap_tensor_shard_memory_module(
                child, process_group, shard_dim=2
            )
            if wrapped is not None and wrapped is not child:
                replacement = wrapped
                state["memory_wrapped"] = True
                changed = True
        if _convert_spikformer_stem_tree(replacement, process_group, mode, state):
            changed = True
        if replacement is not child:
            _replace_child_module(module, child_name, replacement)
    return changed


def parallelize_snn_conv_blocks(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown conv tensor parallel root '{root}'.")
        root_module = named_modules[root]
        if not isinstance(root_module, nn.Sequential):
            raise TypeError(
                f"Conv tensor parallel root '{root}' must be an nn.Sequential, but got {type(root_module)}."
            )

        block_index = 0
        for child_name, child in list(root_module.named_children()):
            mode = "colwise" if block_index % 2 == 0 else "rowwise"
            replacement = child
            changed = _convert_vgg_like_tree(
                replacement, process_group=process_group, mode=mode
            )
            if changed:
                if replacement is not child:
                    root_module[int(child_name)] = replacement
                block_index += 1

    return module


def parallelize_spikformer_blocks(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown Spikformer tensor parallel root '{root}'.")
        root_module = named_modules[root]
        if not isinstance(root_module, (nn.Sequential, nn.ModuleList)):
            raise TypeError(
                f"Spikformer tensor parallel root '{root}' must be an nn.Sequential or nn.ModuleList, "
                f"but got {type(root_module)}."
            )

        for child_name, child in list(root_module.named_children()):
            replacement = child
            changed = False
            changed = (
                _convert_spiking_self_attention_tree(
                    replacement, process_group=process_group
                )
                or changed
            )
            changed = (
                _convert_spikformer_mlp_tree(replacement, process_group=process_group)
                or changed
            )
            if changed and replacement is not child:
                root_module[int(child_name)] = replacement

    return module


def parallelize_spikformer_patch_stem(
    module: nn.Module,
    device_mesh: "DeviceMesh",
    roots: Sequence[str],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    process_group = _resolve_mesh_dim_group(device_mesh, tp_mesh_dim)
    named_modules = dict(module.named_modules())

    for root in roots:
        if root not in named_modules:
            raise KeyError(f"Unknown Spikformer patch stem root '{root}'.")
        root_module = named_modules[root]
        if hasattr(root_module, "stages") and hasattr(
            root_module, "positional_encoding"
        ):
            block_index = 0
            stage_sequence = getattr(root_module, "stages")
            for child_name, child in list(stage_sequence.named_children()):
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement,
                    process_group=process_group,
                    mode=mode,
                )
                if changed:
                    if replacement is not child:
                        stage_sequence[int(child_name)] = replacement
                    block_index += 1
            continue
        if isinstance(root_module, (nn.Sequential, nn.ModuleList)):
            block_index = 0
            for child_name, child in list(root_module.named_children()):
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement, process_group=process_group, mode=mode
                )
                if changed:
                    if replacement is not child:
                        root_module[int(child_name)] = replacement
                    block_index += 1
            continue

        parent_name, _, child_name = root.rpartition(".")
        parent_module = named_modules.get(parent_name) if parent_name else None
        if (
            isinstance(parent_module, (nn.Sequential, nn.ModuleList))
            and child_name.isdigit()
        ):
            child_index = int(child_name)
            child_items = list(parent_module.named_children())
            remaining = len(child_items) - child_index
            convertible_count = remaining if remaining % 2 == 0 else remaining - 1
            if convertible_count < 2:
                raise ValueError(
                    "An isolated Spikformer patch-stem root must include at least two consecutive stem "
                    "blocks so TP can restore full channels before rejoining unsharded modules."
                )
            block_index = 0
            for local_offset in range(convertible_count):
                current_name, child = child_items[child_index + local_offset]
                mode = "colwise" if block_index % 2 == 0 else "rowwise"
                replacement = child
                changed = _convert_spikformer_stem_tree(
                    replacement,
                    process_group=process_group,
                    mode=mode,
                )
                if changed:
                    if replacement is not child:
                        parent_module[int(current_name)] = replacement
                    block_index += 1
            continue

        raise ValueError(
            f"Unsupported isolated Spikformer patch stem root '{root}'. Use 'patch_embed' or a root that "
            "belongs to a sequential stem with at least two consecutive blocks."
        )

    return module


def configure_snn_distributed(
    module: nn.Module,
    config: SNNDistributedConfig,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    if config.device_mesh is None:
        mesh_dim_names = None
        if config.mesh_shape is not None and len(config.mesh_shape) > 1:
            generated_names = [f"mesh_dim_{i}" for i in range(len(config.mesh_shape))]
            generated_names[config.tp_mesh_dim] = "tp"
            if config.dp_mesh_dim is not None:
                generated_names[config.dp_mesh_dim] = "dp"
            mesh_dim_names = tuple(generated_names)
        device_mesh = build_device_mesh(
            device_type=config.device_type,
            mesh_shape=config.mesh_shape,
            mesh_dim_names=mesh_dim_names,
        )
    else:
        device_mesh = config.device_mesh

    mesh_tensor = getattr(device_mesh, "mesh", None)
    mesh_ndim = (
        int(mesh_tensor.ndim)
        if mesh_tensor is not None
        else getattr(device_mesh, "ndim", 1)
    )
    if config.enable_data_parallel and mesh_ndim > 1 and config.dp_mesh_dim is None:
        raise ValueError(
            "dp_mesh_dim must be specified when enable_data_parallel=True on a multi-dimensional DeviceMesh."
        )

    analysis = analyze_snn_distributed_capability(
        module, tensor_parallel_roots=config.tensor_parallel_roots
    )

    if config.experimental_conv_tensor_parallel:
        if not config.conv_tensor_parallel_roots:
            raise ValueError(
                "experimental_conv_tensor_parallel=True requires conv_tensor_parallel_roots."
            )
        module = parallelize_snn_conv_blocks(
            module=module,
            device_mesh=device_mesh,
            roots=config.conv_tensor_parallel_roots,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    if config.experimental_spikformer_tensor_parallel:
        if not config.spikformer_tensor_parallel_roots:
            raise ValueError(
                "experimental_spikformer_tensor_parallel=True requires spikformer_tensor_parallel_roots."
            )
        module = parallelize_spikformer_blocks(
            module=module,
            device_mesh=device_mesh,
            roots=config.spikformer_tensor_parallel_roots,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    if config.experimental_spikformer_patch_stem_tensor_parallel:
        if not config.spikformer_patch_stem_tensor_parallel_roots:
            raise ValueError(
                "experimental_spikformer_patch_stem_tensor_parallel=True requires "
                "spikformer_patch_stem_tensor_parallel_roots."
            )
        module = parallelize_spikformer_patch_stem(
            module=module,
            device_mesh=device_mesh,
            roots=config.spikformer_patch_stem_tensor_parallel_roots,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    should_apply_tp = (
        config.tensor_parallel_plan is not None
        or config.auto_tensor_parallel
        or config.experimental_conv_tensor_parallel
        or config.experimental_spikformer_tensor_parallel
        or config.experimental_spikformer_patch_stem_tensor_parallel
    )
    if should_apply_tp and config.enable_data_parallel and not config.enable_fsdp2:
        raise NotImplementedError(
            "Combining DDP-style data parallelism with DTensor tensor parallelism is not "
            "supported in this implementation because DistributedDataParallel state sync "
            "mixes Tensor and DTensor parameters. Please use FSDP2 + TP instead."
        )
    if should_apply_tp:
        if config.tensor_parallel_plan is not None:
            tp_plan = dict(config.tensor_parallel_plan)
        else:
            tp_plan = auto_build_tensor_parallel_plan(
                module, tensor_parallel_roots=config.tensor_parallel_roots
            )
        tp_group = _resolve_mesh_dim_group(device_mesh, config.tp_mesh_dim)
        module = wrap_tp_memory_modules(
            module=module,
            tensor_parallel_plan=tp_plan,
            process_group=tp_group,
        )
        module = parallelize_snn_module(
            module=module,
            device_mesh=device_mesh,
            tensor_parallel_plan=tp_plan,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    if config.enable_fsdp2:
        fsdp_mesh_dim = config.dp_mesh_dim if config.dp_mesh_dim is not None else 0
        fsdp_mesh = _resolve_mesh_submesh(device_mesh, fsdp_mesh_dim)
        mp_policy = _build_fsdp_mp_policy(config)
        module = fully_shard_snn_module(
            module=module,
            device_mesh=fsdp_mesh,
            shard_roots=config.fsdp_shard_roots,
            shard_module_root=config.fsdp_shard_module_root,
            root_reshard_after_forward=config.fsdp_root_reshard_after_forward,
            mp_policy=mp_policy,
        )

    if config.enable_data_parallel:
        dp_group = _resolve_dp_group_from_mesh(device_mesh, config.dp_mesh_dim)
        device_ids = None
        if config.device_type == "cuda" and torch.cuda.is_available():
            local_rank = int(os.environ.get("LOCAL_RANK", torch.cuda.current_device()))
            device_ids = [local_rank]
        module = prepare_snn_data_parallel(
            module=module,
            process_group=dp_group,
            device_ids=device_ids,
            broadcast_buffers=config.broadcast_buffers,
            find_unused_parameters=config.find_unused_parameters,
            static_graph=config.static_graph,
        )

    return module, device_mesh, analysis


def configure_cifar10dvs_vgg_distributed(
    module: nn.Module,
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    enable_data_parallel: bool = False,
    enable_classifier_tensor_parallel: bool = True,
    enable_experimental_conv_tensor_parallel: bool = True,
    tp_mesh_dim: int = 0,
    dp_mesh_dim: Optional[int] = None,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    config = SNNDistributedConfig(
        device_type=device_type,
        mesh_shape=mesh_shape,
        tensor_parallel_roots=["classifier"]
        if enable_classifier_tensor_parallel
        else None,
        auto_tensor_parallel=enable_classifier_tensor_parallel,
        experimental_conv_tensor_parallel=enable_experimental_conv_tensor_parallel,
        conv_tensor_parallel_roots=["features"]
        if enable_experimental_conv_tensor_parallel
        else None,
        enable_data_parallel=enable_data_parallel,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
    )
    return configure_snn_distributed(module, config)


def configure_cifar10dvs_vgg_fsdp2(
    module: nn.Module,
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    enable_classifier_tensor_parallel: bool = False,
    enable_experimental_conv_tensor_parallel: bool = False,
    tp_mesh_dim: int = 0,
    dp_mesh_dim: Optional[int] = None,
    fsdp_root_reshard_after_forward: Optional[bool] = False,
    fsdp_param_dtype: Optional[torch.dtype] = None,
    fsdp_reduce_dtype: Optional[torch.dtype] = None,
    fsdp_output_dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    tp_enabled = (
        enable_classifier_tensor_parallel or enable_experimental_conv_tensor_parallel
    )
    fsdp_shard_roots = ["features"]
    if not enable_classifier_tensor_parallel:
        fsdp_shard_roots.append("classifier")
    config = SNNDistributedConfig(
        device_type=device_type,
        mesh_shape=mesh_shape,
        enable_fsdp2=True,
        fsdp_shard_roots=fsdp_shard_roots,
        fsdp_shard_module_root=not tp_enabled,
        fsdp_root_reshard_after_forward=fsdp_root_reshard_after_forward,
        fsdp_param_dtype=fsdp_param_dtype,
        fsdp_reduce_dtype=fsdp_reduce_dtype,
        fsdp_output_dtype=fsdp_output_dtype,
        tensor_parallel_roots=["classifier"]
        if enable_classifier_tensor_parallel
        else None,
        auto_tensor_parallel=enable_classifier_tensor_parallel,
        experimental_conv_tensor_parallel=enable_experimental_conv_tensor_parallel,
        conv_tensor_parallel_roots=["features"]
        if enable_experimental_conv_tensor_parallel
        else None,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
    )
    return configure_snn_distributed(module, config)


def configure_spikformer_distributed(
    module: nn.Module,
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    enable_data_parallel: bool = False,
    enable_head_tensor_parallel: bool = True,
    tp_mesh_dim: int = 0,
    dp_mesh_dim: Optional[int] = None,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    config = SNNDistributedConfig(
        device_type=device_type,
        mesh_shape=mesh_shape,
        tensor_parallel_roots=["head"] if enable_head_tensor_parallel else None,
        auto_tensor_parallel=enable_head_tensor_parallel,
        experimental_spikformer_tensor_parallel=enable_head_tensor_parallel,
        spikformer_tensor_parallel_roots=["blocks"]
        if enable_head_tensor_parallel
        else None,
        experimental_spikformer_patch_stem_tensor_parallel=enable_head_tensor_parallel,
        spikformer_patch_stem_tensor_parallel_roots=["patch_embed"]
        if enable_head_tensor_parallel
        else None,
        enable_data_parallel=enable_data_parallel,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
    )
    return configure_snn_distributed(module, config)


def configure_spikformer_fsdp2(
    module: nn.Module,
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    enable_head_tensor_parallel: bool = False,
    tp_mesh_dim: int = 0,
    dp_mesh_dim: Optional[int] = None,
    fsdp_root_reshard_after_forward: Optional[bool] = False,
    fsdp_param_dtype: Optional[torch.dtype] = None,
    fsdp_reduce_dtype: Optional[torch.dtype] = None,
    fsdp_output_dtype: Optional[torch.dtype] = None,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    tp_enabled = enable_head_tensor_parallel
    num_blocks = len(getattr(module, "blocks", ()))
    fsdp_shard_roots = ["patch_embed"] + [f"blocks.{i}" for i in range(num_blocks)]
    if not enable_head_tensor_parallel:
        fsdp_shard_roots.append("head")
    config = SNNDistributedConfig(
        device_type=device_type,
        mesh_shape=mesh_shape,
        enable_fsdp2=True,
        fsdp_shard_roots=fsdp_shard_roots,
        fsdp_shard_module_root=not tp_enabled,
        fsdp_root_reshard_after_forward=fsdp_root_reshard_after_forward,
        fsdp_param_dtype=fsdp_param_dtype,
        fsdp_reduce_dtype=fsdp_reduce_dtype,
        fsdp_output_dtype=fsdp_output_dtype,
        tensor_parallel_roots=["head"] if enable_head_tensor_parallel else None,
        auto_tensor_parallel=enable_head_tensor_parallel,
        experimental_spikformer_tensor_parallel=enable_head_tensor_parallel,
        spikformer_tensor_parallel_roots=["blocks"]
        if enable_head_tensor_parallel
        else None,
        experimental_spikformer_patch_stem_tensor_parallel=enable_head_tensor_parallel,
        spikformer_patch_stem_tensor_parallel_roots=["patch_embed"]
        if enable_head_tensor_parallel
        else None,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
    )
    return configure_snn_distributed(module, config)


def configure_cifar10dvs_vgg_pipeline(
    module: nn.Module,
    example_input: torch.Tensor,
    device: Union[str, torch.device],
    n_microbatches: int,
    pp_schedule: str = "auto",
    pp_virtual_stages: int = 1,
    pp_layout: Optional[Union[str, Sequence[int]]] = None,
    pp_delay_wgrad: bool = False,
    stage_index: Optional[int] = None,
    group=None,
) -> SNNPipelineRuntime:
    physical_num_stages = dist.get_world_size(group) if dist.is_initialized() else 1
    logical_num_stages = physical_num_stages * pp_virtual_stages
    feature_count = len(list(module.features.children()))
    total_units = feature_count + 1
    layout_counts = parse_pipeline_layout(pp_layout, logical_num_stages, total_units)
    pipeline_module = _build_cifar10dvs_vgg_pipeline_module(
        module=module,
        num_logical_stages=logical_num_stages,
        example_input=example_input,
        layout_counts=layout_counts,
    )
    return _build_snn_pipeline_runtime(
        pipeline_module=pipeline_module,
        example_input=example_input,
        device=torch.device(device),
        n_microbatches=n_microbatches,
        stage_index=stage_index,
        model_family="cifar10dvs_vgg",
        schedule_kind=pp_schedule,
        virtual_pipeline_size=pp_virtual_stages,
        delayed_wgrad=pp_delay_wgrad,
        pp_layout=layout_counts,
        group=group,
    )


def configure_spikformer_pipeline(
    module: nn.Module,
    example_input: torch.Tensor,
    device: Union[str, torch.device],
    n_microbatches: int,
    pp_schedule: str = "auto",
    pp_virtual_stages: int = 1,
    pp_layout: Optional[Union[str, Sequence[int]]] = None,
    pp_delay_wgrad: bool = False,
    stage_index: Optional[int] = None,
    group=None,
) -> SNNPipelineRuntime:
    physical_num_stages = dist.get_world_size(group) if dist.is_initialized() else 1
    logical_num_stages = physical_num_stages * pp_virtual_stages
    total_units = len(getattr(module, "blocks", ())) + 2
    layout_counts = parse_pipeline_layout(pp_layout, logical_num_stages, total_units)
    pipeline_module = _build_spikformer_pipeline_module(
        module=module,
        num_logical_stages=logical_num_stages,
        example_input=example_input,
        layout_counts=layout_counts,
    )
    return _build_snn_pipeline_runtime(
        pipeline_module=pipeline_module,
        example_input=example_input,
        device=torch.device(device),
        n_microbatches=n_microbatches,
        stage_index=stage_index,
        model_family="spikformer",
        schedule_kind=pp_schedule,
        virtual_pipeline_size=pp_virtual_stages,
        delayed_wgrad=pp_delay_wgrad,
        pp_layout=layout_counts,
        group=group,
    )
