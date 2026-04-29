import os
import inspect
import copy
from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple, Union

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
        if self.expected_local_dim_size is not None and x.shape[shard_dim] != self.expected_local_dim_size:
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


def ensure_distributed_initialized(
    backend: Optional[str] = None,
    init_method: Optional[str] = None,
    rank: Optional[int] = None,
    world_size: Optional[int] = None,
) -> bool:
    if dist.is_available() and dist.is_initialized():
        return False

    if not dist.is_available():
        raise RuntimeError("torch.distributed is not available in the current PyTorch build.")

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
        if make_output_tensor is not None and getattr(style, "_prepare_output", None) is make_output_tensor:
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
            raise KeyError(f"tensor_parallel_roots contains unknown module path '{root}'.")

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
        elif isinstance(child, (nn.Conv1d, nn.Conv2d, nn.Conv3d, layer.Conv1d, layer.Conv2d, layer.Conv3d)):
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
        notes.append("No Linear-like tensor-parallel candidates were found under the selected roots.")

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
            next_name = f"{parent_name}.{next_index}" if parent_name else str(next_index)
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
        rank = coordinate[0] if coordinate is not None else (dist.get_rank() if dist.is_initialized() else 0)
        return mesh_shape[0], rank

    if dp_mesh_dim < 0 or dp_mesh_dim >= len(mesh_shape):
        raise ValueError(
            f"dp_mesh_dim={dp_mesh_dim} is out of range for a mesh with shape {mesh_shape}."
        )
    if coordinate is None:
        raise ValueError("DeviceMesh does not expose coordinates for data partitioning.")
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
        raise RuntimeError("FSDP2 fully_shard is unavailable in the current PyTorch build.")

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
            fully_shard(module, mesh=device_mesh, reshard_after_forward=root_reshard_after_forward)
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
        raise AttributeError("DeviceMesh does not expose get_dim_groups() or get_all_groups().")

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
        raise AttributeError("DeviceMesh does not expose get_dim_groups() or get_all_groups().")

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
            raise ValueError(f"expected x with shape [T, N, C, H, W], but got {x.shape}!")

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
            bias = source.bias.detach()[start:end].clone() if source.bias is not None else None
            self.local_out_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter("bias", nn.Parameter(bias) if bias is not None else None)
        elif mode == "rowwise":
            _require_even_shard(source.in_channels, self.world_size, "in_channels")
            start, end = _shard_range(source.in_channels, self.rank, self.world_size)
            weight = source.weight.detach()[:, start:end].clone()
            bias = source.bias.detach().clone() if source.bias is not None else None
            self.local_in_channels = end - start
            self.register_parameter("weight", nn.Parameter(weight))
            self.register_parameter("bias", nn.Parameter(bias) if bias is not None else None)
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
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
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
            raise ValueError(f"expected x with shape [T, N, C, H, W], but got {x.shape}!")
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
            self.register_parameter("weight", nn.Parameter(source.weight.detach()[start:end].clone()))
            self.register_parameter("bias", nn.Parameter(source.bias.detach()[start:end].clone()))
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

        if self.track_running_stats:
            self.register_buffer("running_mean", source.running_mean.detach()[start:end].clone())
            self.register_buffer("running_var", source.running_var.detach()[start:end].clone())
            num_batches_tracked = getattr(source, "num_batches_tracked", None)
            if num_batches_tracked is not None:
                self.register_buffer("num_batches_tracked", num_batches_tracked.detach().clone())
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
        if self.training and self.track_running_stats and self.num_batches_tracked is not None:
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


def _try_convert_vgg_like_block(block: nn.Module, process_group, mode: str) -> Optional[nn.Module]:
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


def _convert_seq_to_ann_conv1d_bn(container: nn.Module, process_group, mode: str) -> Optional[nn.Module]:
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


def _try_convert_spikformer_block(block: nn.Module, process_group) -> Optional[nn.Module]:
    if not (hasattr(block, "attn") and hasattr(block, "mlp")):
        return None

    attn = block.attn
    if hasattr(attn, "qkv_conv_bn"):
        converted = _convert_seq_to_ann_conv1d_bn(attn.qkv_conv_bn, process_group, mode="colwise")
        if converted is not None:
            attn.qkv_conv_bn = converted
            if isinstance(attn.qkv_lif, base.MemoryModule):
                attn.qkv_lif = TensorShardMemoryModule(
                    attn.qkv_lif,
                    shard_dim=2,
                    logical_dim_size=getattr(attn, "dim", None) * 3 if getattr(attn, "dim", None) is not None else None,
                    process_group=process_group,
                )
            if isinstance(attn.attn_lif, base.MemoryModule):
                attn.attn_lif = TensorShardMemoryModule(
                    attn.attn_lif,
                    shard_dim=2,
                    logical_dim_size=None,
                    process_group=process_group,
                )
        converted = _convert_seq_to_ann_conv1d_bn(attn.proj_conv_bn, process_group, mode="rowwise")
        if converted is not None:
            attn.proj_conv_bn = converted

    mlp = block.mlp
    if hasattr(mlp, "fc1"):
        converted = _convert_seq_to_ann_conv1d_bn(mlp.fc1, process_group, mode="colwise")
        if converted is not None:
            mlp.fc1 = converted
            if isinstance(mlp.neuron1, base.MemoryModule):
                logical_dim = None
                conv = list(mlp.fc1.children())[0]
                if hasattr(conv, "out_channels"):
                    logical_dim = conv.out_channels if not hasattr(conv, "local_out_channels") else conv.out_channels
                mlp.neuron1 = TensorShardMemoryModule(
                    mlp.neuron1,
                    shard_dim=2,
                    logical_dim_size=logical_dim,
                    process_group=process_group,
                )
    if hasattr(mlp, "fc2"):
        converted = _convert_seq_to_ann_conv1d_bn(mlp.fc2, process_group, mode="rowwise")
        if converted is not None:
            mlp.fc2 = converted

    return block


def _try_convert_spikformer_stem_block(block: nn.Module, process_group, mode: str) -> Optional[nn.Module]:
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
        for child_name, child in root_module.named_children():
            converted = _try_convert_vgg_like_block(
                child,
                process_group=process_group,
                mode="colwise" if block_index % 2 == 0 else "rowwise",
            )
            if converted is not None:
                root_module[int(child_name)] = converted
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

        for child_name, child in root_module.named_children():
            converted = _try_convert_spikformer_block(child, process_group=process_group)
            if converted is not None:
                root_module[int(child_name)] = converted

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
        if isinstance(root_module, (nn.Sequential, nn.ModuleList)):
            block_index = 0
            for child_name, child in root_module.named_children():
                converted = _try_convert_spikformer_stem_block(
                    child,
                    process_group=process_group,
                    mode="colwise" if block_index % 2 == 0 else "rowwise",
                )
                if converted is not None:
                    root_module[int(child_name)] = converted
                    block_index += 1
            continue

        converted = _try_convert_spikformer_stem_block(
            root_module,
            process_group=process_group,
            mode="rowwise",
        )
        if converted is not None:
            _replace_module_by_name(module, root, converted)

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

    should_apply_tp = config.tensor_parallel_plan is not None or config.auto_tensor_parallel
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
        tensor_parallel_roots=["classifier"] if enable_classifier_tensor_parallel else None,
        auto_tensor_parallel=enable_classifier_tensor_parallel,
        experimental_conv_tensor_parallel=enable_experimental_conv_tensor_parallel,
        conv_tensor_parallel_roots=["features"] if enable_experimental_conv_tensor_parallel else None,
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
    tp_enabled = enable_classifier_tensor_parallel or enable_experimental_conv_tensor_parallel
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
        tensor_parallel_roots=["classifier"] if enable_classifier_tensor_parallel else None,
        auto_tensor_parallel=enable_classifier_tensor_parallel,
        experimental_conv_tensor_parallel=enable_experimental_conv_tensor_parallel,
        conv_tensor_parallel_roots=["features"] if enable_experimental_conv_tensor_parallel else None,
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
        spikformer_tensor_parallel_roots=["blocks"] if enable_head_tensor_parallel else None,
        experimental_spikformer_patch_stem_tensor_parallel=enable_head_tensor_parallel,
        spikformer_patch_stem_tensor_parallel_roots=["patch_embed.stages"] if enable_head_tensor_parallel else None,
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
        spikformer_tensor_parallel_roots=["blocks"] if enable_head_tensor_parallel else None,
        experimental_spikformer_patch_stem_tensor_parallel=enable_head_tensor_parallel,
        spikformer_patch_stem_tensor_parallel_roots=["patch_embed.stages"] if enable_head_tensor_parallel else None,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
    )
    return configure_snn_distributed(module, config)
