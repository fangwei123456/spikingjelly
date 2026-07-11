from __future__ import annotations

import inspect
from typing import Optional, Tuple

import torch
import torch.distributed as dist

try:
    from torch.distributed._tensor import DeviceMesh, init_device_mesh

    DTENSOR_AVAILABLE = True
except ImportError:
    DeviceMesh = None
    init_device_mesh = None
    DTENSOR_AVAILABLE = False


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


def _resolve_mesh_submesh(device_mesh: "DeviceMesh", mesh_dim: int) -> "DeviceMesh":
    if getattr(device_mesh, "ndim", 1) == 1:
        return device_mesh
    if getattr(device_mesh, "mesh_dim_names", None):
        return device_mesh[device_mesh.mesh_dim_names[mesh_dim]]
    raise ValueError(
        "A multi-dimensional DeviceMesh requires mesh_dim_names to derive a 1D submesh."
    )


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


def _resolve_dp_group_from_mesh(device_mesh: "DeviceMesh", dp_mesh_dim: Optional[int]):
    if dp_mesh_dim is None:
        return None
    return _resolve_mesh_dim_group(device_mesh, dp_mesh_dim)


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
