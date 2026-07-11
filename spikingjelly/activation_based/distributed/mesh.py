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
    """Initialize the default process group if needed.

    .. admonition:: Chinese

        如果默认 ``torch.distributed`` 进程组尚未初始化，则使用给定参数初始化。

    :param backend: Optional backend name. Defaults to ``"nccl"`` on CUDA and
        ``"gloo"`` otherwise.
    :type backend: str or None
    :param init_method: Optional initialization method passed to PyTorch.
    :type init_method: str or None
    :param rank: Optional rank passed to PyTorch.
    :type rank: int or None
    :param world_size: Optional world size passed to PyTorch.
    :type world_size: int or None
    :return: ``True`` if this call initialized the group, otherwise ``False``.
    :rtype: bool
    """
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
    """Create a PyTorch ``DeviceMesh`` for the initialized process group.

    .. admonition:: Chinese

        基于当前已初始化的进程组创建 PyTorch ``DeviceMesh``，并校验 mesh 大小
        与 ``world_size`` 一致。

    :param device_type: Device type, such as ``"cuda"`` or ``"cpu"``.
    :type device_type: str
    :param mesh_shape: Optional logical mesh shape. Defaults to all ranks in 1D.
    :type mesh_shape: tuple[int, ...] or None
    :param mesh_dim_names: Optional names for mesh dimensions.
    :type mesh_dim_names: tuple[str, ...] or None
    :return: PyTorch DTensor ``DeviceMesh``.
    :rtype: torch.distributed._tensor.DeviceMesh
    """
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
    """Resolve data-parallel partition count and local partition index.

    .. admonition:: Chinese

        根据 ``DeviceMesh`` 和数据并行维度解析数据分片数量以及当前 rank 所属的
        分片编号。

    :param device_mesh: Optional device mesh.
    :param dp_mesh_dim: Data-parallel mesh dimension, or ``None`` for 1D meshes.
    :param sharded_by_data_parallel: Whether data is sharded by data parallelism.
    :return: ``(num_partitions, partition_index)``.
    :rtype: tuple[int, int]
    """
    if not sharded_by_data_parallel or device_mesh is None:
        return 1, 0

    mesh_tensor = getattr(device_mesh, "mesh", None)
    if mesh_tensor is None:
        world_size = dist.get_world_size() if dist.is_initialized() else 1
        rank = dist.get_rank() if dist.is_initialized() else 0
        return world_size, rank

    mesh_shape = tuple(int(v) for v in mesh_tensor.shape)
    raw_coordinate = (
        device_mesh.get_coordinate() if hasattr(device_mesh, "get_coordinate") else None
    )
    coordinate = (
        tuple(int(v) for v in raw_coordinate)
        if raw_coordinate is not None
        else None
    )
    if dp_mesh_dim is None:
        if len(mesh_shape) != 1:
            raise ValueError(
                "dp_mesh_dim must be specified for data-parallel sharding on a multi-dimensional mesh."
            )
        if coordinate is None:
            raise ValueError(
                "Current rank does not belong to the supplied DeviceMesh; "
                "cannot derive a data-parallel partition index."
            )
        return mesh_shape[0], coordinate[0]

    if dp_mesh_dim < 0 or dp_mesh_dim >= len(mesh_shape):
        raise ValueError(
            f"dp_mesh_dim={dp_mesh_dim} is out of range for a mesh with shape {mesh_shape}."
        )
    if coordinate is None:
        raise ValueError(
            "Current rank does not belong to the supplied DeviceMesh; "
            "cannot derive a data-parallel partition index."
        )
    return mesh_shape[dp_mesh_dim], coordinate[dp_mesh_dim]


def resolve_tensor_parallel_group_size(
    device_mesh: Optional["DeviceMesh"],
    tp_mesh_dim: int,
    tensor_parallel_enabled: bool,
) -> int:
    """Resolve the tensor-parallel group size from a mesh.

    .. admonition:: Chinese

        根据 ``DeviceMesh`` 和张量并行维度解析张量并行组大小；未启用张量并行时
        返回 ``1``。

    :param device_mesh: Optional device mesh.
    :param tp_mesh_dim: Tensor-parallel mesh dimension.
    :type tp_mesh_dim: int
    :param tensor_parallel_enabled: Whether tensor parallelism is enabled.
    :type tensor_parallel_enabled: bool
    :return: Tensor-parallel group size.
    :rtype: int
    """
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
