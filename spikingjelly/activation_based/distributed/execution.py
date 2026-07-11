from typing import Optional, Tuple

import torch
import torch.nn as nn

try:
    from torch.distributed._tensor import DeviceMesh
except ImportError:
    DeviceMesh = object

from spikingjelly.activation_based.distributed.analysis import (
    SNNDistributedAnalysis,
    analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.config import (
    EagerParallelPolicy,
    SNNDistributedConfig,
)
from spikingjelly.activation_based.distributed.data_parallel import (
    prepare_snn_data_parallel,
)
from spikingjelly.activation_based.distributed.fsdp import (
    _build_fsdp_mp_policy,
    fully_shard_snn_module,
)
from spikingjelly.activation_based.distributed.mesh import (
    _resolve_dp_group_from_mesh,
    _resolve_mesh_dim_group,
    _resolve_mesh_submesh,
    build_device_mesh,
)
from spikingjelly.activation_based.distributed.tensor_parallel.cifar10dvs_vgg import (
    parallelize_snn_conv_blocks,
)
from spikingjelly.activation_based.distributed.tensor_parallel.linear import (
    auto_build_tensor_parallel_plan,
    parallelize_snn_module,
    wrap_tp_memory_modules,
)
from spikingjelly.activation_based.distributed.tensor_parallel.spikformer import (
    parallelize_spikformer_blocks,
    parallelize_spikformer_patch_stem,
)


def _optional_list(values: Tuple[str, ...]) -> Optional[list[str]]:
    return list(values) if values else None


def build_eager_config(
    *,
    mode: str,
    device_type: str = "cuda",
    mesh_shape: Optional[Tuple[int, ...]] = None,
    device_mesh=None,
    tp_mesh_dim: int = 0,
    dp_mesh_dim: Optional[int] = None,
    policy: Optional[EagerParallelPolicy] = None,
    enable_linear_tensor_parallel: bool = True,
    enable_conv_tensor_parallel: bool = True,
    enable_spikformer_tensor_parallel: bool = True,
    enable_spikformer_patch_stem_tensor_parallel: bool = True,
    auto_tensor_parallel: Optional[bool] = None,
    **config_overrides,
) -> SNNDistributedConfig:
    policy = policy or EagerParallelPolicy()
    mode = mode.lower()

    linear_roots = (
        policy.linear_tensor_parallel_roots
        if enable_linear_tensor_parallel and mode in ("tp", "fsdp2_tp")
        else ()
    )
    conv_roots = (
        policy.conv_tensor_parallel_roots
        if enable_conv_tensor_parallel and mode in ("tp", "fsdp2_tp")
        else ()
    )
    spikformer_roots = (
        policy.spikformer_tensor_parallel_roots
        if enable_spikformer_tensor_parallel and mode in ("tp", "fsdp2_tp")
        else ()
    )
    spikformer_patch_stem_roots = (
        policy.spikformer_patch_stem_tensor_parallel_roots
        if enable_spikformer_patch_stem_tensor_parallel and mode in ("tp", "fsdp2_tp")
        else ()
    )
    if auto_tensor_parallel is None:
        auto_tensor_parallel = bool(linear_roots)

    fsdp_shard_roots = None
    fsdp_shard_module_root = policy.fsdp_shard_module_root
    if mode == "fsdp2":
        fsdp_shard_roots = _optional_list(policy.fsdp_shard_roots)
    elif mode == "fsdp2_tp":
        fsdp2_tp_roots = policy.fsdp2_tp_shard_roots or policy.fsdp_shard_roots
        fsdp_shard_roots = _optional_list(fsdp2_tp_roots)
        fsdp_shard_module_root = (
            policy.fsdp2_tp_shard_module_root
            if policy.fsdp2_tp_shard_roots or policy.fsdp2_tp_shard_module_root
            else policy.fsdp_shard_module_root
        )

    config_kwargs = {
        "device_type": device_type,
        "mesh_shape": mesh_shape,
        "device_mesh": device_mesh,
        "tp_mesh_dim": tp_mesh_dim,
        "dp_mesh_dim": dp_mesh_dim,
        "enable_data_parallel": mode == "dp",
        "enable_fsdp2": mode in ("fsdp2", "fsdp2_tp"),
        "tensor_parallel_roots": _optional_list(linear_roots),
        "auto_tensor_parallel": auto_tensor_parallel,
        "experimental_conv_tensor_parallel": bool(conv_roots),
        "conv_tensor_parallel_roots": _optional_list(conv_roots),
        "experimental_spikformer_tensor_parallel": bool(spikformer_roots),
        "spikformer_tensor_parallel_roots": _optional_list(spikformer_roots),
        "experimental_spikformer_patch_stem_tensor_parallel": bool(
            spikformer_patch_stem_roots
        ),
        "spikformer_patch_stem_tensor_parallel_roots": _optional_list(
            spikformer_patch_stem_roots
        ),
        "fsdp_shard_roots": fsdp_shard_roots,
        "fsdp_shard_module_root": fsdp_shard_module_root,
    }
    config_kwargs.update(config_overrides)
    return SNNDistributedConfig(**config_kwargs)


def configure_snn_distributed(
    module: nn.Module,
    config: SNNDistributedConfig,
) -> Tuple[nn.Module, "DeviceMesh", SNNDistributedAnalysis]:
    r"""
    **API Language** - :ref:`中文 <configure_snn_distributed-cn>` | :ref:`English <configure_snn_distributed-en>`

    ----

    .. _configure_snn_distributed-cn:

    * **中文**

    配置 SNN 分布式训练。

    ----

    .. _configure_snn_distributed-en:

    * **English**

    Configure SNN distributed training.
    """
    should_apply_linear_tp = (
        config.tensor_parallel_plan is not None or config.auto_tensor_parallel
    )
    should_apply_tp = (
        should_apply_linear_tp
        or config.experimental_conv_tensor_parallel
        or config.experimental_spikformer_tensor_parallel
        or config.experimental_spikformer_patch_stem_tensor_parallel
    )

    analysis = analyze_snn_distributed_capability(
        module, tensor_parallel_roots=config.tensor_parallel_roots
    )

    needs_device_mesh = (
        config.device_mesh is not None
        or config.enable_data_parallel
        or config.enable_fsdp2
        or should_apply_tp
    )
    if not needs_device_mesh:
        return module, None, analysis

    if config.device_mesh is None:
        mesh_dim_names = None
        if config.mesh_shape is not None and len(config.mesh_shape) > 1:
            if config.tp_mesh_dim < 0 or config.tp_mesh_dim >= len(config.mesh_shape):
                raise ValueError(
                    f"tp_mesh_dim={config.tp_mesh_dim} is out of range for a mesh with {len(config.mesh_shape)} dimensions."
                )
            if config.dp_mesh_dim is not None and (
                config.dp_mesh_dim < 0 or config.dp_mesh_dim >= len(config.mesh_shape)
            ):
                raise ValueError(
                    f"dp_mesh_dim={config.dp_mesh_dim} is out of range for a mesh with {len(config.mesh_shape)} dimensions."
                )
            if (
                should_apply_tp
                and config.dp_mesh_dim is not None
                and config.tp_mesh_dim == config.dp_mesh_dim
            ):
                raise ValueError(
                    "tp_mesh_dim and dp_mesh_dim must be different when tensor parallelism "
                    f"is enabled, but both are {config.tp_mesh_dim}."
                )
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
    if config.tp_mesh_dim < 0 or config.tp_mesh_dim >= mesh_ndim:
        raise ValueError(
            f"tp_mesh_dim={config.tp_mesh_dim} is out of range for a mesh with {mesh_ndim} dimensions."
        )
    if config.dp_mesh_dim is not None and (
        config.dp_mesh_dim < 0 or config.dp_mesh_dim >= mesh_ndim
    ):
        raise ValueError(
            f"dp_mesh_dim={config.dp_mesh_dim} is out of range for a mesh with {mesh_ndim} dimensions."
        )
    if (
        should_apply_tp
        and config.dp_mesh_dim is not None
        and config.tp_mesh_dim == config.dp_mesh_dim
    ):
        raise ValueError(
            "tp_mesh_dim and dp_mesh_dim must be different when tensor parallelism "
            f"is enabled, but both are {config.tp_mesh_dim}."
        )

    if should_apply_tp and config.enable_data_parallel and not config.enable_fsdp2:
        raise NotImplementedError(
            "Combining DDP-style data parallelism with DTensor tensor parallelism is not "
            "supported in this implementation because DistributedDataParallel state sync "
            "mixes Tensor and DTensor parameters. Please use FSDP2 + TP instead."
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

    if should_apply_linear_tp:
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
            device_ids = [torch.cuda.current_device()]
        module = prepare_snn_data_parallel(
            module=module,
            process_group=dp_group,
            device_ids=device_ids,
            broadcast_buffers=config.broadcast_buffers,
            find_unused_parameters=config.find_unused_parameters,
            static_graph=config.static_graph,
        )

    return module, device_mesh, analysis
