import warnings
from typing import Optional, Tuple

import torch
import torch.nn as nn

from spikingjelly.logger import logger

try:
    from torch.distributed._tensor import DeviceMesh
except ImportError:
    DeviceMesh = object

from spikingjelly.activation_based.distributed.analysis import (
    SNNDistributedAnalysis,
    analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.config import SNNDistributedConfig
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


def _validate_mesh_dims(config: SNNDistributedConfig, mesh_ndim: int) -> None:
    tensor_parallel = config.mode in ("tp", "fsdp2_tp")
    if (
        config.mode in ("dp", "fsdp2", "fsdp2_tp")
        and mesh_ndim > 1
        and config.dp_mesh_dim is None
    ):
        raise ValueError(
            f"dp_mesh_dim must be specified for mode={config.mode!r} on a "
            "multi-dimensional DeviceMesh."
        )
    for name, dim in (
        ("tp_mesh_dim", config.tp_mesh_dim),
        ("dp_mesh_dim", config.dp_mesh_dim),
    ):
        if dim is not None and not 0 <= dim < mesh_ndim:
            raise ValueError(
                f"{name}={dim} is out of range for a mesh with {mesh_ndim} dimensions."
            )
    if tensor_parallel and config.dp_mesh_dim == config.tp_mesh_dim:
        raise ValueError(
            "tp_mesh_dim and dp_mesh_dim must be different when tensor parallelism "
            f"is enabled, but both are {config.tp_mesh_dim}."
        )


def configure_snn_distributed(
    module: nn.Module,
    config: SNNDistributedConfig,
) -> Tuple[nn.Module, Optional["DeviceMesh"], SNNDistributedAnalysis]:
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
    tensor_parallel = config.mode in ("tp", "fsdp2_tp")
    should_apply_linear_tp = tensor_parallel and (
        config.tensor_parallel_plan is not None or config.auto_tensor_parallel
    )
    analysis = analyze_snn_distributed_capability(
        module, tensor_parallel_roots=config.tensor_parallel_roots
    )

    needs_device_mesh = config.device_mesh is not None or config.mode != "none"
    if not needs_device_mesh:
        logger.info(
            "distributed_configuration_summary mode=%s device_type=%s mesh_shape=None "
            "tensor_parallel=%s fsdp2=False data_parallel=False memory_modules=%s "
            "tensor_parallel_candidates=%s",
            config.mode,
            config.device_type,
            tensor_parallel,
            len(analysis.memory_module_names),
            len(analysis.tensor_parallel_candidate_names),
        )
        return module, None, analysis

    if config.device_mesh is None:
        mesh_ndim = len(config.mesh_shape) if config.mesh_shape is not None else 1
        _validate_mesh_dims(config, mesh_ndim)
        mesh_dim_names = None
        if mesh_ndim > 1:
            generated_names = [f"mesh_dim_{i}" for i in range(mesh_ndim)]
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
        if config.mesh_shape is not None:
            warnings.warn(
                "`device_mesh` was supplied explicitly; `mesh_shape` is ignored.",
                UserWarning,
                stacklevel=2,
            )
        device_mesh = config.device_mesh
        _validate_mesh_dims(config, int(device_mesh.mesh.ndim))

    if tensor_parallel and config.conv_tensor_parallel_roots:
        module = parallelize_snn_conv_blocks(
            module=module,
            device_mesh=device_mesh,
            roots=config.conv_tensor_parallel_roots,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    if tensor_parallel and config.spikformer_tensor_parallel_roots:
        module = parallelize_spikformer_blocks(
            module=module,
            device_mesh=device_mesh,
            roots=config.spikformer_tensor_parallel_roots,
            tp_mesh_dim=config.tp_mesh_dim,
        )

    if tensor_parallel and config.spikformer_patch_stem_tensor_parallel_roots:
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

    if config.mode in ("fsdp2", "fsdp2_tp"):
        fsdp_mesh_dim = config.dp_mesh_dim if config.dp_mesh_dim is not None else 0
        fsdp_mesh = _resolve_mesh_submesh(device_mesh, fsdp_mesh_dim)
        mp_policy = _build_fsdp_mp_policy(
            config.fsdp_param_dtype,
            config.fsdp_reduce_dtype,
            config.fsdp_output_dtype,
        )
        module = fully_shard_snn_module(
            module=module,
            device_mesh=fsdp_mesh,
            shard_roots=config.fsdp_shard_roots,
            shard_module_root=config.fsdp_shard_module_root,
            root_reshard_after_forward=config.fsdp_root_reshard_after_forward,
            mp_policy=mp_policy,
        )

    if config.mode == "dp":
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

    mesh_tensor = getattr(device_mesh, "mesh", None)
    mesh_shape = tuple(int(value) for value in mesh_tensor.shape)
    logger.info(
        "distributed_configuration_summary mode=%s device_type=%s mesh_shape=%s tensor_parallel=%s fsdp2=%s data_parallel=%s memory_modules=%s tensor_parallel_candidates=%s",
        config.mode,
        config.device_type,
        mesh_shape,
        tensor_parallel,
        config.mode in ("fsdp2", "fsdp2_tp"),
        config.mode == "dp",
        len(analysis.memory_module_names),
        len(analysis.tensor_parallel_candidate_names),
    )
    return module, device_mesh, analysis
