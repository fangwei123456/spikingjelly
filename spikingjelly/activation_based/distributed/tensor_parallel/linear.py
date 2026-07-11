import inspect
from typing import Dict, Mapping, Optional, Sequence, Union

import torch.nn as nn

from spikingjelly.activation_based import base
from spikingjelly.activation_based.distributed.analysis import (
    LinearLike,
    _iter_named_modules_under_roots,  # noqa: F401
    analyze_snn_distributed_capability,
)
from spikingjelly.activation_based.distributed.tensor_parallel.state import (
    make_tensor_shard_memory_module,
)

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


def _replace_module_by_name(module: nn.Module, module_name: str, new_module: nn.Module):
    parent_name, _, child_name = module_name.rpartition(".")
    parent = module if not parent_name else dict(module.named_modules())[parent_name]
    if isinstance(parent, (nn.Sequential, nn.ModuleList)) and child_name.isdigit():
        parent[int(child_name)] = new_module
    else:
        setattr(parent, child_name, new_module)


def auto_build_tensor_parallel_plan(
    module: nn.Module,
    tensor_parallel_roots: Optional[Sequence[str]] = None,
) -> Dict[str, "ParallelStyle"]:
    r"""
    **API Language** - :ref:`中文 <auto_build_tensor_parallel_plan-cn>` | :ref:`English <auto_build_tensor_parallel_plan-en>`

    ----

    .. _auto_build_tensor_parallel_plan-cn:

    * **中文**

    自动构建张量模型并行计划。

    ----

    .. _auto_build_tensor_parallel_plan-en:

    * **English**

    Auto-build tensor parallel plan.
    """
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
                parent[next_index] = make_tensor_shard_memory_module(
                    next_module,
                    shard_dim=-1,
                    logical_dim_size=source.out_features,
                    process_group=process_group,
                )
                wrapped.add(next_name)
    return module


def parallelize_snn_module(
    module: nn.Module,
    device_mesh,
    tensor_parallel_plan: Mapping[str, Union[str, "ParallelStyle"]],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    r"""
    **API Language** - :ref:`中文 <parallelize_snn_module-cn>` | :ref:`English <parallelize_snn_module-en>`

    ----

    .. _parallelize_snn_module-cn:

    * **中文**

    将 SNN 模块并行化。

    ----

    .. _parallelize_snn_module-en:

    * **English**

    Parallelize an SNN module.
    """
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
