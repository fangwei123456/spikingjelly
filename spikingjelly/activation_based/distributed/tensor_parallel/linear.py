from __future__ import annotations

import inspect
from types import MethodType
from typing import Dict, Mapping, Optional, Sequence, Union

import torch.nn as nn
import torch.nn.functional as F

from ... import base
from ...ann2snn.operators import TDLinear
from ..analysis import (
    LinearLike,
    _iter_named_modules_under_roots,  # noqa: F401
    analyze_snn_distributed_capability,
)
from ..planner import TensorParallelStyle
from .state import make_tensor_shard_memory_module

_STATELESS_TP_MEMORY_LOOKAHEAD = (nn.Dropout, nn.Identity)

try:
    from torch.distributed.tensor.parallel import (
        ColwiseParallel,
        ParallelStyle,
        RowwiseParallel,
        parallelize_module,
    )
    from torch.distributed.tensor import DTensor, Replicate, Shard

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
    DTensor = None
    Replicate = None
    Shard = None
    TENSOR_PARALLEL_AVAILABLE = False


_TDColwiseBase = ColwiseParallel if TENSOR_PARALLEL_AVAILABLE else object
_TDRowwiseBase = RowwiseParallel if TENSOR_PARALLEL_AVAILABLE else object


def _install_tdlinear_ann_forward(
    module: TDLinear,
    device_mesh,
    *,
    desired_input_placement,
) -> TDLinear:
    def ann_forward(self: TDLinear, x):
        distributed_input = DTensor.from_local(
            x,
            device_mesh,
            (Replicate(),),
            run_check=False,
        )
        if not isinstance(desired_input_placement, Replicate):
            distributed_input = distributed_input.redistribute(
                placements=(desired_input_placement,),
            )
        output = F.linear(distributed_input, self.weight, self.bias)
        if output.placements != (Replicate(),):
            output = output.redistribute(placements=(Replicate(),))
        return output.to_local()

    module.ann_forward = MethodType(ann_forward, module)
    return module


class _TDLinearColwiseReplicated(_TDColwiseBase):
    def __init__(self) -> None:
        super().__init__(output_layouts=Replicate(), use_local_output=True)

    def _apply(self, module: nn.Module, device_mesh) -> nn.Module:
        if not isinstance(module, TDLinear):
            raise TypeError("td_colwise_replicated requires a TDLinear module.")
        self._partition_linear_fn("", module, device_mesh)
        return _install_tdlinear_ann_forward(
            module,
            device_mesh,
            desired_input_placement=Replicate(),
        )


class _TDLinearRowwiseReplicated(_TDRowwiseBase):
    def __init__(self) -> None:
        super().__init__(
            input_layouts=Replicate(),
            output_layouts=Replicate(),
            use_local_output=True,
        )

    def _apply(self, module: nn.Module, device_mesh) -> nn.Module:
        if not isinstance(module, TDLinear):
            raise TypeError("td_rowwise_replicated requires a TDLinear module.")
        self._partition_linear_fn("", module, device_mesh)
        return _install_tdlinear_ann_forward(
            module,
            device_mesh,
            desired_input_placement=Shard(-1),
        )


def _make_colwise_parallel(local_output: bool) -> "ParallelStyle":
    if not TENSOR_PARALLEL_AVAILABLE:
        raise RuntimeError("torch.distributed.tensor.parallel is unavailable.")
    signature = inspect.signature(ColwiseParallel)
    if "use_local_output" in signature.parameters:
        return ColwiseParallel(use_local_output=local_output)
    if local_output and make_output_tensor is not None:
        return ColwiseParallel(_prepare_output=make_output_tensor)
    if local_output:
        raise RuntimeError(
            "Cannot produce local output from ColwiseParallel: this PyTorch build "
            "lacks both the 'use_local_output' argument and 'make_output_tensor'. "
            "Please upgrade PyTorch."
        )
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
    if lowered == "td_colwise_replicated":
        return _TDLinearColwiseReplicated()
    if lowered == "td_rowwise_replicated":
        return _TDLinearRowwiseReplicated()

    raise ValueError(
        f"Unsupported tensor parallel style '{style}'. "
        "Expected one of: colwise, colwise_local_output, rowwise, "
        "td_colwise_replicated, td_rowwise_replicated."
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
    parent = module if not parent_name else module.get_submodule(parent_name)
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

    根据分析得到的 Linear-like 候选模块自动构建张量并行计划。``TDLinear``
    使用 replicated-activation style：除最后一个候选使用 rowwise 外，其余候选
    使用 colwise。该启发式仅保证生成可应用的 plan；对模型结构有明确认识时应
    优先传入显式 plan。

    :param module: 待分析的 SNN 模型。
    :type module: torch.nn.Module
    :param tensor_parallel_roots: 限制候选分析范围的可选模块根路径。
    :type tensor_parallel_roots: Sequence[str] or None
    :return: 从模块路径到 PyTorch ``ParallelStyle`` 对象的 mapping。
    :rtype: Dict[str, ParallelStyle]
    :raises RuntimeError: 当当前 PyTorch 不提供 DTensor tensor parallel 时。
    :raises ValueError: 当分析未发现 Linear-like 候选模块时。

    ----

    .. _auto_build_tensor_parallel_plan-en:

    * **English**

    Automatically build a tensor-parallel plan from analyzed Linear-like
    candidates. ``TDLinear`` uses replicated-activation styles: every candidate
    except the last is colwise, and the last is rowwise. This heuristic only
    guarantees an applicable plan; prefer an explicit plan when the model
    structure is known.

    :param module: SNN model to analyze.
    :type module: torch.nn.Module
    :param tensor_parallel_roots: Optional module roots constraining candidate
        analysis.
    :type tensor_parallel_roots: Sequence[str] or None
    :return: Mapping from module paths to PyTorch ``ParallelStyle`` objects.
    :rtype: Dict[str, ParallelStyle]
    :raises RuntimeError: If DTensor tensor parallel is unavailable in this
        PyTorch build.
    :raises ValueError: If analysis finds no Linear-like candidates.
    """
    if not TENSOR_PARALLEL_AVAILABLE:
        raise RuntimeError(
            "torch.distributed.tensor.parallel is unavailable in the current "
            "PyTorch build."
        )
    analysis = analyze_snn_distributed_capability(module, tensor_parallel_roots)
    candidate_names = list(analysis.tensor_parallel_candidate_names)
    named_modules = dict(module.named_modules())

    if not candidate_names:
        raise ValueError("No Linear-like modules were found for tensor parallelism.")

    plan: Dict[str, ParallelStyle] = {}
    if len(candidate_names) == 1:
        candidate = named_modules[candidate_names[0]]
        plan[candidate_names[0]] = (
            _TDLinearColwiseReplicated()
            if isinstance(candidate, TDLinear)
            else _make_colwise_parallel(local_output=False)
        )
        return plan

    for name in candidate_names[:-1]:
        plan[name] = (
            _TDLinearColwiseReplicated()
            if isinstance(named_modules[name], TDLinear)
            else _make_colwise_parallel(local_output=True)
        )
    last_name = candidate_names[-1]
    plan[last_name] = (
        _TDLinearRowwiseReplicated()
        if isinstance(named_modules[last_name], TDLinear)
        else RowwiseParallel()
    )
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
            while next_index < len(parent):
                next_module = parent[next_index]
                if isinstance(next_module, _STATELESS_TP_MEMORY_LOOKAHEAD):
                    next_index += 1
                    continue
                next_name = (
                    f"{parent_name}.{next_index}" if parent_name else str(next_index)
                )
                if next_name in wrapped:
                    break
                if isinstance(next_module, base.MemoryModule):
                    parent[next_index] = make_tensor_shard_memory_module(
                        next_module,
                        shard_dim=-1,
                        logical_dim_size=source.out_features,
                        process_group=process_group,
                    )
                    wrapped.add(next_name)
                break
    return module


def parallelize_snn_module(
    module: nn.Module,
    device_mesh,
    tensor_parallel_plan: Mapping[str, TensorParallelStyle],
    tp_mesh_dim: int = 0,
) -> nn.Module:
    r"""
    **API Language** - :ref:`中文 <parallelize_snn_module-cn>` | :ref:`English <parallelize_snn_module-en>`

    ----

    .. _parallelize_snn_module-cn:

    * **中文**

    按显式 tensor-parallel plan 并行化 SNN 模块。普通 Linear-like 模块支持
    ``"colwise"``、``"colwise_local_output"`` 与 ``"rowwise"``；
    ``TDLinear`` 还支持 ``"td_colwise_replicated"`` 和
    ``"td_rowwise_replicated"``。TD style 只在 ANN GEMM 边界构造 DTensor，
    输入、输出以及 TD cumulative/difference 状态保持 local replicated tensor。

    :param module: 原地应用 tensor parallel 的 SNN 模型。
    :type module: torch.nn.Module
    :param device_mesh: PyTorch ``DeviceMesh``；TP 使用其中 ``tp_mesh_dim`` 维。
    :type device_mesh: torch.distributed.device_mesh.DeviceMesh
    :param tensor_parallel_plan: 模块路径到 TP style 名称或 PyTorch
        ``ParallelStyle`` 对象的 mapping。
    :type tensor_parallel_plan: Mapping[str, TensorParallelStyle]
    :param tp_mesh_dim: 多维 mesh 中用于 tensor parallel 的维度索引。
    :type tp_mesh_dim: int
    :return: 参数已按 plan 分片的输入模型。
    :rtype: torch.nn.Module
    :raises RuntimeError: 当当前 PyTorch 不提供 DTensor tensor parallel 时。
    :raises ValueError: 当 style 不受支持，或旧版 PyTorch 无法从多维 mesh
        解析 TP 维度时。
    :raises TypeError: 当 TD 专用 style 应用于非 ``TDLinear`` 模块时。

    ----

    .. _parallelize_snn_module-en:

    * **English**

    Parallelize an SNN module according to an explicit tensor-parallel plan.
    Ordinary Linear-like modules accept ``"colwise"``,
    ``"colwise_local_output"``, and ``"rowwise"``. ``TDLinear`` additionally
    accepts ``"td_colwise_replicated"`` and ``"td_rowwise_replicated"``.
    TD styles construct DTensors only at the ANN GEMM boundary; their inputs,
    outputs, and TD cumulative/difference state remain local replicated tensors.

    :param module: SNN model modified in place by tensor parallelism.
    :type module: torch.nn.Module
    :param device_mesh: PyTorch ``DeviceMesh`` whose ``tp_mesh_dim`` dimension is
        used for TP.
    :type device_mesh: torch.distributed.device_mesh.DeviceMesh
    :param tensor_parallel_plan: Mapping from module paths to TP style names or
        PyTorch ``ParallelStyle`` objects.
    :type tensor_parallel_plan: Mapping[str, TensorParallelStyle]
    :param tp_mesh_dim: Mesh-dimension index used for tensor parallelism.
    :type tp_mesh_dim: int
    :return: Input model with parameters sharded according to the plan.
    :rtype: torch.nn.Module
    :raises RuntimeError: If DTensor tensor parallel is unavailable in this
        PyTorch build.
    :raises ValueError: If a style is unsupported or an older PyTorch build cannot
        resolve the TP dimension from a multidimensional mesh.
    :raises TypeError: If a TD-specific style is applied to a non-``TDLinear``
        module.
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
        mesh_dim_names = getattr(device_mesh, "mesh_dim_names", None)
        if mesh_dim_names:
            if tp_mesh_dim < 0 or tp_mesh_dim >= len(mesh_dim_names):
                raise ValueError(
                    f"tp_mesh_dim={tp_mesh_dim} is out of range for a mesh with "
                    f"{len(mesh_dim_names)} dimensions."
                )
            mesh_name = mesh_dim_names[tp_mesh_dim]
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
