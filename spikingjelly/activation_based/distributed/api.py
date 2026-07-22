from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import torch.nn as nn

from .adapters import resolve_adapter
from .analysis import SNNDistributedAnalysis, analyze_snn_distributed_capability
from .execution import build_eager_config, configure_snn_distributed
from .planner import (
    DistributedFeatureSet,
    SNN_DISTRIBUTED_PREFERENCES,
    SNNDistributedPlan,
    TensorParallelStyle,
    recommend_snn_distributed_strategy,
)
from .runtime import SNNDistributedRuntime
from .topology import SNNDistributedTopology


def _normalize_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    mode = mode.lower()
    valid_modes = ("none", "dp", "tp", "fsdp2", "fsdp2_tp", "pp")
    if mode not in valid_modes:
        raise ValueError(f"Unsupported mode='{mode}'. Expected one of {valid_modes}.")
    if mode == "pp":
        raise NotImplementedError(
            "Pipeline parallelism ('pp') is not supported by the unified analyze/plan/apply API. "
            "Please use the dedicated pipeline configuration path directly."
        )
    return mode


def analyze(
    model: nn.Module,
    *,
    model_family: Optional[str] = None,
    roots: Optional[Sequence[str]] = None,
) -> SNNDistributedAnalysis:
    """Analyze an SNN model for distributed execution.

    .. admonition:: Chinese

        分析 SNN 模型中可用于分布式执行的状态模块、张量并行候选模块和
        不支持项。

    :param model: Model to inspect.
    :type model: torch.nn.Module
    :param model_family: Optional model-family hint reserved for API symmetry.
    :type model_family: str or None
    :param roots: Optional module roots that constrain tensor-parallel analysis.
    :type roots: sequence[str] or None
    :return: Structured distributed capability analysis.
    :rtype: SNNDistributedAnalysis
    """
    return analyze_snn_distributed_capability(model, tensor_parallel_roots=roots)


def plan(
    *,
    analysis: SNNDistributedAnalysis,
    objective: str,
    topology: Union[Mapping[str, int], SNNDistributedTopology],
    backend: str,
    batch_size: int,
    model_family: Optional[str] = None,
    mode: Optional[str] = None,
    features: Optional[DistributedFeatureSet] = None,
    tensor_parallel_plan: Optional[Mapping[str, TensorParallelStyle]] = None,
) -> SNNDistributedPlan:
    r"""
    **API Language** - :ref:`中文 <distributed-plan-cn>` | :ref:`English <distributed-plan-en>`

    ----

    .. _distributed-plan-cn:

    * **中文**

    根据模型分析结果、优化目标、拓扑和后端构建 eager 分布式执行计划。显式
    ``tensor_parallel_plan`` 的 key 必须属于 ``analysis`` 发现的候选模块；提供
    该 mapping 后，应用阶段不会运行自动 TP 规划。``TDLinear`` 支持
    ``"td_colwise_replicated"`` 和 ``"td_rowwise_replicated"``。

    :param analysis: :func:`analyze` 返回的能力分析结果。
    :type analysis: SNNDistributedAnalysis
    :param objective: 优化目标，例如 ``"speed"``。
    :type objective: str
    :param topology: 逻辑拓扑 mapping 或拓扑对象。
    :type topology: Mapping[str, int] or SNNDistributedTopology
    :param backend: 执行后端名称。
    :type backend: str
    :param batch_size: 规划器使用的单步 batch size。
    :type batch_size: int
    :param model_family: 可选的模型族提示。
    :type model_family: str or None
    :param mode: 可选的显式分布式模式覆盖。
    :type mode: str or None
    :param features: 实验性或可选行为的功能开关。
    :type features: DistributedFeatureSet or None
    :param tensor_parallel_plan: 从已分析模块路径到 TP style 名称或 PyTorch
        ``ParallelStyle`` 对象的可选 mapping。
    :type tensor_parallel_plan: Mapping[str, TensorParallelStyle] or None
    :return: 不可变的分布式执行计划。
    :rtype: SNNDistributedPlan
    :raises ValueError: 当目标、模式、拓扑或显式 TP plan 不合法，或 plan 引用
        analysis 候选集合之外的模块时。
    :raises NotImplementedError: 当请求统一 API 尚不支持的 pipeline 模式时。

    ----

    .. _distributed-plan-en:

    * **English**

    Build an eager distributed execution plan from model analysis, objective,
    topology, and backend. Keys in an explicit ``tensor_parallel_plan`` must be
    candidates found by ``analysis``; supplying the mapping disables automatic
    TP planning during apply. ``TDLinear`` accepts
    ``"td_colwise_replicated"`` and ``"td_rowwise_replicated"``.

    :param analysis: Capability analysis returned by :func:`analyze`.
    :type analysis: SNNDistributedAnalysis
    :param objective: Optimization objective, for example ``"speed"``.
    :type objective: str
    :param topology: Logical topology mapping or topology object.
    :type topology: Mapping[str, int] or SNNDistributedTopology
    :param backend: Execution backend name.
    :type backend: str
    :param batch_size: Per-step batch size used by the planner.
    :type batch_size: int
    :param model_family: Optional model-family hint.
    :type model_family: str or None
    :param mode: Optional explicit distributed-mode override.
    :type mode: str or None
    :param features: Feature gates for experimental or optional behavior.
    :type features: DistributedFeatureSet or None
    :param tensor_parallel_plan: Optional mapping from analyzed module paths to TP
        style names or PyTorch ``ParallelStyle`` objects.
    :type tensor_parallel_plan: Mapping[str, TensorParallelStyle] or None
    :return: Immutable distributed execution plan.
    :rtype: SNNDistributedPlan
    :raises ValueError: If the objective, mode, topology, or explicit TP plan is
        invalid, or the plan references modules outside the analyzed candidates.
    :raises NotImplementedError: If the unsupported unified pipeline mode is
        requested.
    """
    objective = objective.lower()
    if objective not in SNN_DISTRIBUTED_PREFERENCES:
        raise ValueError(
            f"Unsupported objective='{objective}'. Expected one of {SNN_DISTRIBUTED_PREFERENCES}."
        )
    resolved_topology = (
        topology
        if isinstance(topology, SNNDistributedTopology)
        else SNNDistributedTopology.from_mapping(topology)
    )
    features = features or DistributedFeatureSet()
    resolved_model_family = model_family or "generic"
    tp_mesh_dim = (
        resolved_topology.ordered_dim_names.index("tp")
        if "tp" in resolved_topology.ordered_dim_names
        else 0
    )
    dp_mesh_dim = (
        resolved_topology.ordered_dim_names.index("dp")
        if "dp" in resolved_topology.ordered_dim_names
        else None
    )
    recommendation = recommend_snn_distributed_strategy(
        model=resolved_model_family,
        world_size=resolved_topology.world_size,
        prefer=objective,
        batch_size=batch_size,
        backend=backend,
        pipelining_available=False,
    )
    mode = _normalize_mode(mode)
    notes = list(analysis.notes)
    selected_mode = mode or recommendation.mode
    if selected_mode == "fsdp2_tp":
        if (
            "dp" not in resolved_topology.ordered_dim_names
            or "tp" not in resolved_topology.ordered_dim_names
        ):
            raise ValueError(
                "Hybrid 'fsdp2_tp' mode requires both 'dp' and 'tp' dimensions in the topology."
            )
    if (
        selected_mode in ("tp", "fsdp2_tp")
        and not analysis.tensor_parallel_candidate_names
    ):
        raise ValueError(
            f"mode='{selected_mode}' requires at least one tensor-parallel candidate, but analysis found none."
        )
    if tensor_parallel_plan is not None:
        if selected_mode not in ("tp", "fsdp2_tp"):
            raise ValueError(
                "tensor_parallel_plan requires mode='tp' or mode='fsdp2_tp'."
            )
        if not tensor_parallel_plan:
            raise ValueError("tensor_parallel_plan must not be empty.")
        unknown = set(tensor_parallel_plan) - set(
            analysis.tensor_parallel_candidate_names
        )
        if unknown:
            raise ValueError(
                "tensor_parallel_plan contains modules outside the analysis "
                f"candidates: {sorted(unknown)!r}."
            )
    optimizer_strategy = recommendation.optimizer_sharding
    if selected_mode != "dp" or not features.allow_zero_optimizer:
        optimizer_strategy = "none"
        if selected_mode == "dp" and not features.allow_zero_optimizer:
            notes.append(
                "Zero optimizer was disabled by DistributedFeatureSet; planner fell back to optimizer_strategy='none'."
            )
    mesh_shape = resolved_topology.mesh_shape
    pp_microbatches = recommendation.pp_microbatches
    pp_schedule = recommendation.pp_schedule
    pp_virtual_stages = recommendation.pp_virtual_stages
    pp_layout = recommendation.pp_layout
    pp_delay_wgrad = recommendation.pp_delay_wgrad
    return SNNDistributedPlan(
        mode=selected_mode,
        objective=objective,
        topology=resolved_topology,
        model_family=resolved_model_family,
        backend=backend,
        batch_size=batch_size,
        optimizer_strategy=optimizer_strategy,
        memopt_level=recommendation.memopt_level,
        rationale=tuple(recommendation.rationale),
        notes=tuple(notes),
        tensor_parallel_roots=analysis.tensor_parallel_roots,
        tensor_parallel_plan=tensor_parallel_plan,
        mesh_shape=mesh_shape,
        tp_mesh_dim=tp_mesh_dim,
        dp_mesh_dim=dp_mesh_dim,
        pp_microbatches=pp_microbatches,
        pp_schedule=pp_schedule,
        pp_virtual_stages=pp_virtual_stages,
        pp_layout=pp_layout,
        pp_delay_wgrad=pp_delay_wgrad,
        experimental_features=features,
    )


def apply(
    *,
    model: nn.Module,
    plan: SNNDistributedPlan,
    device_type: str = "cuda",
    device_mesh=None,
) -> SNNDistributedRuntime:
    """Apply an eager distributed plan to a model.

    .. admonition:: Chinese

        将 :class:`SNNDistributedPlan` 应用到模型并返回包含已包装模型、mesh 和
        分析结果的运行时对象。

    :param model: Model to configure. DDP-style ``.module`` wrappers are unwrapped.
    :type model: torch.nn.Module
    :param plan: Plan returned by :func:`plan`.
    :type plan: SNNDistributedPlan
    :param device_type: Device type used when constructing a mesh.
    :type device_type: str
    :param device_mesh: Optional pre-built PyTorch ``DeviceMesh``.
    :return: Runtime wrapper for the configured model.
    :rtype: SNNDistributedRuntime
    """
    wrapped = getattr(model, "module", None)
    if isinstance(wrapped, nn.Module):
        model = wrapped

    topology = (
        plan.topology
        if isinstance(plan.topology, SNNDistributedTopology)
        else SNNDistributedTopology.from_mapping(plan.topology)
    )
    if device_mesh is not None:
        mesh_tensor = getattr(device_mesh, "mesh", None)
        mesh_volume = None
        if mesh_tensor is not None:
            mesh_volume = int(mesh_tensor.numel())
        elif hasattr(device_mesh, "size"):
            try:
                mesh_volume = int(device_mesh.size())
            except TypeError:
                mesh_volume = None
        if mesh_volume is not None and mesh_volume != topology.world_size:
            raise ValueError(
                f"device_mesh spans {mesh_volume} ranks, but plan.topology.world_size={topology.world_size}."
            )
    if plan.mode == "pp":
        raise NotImplementedError(
            "Pipeline parallelism ('pp') is not supported via the unified `apply` API "
            "because it requires an `example_input` to partition the model and measure stage costs. "
            "Please use the dedicated pipeline configuration path directly."
        )
    use_adapter = plan.mode in ("tp", "fsdp2", "fsdp2_tp") or (
        plan.experimental_features.allow_experimental_conv_tp
        or plan.experimental_features.allow_experimental_spikformer_tp
    )
    adapter = resolve_adapter(model, plan.model_family) if use_adapter else None
    if adapter is not None:
        return adapter.apply(
            model,
            plan,
            device_type=device_type,
            device_mesh=device_mesh,
        )
    config = build_eager_config(
        mode=plan.mode,
        device_type=device_type,
        mesh_shape=plan.mesh_shape or topology.mesh_shape,
        device_mesh=device_mesh,
        tp_mesh_dim=plan.tp_mesh_dim,
        dp_mesh_dim=plan.dp_mesh_dim,
        tensor_parallel_roots=plan.tensor_parallel_roots,
        tensor_parallel_plan=plan.tensor_parallel_plan,
        auto_tensor_parallel=(
            plan.mode in ("tp", "fsdp2_tp") and plan.tensor_parallel_plan is None
        ),
    )
    configured_model, mesh, analysis = configure_snn_distributed(model, config)
    return SNNDistributedRuntime(
        kind="eager",
        model=configured_model,
        mesh=mesh,
        analysis=analysis,
        plan=plan,
    )
