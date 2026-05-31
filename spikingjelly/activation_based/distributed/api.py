from __future__ import annotations

from typing import Mapping, Optional, Sequence, Union

import torch.nn as nn

from .adapters import resolve_adapter
from .dtensor import (
    SNNDistributedAnalysis,
    SNNDistributedConfig,
    SNN_DISTRIBUTED_PREFERENCES,
    analyze_snn_distributed_capability,
    configure_snn_distributed,
    recommend_snn_distributed_strategy,
)
from .planner import (
    DistributedFeatureSet,
    SNNDistributedPlan,
)
from .runtime import SNNDistributedRuntime
from .topology import SNNDistributedTopology


def analyze(
    model: nn.Module,
    *,
    model_family: Optional[str] = None,
    roots: Optional[Sequence[str]] = None,
) -> SNNDistributedAnalysis:
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
) -> SNNDistributedPlan:
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
    if mode is not None:
        mode = mode.lower()
        valid_modes = ("none", "dp", "tp", "fsdp2", "fsdp2_tp", "pp")
        if mode not in valid_modes:
            raise ValueError(
                f"Unsupported mode='{mode}'. Expected one of {valid_modes}."
            )
        if mode == "pp":
            raise NotImplementedError(
                "Pipeline parallelism ('pp') is not supported by the unified analyze/plan/apply API. "
                "Please use the dedicated pipeline configuration path directly."
            )
    notes = list(analysis.notes)
    selected_mode = mode or recommendation.mode
    if (
        selected_mode in ("tp", "fsdp2_tp")
        and not analysis.tensor_parallel_candidate_names
    ):
        raise ValueError(
            f"mode='{selected_mode}' requires at least one tensor-parallel candidate, but analysis found none."
        )
    optimizer_strategy = recommendation.optimizer_sharding
    if selected_mode != "dp" or not features.allow_zero_optimizer:
        optimizer_strategy = "none"
        if selected_mode == "dp" and not features.allow_zero_optimizer:
            notes.append(
                "Zero optimizer was disabled by DistributedFeatureSet; planner fell back to optimizer_strategy='none'."
            )
    mesh_shape = recommendation.mesh_shape or resolved_topology.mesh_shape
    pp_microbatches = recommendation.pp_microbatches
    pp_schedule = recommendation.pp_schedule
    pp_virtual_stages = recommendation.pp_virtual_stages
    pp_layout = recommendation.pp_layout
    pp_delay_wgrad = recommendation.pp_delay_wgrad
    if selected_mode in ("tp", "fsdp2", "fsdp2_tp", "none"):
        mesh_shape = resolved_topology.mesh_shape
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
    config = SNNDistributedConfig(
        device_type=device_type,
        mesh_shape=plan.mesh_shape or topology.mesh_shape,
        device_mesh=device_mesh,
        tp_mesh_dim=plan.tp_mesh_dim,
        dp_mesh_dim=plan.dp_mesh_dim,
        enable_data_parallel=plan.mode == "dp",
        enable_fsdp2=plan.mode in ("fsdp2", "fsdp2_tp"),
        tensor_parallel_roots=plan.tensor_parallel_roots,
        auto_tensor_parallel=plan.mode in ("tp", "fsdp2_tp"),
    )
    configured_model, mesh, analysis = configure_snn_distributed(model, config)
    return SNNDistributedRuntime(
        kind="eager",
        model=configured_model,
        mesh=mesh,
        analysis=analysis,
        plan=plan,
    )
