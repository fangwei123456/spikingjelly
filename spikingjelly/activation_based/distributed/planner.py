from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

from .topology import SNNDistributedTopology


@dataclass(frozen=True)
class DistributedFeatureSet:
    allow_experimental_conv_tp: bool = False
    allow_experimental_spikformer_tp: bool = False
    allow_pipeline: bool = True
    allow_zero_optimizer: bool = True


@dataclass(frozen=True)
class SNNDistributedPlan:
    mode: str
    objective: str
    topology: SNNDistributedTopology
    model_family: str
    backend: str
    batch_size: int
    optimizer_strategy: str
    memopt_level: int
    rationale: Tuple[str, ...]
    notes: Tuple[str, ...]
    mesh_shape: Optional[Tuple[int, ...]] = None
    tp_mesh_dim: int = 0
    dp_mesh_dim: Optional[int] = None
    pp_microbatches: Optional[int] = None
    pp_schedule: str = "1f1b"
    pp_virtual_stages: int = 1
    pp_layout: Optional[Tuple[int, ...]] = None
    pp_delay_wgrad: bool = False
    experimental_features: DistributedFeatureSet = DistributedFeatureSet()
