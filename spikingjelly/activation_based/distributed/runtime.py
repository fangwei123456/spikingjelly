from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from spikingjelly.activation_based import functional

from .dtensor import (
    SNNDistributedAnalysis,
    SNNPipelineRuntime,
    build_snn_optimizer,
    resolve_data_parallel_partition,
)
from .metrics import PreparedModelOutput, prepare_classification_output
from .planner import DistributedFeatureSet, SNNDistributedPlan
from .topology import SNNDistributedTopology


@dataclass
class SNNDistributedRuntime:
    kind: str
    model: nn.Module
    mesh: Optional[object]
    analysis: Optional[SNNDistributedAnalysis]
    plan: Optional[SNNDistributedPlan] = None
    mode: str = "none"
    pipeline_runtime: Optional[SNNPipelineRuntime] = None

    @classmethod
    def from_legacy(
        cls,
        *,
        kind: str,
        model: nn.Module,
        mesh: Optional[object],
        analysis: Optional[SNNDistributedAnalysis],
        mode: str,
        pipeline_runtime: Optional[SNNPipelineRuntime] = None,
    ) -> "SNNDistributedRuntime":
        topology = SNNDistributedTopology.from_mapping({"dp": 1})
        if mesh is not None:
            mesh_shape = getattr(mesh, "shape", None)
            if mesh_shape is None:
                mesh_tensor = getattr(mesh, "mesh", None)
                mesh_shape = getattr(mesh_tensor, "shape", None)
            if mesh_shape is not None:
                dims = tuple(int(size) for size in mesh_shape)
                if dims:
                    dim_names = ("dp", "tp", "pp", "vpp")
                    mapping = {
                        dim_names[idx] if idx < len(dim_names) else f"dim{idx}": size
                        for idx, size in enumerate(dims)
                    }
                    topology = SNNDistributedTopology.from_mapping(mapping)
        plan = SNNDistributedPlan(
            mode=mode,
            objective="legacy",
            topology=topology,
            model_family="legacy",
            backend="legacy",
            batch_size=0,
            optimizer_strategy="none",
            memopt_level=0,
            rationale=(),
            notes=("Constructed from legacy runtime bridge.",),
            experimental_features=DistributedFeatureSet(),
        )
        return cls(
            kind=kind,
            model=model,
            mesh=mesh,
            analysis=analysis,
            plan=plan,
            mode=mode,
            pipeline_runtime=pipeline_runtime,
        )

    def build_optimizer(
        self,
        optimizer_cls=torch.optim.Adam,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        target = self.model
        if self.kind == "pipeline" and self.pipeline_runtime is not None:
            target = self.pipeline_runtime.stage_module
        mode = self.plan.mode if self.plan is not None else self.mode
        optimizer_strategy = (
            self.plan.optimizer_strategy if self.plan is not None else "none"
        )
        return build_snn_optimizer(
            target,
            mode=mode,
            optimizer_cls=optimizer_cls,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_sharding=optimizer_strategy,
            **kwargs,
        )

    def reset_state(self):
        target = self.model
        if self.kind == "pipeline" and self.pipeline_runtime is not None:
            target = self.pipeline_runtime.stage_module
        functional.reset_net(target)

    @staticmethod
    def reduce_classification_output(
        outputs: torch.Tensor,
        labels: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        prepared = prepare_classification_output(
            outputs,
            labels,
            require_full_logits=False,
        )
        return prepared.logits, prepared.target

    def prepare_classification_output(
        self,
        outputs,
        labels: torch.Tensor,
        *,
        return_metadata: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor] | PreparedModelOutput:
        prepared = prepare_classification_output(
            outputs,
            labels,
            require_full_logits=True,
        )
        if return_metadata:
            return prepared
        return prepared.logits, prepared.target

    def forward_loss(
        self,
        criterion,
        images: torch.Tensor,
        labels: torch.Tensor,
    ):
        try:
            param = next(self.model.parameters())
            dtype = param.dtype
            device = param.device
        except StopIteration:
            dtype = torch.float32
            device = torch.device("cpu")
        outputs = self.model(images.to(device=device, dtype=dtype))
        outputs, labels = self.prepare_classification_output(outputs, labels)
        loss = criterion(outputs, labels)
        return outputs, labels, loss

    def prepare_dataloader(
        self,
        *,
        dataset,
        batch_size: int,
        shuffle: bool,
        num_workers: int,
        drop_last: bool,
        pin_memory: bool = True,
    ) -> DataLoader:
        sampler = None
        if dist.is_initialized():
            if self.kind == "pipeline":
                sampler = DistributedSampler(
                    dataset, num_replicas=1, rank=0, shuffle=shuffle
                )
            else:
                dp_mesh_dim = self.plan.dp_mesh_dim if self.plan is not None else None
                sharded = (
                    self.plan.mode in ("dp", "fsdp2", "fsdp2_tp")
                    if self.plan is not None
                    else self.mode in ("dp", "fsdp2", "fsdp2_tp")
                )
                replicas, rank = resolve_data_parallel_partition(
                    self.mesh,
                    dp_mesh_dim=dp_mesh_dim,
                    sharded_by_data_parallel=sharded,
                )
                if replicas > 1:
                    sampler = DistributedSampler(
                        dataset,
                        num_replicas=replicas,
                        rank=rank,
                        shuffle=shuffle,
                    )
        return DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
