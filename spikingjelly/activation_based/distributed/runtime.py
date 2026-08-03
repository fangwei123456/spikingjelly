from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from spikingjelly.activation_based import functional
from spikingjelly.logger import logger

from .analysis import SNNDistributedAnalysis
from .metrics import PreparedModelOutput, prepare_classification_output
from .mesh import resolve_data_parallel_partition
from .optimizer import build_snn_optimizer
from .planner import SNNDistributedPlan


@dataclass
class SNNDistributedRuntime:
    model: nn.Module
    mesh: Optional[object]
    analysis: Optional[SNNDistributedAnalysis]
    plan: SNNDistributedPlan

    def build_optimizer(
        self,
        optimizer_cls=torch.optim.Adam,
        lr: float = 1e-3,
        weight_decay: float = 0.0,
        **kwargs,
    ):
        return build_snn_optimizer(
            self.model,
            mode=self.plan.mode,
            optimizer_cls=optimizer_cls,
            lr=lr,
            weight_decay=weight_decay,
            optimizer_sharding=self.plan.optimizer_strategy,
            **kwargs,
        )

    def reset_state(self):
        r"""
        **API Language** - :ref:`中文 <reset_state-cn>` | :ref:`English <reset_state-en>`

        ----

        .. _reset_state-cn:

        * **中文**

        重置模型中所有有状态模块（如神经元膜电位）。

        ----

        .. _reset_state-en:

        * **English**

        Reset all stateful modules in the model (e.g. neuron membrane potentials).
        """
        functional.reset_net(self.model)

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
        replicas = 1
        rank = 0
        if dist.is_initialized():
            replicas, rank = resolve_data_parallel_partition(
                self.mesh,
                dp_mesh_dim=self.plan.dp_mesh_dim,
                sharded_by_data_parallel=self.plan.mode in ("dp", "fsdp2", "fsdp2_tp"),
            )
            if replicas > 1:
                sampler = DistributedSampler(
                    dataset,
                    num_replicas=replicas,
                    rank=rank,
                    shuffle=shuffle,
                )
        data_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=sampler,
            shuffle=shuffle if sampler is None else False,
            num_workers=num_workers,
            pin_memory=pin_memory,
            drop_last=drop_last,
        )
        logger.info(
            "distributed_dataloader_prepared dataset=%s batch_size=%s num_workers=%s sampler=%s replicas=%s rank=%s drop_last=%s pin_memory=%s",
            type(dataset).__name__,
            batch_size,
            num_workers,
            type(sampler).__name__ if sampler is not None else None,
            replicas,
            rank,
            drop_last,
            pin_memory,
        )
        return data_loader
