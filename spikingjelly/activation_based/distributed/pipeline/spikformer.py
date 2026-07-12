from typing import Optional, Sequence, Union

import torch
import torch.distributed as dist
import torch.nn as nn

from spikingjelly.activation_based.distributed.pipeline.partition import (
    _partition_costs_contiguously,
    parse_pipeline_layout,
)
from spikingjelly.activation_based.distributed.pipeline.runtime import (
    SNNPipelineRuntime,
    _PipelineSequentialModule,
    _build_snn_pipeline_runtime,
    _measure_module_cost,
)


class _SpikformerPipelineStage(nn.Module):
    def __init__(
        self,
        *,
        patch_embed: Optional[nn.Module] = None,
        blocks: Sequence[nn.Module] = (),
        head: Optional[nn.Module] = None,
        T: Optional[int] = None,
    ):
        super().__init__()
        self.patch_embed = patch_embed
        self.blocks = nn.ModuleList(list(blocks))
        self.head = head
        self.T = T

    @staticmethod
    def _pool_features(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(3).mean(dim=-1)

    def forward(self, x: torch.Tensor):
        if self.patch_embed is not None:
            if x.ndim == 4:
                if self.T is None:
                    raise RuntimeError(
                        "Spikformer pipeline stage requires T for 4D inputs."
                    )
                x = x.unsqueeze(0).expand(self.T, -1, -1, -1, -1)
            elif x.ndim != 5:
                raise ValueError(
                    f"expected 4D [N, C, H, W] or 5D [T, N, C, H, W] input, but got {tuple(x.shape)}"
                )
            x = self.patch_embed(x)

        if self.blocks and x.ndim != 5:
            raise ValueError(
                "Spikformer pipeline stage expects 5D [T, N, D, H', W'] "
                f"input before blocks, but got {tuple(x.shape)}."
            )
        for block in self.blocks:
            x = block(x)

        if self.head is not None:
            x = self._pool_features(x)
            x = self.head(x)
        return x


def _build_spikformer_pipeline_module(
    module: nn.Module,
    num_logical_stages: int,
    example_input: torch.Tensor,
    layout_counts: Optional[Sequence[int]] = None,
) -> _PipelineSequentialModule:
    if num_logical_stages < 2:
        raise ValueError("Spikformer pipeline parallel requires at least 2 stages.")
    if not (
        hasattr(module, "patch_embed")
        and hasattr(module, "blocks")
        and hasattr(module, "head")
    ):
        raise TypeError(
            "Expected a Spikformer-like module with patch_embed, blocks, and head."
        )

    blocks = list(module.blocks)
    current = example_input
    if current.ndim == 4:
        stage_T = getattr(module, "T", None)
        if stage_T is None:
            raise RuntimeError(
                "Spikformer pipeline stage requires module.T to be set when a 4D example input is provided."
            )
        current = current.unsqueeze(0).expand(int(stage_T), -1, -1, -1, -1)
    elif current.ndim != 5:
        raise ValueError(
            f"expected 4D [N, C, H, W] or 5D [T, N, C, H, W] example input, but got {tuple(current.shape)}"
        )
    unit_costs: list[float] = []
    current, patch_cost = _measure_module_cost(module.patch_embed, current)
    unit_costs.append(patch_cost)
    for block in blocks:
        current, block_cost = _measure_module_cost(block, current)
        unit_costs.append(block_cost)
    head_input = _SpikformerPipelineStage._pool_features(current)
    _, head_cost = _measure_module_cost(module.head, head_input)
    unit_costs.append(head_cost)
    block_counts = (
        list(layout_counts)
        if layout_counts is not None
        else _partition_costs_contiguously(unit_costs, num_logical_stages)
    )
    first_active_stage_idx = next(
        (idx for idx, count in enumerate(block_counts) if count > 0), None
    )
    stages: list[nn.Module] = []
    cursor = 0
    unit_cursor = 0
    stage_costs: list[float] = []
    for stage_idx, count in enumerate(block_counts):
        patch_embed = (
            module.patch_embed
            if stage_idx == first_active_stage_idx and count > 0
            else None
        )
        units_remaining = count
        if patch_embed is not None:
            units_remaining -= 1
        block_take = min(max(units_remaining, 0), len(blocks) - cursor)
        stage_blocks = blocks[cursor : cursor + block_take]
        cursor += block_take
        units_remaining -= block_take
        head = module.head if units_remaining > 0 else None
        stages.append(
            _SpikformerPipelineStage(
                patch_embed=patch_embed,
                blocks=stage_blocks,
                head=head,
                T=getattr(module, "T", None) if patch_embed is not None else None,
            )
        )
        stage_costs.append(
            sum(float(cost) for cost in unit_costs[unit_cursor : unit_cursor + count])
        )
        unit_cursor += count
    pipeline_module = _PipelineSequentialModule(stages)
    pipeline_module.stage_costs = tuple(stage_costs)
    return pipeline_module


def configure_spikformer_pipeline(
    module: nn.Module,
    example_input: torch.Tensor,
    device: Union[str, torch.device],
    n_microbatches: int,
    pp_schedule: str = "auto",
    pp_virtual_stages: int = 1,
    pp_layout: Optional[Union[str, Sequence[int]]] = None,
    pp_delay_wgrad: bool = False,
    stage_index: Optional[int] = None,
    group=None,
) -> SNNPipelineRuntime:
    physical_num_stages = dist.get_world_size(group) if dist.is_initialized() else 1
    logical_num_stages = physical_num_stages * pp_virtual_stages
    total_units = len(getattr(module, "blocks", ())) + 2
    layout_counts = parse_pipeline_layout(pp_layout, logical_num_stages, total_units)
    pipeline_module = _build_spikformer_pipeline_module(
        module=module,
        num_logical_stages=logical_num_stages,
        example_input=example_input,
        layout_counts=layout_counts,
    )
    return _build_snn_pipeline_runtime(
        pipeline_module=pipeline_module,
        example_input=example_input,
        device=torch.device(device),
        n_microbatches=n_microbatches,
        stage_index=stage_index,
        model_family="spikformer",
        schedule_kind=pp_schedule,
        virtual_pipeline_size=pp_virtual_stages,
        delayed_wgrad=pp_delay_wgrad,
        pp_layout=layout_counts,
        group=group,
    )
