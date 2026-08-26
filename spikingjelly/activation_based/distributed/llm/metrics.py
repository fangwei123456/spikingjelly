from __future__ import annotations

from typing import Any

import torch


def _loss_totals(
    losses: list[dict[str, torch.Tensor]],
) -> dict[str, tuple[float, float]]:
    totals: dict[str, float] = {}
    counts: dict[str, float] = {}
    for loss in losses:
        for name, value in loss.items():
            values = value.detach().float().reshape(-1)
            if values.numel() == 2:
                total, count = values.tolist()
            else:
                total, count = values.mean().item(), 1.0
            totals[name] = totals.get(name, 0.0) + total
            counts[name] = counts.get(name, 0.0) + count
    return {name: (total, counts[name]) for name, total in totals.items()}


def _broadcast_pipeline_metrics(
    metrics: dict[str, tuple[float, float]],
    parallel_state: Any,
    device: torch.device,
) -> dict[str, tuple[float, float]]:
    values: list[Any] = [metrics if parallel_state.is_pipeline_last_stage() else None]
    torch.distributed.broadcast_object_list(
        values,
        src=parallel_state.get_pipeline_model_parallel_last_rank(),
        group=parallel_state.get_pipeline_model_parallel_group(),
        device=device,
    )
    return values[0]


def _reduce_data_parallel_metrics(
    totals: dict[str, tuple[float, float]],
    parallel_state: Any,
    device: torch.device,
) -> dict[str, float]:
    if not totals:
        return {}
    names = sorted(totals)
    values = torch.tensor([totals[name] for name in names], device=device)
    torch.distributed.all_reduce(
        values,
        group=parallel_state.get_data_parallel_group(with_context_parallel=True),
    )
    if torch.any(values[:, 1] <= 0):
        raise ValueError("Loss metric counts must be positive.")
    return dict(zip(names, (values[:, 0] / values[:, 1]).tolist(), strict=True))
