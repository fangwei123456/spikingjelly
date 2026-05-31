from __future__ import annotations

from typing import NamedTuple

import torch

from .dtensor import materialize_dtensor_output


class PreparedModelOutput(NamedTuple):
    logits: torch.Tensor
    target: torch.Tensor
    materialized: bool


def prepare_classification_output(
    output,
    target: torch.Tensor,
    *,
    require_full_logits: bool = True,
) -> PreparedModelOutput:
    # Reduce the time dimension before optional materialization so callers can
    # share one helper for eager tensors and DTensor outputs. The exact
    # communication behavior then follows the target PyTorch DTensor build.
    if output.ndim >= 3:
        output = output.mean(dim=0)
    if target.ndim > 1:
        target = target.squeeze(-1)
        if target.ndim > 1:
            target = target.argmax(dim=-1)
            if target.ndim > 1:
                target = target.squeeze(-1)
    output_device = getattr(output, "device", None)
    if output_device is not None and target.device != output_device:
        target = target.to(device=output_device)
    materialized = False
    if require_full_logits:
        materialized_output = materialize_dtensor_output(output)
        materialized = materialized_output is not output
        output = materialized_output
    return PreparedModelOutput(logits=output, target=target, materialized=materialized)
