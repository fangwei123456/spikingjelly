from __future__ import annotations

from typing import TYPE_CHECKING

import torch

from spikingjelly.activation_based.distributed.llm.temporal import (
    _reduce_time_batch,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator

    from megatron.core.transformer import MegatronModule


def forward_step(
    data_iterator: "Iterator[dict[str, torch.Tensor]]",
    model: "MegatronModule",
) -> tuple[torch.Tensor, "Callable"]:
    r"""Run one temporal causal-language-model microbatch under MCore."""
    from megatron.core import parallel_state
    from megatron.core.utils import get_attr_wrapped_model, get_batch_on_this_cp_rank

    batch = next(data_iterator)
    device = torch.device("cuda", torch.cuda.current_device())
    batch = {key: value.to(device, non_blocking=True) for key, value in batch.items()}
    batch["position_ids"] = torch.arange(
        batch["input_ids"].shape[1], device=device
    ).expand_as(batch["input_ids"])
    if parallel_state.get_context_parallel_world_size() > 1:
        batch = get_batch_on_this_cp_rank(
            batch,
            is_hybrid_cp=False,
            cp_group=parallel_state.get_context_parallel_group(),
        )
    input_ids = batch["input_ids"]
    labels = batch["labels"]
    loss_mask = batch.get("loss_mask")
    if loss_mask is None:
        loss_mask = torch.ones_like(labels, dtype=torch.float32)
    position_ids = batch["position_ids"]
    time_steps = get_attr_wrapped_model(model, "snn_model_config").time_steps
    output = model(
        input_ids=input_ids.repeat(time_steps, 1),
        position_ids=position_ids.repeat(time_steps, 1),
        attention_mask=None,
        labels=None,
    )
    if parallel_state.is_pipeline_last_stage():
        output = _reduce_time_batch(
            output,
            time_steps,
            str(get_attr_wrapped_model(model, "temporal_output_reduction")),
        )

    def loss_function(semantic_logits: torch.Tensor):
        compute_loss = get_attr_wrapped_model(model, "compute_language_model_loss")
        token_losses = compute_loss(
            labels, semantic_logits.transpose(0, 1).contiguous()
        )
        loss_sum = (token_losses * loss_mask).sum()
        token_count = loss_mask.sum(dtype=torch.int32)
        return (
            loss_sum,
            token_count,
            {
                "lm_loss": torch.stack(
                    (loss_sum.detach(), token_count.detach().to(loss_sum.dtype))
                )
            },
        )

    return output, loss_function
