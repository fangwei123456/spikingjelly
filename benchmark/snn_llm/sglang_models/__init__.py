"""SGLang external model package for benchmark SNN LLM recipes."""

from collections.abc import Iterable

import torch
import torch.nn as nn

from sglang.srt.layers.utils import get_layer_id
from sglang.srt.model_loader.weight_utils import default_weight_loader


def _load_stage_weights(
    module: nn.Module,
    weights: Iterable[tuple[str, torch.Tensor]],
    model_name: str,
) -> None:
    params = dict(module.named_parameters())
    buffers = dict(module.named_buffers())
    for name, value in weights:
        layer_id = get_layer_id(name)
        if layer_id is not None and not (
            module.model.start_layer <= layer_id < module.model.end_layer
        ):
            continue
        target = params.get(name, buffers.get(name))
        if target is None:
            missing_embedding = not module.pp_group.is_first_rank and name.startswith(
                "model.embedding."
            )
            missing_output = not module.pp_group.is_last_rank and name.startswith(
                ("model.final_norm.", "lm_head.")
            )
            if missing_embedding or missing_output:
                continue
            raise KeyError(f"Unknown {model_name} artifact tensor: {name}")
        getattr(target, "weight_loader", default_weight_loader)(target, value)
