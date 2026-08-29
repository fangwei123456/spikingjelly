from __future__ import annotations

from collections.abc import Callable
from typing import Any, Literal, Optional

import torch
from torch.distributed import ProcessGroup
import torch.nn as nn

from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
)
from spikingjelly.activation_based.precision.config import PrecisionConfig


def _classification_logits(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output.mean(0)
    raise ValueError(
        "Vision classification models must return [N, C] or [T, N, C], "
        f"got {tuple(output.shape)}."
    )


def _classification_sequence(
    images: torch.Tensor,
    time_steps: int,
    input_layout: str,
    *,
    batch_first: bool = False,
) -> torch.Tensor:
    if input_layout == "NCHW":
        if images.ndim != 4:
            raise ValueError(
                "input_layout='NCHW' requires image batches shaped [N, C, H, W]."
            )
        time_dim = 1 if batch_first else 0
        shape = list(images.shape)
        shape.insert(time_dim, time_steps)
        return images.unsqueeze(time_dim).expand(*shape).contiguous()
    if images.ndim != 5:
        raise ValueError(
            "input_layout='NTCHW' requires image batches shaped [N, T, C, H, W]."
        )
    if images.shape[1] != time_steps:
        raise ValueError(
            f"input time dimension {images.shape[1]} does not match "
            f"model.time_steps={time_steps}."
        )
    return images.contiguous() if batch_first else images.transpose(0, 1).contiguous()


def _forward_classification(
    model: nn.Module,
    images: torch.Tensor,
    time_steps: int,
    step_mode: str,
    input_layout: str,
) -> torch.Tensor:
    if step_mode == "s" and input_layout == "NCHW":
        if images.ndim != 4:
            raise ValueError(
                "input_layout='NCHW' requires image batches shaped [N, C, H, W]."
            )
        return _classification_logits(
            torch.stack([model(images) for _ in range(time_steps)])
        )
    sequence = _classification_sequence(images, time_steps, input_layout)
    output = (
        torch.stack([model(x) for x in sequence])
        if step_mode == "s"
        else model(sequence)
    )
    return _classification_logits(output)


def _classification_loss(
    logits: torch.Tensor,
    targets: torch.Tensor,
    loss_function: Callable[..., torch.Tensor],
) -> torch.Tensor:
    loss = loss_function(logits, targets)
    if not isinstance(loss, torch.Tensor):
        raise TypeError("loss_function must return a torch.Tensor.")
    if loss.ndim != 0:
        raise ValueError("loss_function must return a scalar tensor.")
    return loss


def _pipeline_loss(
    output: torch.Tensor,
    target: torch.Tensor,
    *,
    loss_function: Callable[..., torch.Tensor],
) -> torch.Tensor:
    return _classification_loss(_classification_logits(output), target, loss_function)


def _wrap_data_parallel(
    model: nn.Module,
    *,
    data_parallel: Literal["ddp", "fsdp2"],
    pipeline_parallel_size: int,
    step_mode: str,
    precision: PrecisionConfig,
    device: torch.device,
    dp_size: int,
    dp_group: Optional[ProcessGroup],
    dp_mesh: Any,
    fsdp_roots: tuple[str, ...],
) -> nn.Module:
    if dp_size == 1:
        return model
    if data_parallel == "ddp":
        if pipeline_parallel_size > 1:
            return model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            process_group=dp_group,
            broadcast_buffers=step_mode == "m",
        )

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dtype = {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[precision.mode]
    policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=dtype,
        output_dtype=dtype,
    )
    if dtype is not None:
        batch_norm_policy = MixedPrecisionPolicy(output_dtype=dtype)
        for module in model.modules():
            if isinstance(
                module,
                (
                    nn.modules.batchnorm._BatchNorm,
                    ChannelShardBatchNorm1d,
                    ChannelShardBatchNorm2d,
                ),
            ):
                fully_shard(module, mesh=dp_mesh, mp_policy=batch_norm_policy)
    named_modules = dict(model.named_modules())
    for name in fsdp_roots:
        if name not in named_modules:
            raise KeyError(f"Unknown FSDP2 root {name!r} returned by model builder.")
        fully_shard(named_modules[name], mesh=dp_mesh, mp_policy=policy)
    fully_shard(model, mesh=dp_mesh, mp_policy=policy, reshard_after_forward=False)
    return model
