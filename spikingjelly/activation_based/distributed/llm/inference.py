"""Distributed inference helpers for SNN language models."""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import torch

from .temporal import _reduce_time_batch

if TYPE_CHECKING:
    from collections.abc import Callable

    from megatron.core.transformer import MegatronModule, TransformerConfig


def load_for_inference(
    transformer_config: "TransformerConfig",
    model_provider: "Callable[[bool, bool], MegatronModule]",
    checkpoint: Path,
) -> "MegatronModule":
    r"""Load one initialized MCore pipeline stage for inference.

    **API Language** - 中文 | English

    **中文：** 在调用方已初始化的 MCore process groups 上构建当前 PP stage，加载
    sharded model checkpoint，并返回 eval model。本函数不加载 optimizer、scheduler
    或训练 RNG，也不拥有 process-group 生命周期。

    :param transformer_config: 与 checkpoint 兼容的 MCore Transformer 配置。
    :type transformer_config: megatron.core.transformer.TransformerConfig
    :param model_provider: 接收 ``pre_process`` 和 ``post_process`` 的 model provider。
    :type model_provider: Callable
    :param checkpoint: optimizer-step checkpoint 目录。
    :type checkpoint: pathlib.Path
    :return: 当前 PP stage 的已加载模型。
    :rtype: megatron.core.transformer.MegatronModule
    :raises RuntimeError: torch.distributed 或 MCore process groups 尚未初始化。
    :raises ValueError: checkpoint recipe 与模型不匹配。

    **English:** Build the current PP stage on caller-initialized MCore process
    groups, load its sharded model checkpoint, and return an eval model. This
    function loads no optimizer, scheduler, or training RNG and does not own the
    process-group lifecycle.

    :param transformer_config: MCore Transformer configuration compatible with the checkpoint.
    :type transformer_config: megatron.core.transformer.TransformerConfig
    :param model_provider: Model provider accepting ``pre_process`` and ``post_process``.
    :type model_provider: Callable
    :param checkpoint: Optimizer-boundary checkpoint directory.
    :type checkpoint: pathlib.Path
    :return: Loaded model for the current PP stage.
    :rtype: megatron.core.transformer.MegatronModule
    :raises RuntimeError: If torch.distributed or MCore process groups are uninitialized.
    :raises ValueError: If the checkpoint recipe does not match the model.
    """
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.transformer.module import Float16Module
    from megatron.core.utils import unwrap_model

    if (
        not torch.distributed.is_initialized()
        or not parallel_state.model_parallel_is_initialized()
    ):
        raise RuntimeError(
            "Inference loading requires initialized MCore process groups."
        )
    model = model_provider(
        parallel_state.is_pipeline_first_stage(),
        parallel_state.is_pipeline_last_stage(),
    )
    model.cuda(torch.cuda.current_device())
    if transformer_config.fp16 or transformer_config.bf16:
        model = Float16Module(transformer_config, model)
    metadata = {
        "dp_cp_group": parallel_state.get_data_parallel_group(
            with_context_parallel=True
        )
    }
    unwrapped = unwrap_model(model)
    state = dist_checkpointing.load(
        {"model": unwrapped.sharded_state_dict(metadata=metadata)}, str(checkpoint)
    )
    expected = getattr(unwrapped, "checkpoint_metadata", None)
    if isinstance(expected, dict):
        model_config = unwrapped.snn_model_config
        expected = {
            **expected,
            "model": model_config._checkpoint_metadata(),
            "use_snn_memopt": unwrapped.snn_memopt_enabled,
            "mcore_recompute_granularity": transformer_config.recompute_granularity,
            "mcore_recompute_modules": transformer_config.recompute_modules,
        }
    recipe = state.get("recipe", {})
    if not isinstance(expected, dict) or any(
        recipe.get(key) != value for key, value in expected.items()
    ):
        raise ValueError("Checkpoint recipe does not match the inference model.")
    unwrapped.load_state_dict(state["model"])
    model.eval()
    return model


def generate(
    transformer_config: "TransformerConfig",
    model_provider: "Callable[[bool, bool], MegatronModule]",
    checkpoint: Path,
    input_ids: torch.Tensor,
    max_new_tokens: int,
) -> Optional[torch.Tensor]:
    r"""Greedily generate tokens with MCore TP/PP execution and a KV cache.

    **API Language** - 中文 | English

    **中文：** 广播 rank 0 的 ``[B,S]`` prompt，使用 MCore static inference context
    完成 prefill 与 cached decode。每次模型调用包含完整 ``T``，神经元状态在调用后
    丢弃；KV cache 跨 decode-step ``D`` 保留。仅 global rank 0 返回 CPU token。

    :param transformer_config: 推理用 MCore 配置；当前要求 CP=1 且关闭 sequence parallel。
    :type transformer_config: megatron.core.transformer.TransformerConfig
    :param model_provider: MCore model provider。
    :type model_provider: Callable
    :param checkpoint: 要加载的 sharded checkpoint。
    :type checkpoint: pathlib.Path
    :param input_ids: rank 0 上形状 ``[B,S]`` 的整数 prompt；其他 rank 内容被覆盖。
    :type input_ids: torch.Tensor
    :param max_new_tokens: 要生成的正整数 token 数。
    :type max_new_tokens: int
    :return: global rank 0 上的 ``[B,S+max_new_tokens]`` CPU tensor；其他 rank 为 ``None``。
    :rtype: Optional[torch.Tensor]
    :raises ValueError: 输入、上下文长度或推理并行配置无效。
    :raises RuntimeError: CUDA 或 MCore 不可用。

    **English:** Broadcast rank 0's ``[B,S]`` prompt and run prefill plus cached
    decode with MCore's static inference context. Every model call contains a full
    ``T`` window and discards neuron state afterwards, while the KV cache persists
    across decode steps ``D``. Only global rank 0 returns CPU tokens.

    :param transformer_config: Inference MCore config; CP must be one and sequence parallel disabled.
    :type transformer_config: megatron.core.transformer.TransformerConfig
    :param model_provider: MCore model provider.
    :type model_provider: Callable
    :param checkpoint: Sharded checkpoint to load.
    :type checkpoint: pathlib.Path
    :param input_ids: Integer ``[B,S]`` prompt on rank 0; other-rank contents are overwritten.
    :type input_ids: torch.Tensor
    :param max_new_tokens: Positive number of tokens to generate.
    :type max_new_tokens: int
    :return: ``[B,S+max_new_tokens]`` CPU tensor on global rank 0, otherwise ``None``.
    :rtype: Optional[torch.Tensor]
    :raises ValueError: If inputs, context length, or inference parallelism are invalid.
    :raises RuntimeError: If CUDA or MCore is unavailable.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MCore distributed inference requires CUDA.")
    if max_new_tokens <= 0:
        raise ValueError("max_new_tokens must be positive.")
    if transformer_config.context_parallel_size != 1:
        raise ValueError("Cached decode currently requires context_parallel_size=1.")
    if transformer_config.sequence_parallel:
        raise ValueError("MCore static inference requires sequence_parallel=False.")

    from megatron.core import parallel_state
    from megatron.core.inference.contexts import StaticInferenceContext
    from megatron.core.inference.model_inference_wrappers.gpt.gpt_inference_wrapper import (
        GPTInferenceWrapper,
    )
    from megatron.core.inference.utils import InferenceMode
    from megatron.core.utils import get_attr_wrapped_model

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    owns_distributed = not torch.distributed.is_initialized()
    if owns_distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=device)
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    try:
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=transformer_config.tensor_model_parallel_size,
                pipeline_model_parallel_size=transformer_config.pipeline_model_parallel_size,
                context_parallel_size=1,
                expert_model_parallel_size=1,
            )
        elif (
            parallel_state.get_tensor_model_parallel_world_size()
            != transformer_config.tensor_model_parallel_size
            or parallel_state.get_pipeline_model_parallel_world_size()
            != transformer_config.pipeline_model_parallel_size
            or parallel_state.get_context_parallel_world_size() != 1
        ):
            raise ValueError(
                "Existing MCore process groups do not match the inference topology."
            )
        model = load_for_inference(transformer_config, model_provider, checkpoint)
        model_config = get_attr_wrapped_model(model, "snn_model_config")
        time_steps = model_config.time_steps
        reduction = str(get_attr_wrapped_model(model, "temporal_output_reduction"))
        rank = torch.distributed.get_rank()
        valid_input = (
            rank == 0
            and input_ids.dim() == 2
            and input_ids.numel() > 0
            and input_ids.dtype
            in (torch.uint8, torch.int8, torch.int16, torch.int32, torch.int64)
        )
        shape = torch.tensor(
            (*input_ids.shape, 1) if valid_input else (0, 0, 0),
            device=device,
            dtype=torch.long,
        )
        torch.distributed.broadcast(shape, src=0)
        batch_size, prompt_length, valid_input = shape.tolist()
        if not valid_input:
            raise ValueError(
                "input_ids must be a non-empty integer [B,S] tensor on rank 0."
            )
        max_sequence_length = model_config.max_sequence_length
        if prompt_length + max_new_tokens > max_sequence_length:
            raise ValueError("Requested generation exceeds the model context window.")
        tokens = (
            input_ids.to(device=device, dtype=torch.long)
            if rank == 0
            else torch.empty(
                (batch_size, prompt_length), device=device, dtype=torch.long
            )
        )
        torch.distributed.broadcast(tokens, src=0)
        context = StaticInferenceContext(
            max_batch_size=time_steps * batch_size,
            max_sequence_length=max_sequence_length,
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.prep_model_for_inference()
        temporal_prompt = tokens.repeat(time_steps, 1)
        inference_input = wrapper.prep_inference_input(temporal_prompt)
        output_rank = torch.distributed.get_world_size() - 1
        with InferenceMode.active():
            logits = wrapper.run_one_forward_step(inference_input)
            context.enable_decode_mode()
            generated = tokens
            for step in range(max_new_tokens):
                next_token = torch.empty(batch_size, device=device, dtype=torch.long)
                if rank == output_rank:
                    semantic_logits = _reduce_time_batch(
                        logits[:, -1], time_steps, reduction
                    )
                    next_token.copy_(semantic_logits.argmax(dim=-1))
                torch.distributed.broadcast(next_token, src=output_rank)
                generated = torch.cat((generated, next_token[:, None]), dim=1)
                if step + 1 == max_new_tokens:
                    break
                position = context.sequence_len_offset
                decode_input = {
                    "tokens": next_token[:, None].repeat(time_steps, 1),
                    "position_ids": torch.full(
                        (time_steps * batch_size, 1),
                        position,
                        device=device,
                        dtype=torch.long,
                    ),
                    "attention_mask": None,
                }
                logits = wrapper.run_one_forward_step(decode_input)
        return generated.cpu() if rank == 0 else None
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


__all__ = ["generate", "load_for_inference"]
