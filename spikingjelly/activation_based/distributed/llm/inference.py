"""Distributed inference helpers for SNN language models."""

from __future__ import annotations

import importlib
import math
import os
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any, Optional

import torch
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from .config import EvaluationConfig, MCoreGenerationConfig
from .metrics import (
    _broadcast_pipeline_metrics,
    _loss_totals,
    _reduce_data_parallel_metrics,
)
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


def _import_object(path: str) -> Any:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


class _EvaluationDataset(Dataset):
    def __init__(
        self, dataset: Dataset, padded_size: int, sequence_length: int
    ) -> None:
        self.dataset = dataset
        self.padded_size = padded_size
        self.sequence_length = sequence_length

    def __len__(self) -> int:
        return self.padded_size

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        valid = index < len(self.dataset)
        item = dict(self.dataset[index if valid else 0])
        if set(item) - {"input_ids", "labels", "loss_mask"} or not {
            "input_ids",
            "labels",
        } <= set(item):
            raise ValueError(
                "Evaluation samples must contain input_ids, labels, and optional loss_mask."
            )
        if any(
            not isinstance(item[name], torch.Tensor)
            or item[name].shape != (self.sequence_length,)
            for name in ("input_ids", "labels")
        ):
            raise ValueError(
                "input_ids and labels must be one-dimensional [S] tensors."
            )
        mask = item.get("loss_mask")
        if mask is None:
            mask = torch.ones(self.sequence_length, dtype=torch.float32)
        elif not isinstance(mask, torch.Tensor) or mask.shape != (
            self.sequence_length,
        ):
            raise ValueError("loss_mask must be a one-dimensional [S] tensor.")
        item["loss_mask"] = mask if valid else torch.zeros_like(mask)
        return item


def _perplexity(loss: float) -> float:
    try:
        return math.exp(loss)
    except OverflowError:
        return float("inf")


def evaluate(config: EvaluationConfig) -> dict[str, float]:
    r"""Evaluate loss and perplexity with MCore DP/TP/PP/CP execution.

    **API Language** - 中文 | English

    **中文：** 初始化 NCCL 与 MCore process groups，仅恢复 checkpoint
    中的 model state，对 dataset builder 返回的完整数据集执行
    forward-only schedule。填充样本的 ``loss_mask`` 为零，不进入 loss、
    perplexity 或有效 token 计数。

    **English:** Initialize NCCL and MCore process groups, restore model state
    only, and run a forward-only schedule over the complete dataset returned by
    the configured builder. Padded samples receive a zero ``loss_mask`` and do
    not contribute to loss, perplexity, or valid-token counts.

    :param config: 评测配置。 / Evaluation configuration.
    :type config: spikingjelly.activation_based.distributed.llm.EvaluationConfig
    :return: Loss, perplexity, throughput, memory, and topology metrics.
    :rtype: dict[str, float]
    :raises RuntimeError: CUDA or MCore is unavailable.
    :raises ValueError: The topology, dataset, or sample tensors are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MCore distributed evaluation requires CUDA.")
    builder_cls = config.model.get_builder_cls()
    model_provider, forward_step = builder_cls(config.model).build(
        resume=True,
    )
    dataset = _import_object(config.dataset_builder)(**config.dataset_kwargs)
    if not isinstance(dataset, Dataset) or len(dataset) == 0:
        raise ValueError("dataset_builder must return one non-empty Dataset.")

    try:
        from megatron.core import parallel_state
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    except ImportError as error:
        raise RuntimeError(
            "MCore evaluation requires Python 3.12 and spikingjelly[megatron]."
        ) from error
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    owns_distributed = not torch.distributed.is_initialized()
    if owns_distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=device)
    owns_model_parallel = not parallel_state.model_parallel_is_initialized()
    try:
        transformer = config.model.transformer
        if owns_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=transformer.tensor_model_parallel_size,
                pipeline_model_parallel_size=transformer.pipeline_model_parallel_size,
                context_parallel_size=transformer.context_parallel_size,
                expert_model_parallel_size=1,
            )
        elif (
            parallel_state.get_tensor_model_parallel_world_size()
            != transformer.tensor_model_parallel_size
            or parallel_state.get_pipeline_model_parallel_world_size()
            != transformer.pipeline_model_parallel_size
            or parallel_state.get_context_parallel_world_size()
            != transformer.context_parallel_size
        ):
            raise ValueError(
                "Existing MCore process groups do not match evaluation TP/PP/CP."
            )
        model_parallel_size = (
            transformer.tensor_model_parallel_size
            * transformer.pipeline_model_parallel_size
            * transformer.context_parallel_size
        )
        world_size = torch.distributed.get_world_size()
        if world_size % model_parallel_size:
            raise ValueError("world_size must be divisible by TP * PP * CP.")
        data_parallel_size = world_size // model_parallel_size
        local_batch_size = config.micro_batch_size * config.pipeline_microbatches
        multiple = data_parallel_size * local_batch_size
        padded_size = ((len(dataset) + multiple - 1) // multiple) * multiple
        loader = DataLoader(
            _EvaluationDataset(dataset, padded_size, config.sequence_length),
            batch_size=config.micro_batch_size,
            sampler=DistributedSampler(
                range(padded_size),
                num_replicas=data_parallel_size,
                rank=parallel_state.get_data_parallel_rank(),
                shuffle=False,
                drop_last=False,
            ),
            drop_last=True,
        )

        model_parallel_cuda_manual_seed(config.seed)
        model = load_for_inference(transformer, model_provider, config.checkpoint)
        models = [model]
        schedule = get_forward_backward_func()
        losses = []
        schedule_kwargs = {
            "forward_step_func": forward_step,
            "model": models,
            "num_microbatches": config.pipeline_microbatches,
            "seq_length": config.sequence_length,
            "micro_batch_size": config.micro_batch_size * config.model.time_steps,
            "decoder_seq_length": config.sequence_length,
            "forward_only": True,
        }

        def next_schedule(data_iterator):
            batches = [next(data_iterator) for _ in range(config.pipeline_microbatches)]
            return schedule(data_iterator=iter(batches), **schedule_kwargs)

        elapsed_seconds = 0.0
        with torch.no_grad():
            for _ in range(config.timing_warmup_batches):
                next_schedule(iter(loader))
            torch.cuda.synchronize(device)
            torch.cuda.reset_peak_memory_stats(device)
            torch.distributed.barrier()
            iterator = iter(loader)
            for _ in range(len(loader) // config.pipeline_microbatches):
                batches = [next(iterator) for _ in range(config.pipeline_microbatches)]
                torch.cuda.synchronize(device)
                started = time.perf_counter()
                losses.extend(schedule(data_iterator=iter(batches), **schedule_kwargs))
                torch.cuda.synchronize(device)
                elapsed_seconds += time.perf_counter() - started
        metrics_started = time.perf_counter()
        totals = _broadcast_pipeline_metrics(
            _loss_totals(losses) if parallel_state.is_pipeline_last_stage() else {},
            parallel_state,
            device,
        )
        metrics = _reduce_data_parallel_metrics(totals, parallel_state, device)
        if "lm_loss" not in metrics:
            raise RuntimeError("Evaluation model did not return lm_loss.")
        token_count = torch.tensor(
            totals["lm_loss"][1], device=device, dtype=torch.float64
        )
        torch.distributed.all_reduce(
            token_count,
            group=parallel_state.get_data_parallel_group(with_context_parallel=True),
        )
        peak = torch.tensor(
            float(torch.cuda.max_memory_allocated(device)), device=device
        )
        torch.distributed.all_reduce(peak, op=torch.distributed.ReduceOp.MAX)
        torch.cuda.synchronize(device)
        elapsed = torch.tensor(
            elapsed_seconds + time.perf_counter() - metrics_started, device=device
        )
        torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
        loss = metrics["lm_loss"]
        return {
            "lm_loss": loss,
            "perplexity": _perplexity(loss),
            "valid_tokens": token_count.item(),
            "evaluation_seconds": elapsed.item(),
            "semantic_tokens_per_second": token_count.item() / elapsed.item(),
            "peak_memory_bytes": peak.item(),
            "data_parallel_size": float(data_parallel_size),
            "tensor_parallel_size": float(transformer.tensor_model_parallel_size),
            "pipeline_parallel_size": float(transformer.pipeline_model_parallel_size),
            "context_parallel_size": float(transformer.context_parallel_size),
            "micro_batch_size": float(config.micro_batch_size),
            "pipeline_microbatches": float(config.pipeline_microbatches),
            "local_batch_size": float(local_batch_size),
        }
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def generate_mcore(
    transformer_config: "TransformerConfig",
    model_provider: "Callable[[bool, bool], MegatronModule]",
    checkpoint: Path,
    input_ids: torch.Tensor,
    max_new_tokens: int,
    eos_token_id: Optional[int] = None,
    seed: int = 1234,
) -> Optional[torch.Tensor]:
    r"""Greedily generate tokens with MCore TP/PP execution and a KV cache.

    **API Language** - 中文 | English

    **中文：** 广播 rank 0 的 ``[B,S]`` global prompt batch，沿 DP replicas 切分，
    并使用 MCore static inference context 完成 prefill 与 cached decode。每次模型调用
    包含完整 ``T``，神经元状态在调用后丢弃；KV cache 跨 decode-step ``D`` 保留。
    仅 global rank 0 返回按原顺序合并的 CPU token。

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
    :param eos_token_id: 可选 EOS token ID；已完成序列后续填充该值。
    :type eos_token_id: Optional[int]
    :param seed: MCore model seed。
    :type seed: int
    :return: global rank 0 上的 ``[B,S+max_new_tokens]`` CPU tensor；其他 rank 为 ``None``。
    :rtype: Optional[torch.Tensor]
    :raises ValueError: 输入、上下文长度或推理并行配置无效。
    :raises RuntimeError: CUDA 或 MCore 不可用。

    **English:** Broadcast rank 0's global ``[B,S]`` prompt batch, shard it over
    DP replicas, and run prefill plus cached decode with MCore's static inference
    context. Every model call contains a full ``T`` window and discards neuron
    state afterwards, while the KV cache persists across decode steps ``D``.
    Only global rank 0 returns CPU tokens merged in input order.

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
    :param eos_token_id: Optional EOS token ID used to pad completed sequences.
    :type eos_token_id: Optional[int]
    :param seed: MCore model seed.
    :type seed: int
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
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
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
        model_parallel_cuda_manual_seed(seed)
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
        global_tokens = (
            input_ids.to(device=device, dtype=torch.long)
            if rank == 0
            else torch.empty(
                (batch_size, prompt_length), device=device, dtype=torch.long
            )
        )
        torch.distributed.broadcast(global_tokens, src=0)
        data_parallel_size = parallel_state.get_data_parallel_world_size()
        data_parallel_rank = parallel_state.get_data_parallel_rank()
        local_batch_size = (batch_size + data_parallel_size - 1) // data_parallel_size
        start = data_parallel_rank * local_batch_size
        stop = min(start + local_batch_size, batch_size)
        valid_batch_size = max(0, stop - start)
        tokens = torch.empty(
            (local_batch_size, prompt_length), device=device, dtype=torch.long
        )
        if valid_batch_size:
            tokens[:valid_batch_size].copy_(global_tokens[start:stop])
        if valid_batch_size < local_batch_size:
            tokens[valid_batch_size:].zero_()
        context = StaticInferenceContext(
            max_batch_size=time_steps * local_batch_size,
            max_sequence_length=max_sequence_length,
        )
        wrapper = GPTInferenceWrapper(model, context)
        wrapper.prep_model_for_inference()
        temporal_prompt = tokens.repeat(time_steps, 1)
        inference_input = wrapper.prep_inference_input(temporal_prompt)
        model_group = parallel_state.get_model_parallel_group()
        output_rank = max(torch.distributed.get_process_group_ranks(model_group))
        with InferenceMode.active():
            logits = wrapper.run_one_forward_step(inference_input)
            context.enable_decode_mode()
            generated = tokens
            finished = torch.zeros(local_batch_size, device=device, dtype=torch.bool)
            for step in range(max_new_tokens):
                next_token = torch.empty(
                    local_batch_size, device=device, dtype=torch.long
                )
                if rank == output_rank:
                    semantic_logits = _reduce_time_batch(
                        logits[:, -1], time_steps, reduction
                    )
                    next_token.copy_(semantic_logits.argmax(dim=-1))
                    if eos_token_id is not None:
                        next_token.masked_fill_(finished, eos_token_id)
                torch.distributed.broadcast(
                    next_token, src=output_rank, group=model_group
                )
                generated = torch.cat((generated, next_token[:, None]), dim=1)
                if eos_token_id is not None:
                    finished |= next_token == eos_token_id
                if step + 1 == max_new_tokens:
                    break
                position = context.sequence_len_offset
                decode_input = {
                    "tokens": next_token[:, None].repeat(time_steps, 1),
                    "position_ids": torch.full(
                        (time_steps * local_batch_size, 1),
                        position,
                        device=device,
                        dtype=torch.long,
                    ),
                    "attention_mask": None,
                }
                logits = wrapper.run_one_forward_step(decode_input)
        payload = (
            (start, generated[:valid_batch_size].cpu())
            if rank == output_rank and valid_batch_size
            else None
        )
        gathered = [None] * torch.distributed.get_world_size() if rank == 0 else None
        torch.distributed.gather_object(payload, gathered, dst=0)
        if rank != 0:
            return None
        pieces = sorted(
            (item for item in gathered if item is not None), key=lambda x: x[0]
        )
        return torch.cat([piece for _, piece in pieces], dim=0)
    finally:
        if owns_model_parallel and parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        if owns_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()


def generate(
    config: MCoreGenerationConfig, input_ids: torch.Tensor
) -> Optional[torch.Tensor]:
    r"""Generate token IDs with the high-level MCore inference configuration.

    **中文：** 从 ``config.model`` 解析 model provider，使用同一 checkpoint
    在 DP/TP/PP 拓扑上生成 token。

    **English:** Resolve the model provider from ``config.model`` and generate
    tokens from the checkpoint over the configured DP/TP/PP topology.

    :param config: MCore generation configuration. / MCore generation configuration.
    :type config: MCoreGenerationConfig
    :param input_ids: Integer prompts shaped ``[B, S]`` on global rank zero.
    :type input_ids: torch.Tensor
    :return: Generated CPU token IDs on global rank zero, otherwise ``None``.
    :rtype: Optional[torch.Tensor]
    :raises TypeError: If ``config`` is not :class:`MCoreGenerationConfig`; use
        :func:`generate_mcore` for the low-level callback API.
    """
    if not isinstance(config, MCoreGenerationConfig):
        raise TypeError(
            "generate requires MCoreGenerationConfig; use generate_mcore for "
            "the low-level callback API."
        )
    builder_cls = config.model.get_builder_cls()
    model_provider, _ = builder_cls(config.model).build(
        resume=True,
    )
    return generate_mcore(
        config.model.transformer,
        model_provider,
        config.checkpoint,
        input_ids,
        config.max_new_tokens,
        config.eos_token_id,
        config.seed,
    )


__all__ = ["evaluate", "generate", "generate_mcore", "load_for_inference"]
