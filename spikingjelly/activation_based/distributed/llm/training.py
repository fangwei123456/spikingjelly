"""Megatron Core training loop for SNN language models."""

from __future__ import annotations

import functools
import importlib
import os
import time
from collections.abc import Iterator
from typing import Any

import torch
from torch.utils.data import DataLoader, DistributedSampler

from spikingjelly.logger import logger

from .config import TrainingConfig
from .metrics import (
    _broadcast_pipeline_metrics,
    _loss_totals,
    _reduce_data_parallel_metrics,
)


def _import_object(path: str) -> Any:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def _build_training_inputs(config: TrainingConfig) -> tuple[Any, Any, Any]:
    builder_cls = config.model.get_builder_cls()
    model_provider, forward_step = builder_cls(config.model).build(
        memopt_level=config.memopt_level,
        memopt_checkpoint_budget=config.memopt_checkpoint_budget,
        memopt_compress_inputs=config.memopt_compress_inputs,
        resume=config.resume is not None,
    )
    dataset_provider = functools.partial(
        _import_object(config.dataset_builder), **config.dataset_kwargs
    )
    return model_provider, dataset_provider, forward_step


def _iterator(loader: DataLoader, consumed_batches: int = 0) -> Iterator[Any]:
    batches_per_epoch = len(loader)
    if batches_per_epoch == 0:
        raise ValueError(
            "Dataset is too small for one complete distributed microbatch."
        )
    epoch, offset = divmod(consumed_batches, batches_per_epoch)
    while True:
        if isinstance(loader.sampler, DistributedSampler):
            loader.sampler.set_epoch(epoch)
        batches = iter(loader)
        for _ in range(offset):
            next(batches)
        yield from batches
        epoch += 1
        offset = 0


def train(
    config: TrainingConfig,
) -> dict[str, float]:
    r"""Train an SNN Transformer with Megatron Core.

    **API Language** - :ref:`中文 <distributed-train-cn>` | :ref:`English <distributed-train-en>`

    ----

    .. _distributed-train-cn:

    * **中文**

    初始化 NCCL 与 MCore 并行组，从 ``config.model`` 和
    ``config.dataset_builder`` 构建当前 PP stage 与数据集，再使用 MCore DDP、
    distributed optimizer 与 pipeline schedule 运行训练。模型必须公开与配置相同的
    ``snn_model_config`` 和 ``snn_memopt_level``；有效 MCore microbatch 的 ``T``
    由 ``config.model.time_steps`` 唯一决定。

    :param config: 训练配置。
    :type config: TrainingConfig
    :return: 最后一步 loss、吞吐、梯度范数、峰值显存、checkpoint 时间与训练进度指标。
    :rtype: dict[str, float]
    :raises RuntimeError: CUDA、torch.distributed 或 MCore 运行环境不可用，或模型
        元数据与配置不一致。
    :raises ValueError: world size、batch 拓扑或数据集为空。

    ----

    .. _distributed-train-en:

    * **English**

    Initializes NCCL and MCore process groups, builds the local PP stage and datasets
    from ``config.model`` and ``config.dataset_builder``, and runs MCore DDP,
    distributed optimizer, and pipeline schedules. The model must expose
    ``snn_model_config`` and ``snn_memopt_level`` values matching the configuration.
    ``config.model.time_steps`` is the sole source of ``T`` for the effective MCore
    microbatch ``T × B``.

    :param config: Training configuration.
    :type config: TrainingConfig
    :return: Final loss, throughput, gradient norm, peak memory, checkpoint timing, and progress metrics.
    :rtype: dict[str, float]
    :raises RuntimeError: If CUDA, torch.distributed, or MCore is unavailable, or
        model metadata disagrees with the configuration.
    :raises ValueError: If world size, batch topology, or datasets are invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("MCore distributed training requires CUDA.")

    model_provider, dataset_provider, forward_step = _build_training_inputs(config)

    try:
        from megatron.core import parallel_state, tensor_parallel
        from megatron.core.distributed import (
            DistributedDataParallel,
            DistributedDataParallelConfig,
            finalize_model_grads,
        )
        from megatron.core.optimizer import get_megatron_optimizer
        from megatron.core.optimizer_param_scheduler import OptimizerParamScheduler
        from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
        from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
        from megatron.core.transformer.module import Float16Module
    except ImportError as error:
        raise RuntimeError(
            "MCore training requires Python 3.12 and spikingjelly[megatron]."
        ) from error

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_distributed = not torch.distributed.is_initialized()
    if initialized_distributed:
        torch.distributed.init_process_group(backend="nccl", device_id=device)

    initialized_model_parallel = not parallel_state.model_parallel_is_initialized()
    try:
        transformer = config.model.transformer
        if initialized_model_parallel:
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
                "Existing MCore process groups do not match transformer TP/PP/CP."
            )

        world_size = torch.distributed.get_world_size()
        model_parallel_size = (
            transformer.tensor_model_parallel_size
            * transformer.pipeline_model_parallel_size
            * transformer.context_parallel_size
        )
        if world_size % model_parallel_size:
            raise ValueError(
                "world_size="
                f"{world_size} is not divisible by TP×PP×CP={model_parallel_size}."
            )
        data_parallel_size = world_size // model_parallel_size
        samples_per_microbatch = config.micro_batch_size * data_parallel_size
        if config.global_batch_size % samples_per_microbatch:
            raise ValueError(
                "global_batch_size must be divisible by micro_batch_size × DP size."
            )
        num_microbatches = config.global_batch_size // samples_per_microbatch

        model_parallel_cuda_manual_seed(config.seed)
        model = model_provider(
            parallel_state.is_pipeline_first_stage(),
            parallel_state.is_pipeline_last_stage(),
        )
        if getattr(model, "snn_model_config", None) is not config.model:
            raise RuntimeError(
                "model_provider returned a model whose snn_model_config does not "
                "match config.model."
            )
        if getattr(model, "snn_memopt_level", None) != config.memopt_level:
            raise RuntimeError(
                "model_provider returned a model whose snn_memopt_level does "
                "not match config.memopt_level."
            )
        time_steps = config.model.time_steps
        recipe = getattr(model, "checkpoint_metadata", {})
        if not isinstance(recipe, dict):
            raise RuntimeError("model.checkpoint_metadata must be a dictionary.")
        recipe = {
            **recipe,
            "model": config.model._checkpoint_metadata(),
            "dataset_builder": config.dataset_builder,
            "dataset_kwargs": config.dataset_kwargs,
            "memopt_level": config.memopt_level,
            "memopt_checkpoint_budget": config.memopt_checkpoint_budget,
            "memopt_compress_inputs": config.memopt_compress_inputs,
            "mcore_recompute_granularity": transformer.recompute_granularity,
            "mcore_recompute_modules": transformer.recompute_modules,
            "sequence_length": config.sequence_length,
            "temporal_semantics_version": 1,
        }

        model.cuda(device)
        if transformer.fp16 or transformer.bf16:
            model = Float16Module(transformer, model)
        for parameter in model.parameters():
            tensor_parallel.set_defaults_if_not_set_tensor_model_parallel_attributes(
                parameter
            )

        ddp_config = DistributedDataParallelConfig(
            grad_reduce_in_fp32=True,
            use_distributed_optimizer=True,
        )
        model = DistributedDataParallel(transformer, ddp_config, model)
        model.broadcast_params()
        models = [model]

        optimizer = get_megatron_optimizer(config.optimizer, models)
        transformer.grad_scale_func = optimizer.scale_loss
        transformer.finalize_model_grads_func = finalize_model_grads
        scheduler = OptimizerParamScheduler(
            optimizer,
            init_lr=0.0 if config.lr_warmup_steps else config.optimizer.lr,
            max_lr=config.optimizer.lr,
            min_lr=config.optimizer.min_lr,
            lr_warmup_steps=config.lr_warmup_steps * config.global_batch_size,
            lr_decay_steps=config.lr_decay_steps * config.global_batch_size,
            lr_decay_style=config.lr_decay_style,
            start_wd=config.optimizer.weight_decay,
            end_wd=config.optimizer.weight_decay,
            wd_incr_steps=config.train_steps * config.global_batch_size,
            wd_incr_style="constant",
        )
        start_step = 0
        if config.resume is not None:
            from .checkpoint import load_checkpoint

            restored = load_checkpoint(config.resume, model, optimizer, scheduler)
            if restored["recipe"] != recipe:
                raise ValueError("Checkpoint recipe does not match the current model.")
            start_step = int(restored["progress"]["optimizer_step"])
            consumed_samples = int(restored["progress"]["consumed_samples"])
            if consumed_samples != start_step * config.global_batch_size:
                raise ValueError(
                    "Checkpoint progress is inconsistent with global_batch_size."
                )
            if not 0 <= start_step <= config.train_steps:
                raise ValueError(
                    "Checkpoint optimizer_step is outside this training run."
                )

        validation_events = (
            config.train_steps // config.eval_interval if config.eval_interval else 0
        )
        datasets = dataset_provider(
            (
                config.train_steps * config.global_batch_size,
                validation_events * config.eval_steps * config.global_batch_size,
                0,
            )
        )
        if len(datasets) != 3 or len(datasets[0]) == 0:
            raise ValueError(
                "dataset_provider must return three datasets and non-empty training data."
            )
        loaders = [
            DataLoader(
                dataset,
                batch_size=config.micro_batch_size,
                sampler=DistributedSampler(
                    dataset,
                    num_replicas=data_parallel_size,
                    rank=parallel_state.get_data_parallel_rank(),
                    shuffle=index == 0,
                    seed=config.seed,
                    drop_last=True,
                ),
                drop_last=True,
            )
            for index, dataset in enumerate(datasets[:2])
            if len(dataset)
        ]
        train_iterator = _iterator(loaders[0], start_step * num_microbatches)
        completed_evaluations = (
            start_step // config.eval_interval if config.eval_interval else 0
        )
        valid_iterator = (
            _iterator(
                loaders[1],
                completed_evaluations * config.eval_steps * num_microbatches,
            )
            if len(loaders) == 2
            else None
        )
        schedule = get_forward_backward_func()
        metrics: dict[str, float] = {
            "optimizer_step": float(start_step),
            "consumed_samples": float(start_step * config.global_batch_size),
        }

        torch.cuda.reset_peak_memory_stats(device)
        model.train()
        training_seconds = 0.0
        timed_steps = 0
        for step in range(start_step + 1, config.train_steps + 1):
            model.zero_grad_buffer()
            optimizer.zero_grad()
            torch.cuda.synchronize(device)
            started = time.perf_counter()
            losses = schedule(
                forward_step_func=forward_step,
                data_iterator=train_iterator,
                model=models,
                num_microbatches=num_microbatches,
                seq_length=config.sequence_length,
                micro_batch_size=config.micro_batch_size * time_steps,
                decoder_seq_length=config.sequence_length,
                forward_only=False,
            )
            update_successful, grad_norm, _ = optimizer.step()
            success = torch.tensor(int(update_successful), device=device)
            torch.distributed.all_reduce(success, op=torch.distributed.ReduceOp.MIN)
            if success.item():
                scheduler.step(config.global_batch_size)
            torch.cuda.synchronize(device)
            elapsed = torch.tensor(time.perf_counter() - started, device=device)
            torch.distributed.all_reduce(elapsed, op=torch.distributed.ReduceOp.MAX)
            if step - start_step > config.timing_warmup_steps:
                training_seconds += elapsed.item()
                timed_steps += 1

            metrics = _reduce_data_parallel_metrics(
                _broadcast_pipeline_metrics(
                    _loss_totals(losses)
                    if parallel_state.is_pipeline_last_stage()
                    else {},
                    parallel_state,
                    device,
                ),
                parallel_state,
                device,
            )
            metrics.update(
                {
                    "optimizer_step": float(step),
                    "consumed_samples": float(step * config.global_batch_size),
                    "timed_steps": float(timed_steps),
                    "training_seconds": training_seconds,
                    "update_successful": float(success.item()),
                }
            )
            if timed_steps:
                metrics.update(
                    {
                        "semantic_tokens_per_second": timed_steps
                        * config.global_batch_size
                        * config.sequence_length
                        / training_seconds,
                        "neural_steps_per_second": timed_steps
                        * config.global_batch_size
                        * config.sequence_length
                        * time_steps
                        / training_seconds,
                    }
                )
            if grad_norm is not None:
                metrics["grad_norm"] = float(grad_norm)
            if step % config.log_interval == 0 and torch.distributed.get_rank() == 0:
                logger.info("MCore SNN training step {}: {}", step, metrics)

            if config.eval_interval and step % config.eval_interval == 0:
                if valid_iterator is None:
                    raise ValueError(
                        "Validation is enabled but validation data is empty."
                    )
                model.eval()
                with torch.no_grad():
                    validation_losses = []
                    for _ in range(config.eval_steps):
                        validation_losses.extend(
                            schedule(
                                forward_step_func=forward_step,
                                data_iterator=valid_iterator,
                                model=models,
                                num_microbatches=num_microbatches,
                                seq_length=config.sequence_length,
                                micro_batch_size=config.micro_batch_size * time_steps,
                                decoder_seq_length=config.sequence_length,
                                forward_only=True,
                            )
                        )
                validation_metrics = _reduce_data_parallel_metrics(
                    _broadcast_pipeline_metrics(
                        _loss_totals(validation_losses)
                        if parallel_state.is_pipeline_last_stage()
                        else {},
                        parallel_state,
                        device,
                    ),
                    parallel_state,
                    device,
                )
                metrics.update(
                    {
                        f"validation_{key}": value
                        for key, value in validation_metrics.items()
                    }
                )
                model.train()

            if config.checkpoint_interval and step % config.checkpoint_interval == 0:
                from .checkpoint import save_checkpoint

                torch.cuda.synchronize(device)
                checkpoint_started = time.perf_counter()
                save_checkpoint(
                    config.checkpoint_dir / f"step_{step:07d}",
                    model,
                    optimizer,
                    scheduler,
                    step,
                    step * config.global_batch_size,
                    recipe,
                )
                torch.cuda.synchronize(device)
                checkpoint_seconds = torch.tensor(
                    time.perf_counter() - checkpoint_started, device=device
                )
                torch.distributed.all_reduce(
                    checkpoint_seconds, op=torch.distributed.ReduceOp.MAX
                )
                metrics["checkpoint_seconds"] = checkpoint_seconds.item()

        local_memory = torch.tensor(
            [
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ],
            device=device,
            dtype=torch.float64,
        )
        maximum_memory = local_memory.clone()
        torch.distributed.all_reduce(maximum_memory, op=torch.distributed.ReduceOp.MAX)
        total_memory = local_memory.clone()
        torch.distributed.all_reduce(total_memory, op=torch.distributed.ReduceOp.SUM)
        metrics.update(
            {
                "peak_memory_bytes": maximum_memory[0].item(),
                "total_peak_memory_bytes": total_memory[0].item(),
                "peak_reserved_memory_bytes": maximum_memory[1].item(),
                "total_peak_reserved_memory_bytes": total_memory[1].item(),
                "tensor_parallel_size": float(transformer.tensor_model_parallel_size),
                "pipeline_parallel_size": float(
                    transformer.pipeline_model_parallel_size
                ),
                "context_parallel_size": float(transformer.context_parallel_size),
                "data_parallel_size": float(data_parallel_size),
                "memopt_level": float(config.memopt_level),
                "selective_recompute": float(
                    transformer.recompute_granularity == "selective"
                ),
            }
        )
        return metrics
    finally:
        if (
            initialized_model_parallel
            and parallel_state.model_parallel_is_initialized()
        ):
            parallel_state.destroy_model_parallel()
        if initialized_distributed and torch.distributed.is_initialized():
            torch.distributed.destroy_process_group()
