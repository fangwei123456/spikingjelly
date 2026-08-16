from __future__ import annotations

import importlib
import json
import os
import random
import time
from pathlib import Path
from typing import Any, Optional

import numpy as np
import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.distributed.tensor_parallel import (
    ChannelShardBatchNorm1d,
    ChannelShardBatchNorm2d,
)

from .config import ModelBuilder, TrainingConfig


def _import_object(path: str) -> Any:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def _seed_worker(worker_id: int) -> None:
    seed = torch.initial_seed() % 2**32
    np.random.seed(seed)
    random.seed(seed)


def _classification_logits(output: torch.Tensor) -> torch.Tensor:
    if output.ndim == 2:
        return output
    if output.ndim == 3:
        return output.mean(0)
    raise ValueError(
        "Vision classification models must return [N, C] or [T, N, C], "
        f"got {tuple(output.shape)}."
    )


class _ResetAfterForward(nn.Module):
    def __init__(self, module: nn.Module, batch_first: bool = False) -> None:
        super().__init__()
        self.module = module
        self.batch_first = batch_first

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.batch_first:
            x = x.transpose(0, 1).contiguous()
        output = self.module(x)
        functional.reset_net(self.module)
        return output


def _pipeline_loss(output: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return nn.functional.cross_entropy(_classification_logits(output), target)


def _recipe(config: TrainingConfig) -> dict[str, Any]:
    recipe = config.as_dict()
    for name in (
        "epochs",
        "max_steps",
        "checkpoint_dir",
        "checkpoint_interval",
        "resume",
    ):
        recipe.pop(name, None)
    return recipe


def _state_dict_options():
    from torch.distributed.checkpoint.state_dict import StateDictOptions

    return StateDictOptions(full_state_dict=True)


def _save_checkpoint(
    path: Path,
    *,
    config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.amp.GradScaler,
    step: int,
    epoch: int,
    batch_in_epoch: int,
    tp_rank: int,
    pp_rank: int,
    dp_rank: int,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict

    occupied = torch.tensor(
        int(dist.get_rank() == 0 and path.exists()),
        device=torch.cuda.current_device(),
    )
    dist.broadcast(occupied, src=0)
    if occupied.item():
        raise FileExistsError(f"Checkpoint directory already exists: {path}")
    config_json = json.dumps(_recipe(config), indent=2, sort_keys=True)
    creation_failed = torch.zeros_like(occupied)
    creation_error = None
    if dist.get_rank() == 0:
        try:
            path.mkdir(parents=True, exist_ok=False)
            (path / "config.json").write_text(config_json, encoding="utf-8")
        except OSError as error:
            creation_failed.fill_(1)
            creation_error = error
    dist.broadcast(creation_failed, src=0)
    if creation_failed.item():
        if creation_error is not None:
            raise creation_error
        raise RuntimeError(f"Rank 0 could not create checkpoint directory: {path}")
    dist.barrier()

    model_state, optimizer_state = get_state_dict(
        model, optimizer, options=_state_dict_options()
    )
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "scaler": scaler.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
        "progress": {
            "step": torch.tensor(step, dtype=torch.int64),
            "epoch": torch.tensor(epoch, dtype=torch.int64),
            "batch_in_epoch": torch.tensor(batch_in_epoch, dtype=torch.int64),
        },
    }
    if dp_rank == 0:
        dcp.save(
            state,
            checkpoint_id=path / f"pp_{pp_rank:04d}_tp_{tp_rank:04d}",
            no_dist=True,
        )
    dist.barrier()


def _load_checkpoint(
    path: Path,
    *,
    config: TrainingConfig,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    scheduler: Optional[Any],
    scaler: torch.amp.GradScaler,
    tp_rank: int,
    pp_rank: int,
) -> tuple[int, int, int]:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import get_state_dict, set_state_dict

    saved_recipe = json.loads((path / "config.json").read_text(encoding="utf-8"))
    if saved_recipe != _recipe(config):
        raise ValueError("Checkpoint configuration does not match this training run.")

    options = _state_dict_options()
    model_state, optimizer_state = get_state_dict(model, optimizer, options=options)
    state = {
        "model": model_state,
        "optimizer": optimizer_state,
        "scheduler": scheduler.state_dict() if scheduler is not None else {},
        "scaler": scaler.state_dict(),
        "rng": {
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state(),
        },
        "progress": {
            "step": torch.tensor(0, dtype=torch.int64),
            "epoch": torch.tensor(0, dtype=torch.int64),
            "batch_in_epoch": torch.tensor(0, dtype=torch.int64),
        },
    }
    dcp.load(
        state,
        checkpoint_id=path / f"pp_{pp_rank:04d}_tp_{tp_rank:04d}",
        no_dist=True,
    )
    set_state_dict(
        model,
        optimizer,
        model_state_dict=state["model"],
        optim_state_dict=state["optimizer"],
        options=options,
    )
    if scheduler is not None:
        scheduler.load_state_dict(state["scheduler"])
    scaler.load_state_dict(state["scaler"])
    torch.set_rng_state(state["rng"]["torch"])
    torch.cuda.set_rng_state(state["rng"]["cuda"])
    progress = state["progress"]
    return (
        int(progress["step"].item()),
        int(progress["epoch"].item()),
        int(progress["batch_in_epoch"].item()),
    )


def _wrap_data_parallel(
    model: nn.Module,
    *,
    config: TrainingConfig,
    device: torch.device,
    dp_size: int,
    dp_group: Optional[Any],
    dp_mesh: Any,
    fsdp_roots: tuple[str, ...],
) -> nn.Module:
    if dp_size == 1:
        return model
    if config.data_parallel == "ddp":
        if config.pipeline_parallel_size > 1:
            return model
        return torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[device.index],
            process_group=dp_group,
        )

    from torch.distributed.fsdp import MixedPrecisionPolicy, fully_shard

    dtype = {
        "fp32": None,
        "bf16": torch.bfloat16,
        "fp16": torch.float16,
    }[config.precision]
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


def _build_loaders(
    config: TrainingConfig,
    *,
    dp_size: int,
    dp_rank: int,
) -> tuple[DataLoader, DataLoader, DistributedSampler, DistributedSampler]:
    builder = _import_object(config.dataset_builder)
    datasets = builder(**config.dataset_kwargs)
    if not isinstance(datasets, (tuple, list)) or len(datasets) != 2:
        raise TypeError(
            "dataset_builder must return (train_dataset, validation_dataset)."
        )
    train_dataset, validation_dataset = datasets
    if not isinstance(train_dataset, Dataset) or not isinstance(
        validation_dataset, Dataset
    ):
        raise TypeError("dataset_builder outputs must be torch Dataset instances.")
    if len(train_dataset) == 0 or len(validation_dataset) == 0:
        raise ValueError("Training and validation datasets must be non-empty.")

    train_sampler = DistributedSampler(
        train_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=True,
        seed=config.seed,
    )
    validation_sampler = DistributedSampler(
        validation_dataset,
        num_replicas=dp_size,
        rank=dp_rank,
        shuffle=False,
    )
    generator = torch.Generator().manual_seed(config.seed + dp_rank)
    kwargs = dict(
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=True,
        worker_init_fn=_seed_worker,
        generator=generator,
        persistent_workers=config.workers > 0,
        drop_last=config.pipeline_parallel_size > 1,
    )
    return (
        DataLoader(train_dataset, sampler=train_sampler, **kwargs),
        DataLoader(validation_dataset, sampler=validation_sampler, **kwargs),
        train_sampler,
        validation_sampler,
    )


def _evaluate(
    model: nn.Module,
    loader: DataLoader,
    *,
    time_steps: int,
    device: torch.device,
    precision: str,
    dp_group: Optional[Any],
    dp_size: int,
) -> tuple[float, float]:
    model.eval()
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.inference_mode():
        for images, targets in loader:
            if images.ndim != 4:
                raise ValueError(
                    "Vision training expects image batches shaped [N, C, H, W]."
                )
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sequence = (
                images.unsqueeze(0).expand(time_steps, *images.shape).contiguous()
            )
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=precision != "fp32",
            ):
                logits = _classification_logits(model(sequence))
                loss = nn.functional.cross_entropy(logits, targets)
            totals[0] += loss.double() * targets.numel()
            totals[1] += (logits.argmax(1) == targets).sum().double()
            totals[2] += targets.numel()
            functional.reset_net(model)
    if dp_size > 1:
        dist.all_reduce(totals, group=dp_group)
    model.train()
    return (totals[0] / totals[2]).item(), (totals[1] / totals[2]).item()


def _evaluate_pipeline(
    schedule: Any,
    loader: DataLoader,
    *,
    time_steps: int,
    device: torch.device,
    precision: str,
    dp_group: Optional[Any],
    dp_size: int,
    pp_group: Any,
    pp_rank: int,
    pp_size: int,
    tp_size: int,
) -> tuple[float, float]:
    totals = torch.zeros(3, device=device, dtype=torch.float64)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=True)
            targets = targets.to(device, non_blocking=True)
            sequence = (
                images.unsqueeze(1)
                .expand(images.shape[0], time_steps, *images.shape[1:])
                .contiguous()
            )
            with torch.autocast(
                device_type="cuda",
                dtype=dtype,
                enabled=precision != "fp32",
            ):
                if pp_rank == 0:
                    output = schedule.step(sequence)
                else:
                    output = schedule.step()
            if pp_rank == pp_size - 1:
                logits = _classification_logits(output)
                totals[0] += (
                    nn.functional.cross_entropy(logits, targets).double()
                    * targets.numel()
                )
                totals[1] += (logits.argmax(1) == targets).sum().double()
                totals[2] += targets.numel()
    if pp_rank == pp_size - 1 and dp_size > 1:
        dist.all_reduce(totals, group=dp_group)
    last_stage_rank = dist.get_rank() + (pp_size - 1 - pp_rank) * tp_size
    dist.broadcast(totals, src=last_stage_rank, group=pp_group)
    return (totals[0] / totals[2]).item(), (totals[1] / totals[2]).item()


def build_imagefolder_datasets(
    root: str | Path,
    image_size: int = 224,
    train_subdirectory: str = "train",
    validation_subdirectory: str = "val",
) -> tuple[Dataset, Dataset]:
    r"""Build ImageNet-style image-folder datasets.

    **API Language** - 中文 | English

    **中文：** 从 ``root/train`` 和 ``root/val`` 构建 torchvision ImageFolder，
    使用标准 ImageNet crop、flip 和 normalization。

    **English:** Build torchvision ImageFolder datasets from ``root/train`` and
    ``root/val`` with standard ImageNet crop, flip, and normalization.

    :param root: 数据集根目录。 / Dataset root.
    :type root: str or pathlib.Path
    :param image_size: 模型输入边长。 / Model input side length.
    :type image_size: int
    :param train_subdirectory: 训练子目录。 / Training subdirectory.
    :type train_subdirectory: str
    :param validation_subdirectory: 验证子目录。 / Validation subdirectory.
    :type validation_subdirectory: str
    :return: train 与 validation Dataset。 / Train and validation datasets.
    :rtype: tuple[torch.utils.data.Dataset, torch.utils.data.Dataset]
    :raises ValueError: ``image_size`` 非正数。 / If ``image_size`` is not positive.
    """
    if image_size <= 0:
        raise ValueError("image_size must be positive.")
    from torchvision import datasets, transforms

    root = Path(root)
    normalization = transforms.Normalize(
        mean=(0.485, 0.456, 0.406),
        std=(0.229, 0.224, 0.225),
    )
    train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(image_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalization,
        ]
    )
    validation_transform = transforms.Compose(
        [
            transforms.Resize(int(image_size / 0.875)),
            transforms.CenterCrop(image_size),
            transforms.ToTensor(),
            normalization,
        ]
    )
    return (
        datasets.ImageFolder(root / train_subdirectory, train_transform),
        datasets.ImageFolder(root / validation_subdirectory, validation_transform),
    )


def train(config: TrainingConfig) -> dict[str, float]:
    r"""Train a vision SNN with native PyTorch distributed execution.

    **API Language** - 中文 | English

    **中文：** 从可序列化 config 构建数据和 architecture-specific 模型，初始化
    ``DP × PP × TP`` DeviceMesh，运行 DDP 或 FSDP2 图像分类训练，并在每个 batch 后重置
    SNN 状态。所有 TP rank 使用同一个 DP sampler rank。

    **English:** Build data and an architecture-specific model from a serializable
    config, initialize a ``DP × PP × TP`` DeviceMesh, run DDP or FSDP2 image
    classification, and reset SNN state after every batch. TP peers share the same
    DP sampler rank.

    :param config: 可直接运行的训练配置。 / Train-ready configuration.
    :type config: TrainingConfig
    :return: 最终 loss、accuracy、step、训练 step 吞吐和跨 rank 峰值显存；吞吐不含
        数据加载、验证与 checkpoint。 / Final loss, accuracy, step, training-step
        throughput, and maximum peak memory across ranks; throughput excludes data
        loading, validation, and checkpointing.
    :rtype: dict[str, float]
    :raises RuntimeError: CUDA、NCCL 或 FSDP2 不可用。 / If CUDA, NCCL, or FSDP2 is unavailable.
    :raises ValueError: world size、模型输出或输入布局无效。 / If world size,
        model output, or input layout is invalid.
    """
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed vision training requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", device_id=device)

    try:
        world_size = dist.get_world_size()
        rank = dist.get_rank()
        model_parallel_size = (
            config.pipeline_parallel_size * config.tensor_parallel_size
        )
        if world_size % model_parallel_size:
            raise ValueError("world_size must be divisible by PP × TP.")
        dp_size = world_size // model_parallel_size
        tp_rank = rank % config.tensor_parallel_size
        pp_rank = (rank // config.tensor_parallel_size) % config.pipeline_parallel_size
        dp_rank = rank // model_parallel_size

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            (
                dp_size,
                config.pipeline_parallel_size,
                config.tensor_parallel_size,
            ),
            mesh_dim_names=("dp", "pp", "tp"),
        )
        dp_mesh = mesh["dp"]
        pp_mesh = mesh["pp"]
        tp_mesh = mesh["tp"]
        dp_group = dp_mesh.get_group() if dp_size > 1 else None
        pp_group = pp_mesh.get_group() if config.pipeline_parallel_size > 1 else None
        tp_group = tp_mesh.get_group() if config.tensor_parallel_size > 1 else None

        torch.manual_seed(config.seed)
        builder_cls = config.model.get_builder_cls()
        if not isinstance(builder_cls, type) or not issubclass(
            builder_cls, ModelBuilder
        ):
            raise TypeError("model builder must inherit vision.ModelBuilder.")
        model, fsdp_roots, pipeline_input_shape, pipeline_output_shape = builder_cls(
            config.model
        ).build(
            process_group=tp_group,
            pipeline_rank=pp_rank,
            pipeline_size=config.pipeline_parallel_size,
            pipeline_microbatches=config.pipeline_microbatches,
            device=device,
            micro_batch_size=config.batch_size,
            memopt_level=config.memopt_level,
            memopt_compress_inputs=config.memopt_compress_inputs,
        )
        model = _wrap_data_parallel(
            model,
            config=config,
            device=device,
            dp_size=dp_size,
            dp_group=dp_group,
            dp_mesh=dp_mesh,
            fsdp_roots=fsdp_roots,
        )

        train_schedule = evaluation_schedule = None
        if config.pipeline_parallel_size > 1:
            from torch.distributed.pipelining import PipelineStage, ScheduleGPipe
            from torch.distributed.pipelining.microbatch import TensorChunkSpec

            schedule_kwargs = {
                "n_microbatches": config.pipeline_microbatches,
                "output_merge_spec": TensorChunkSpec(1),
            }
            compute_dtype = {
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
                "fp16": torch.float16,
            }[config.precision]
            input_args = torch.empty(
                pipeline_input_shape,
                device=device,
                dtype=torch.float32 if pp_rank == 0 else compute_dtype,
            )
            output_args = torch.empty(
                pipeline_output_shape, device=device, dtype=compute_dtype
            )
            train_schedule = ScheduleGPipe(
                PipelineStage(
                    _ResetAfterForward(model, batch_first=pp_rank == 0),
                    pp_rank,
                    config.pipeline_parallel_size,
                    device,
                    input_args=input_args,
                    output_args=output_args,
                    group=pp_group,
                ),
                loss_fn=_pipeline_loss,
                **schedule_kwargs,
            )
            evaluation_schedule = ScheduleGPipe(
                PipelineStage(
                    _ResetAfterForward(model, batch_first=pp_rank == 0),
                    pp_rank,
                    config.pipeline_parallel_size,
                    device,
                    input_args=input_args,
                    output_args=output_args,
                    group=pp_group,
                ),
                **schedule_kwargs,
            )

        optimizer_cls = _import_object(config.optimizer)
        optimizer = optimizer_cls(model.parameters(), **config.optimizer_kwargs)
        scheduler = (
            _import_object(config.scheduler)(optimizer, **config.scheduler_kwargs)
            if config.scheduler is not None
            else None
        )
        train_loader, validation_loader, train_sampler, validation_sampler = (
            _build_loaders(config, dp_size=dp_size, dp_rank=dp_rank)
        )
        scaler = torch.amp.GradScaler("cuda", enabled=config.precision == "fp16")

        start_step = start_epoch = start_batch = 0
        if config.resume is not None:
            start_step, start_epoch, start_batch = _load_checkpoint(
                config.resume,
                config=config,
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                scaler=scaler,
                tp_rank=tp_rank,
                pp_rank=pp_rank,
            )
            start_epoch += start_batch // len(train_loader)
            start_batch %= len(train_loader)

        autocast_dtype = torch.bfloat16 if config.precision == "bf16" else torch.float16
        torch.cuda.reset_peak_memory_stats(device)
        training_seconds = 0.0
        timed_steps = 0
        global_step = start_step
        processed_images = 0
        train_loss = float("nan")
        validation_loss = validation_accuracy = float("nan")
        stop = config.max_steps is not None and global_step >= config.max_steps

        model.train()
        for epoch in range(start_epoch, config.epochs):
            if stop:
                break
            train_sampler.set_epoch(epoch)
            validation_sampler.set_epoch(epoch)
            totals = torch.zeros(2, device=device, dtype=torch.float64)
            for batch_index, (images, targets) in enumerate(train_loader):
                if epoch == start_epoch and batch_index < start_batch:
                    continue
                measure_step = global_step - start_step >= config.timing_warmup_steps
                torch.cuda.synchronize(device)
                step_started = time.perf_counter() if measure_step else 0.0
                if images.ndim != 4:
                    raise ValueError(
                        "Vision training expects image batches shaped [N, C, H, W]."
                    )
                images = images.to(device, non_blocking=True)
                targets = targets.to(device, non_blocking=True)
                if train_schedule is None:
                    sequence = (
                        images.unsqueeze(0)
                        .expand(config.model.time_steps, *images.shape)
                        .contiguous()
                    )
                else:
                    sequence = (
                        images.unsqueeze(1)
                        .expand(
                            images.shape[0],
                            config.model.time_steps,
                            *images.shape[1:],
                        )
                        .contiguous()
                    )
                optimizer.zero_grad(set_to_none=True)
                if train_schedule is not None:
                    losses = []
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=config.precision != "fp32",
                    ):
                        if pp_rank == 0:
                            train_schedule.step(sequence)
                        elif pp_rank == config.pipeline_parallel_size - 1:
                            train_schedule.step(target=targets, losses=losses)
                        else:
                            train_schedule.step()
                    loss = (
                        torch.stack(losses).mean()
                        if losses
                        else torch.zeros((), device=device)
                    )
                    last_stage_rank = (
                        rank
                        + (config.pipeline_parallel_size - 1 - pp_rank)
                        * config.tensor_parallel_size
                    )
                    dist.broadcast(loss, src=last_stage_rank, group=pp_group)
                    if dp_size > 1 and config.data_parallel == "ddp":
                        for parameter in model.parameters():
                            if parameter.grad is not None:
                                dist.all_reduce(parameter.grad, group=dp_group)
                                parameter.grad.div_(dp_size)
                    optimizer.step()
                    update_successful = True
                else:
                    with torch.autocast(
                        device_type="cuda",
                        dtype=autocast_dtype,
                        enabled=config.precision != "fp32",
                    ):
                        logits = _classification_logits(model(sequence))
                        loss = nn.functional.cross_entropy(logits, targets)
                    if scaler.is_enabled():
                        scale = scaler.get_scale()
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()
                        update_successful = scaler.get_scale() >= scale
                    else:
                        loss.backward()
                        optimizer.step()
                        update_successful = True
                if scheduler is not None and update_successful:
                    scheduler.step()
                if train_schedule is None:
                    functional.reset_net(model)
                torch.cuda.synchronize(device)
                if measure_step:
                    training_seconds += time.perf_counter() - step_started
                    timed_steps += 1

                global_step += 1
                batch_samples = targets.numel()
                totals[0] += loss.detach().double() * batch_samples
                totals[1] += batch_samples
                if measure_step:
                    processed_images += batch_samples * dp_size

                if (
                    config.checkpoint_interval
                    and global_step % config.checkpoint_interval == 0
                ):
                    _save_checkpoint(
                        config.checkpoint_dir / f"step_{global_step:08d}",
                        config=config,
                        model=model,
                        optimizer=optimizer,
                        scheduler=scheduler,
                        scaler=scaler,
                        step=global_step,
                        epoch=epoch,
                        batch_in_epoch=batch_index + 1,
                        tp_rank=tp_rank,
                        pp_rank=pp_rank,
                        dp_rank=dp_rank,
                    )
                if config.max_steps is not None and global_step >= config.max_steps:
                    stop = True
                    break

            if dp_size > 1:
                dist.all_reduce(totals, group=dp_group)
            train_loss = (totals[0] / totals[1]).item()
            if evaluation_schedule is None:
                validation_loss, validation_accuracy = _evaluate(
                    model,
                    validation_loader,
                    time_steps=config.model.time_steps,
                    device=device,
                    precision=config.precision,
                    dp_group=dp_group,
                    dp_size=dp_size,
                )
            else:
                model.eval()
                validation_loss, validation_accuracy = _evaluate_pipeline(
                    evaluation_schedule,
                    validation_loader,
                    time_steps=config.model.time_steps,
                    device=device,
                    precision=config.precision,
                    dp_group=dp_group,
                    dp_size=dp_size,
                    pp_group=pp_group,
                    pp_rank=pp_rank,
                    pp_size=config.pipeline_parallel_size,
                    tp_size=config.tensor_parallel_size,
                )
                model.train()
            start_batch = 0
            if stop:
                break

        if validation_accuracy != validation_accuracy:
            if evaluation_schedule is None:
                validation_loss, validation_accuracy = _evaluate(
                    model,
                    validation_loader,
                    time_steps=config.model.time_steps,
                    device=device,
                    precision=config.precision,
                    dp_group=dp_group,
                    dp_size=dp_size,
                )
            else:
                model.eval()
                validation_loss, validation_accuracy = _evaluate_pipeline(
                    evaluation_schedule,
                    validation_loader,
                    time_steps=config.model.time_steps,
                    device=device,
                    precision=config.precision,
                    dp_group=dp_group,
                    dp_size=dp_size,
                    pp_group=pp_group,
                    pp_rank=pp_rank,
                    pp_size=config.pipeline_parallel_size,
                    tp_size=config.tensor_parallel_size,
                )
                model.train()

        local_metrics = torch.tensor(
            [
                training_seconds,
                torch.cuda.max_memory_allocated(device),
                torch.cuda.max_memory_reserved(device),
            ],
            device=device,
            dtype=torch.float64,
        )
        maximum_metrics = local_metrics.clone()
        dist.all_reduce(maximum_metrics, op=dist.ReduceOp.MAX)
        total_memory = local_metrics[1:].clone()
        dist.all_reduce(total_memory, op=dist.ReduceOp.SUM)
        elapsed = maximum_metrics[0].item()
        return {
            "train_loss": train_loss,
            "validation_loss": validation_loss,
            "validation_accuracy": validation_accuracy,
            "optimizer_step": float(global_step),
            "timed_steps": float(timed_steps),
            "training_seconds": elapsed,
            "images_per_second": processed_images / elapsed
            if elapsed
            else float("nan"),
            "peak_memory_bytes": maximum_metrics[1].item(),
            "total_peak_memory_bytes": total_memory[0].item(),
            "peak_reserved_memory_bytes": maximum_metrics[2].item(),
            "total_peak_reserved_memory_bytes": total_memory[1].item(),
            "data_parallel_size": float(dp_size),
            "pipeline_parallel_size": float(config.pipeline_parallel_size),
            "tensor_parallel_size": float(config.tensor_parallel_size),
            "global_batch_size": float(config.batch_size * dp_size),
        }
    finally:
        if initialized_here:
            dist.destroy_process_group()
