from __future__ import annotations

import functools
import importlib
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Literal, Optional

import h5py
import numpy as np
import torch
import torch.distributed as dist
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.distributed import DistributedSampler

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.precision import prepare_model_for_precision

from .config import (
    EvaluationConfig,
    ModelConfig,
    PredictionConfig,
    TrainingConfig,
)
from .execution import (
    _classification_logits,
    _classification_sequence,
    _forward_classification,
    _wrap_data_parallel,
)

_ARTIFACT_SCHEMA_VERSION = 1


def _import_object(path: str) -> Any:
    module_name, name = path.rsplit(".", 1)
    return getattr(importlib.import_module(module_name), name)


def _load_checkpoint_model(
    checkpoint: Path,
    model: torch.nn.Module,
    *,
    pipeline_rank: int,
    tensor_rank: int,
) -> None:
    import torch.distributed.checkpoint as dcp
    from torch.distributed.checkpoint.state_dict import (
        StateDictOptions,
        get_model_state_dict,
        set_model_state_dict,
    )

    options = StateDictOptions(full_state_dict=True)
    model_state = get_model_state_dict(model, options=options)
    state = {"model": model_state}
    dcp.load(
        state,
        checkpoint_id=checkpoint / f"pp_{pipeline_rank:04d}_tp_{tensor_rank:04d}",
        no_dist=True,
    )
    set_model_state_dict(model, state["model"], options=options)


def _sync_rank_zero_error(
    error: Optional[BaseException], device: torch.device, message: str
) -> None:
    failed = torch.tensor(int(error is not None), dtype=torch.uint8, device=device)
    dist.broadcast(failed, src=0)
    if failed.item():
        if error is not None:
            raise error
        raise RuntimeError(message)


def export_inference_artifact(checkpoint: Path, output: Path) -> None:
    r"""Export a distributed vision checkpoint as one canonical artifact.

    **API Language** - 中文 | English

    **中文：** 使用训练 checkpoint 记录的 TP/PP 拓扑构建各 rank stage，
    仅恢复 model state，然后在 global rank 0 合并为与目标推理拓扑
    无关的 CPU artifact。运行时 world size 必须等于源 ``TP * PP``。

    **English:** Build every source TP/PP stage recorded by the training
    checkpoint, restore model state only, and merge a topology-independent CPU
    artifact on global rank zero. The launch world size must equal source
    ``TP * PP``.

    :param checkpoint: 训练 checkpoint 目录。 / Training checkpoint directory.
    :type checkpoint: pathlib.Path
    :param output: 新 artifact 文件。 / New artifact file.
    :type output: pathlib.Path
    :raises FileExistsError: ``output`` 已存在。 / If ``output`` exists.
    :raises RuntimeError: CUDA/NCCL 不可用或 world size 不匹配。 /
        If CUDA/NCCL is unavailable or the world size is incompatible.
    """
    checkpoint = Path(checkpoint)
    output = Path(output)
    if not checkpoint.is_dir():
        raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint}")
    if output.exists():
        raise FileExistsError(f"Artifact already exists: {output}")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed vision artifact export requires CUDA.")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", device_id=device)
    try:
        config = TrainingConfig.from_dict(
            json.loads((checkpoint / "config.json").read_text(encoding="utf-8"))
        )
        expected_world_size = (
            config.tensor_parallel_size * config.pipeline_parallel_size
        )
        if dist.get_world_size() != expected_world_size:
            raise RuntimeError(
                "Artifact export world size must equal the checkpoint TP * PP."
            )
        rank = dist.get_rank()
        tensor_rank = rank % config.tensor_parallel_size
        pipeline_rank = rank // config.tensor_parallel_size

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            (config.pipeline_parallel_size, config.tensor_parallel_size),
            mesh_dim_names=("pp", "tp"),
        )
        tensor_group = (
            mesh["tp"].get_group() if config.tensor_parallel_size > 1 else None
        )
        builder_cls = config.model.get_builder_cls()
        builder = builder_cls(config.model)
        model, _, _, _ = builder.build(
            process_group=tensor_group,
            memopt_process_group=None,
            pipeline_rank=pipeline_rank,
            pipeline_size=config.pipeline_parallel_size,
            pipeline_microbatches=config.pipeline_microbatches,
            device=device,
            micro_batch_size=config.batch_size,
            memopt_level=0,
            memopt_compress_inputs=False,
            memopt_checkpoint_budget="memory",
        )
        _load_checkpoint_model(
            checkpoint,
            model,
            pipeline_rank=pipeline_rank,
            tensor_rank=tensor_rank,
        )
        payload = (
            pipeline_rank,
            tensor_rank,
            {name: value.detach().cpu() for name, value in model.state_dict().items()},
        )
        gathered = [None] * expected_world_size if rank == 0 else None
        dist.gather_object(payload, gathered, dst=0)

        error: Optional[BaseException] = None
        if rank == 0:
            temporary = output.with_name(f".{output.name}.tmp")
            try:
                canonical = builder.merge_state_dicts(
                    gathered,
                    pipeline_size=config.pipeline_parallel_size,
                    tensor_size=config.tensor_parallel_size,
                )
                output.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "schema_version": _ARTIFACT_SCHEMA_VERSION,
                        "model_config": config.model.as_dict(),
                        "state_dict": canonical,
                        "source": {
                            "checkpoint": str(checkpoint),
                            "pipeline_parallel_size": config.pipeline_parallel_size,
                            "tensor_parallel_size": config.tensor_parallel_size,
                        },
                    },
                    temporary,
                )
                temporary.replace(output)
            except BaseException as exception:
                error = exception
                if temporary.exists():
                    temporary.unlink()
        _sync_rank_zero_error(
            error, device, "Rank 0 could not export the inference artifact."
        )
        dist.barrier()
    finally:
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def load_inference_artifact(
    path: Path,
) -> tuple[ModelConfig, dict[str, torch.Tensor], dict[str, Any]]:
    r"""Load and validate a vision inference artifact on CPU.

    **中文：** 返回 model config、canonical state 和来源元数据。
    **English:** Return the model configuration, canonical state, and source
    metadata.

    :param path: Artifact 文件。 / Artifact file.
    :type path: pathlib.Path
    :return: ``(model_config, state_dict, source)``。
    :rtype: tuple[ModelConfig, dict[str, torch.Tensor], dict[str, Any]]
    :raises ValueError: schema 或内容无效。 / If the schema or payload is invalid.
    """
    artifact = torch.load(Path(path), map_location="cpu", weights_only=True)
    if (
        not isinstance(artifact, dict)
        or artifact.get("schema_version") != _ARTIFACT_SCHEMA_VERSION
        or not isinstance(artifact.get("state_dict"), dict)
        or not isinstance(artifact.get("source"), dict)
    ):
        raise ValueError("Invalid vision inference artifact.")
    model_config = ModelConfig.from_dict(artifact["model_config"])
    state = artifact["state_dict"]
    if not state or any(
        not isinstance(value, torch.Tensor) for value in state.values()
    ):
        raise ValueError("Artifact state_dict must contain tensors.")
    return model_config, state, artifact["source"]


class _IndexedDataset(Dataset):
    def __init__(self, dataset: Dataset, padded_size: int) -> None:
        self.dataset = dataset
        self.padded_size = padded_size

    def __len__(self) -> int:
        return self.padded_size

    def __getitem__(self, index: int):
        valid = index < len(self.dataset)
        source_index = index if valid else 0
        item = self.dataset[source_index]
        if isinstance(item, (tuple, list)) and len(item) == 2:
            image, target = item
            has_target = True
        else:
            image, target, has_target = item, -1, False
        return image, target, source_index, valid, has_target


def _build_loader(
    config: PredictionConfig, *, data_parallel_size: int, data_parallel_rank: int
) -> tuple[DataLoader, int]:
    dataset = _import_object(config.dataset_builder)(**config.dataset_kwargs)
    if not isinstance(dataset, Dataset) or len(dataset) == 0:
        raise ValueError("dataset_builder must return one non-empty Dataset.")
    multiple = data_parallel_size * config.batch_size
    padded_size = ((len(dataset) + multiple - 1) // multiple) * multiple
    indexed = _IndexedDataset(dataset, padded_size)
    sampler = DistributedSampler(
        indexed,
        num_replicas=data_parallel_size,
        rank=data_parallel_rank,
        shuffle=False,
        drop_last=False,
    )
    return (
        DataLoader(
            indexed,
            batch_size=config.batch_size,
            sampler=sampler,
            num_workers=config.workers,
            pin_memory=True,
            persistent_workers=config.workers > 0,
        ),
        len(dataset),
    )


def _open_prediction_shard(path: Path, num_classes: int):
    handle = h5py.File(path, "w", locking=False)
    handle.create_dataset("index", shape=(0,), maxshape=(None,), dtype="i8")
    handle.create_dataset(
        "logits",
        shape=(0, num_classes),
        maxshape=(None, num_classes),
        dtype="f4",
    )
    return handle


def _append_predictions(
    handle: h5py.File,
    indices: torch.Tensor,
    logits: torch.Tensor,
) -> None:
    size = len(indices)
    if not size:
        return
    start = handle["index"].shape[0]
    end = start + size
    for name in ("index", "logits"):
        shape = list(handle[name].shape)
        shape[0] = end
        handle[name].resize(tuple(shape))
    handle["index"][start:end] = indices.numpy()
    handle["logits"][start:end] = logits.float().numpy()


def _merge_prediction_shards(
    output: Path,
    shard_paths: list[Path],
    *,
    dataset_size: int,
    num_classes: int,
    attributes: dict[str, Any],
) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.name}.tmp")
    seen = np.zeros(dataset_size, dtype=np.bool_)
    try:
        with h5py.File(temporary, "w", locking=False) as target:
            target.create_dataset("index", data=np.arange(dataset_size, dtype=np.int64))
            logits = target.create_dataset(
                "logits", (dataset_size, num_classes), dtype="f4"
            )
            for key, value in attributes.items():
                target.attrs[key] = value
            for path in shard_paths:
                with h5py.File(path, "r", locking=False) as shard:
                    indices = shard["index"][:]
                    if len(np.unique(indices)) != len(indices) or np.any(seen[indices]):
                        raise ValueError(
                            "Prediction shards contain duplicate dataset indices."
                        )
                    seen[indices] = True
                    order = np.argsort(indices)
                    logits[indices[order]] = shard["logits"][:][order]
            if not seen.all():
                raise ValueError("Prediction shards do not cover the complete dataset.")
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    for path in shard_paths:
        path.unlink()


# ScheduleGPipe retains training state across repeated calls; inference needs one
# bounded send/recv lifecycle per batch.
class _ForwardPipeline:
    def __init__(
        self,
        module: torch.nn.Module,
        *,
        process_group: Any,
        pipeline_rank: int,
        pipeline_size: int,
        microbatches: int,
        input_shape: tuple[int, ...],
        communication_dtype: torch.dtype,
        device: torch.device,
        batch_first: bool = False,
        time_steps: Optional[int] = None,
    ) -> None:
        self.module = module
        self.batch_first = batch_first
        self.time_steps = time_steps
        self.process_group = process_group
        self.pipeline_rank = pipeline_rank
        self.pipeline_size = pipeline_size
        self.microbatches = microbatches
        self.communication_dtype = communication_dtype
        self.source = (
            dist.get_global_rank(process_group, pipeline_rank - 1)
            if pipeline_rank > 0
            else None
        )
        self.destination = (
            dist.get_global_rank(process_group, pipeline_rank + 1)
            if pipeline_rank + 1 < pipeline_size
            else None
        )
        self.receive_buffer = (
            torch.empty(input_shape, device=device, dtype=communication_dtype)
            if pipeline_rank > 0
            else None
        )

    def step(self, value: Optional[torch.Tensor] = None) -> Optional[torch.Tensor]:
        if self.pipeline_rank == 0:
            if value is None or value.shape[0] % self.microbatches:
                raise ValueError(
                    "Pipeline input batch must be divisible by pipeline_microbatches."
                )
            inputs = value.chunk(self.microbatches, dim=0)
        else:
            inputs = (self.receive_buffer,) * self.microbatches

        outputs = []
        for input_value in inputs:
            if self.source is not None:
                dist.recv(input_value, src=self.source, group=self.process_group)
            if self.time_steps is not None:
                input_value = _classification_sequence(
                    input_value, self.time_steps, "NCHW"
                )
            elif self.batch_first:
                input_value = input_value.transpose(0, 1).contiguous()
            output = self.module(input_value)
            functional.reset_net(self.module)
            if self.destination is not None:
                dist.send(
                    output.to(self.communication_dtype, copy=False).contiguous(),
                    dst=self.destination,
                    group=self.process_group,
                )
            else:
                outputs.append(_classification_logits(output))
        if self.pipeline_size > 1:
            dist.barrier(group=self.process_group)
        return torch.cat(outputs) if outputs else None


def _run_classification(
    config: PredictionConfig,
    *,
    mode: Literal["evaluate", "predict"],
    output: Optional[Path] = None,
) -> Optional[dict[str, float]]:
    evaluate = mode == "evaluate"
    if (evaluate and output is not None) or (not evaluate and output is None):
        raise ValueError("evaluate mode cannot write output; predict mode requires it.")
    if not torch.cuda.is_available():
        raise RuntimeError("Distributed vision inference requires CUDA.")
    if not evaluate and output.exists():
        raise FileExistsError(f"Prediction output already exists: {output}")

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_here = not dist.is_initialized()
    if initialized_here:
        dist.init_process_group("nccl", device_id=device)
    prediction_shard = None
    shard_path = None
    try:
        world_size = dist.get_world_size()
        model_parallel_size = (
            config.pipeline_parallel_size * config.tensor_parallel_size
        )
        if world_size % model_parallel_size:
            raise ValueError("world_size must be divisible by PP * TP.")
        rank = dist.get_rank()
        data_parallel_size = world_size // model_parallel_size
        tensor_rank = rank % config.tensor_parallel_size
        pipeline_rank = (
            rank // config.tensor_parallel_size
        ) % config.pipeline_parallel_size
        data_parallel_rank = rank // model_parallel_size

        from torch.distributed.device_mesh import init_device_mesh

        mesh = init_device_mesh(
            "cuda",
            (
                data_parallel_size,
                config.pipeline_parallel_size,
                config.tensor_parallel_size,
            ),
            mesh_dim_names=("dp", "pp", "tp"),
        )
        data_mesh, pipeline_mesh, tensor_mesh = mesh["dp"], mesh["pp"], mesh["tp"]
        data_group = data_mesh.get_group() if data_parallel_size > 1 else None
        pipeline_group = (
            pipeline_mesh.get_group() if config.pipeline_parallel_size > 1 else None
        )
        tensor_group = (
            tensor_mesh.get_group() if config.tensor_parallel_size > 1 else None
        )

        model_config, state_dict, source = load_inference_artifact(config.artifact)
        if config.pipeline_parallel_size > 1 and model_config.step_mode == "s":
            raise ValueError("Vision PP currently requires step_mode='m'.")
        torch.manual_seed(config.seed)
        builder_cls = model_config.get_builder_cls()
        builder = builder_cls(model_config)
        model, fsdp_roots, pipeline_input_shape, _ = builder.build_for_inference(
            state_dict,
            process_group=tensor_group,
            pipeline_rank=pipeline_rank,
            pipeline_size=config.pipeline_parallel_size,
            pipeline_microbatches=config.pipeline_microbatches,
            device=device,
            micro_batch_size=config.batch_size,
        )
        precision = prepare_model_for_precision(model, device, config.precision)
        model = precision.model
        if config.data_parallel == "fsdp2":
            model = _wrap_data_parallel(
                model,
                data_parallel="fsdp2",
                pipeline_parallel_size=config.pipeline_parallel_size,
                step_mode=model_config.step_mode,
                precision=config.precision,
                device=device,
                dp_size=data_parallel_size,
                dp_group=data_group,
                dp_mesh=data_mesh,
                fsdp_roots=fsdp_roots,
            )
        if config.compile:
            model = torch.compile(model)
        model.eval()

        schedule = None
        if config.pipeline_parallel_size > 1:
            stage_dtype = {
                "fp32": torch.float32,
                "bf16": torch.bfloat16,
            }[config.precision.mode]
            static_input = pipeline_rank == 0 and config.input_layout == "NCHW"
            stage_input_shape = (
                (pipeline_input_shape[0], *pipeline_input_shape[2:])
                if static_input
                else pipeline_input_shape
            )
            schedule = _ForwardPipeline(
                model,
                process_group=pipeline_group,
                pipeline_rank=pipeline_rank,
                pipeline_size=config.pipeline_parallel_size,
                microbatches=config.pipeline_microbatches,
                input_shape=stage_input_shape,
                communication_dtype=stage_dtype,
                device=device,
                batch_first=pipeline_rank == 0 and not static_input,
                time_steps=model_config.time_steps if static_input else None,
            )

        loader, dataset_size = _build_loader(
            config,
            data_parallel_size=data_parallel_size,
            data_parallel_rank=data_parallel_rank,
        )
        if evaluate:
            loss_function = functools.partial(
                _import_object(config.loss_function), **config.loss_kwargs
            )
        else:
            loss_function = None
        totals = torch.zeros(3, device=device, dtype=torch.float64)
        output_rank = (
            pipeline_rank == config.pipeline_parallel_size - 1 and tensor_rank == 0
        )
        if not evaluate and output_rank:
            shard_path = output.with_name(f".{output.name}.rank-{rank:05d}.h5")
            if shard_path.exists():
                raise FileExistsError(f"Prediction shard already exists: {shard_path}")
            prediction_shard = _open_prediction_shard(
                shard_path, model_config.num_classes
            )

        def forward_batch(batch):
            images, targets, indices, valid, has_target = batch
            if schedule is None or pipeline_rank == 0:
                images = images.to(device, non_blocking=True)
            if evaluate and output_rank:
                targets = targets.to(device, non_blocking=True)
            with precision.autocast_context(group=data_group):
                if schedule is None:
                    logits = _forward_classification(
                        model,
                        images,
                        model_config.time_steps,
                        model_config.step_mode,
                        config.input_layout,
                    )
                else:
                    sequence = (
                        images
                        if config.input_layout == "NCHW"
                        else _classification_sequence(
                            images,
                            model_config.time_steps,
                            config.input_layout,
                            batch_first=True,
                        )
                    )
                    logits = (
                        schedule.step(sequence)
                        if pipeline_rank == 0
                        else schedule.step()
                    )
            if schedule is None:
                functional.reset_net(model)
            return logits, targets, indices, valid, has_target

        with torch.inference_mode():
            if evaluate and config.timing_warmup_batches:
                if config.timing_warmup_batches > len(loader):
                    raise ValueError(
                        "timing_warmup_batches cannot exceed evaluation batches."
                    )
                warmup_iterator = iter(loader)
                for _ in range(config.timing_warmup_batches):
                    forward_batch(next(warmup_iterator))
                torch.cuda.synchronize(device)

        if evaluate:
            torch.cuda.reset_peak_memory_stats(device)
        dist.barrier()
        elapsed_seconds = 0.0
        with torch.inference_mode():
            for batch in loader:
                batch_started = time.perf_counter() if evaluate else 0.0
                logits, targets, indices, valid, has_target = forward_batch(batch)
                if output_rank:
                    valid_device = valid.to(device=device, dtype=torch.bool)
                    if evaluate:
                        target_device = has_target.to(device=device, dtype=torch.bool)
                        selected = valid_device & target_device
                        if selected.any():
                            count = selected.sum()
                            loss = loss_function(logits[selected], targets[selected])
                            totals[0] += loss.double() * count
                            totals[1] += (
                                logits[selected].argmax(1) == targets[selected]
                            ).sum()
                            totals[2] += count
                    if prediction_shard is not None:
                        selected_cpu = valid.to(torch.bool)
                        _append_predictions(
                            prediction_shard,
                            indices[selected_cpu],
                            logits[selected_cpu.to(device)].detach().float().cpu(),
                        )
                torch.cuda.synchronize(device)
                if evaluate:
                    elapsed_seconds += time.perf_counter() - batch_started
        if prediction_shard is not None:
            prediction_shard.close()
            prediction_shard = None

        if not evaluate:
            dist.barrier()
            error = None
            if rank == 0:
                try:
                    output_ranks = [
                        dp_rank * model_parallel_size
                        + (config.pipeline_parallel_size - 1)
                        * config.tensor_parallel_size
                        for dp_rank in range(data_parallel_size)
                    ]
                    _merge_prediction_shards(
                        output,
                        [
                            output.with_name(
                                f".{output.name}.rank-{last_stage_rank:05d}.h5"
                            )
                            for last_stage_rank in output_ranks
                        ],
                        dataset_size=dataset_size,
                        num_classes=model_config.num_classes,
                        attributes={
                            "artifact": str(config.artifact),
                            "precision": json.dumps(
                                asdict(config.precision), sort_keys=True
                            ),
                            "source_checkpoint": source["checkpoint"],
                        },
                    )
                except BaseException as exception:
                    error = exception
            _sync_rank_zero_error(
                error, device, "Rank 0 could not merge prediction shards."
            )
            return None

        elapsed = torch.tensor(elapsed_seconds, device=device)
        dist.all_reduce(elapsed, op=dist.ReduceOp.MAX)
        dist.all_reduce(totals)
        peak = torch.tensor(
            float(torch.cuda.max_memory_allocated(device)), device=device
        )
        dist.all_reduce(peak, op=dist.ReduceOp.MAX)
        valid_targets = int(totals[2].item())
        if valid_targets != dataset_size:
            raise ValueError(
                "evaluate_classification requires a target for every dataset sample."
            )
        return {
            "loss": (totals[0] / totals[2]).item(),
            "accuracy": (totals[1] / totals[2]).item(),
            "samples": float(dataset_size),
            "inference_seconds": elapsed.item(),
            "images_per_second": dataset_size / elapsed.item(),
            "peak_memory_bytes": peak.item(),
            "data_parallel_size": float(data_parallel_size),
            "pipeline_parallel_size": float(config.pipeline_parallel_size),
            "tensor_parallel_size": float(config.tensor_parallel_size),
        }
    finally:
        if prediction_shard is not None:
            prediction_shard.close()
        if shard_path is not None and shard_path.exists():
            shard_path.unlink()
        if initialized_here and dist.is_initialized():
            dist.destroy_process_group()


def evaluate_classification(config: EvaluationConfig) -> dict[str, float]:
    r"""Evaluate one distributed vision classification dataset.

    **中文：** 对 dataset builder 返回的 ``(image, target)`` 数据集计算
    全局 loss、accuracy 和性能指标。

    **English:** Compute global loss, accuracy, and performance metrics for the
    ``(image, target)`` dataset returned by the configured builder.

    :param config: 推理配置。 / Inference configuration.
    :type config: spikingjelly.activation_based.distributed.vision.EvaluationConfig
    :return: 全局评测与性能指标。 / Global evaluation and performance metrics.
    :rtype: dict[str, float]
    :raises TypeError: ``config`` 不是 :class:`EvaluationConfig`。 / If ``config``
        is not :class:`EvaluationConfig`.
    """
    if not isinstance(config, EvaluationConfig):
        raise TypeError("evaluate_classification requires EvaluationConfig.")
    return _run_classification(config, mode="evaluate")


def predict_classification(config: PredictionConfig, output: Path) -> None:
    r"""Predict one distributed vision dataset and write one ordered HDF5 file.

    **中文：** 按 dataset index 保存 logits。dataset 可以返回 image 或
    ``(image, target)``；target 被忽略，函数不计算或返回评测指标。填充样本
    不会写入结果。

    **English:** Save logits by dataset index. The dataset may return an image or
    ``(image, target)``; targets are ignored and no evaluation metrics are
    computed or returned. Padded samples are excluded.

    :param config: 推理配置。 / Inference configuration.
    :type config: spikingjelly.activation_based.distributed.vision.PredictionConfig
    :param output: 新 HDF5 文件。 / New HDF5 file.
    :type output: pathlib.Path
    :return: 无。 / None.
    :rtype: None
    :raises TypeError: ``config`` 不是 :class:`PredictionConfig`，或是
        :class:`EvaluationConfig`。 / If ``config`` is not a prediction-only
        :class:`PredictionConfig`.
    """
    if not isinstance(config, PredictionConfig) or isinstance(config, EvaluationConfig):
        raise TypeError("predict_classification requires PredictionConfig.")
    _run_classification(config, mode="predict", output=Path(output))


__all__ = [
    "evaluate_classification",
    "export_inference_artifact",
    "load_inference_artifact",
    "predict_classification",
]
