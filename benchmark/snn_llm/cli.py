from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset

from spikingjelly.logger import logger


class _TokenDataset(Dataset):
    def __init__(
        self, tokens: np.ndarray | None, sample_count: int, sequence_length: int
    ):
        if sample_count and (
            tokens is None
            or tokens.ndim != 2
            or tokens.shape[0] == 0
            or tokens.shape[1] != sequence_length + 1
            or not np.issubdtype(tokens.dtype, np.integer)
        ):
            raise ValueError(
                f"Token data must be an integer array with shape "
                f"[N, {sequence_length + 1}] and N > 0."
            )
        self.tokens = tokens
        self.sample_count = sample_count

    def __len__(self) -> int:
        return self.sample_count

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        row = torch.from_numpy(
            np.array(self.tokens[index % len(self.tokens)], copy=True)
        ).long()
        return {"input_ids": row[:-1], "labels": row[1:]}


def _dataset_provider(
    sample_counts: tuple[int, int, int],
    *,
    data_dir: Path,
    sequence_length: int,
) -> tuple[_TokenDataset, _TokenDataset, _TokenDataset]:
    paths = [data_dir / f"{split}.npy" for split in ("train", "valid", "test")]
    arrays = [
        np.load(path, mmap_mode="r") if count and path.is_file() else None
        for path, count in zip(paths, sample_counts, strict=True)
    ]
    return tuple(
        _TokenDataset(array, count, sequence_length)
        for array, count in zip(arrays, sample_counts, strict=True)
    )


def _report(metrics: dict[str, float], output: Path) -> None:
    if int(os.environ.get("RANK", "0")) == 0:
        output.mkdir(parents=True, exist_ok=True)
        (output / "metrics.json").write_text(
            json.dumps(metrics, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        logger.info("{}", metrics)


def _run_spikelm(args: argparse.Namespace) -> None:
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import TransformerConfig

    from spikingjelly.activation_based.distributed.llm import (
        TrainingConfig,
        plan_training,
        train,
    )

    from .spikelm import SpikeLMConfig

    explicit_topology = any(
        size is not None
        for size in (
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            args.context_parallel_size,
        )
    )
    transformer = TransformerConfig(
        num_layers=24,
        hidden_size=2048,
        num_attention_heads=16,
        ffn_hidden_size=8192,
        tensor_model_parallel_size=args.tensor_parallel_size
        if args.tensor_parallel_size is not None
        else 1,
        pipeline_model_parallel_size=args.pipeline_parallel_size
        if args.pipeline_parallel_size is not None
        else 1,
        context_parallel_size=args.context_parallel_size
        if args.context_parallel_size is not None
        else 1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
    )
    model = SpikeLMConfig(
        transformer=transformer,
        vocab_size=50304,
        max_sequence_length=128,
        time_steps=4,
    )
    optimizer = OptimizerConfig(
        lr=3e-4,
        min_lr=3e-5,
        weight_decay=0.01,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    training = TrainingConfig(
        model=model,
        optimizer=optimizer,
        dataset_builder="benchmark.snn_llm.cli._dataset_provider",
        dataset_kwargs={
            "data_dir": args.data,
            "sequence_length": model.max_sequence_length,
        },
        sequence_length=model.max_sequence_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        train_steps=args.train_steps,
        timing_warmup_steps=args.timing_warmup_steps,
        log_interval=10,
        lr_warmup_steps=min(10, args.train_steps - 1),
        checkpoint_dir=args.output,
        checkpoint_interval=args.checkpoint_interval,
        resume=args.resume,
        use_snn_memopt=args.memopt,
    )
    if not explicit_topology:
        training = plan_training(
            training,
            world_size=int(os.environ["WORLD_SIZE"]),
            device_memory_bytes=int(args.device_memory_gib * 1024**3)
            if args.device_memory_gib is not None
            else torch.cuda.get_device_properties(
                torch.cuda.current_device()
            ).total_memory,
            objective=args.objective,
            memory_fraction=args.memory_fraction,
        )
    metrics = train(training)
    _report(metrics, args.output)


def _run_qwen2(args: argparse.Namespace) -> None:
    from megatron.core.optimizer import OptimizerConfig
    from megatron.core.transformer import TransformerConfig
    from transformers import AutoConfig

    from spikingjelly.activation_based.ann2snn.recipes.qwen2 import Qwen2SNNCalibration
    from spikingjelly.activation_based.distributed.llm import (
        TrainingConfig,
        plan_training,
        train,
    )

    from .qwen2 import Qwen2Config

    source_config = AutoConfig.from_pretrained(args.source)
    calibration = Qwen2SNNCalibration.from_state_dict(
        torch.load(args.calibration, map_location="cpu", weights_only=True)
    )
    train_tokens = np.load(args.data / "train.npy", mmap_mode="r")
    if train_tokens.ndim != 2:
        raise ValueError("Qwen token data must be a two-dimensional NPY array.")
    sequence_length = train_tokens.shape[1] - 1
    if not 0 < sequence_length <= source_config.max_position_embeddings:
        raise ValueError("Token sequence length must fit the Qwen context window.")
    transformer = TransformerConfig(
        num_layers=source_config.num_hidden_layers,
        hidden_size=source_config.hidden_size,
        num_attention_heads=source_config.num_attention_heads,
        num_query_groups=source_config.num_key_value_heads,
        kv_channels=source_config.hidden_size // source_config.num_attention_heads,
        ffn_hidden_size=source_config.intermediate_size,
        tensor_model_parallel_size=1,
        pipeline_model_parallel_size=1,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        normalization="RMSNorm",
        layernorm_epsilon=source_config.rms_norm_eps,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=bool(getattr(source_config, "attention_bias", True)),
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        fp8="hybrid",
        fp8_recipe="delayed",
    )
    model = Qwen2Config(
        transformer=transformer,
        vocab_size=int(source_config.vocab_size),
        max_sequence_length=int(source_config.max_position_embeddings),
        time_steps=calibration.time_steps,
        share_embeddings_and_output_weights=bool(source_config.tie_word_embeddings),
        source_path=args.source,
        calibration_path=args.calibration,
    )
    optimizer = OptimizerConfig(
        lr=2e-5,
        min_lr=2e-6,
        weight_decay=0.1,
        bf16=True,
        params_dtype=torch.bfloat16,
        fp8_recipe="delayed",
        use_distributed_optimizer=True,
    )
    training = TrainingConfig(
        model=model,
        optimizer=optimizer,
        dataset_builder="benchmark.snn_llm.cli._dataset_provider",
        dataset_kwargs={
            "data_dir": args.data,
            "sequence_length": sequence_length,
        },
        sequence_length=sequence_length,
        micro_batch_size=1,
        global_batch_size=8,
        train_steps=100,
        log_interval=5,
        lr_warmup_steps=5,
        checkpoint_dir=args.output,
        checkpoint_interval=50,
        resume=args.resume,
    )
    training = plan_training(
        training,
        world_size=int(os.environ["WORLD_SIZE"]),
        device_memory_bytes=int(args.device_memory_gib * 1024**3)
        if args.device_memory_gib is not None
        else torch.cuda.get_device_properties(torch.cuda.current_device()).total_memory,
        objective=args.objective,
        memory_fraction=args.memory_fraction,
    )
    metrics = train(training)
    _report(metrics, args.output)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCore SNN Transformer recipes")
    commands = parser.add_subparsers(dest="command", required=True)
    spikelm = commands.add_parser("spikelm-pretrain")
    spikelm.add_argument("--data", required=True, type=Path)
    spikelm.add_argument("--output", required=True, type=Path)
    spikelm.add_argument("--resume", type=Path)
    spikelm.add_argument(
        "--objective", choices=("throughput", "memory"), default="throughput"
    )
    spikelm.add_argument("--memory-fraction", type=float, default=0.9)
    spikelm.add_argument("--device-memory-gib", type=float)
    spikelm.add_argument("--tensor-parallel-size", type=int)
    spikelm.add_argument("--pipeline-parallel-size", type=int)
    spikelm.add_argument("--context-parallel-size", type=int)
    spikelm.add_argument("--train-steps", type=int, default=200)
    spikelm.add_argument("--micro-batch-size", type=int, default=1)
    spikelm.add_argument("--global-batch-size", type=int, default=8)
    spikelm.add_argument("--timing-warmup-steps", type=int, default=0)
    spikelm.add_argument("--checkpoint-interval", type=int, default=100)
    spikelm.add_argument("--memopt", action="store_true")
    qwen2 = commands.add_parser("qwen2-finetune")
    qwen2.add_argument("--data", required=True, type=Path)
    qwen2.add_argument("--source", required=True, type=Path)
    qwen2.add_argument("--calibration", required=True, type=Path)
    qwen2.add_argument("--output", required=True, type=Path)
    qwen2.add_argument("--resume", type=Path)
    qwen2.add_argument(
        "--objective", choices=("throughput", "memory"), default="throughput"
    )
    qwen2.add_argument("--memory-fraction", type=float, default=0.9)
    qwen2.add_argument("--device-memory-gib", type=float)
    return parser.parse_args()


def main() -> None:
    logger.enable("spikingjelly")
    args = _parse_args()
    if args.command == "spikelm-pretrain":
        _run_spikelm(args)
    else:
        _run_qwen2(args)


if __name__ == "__main__":
    main()
