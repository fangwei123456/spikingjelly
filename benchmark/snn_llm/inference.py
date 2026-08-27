from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from benchmark.snn_llm.cli import _TokenDataset
from benchmark.snn_llm.spikelm import SpikeLMConfig, export_sglang
from spikingjelly.activation_based.distributed import llm


def build_token_dataset(
    data_dir: Path,
    sequence_length: int,
    split: str = "valid",
    samples: int | None = None,
) -> Dataset:
    tokens = np.load(Path(data_dir) / f"{split}.npy", mmap_mode="r")
    return _TokenDataset(
        tokens, len(tokens) if samples is None else samples, sequence_length
    )


def build_training_datasets(sample_counts, **kwargs):
    from benchmark.snn_llm.cli import _dataset_provider

    return _dataset_provider(sample_counts, **kwargs)


def _model(args: argparse.Namespace):
    from megatron.core.transformer import TransformerConfig

    transformer = TransformerConfig(
        num_layers=args.layers,
        hidden_size=args.hidden_size,
        num_attention_heads=args.heads,
        ffn_hidden_size=args.ffn_hidden_size,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return SpikeLMConfig(
        transformer=transformer,
        vocab_size=args.vocab_size,
        max_sequence_length=args.sequence_length,
        time_steps=args.time_steps,
    )


def _train(args: argparse.Namespace, model: SpikeLMConfig) -> dict[str, float]:
    from megatron.core.optimizer import OptimizerConfig

    optimizer = OptimizerConfig(
        lr=1e-3,
        min_lr=1e-4,
        bf16=True,
        params_dtype=torch.bfloat16,
        use_distributed_optimizer=True,
    )
    return llm.train(
        llm.TrainingConfig(
            model=model,
            optimizer=optimizer,
            dataset_builder=f"{__name__}.build_training_datasets",
            dataset_kwargs={
                "data_dir": args.data,
                "sequence_length": args.sequence_length,
            },
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            global_batch_size=args.global_batch_size,
            train_steps=args.steps,
            eval_interval=1,
            eval_steps=1,
            log_interval=1,
            checkpoint_dir=args.output,
            checkpoint_interval=args.steps,
        )
    )


def _evaluate(args: argparse.Namespace, model: SpikeLMConfig) -> dict[str, float]:
    return llm.evaluate(
        llm.EvaluationConfig(
            model=model,
            checkpoint=args.checkpoint,
            dataset_builder=f"{__name__}.build_token_dataset",
            dataset_kwargs={
                "data_dir": args.data,
                "sequence_length": args.sequence_length,
            },
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
        )
    )


def _generate(args: argparse.Namespace, model: SpikeLMConfig):
    prompts = torch.from_numpy(
        np.array(
            np.load(args.data / "valid.npy", mmap_mode="r")[
                : args.prompt_count, : args.prompt_length
            ],
            copy=True,
        )
    ).long()
    return llm.generate(
        llm.MCoreGenerationConfig(
            model=model,
            checkpoint=args.checkpoint,
            max_new_tokens=args.max_new_tokens,
        ),
        prompts,
    )


def _initialize_checkpoint(
    args: argparse.Namespace, model_config: SpikeLMConfig
) -> dict[str, object]:
    import torch.distributed as dist
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    initialized_distributed = not dist.is_initialized()
    if initialized_distributed:
        dist.init_process_group("nccl", device_id=device)
    initialized_model_parallel = not parallel_state.model_parallel_is_initialized()
    try:
        if initialized_model_parallel:
            parallel_state.initialize_model_parallel(
                tensor_model_parallel_size=args.tensor_parallel_size,
                pipeline_model_parallel_size=args.pipeline_parallel_size,
                context_parallel_size=args.context_parallel_size,
                expert_model_parallel_size=1,
            )
        model_parallel_cuda_manual_seed(1234)
        provider, _ = model_config.get_builder_cls()(model_config).build(
            resume=True,
        )
        model = provider(
            parallel_state.is_pipeline_first_stage(),
            parallel_state.is_pipeline_last_stage(),
        ).cuda(device)
        metadata = {
            "dp_cp_group": parallel_state.get_data_parallel_group(
                with_context_parallel=True
            )
        }
        recipe = {
            **model.checkpoint_metadata,
            "model": model_config._checkpoint_metadata(),
            "mcore_recompute_granularity": model_config.transformer.recompute_granularity,
            "mcore_recompute_modules": model_config.transformer.recompute_modules,
        }
        if dist.get_rank() == 0:
            args.output.mkdir(parents=True, exist_ok=True)
        dist.barrier()
        dist_checkpointing.save(
            {
                "model": model.sharded_state_dict(metadata=metadata),
                "recipe": recipe,
            },
            str(args.output),
        )
        return {"checkpoint": str(args.output), "recipe": recipe}
    finally:
        if (
            initialized_model_parallel
            and parallel_state.model_parallel_is_initialized()
        ):
            parallel_state.destroy_model_parallel()
        if initialized_distributed and dist.is_initialized():
            dist.destroy_process_group()


def _export_sglang(
    args: argparse.Namespace, model_config: SpikeLMConfig
) -> dict[str, object]:
    provider, _ = model_config.get_builder_cls()(model_config).build(
        resume=True,
    )
    export_sglang(
        model_config,
        provider,
        args.checkpoint,
        args.output,
    )
    index = json.loads(
        (args.output / "model.safetensors.index.json").read_text(encoding="utf-8")
    )
    return {
        "artifact": str(args.output),
        "tensor_count": len(index["weight_map"]),
        "parameter_count": index["metadata"]["parameter_count"],
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCore distributed inference smoke")
    parser.add_argument(
        "command",
        choices=("train", "evaluate", "generate", "initialize", "export-sglang"),
    )
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--layers", type=int, default=2)
    parser.add_argument("--hidden-size", type=int, default=64)
    parser.add_argument("--heads", type=int, default=4)
    parser.add_argument("--ffn-hidden-size", type=int, default=128)
    parser.add_argument("--vocab-size", type=int, default=128)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--time-steps", type=int, default=2)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=2)
    parser.add_argument("--steps", type=int, default=1)
    parser.add_argument("--prompt-count", type=int, default=4)
    parser.add_argument("--prompt-length", type=int, default=8)
    parser.add_argument("--max-new-tokens", type=int, default=4)
    args = parser.parse_args()
    if args.command in {"train", "initialize", "export-sglang"} and args.output is None:
        parser.error("train, initialize, and export-sglang require --output")
    if args.command not in {"train", "initialize"} and args.checkpoint is None:
        parser.error("evaluate, generate, and export-sglang require --checkpoint")
    return args


def main() -> None:
    args = _parse_args()
    model = _model(args)
    if args.command == "train":
        result = _train(args, model)
    elif args.command == "evaluate":
        result = _evaluate(args, model)
    elif args.command == "generate":
        started = time.perf_counter()
        generated = _generate(args, model)
        elapsed = time.perf_counter() - started
        result = (
            {
                "outputs": generated.tolist(),
                "inference_seconds": elapsed,
                "generated_tokens_per_second": generated.shape[0]
                * args.max_new_tokens
                / elapsed,
            }
            if generated is not None
            else None
        )
    elif args.command == "initialize":
        result = _initialize_checkpoint(args, model)
    else:
        result = _export_sglang(args, model)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
