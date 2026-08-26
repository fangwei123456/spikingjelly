from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from benchmark.snn_llm.qwen2 import Qwen2Config, export_sglang
from spikingjelly.activation_based.ann2snn.recipes.qwen2 import Qwen2SNNCalibration
from spikingjelly.activation_based.distributed import llm


def _model(args: argparse.Namespace) -> Qwen2Config:
    from megatron.core.transformer import TransformerConfig
    from transformers import AutoConfig

    source = AutoConfig.from_pretrained(args.source)
    calibration = Qwen2SNNCalibration.from_state_dict(
        torch.load(args.calibration, map_location="cpu", weights_only=True)
    )
    transformer = TransformerConfig(
        num_layers=source.num_hidden_layers,
        hidden_size=source.hidden_size,
        num_attention_heads=source.num_attention_heads,
        num_query_groups=source.num_key_value_heads,
        kv_channels=source.hidden_size // source.num_attention_heads,
        ffn_hidden_size=source.intermediate_size,
        tensor_model_parallel_size=args.tensor_parallel_size,
        pipeline_model_parallel_size=args.pipeline_parallel_size,
        context_parallel_size=args.context_parallel_size,
        sequence_parallel=False,
        calculate_per_token_loss=True,
        normalization="RMSNorm",
        layernorm_epsilon=source.rms_norm_eps,
        gated_linear_unit=True,
        activation_func=F.silu,
        add_bias_linear=False,
        add_qkv_bias=bool(getattr(source, "attention_bias", True)),
        params_dtype=torch.bfloat16,
        pipeline_dtype=torch.bfloat16,
        bf16=True,
        hidden_dropout=0.0,
        attention_dropout=0.0,
    )
    return Qwen2Config(
        transformer=transformer,
        vocab_size=int(source.vocab_size),
        max_sequence_length=int(source.max_position_embeddings),
        time_steps=calibration.time_steps,
        share_embeddings_and_output_weights=bool(source.tie_word_embeddings),
        source_path=args.source,
        calibration_path=args.calibration,
    )


def _train(args: argparse.Namespace, model: Qwen2Config) -> dict[str, float]:
    from megatron.core.optimizer import OptimizerConfig

    return llm.train(
        llm.TrainingConfig(
            model=model,
            optimizer=OptimizerConfig(
                lr=1e-4,
                min_lr=1e-5,
                bf16=True,
                params_dtype=torch.bfloat16,
                use_distributed_optimizer=True,
            ),
            dataset_builder="benchmark.snn_llm.inference.build_training_datasets",
            dataset_kwargs={
                "data_dir": args.data,
                "sequence_length": args.sequence_length,
            },
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            timing_warmup_steps=args.timing_warmup_batches,
            global_batch_size=args.global_batch_size,
            train_steps=args.steps,
            eval_interval=1,
            eval_steps=1,
            log_interval=1,
            checkpoint_dir=args.output,
            checkpoint_interval=args.steps,
        )
    )


def _evaluate(args: argparse.Namespace, model: Qwen2Config) -> dict[str, float]:
    return llm.evaluate(
        llm.EvaluationConfig(
            model=model,
            checkpoint=args.checkpoint,
            dataset_builder="benchmark.snn_llm.inference.build_token_dataset",
            dataset_kwargs={
                "data_dir": args.data,
                "sequence_length": args.sequence_length,
                "samples": args.evaluation_samples,
            },
            sequence_length=args.sequence_length,
            micro_batch_size=args.micro_batch_size,
            pipeline_microbatches=args.pipeline_microbatches,
            timing_warmup_batches=args.timing_warmup_batches,
        )
    )


def _generate(args: argparse.Namespace, model: Qwen2Config):
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
    args: argparse.Namespace, model_config: Qwen2Config
) -> dict[str, object]:
    import torch.distributed as dist
    from megatron.core import dist_checkpointing, parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed

    if any(
        size != 1
        for size in (
            args.tensor_parallel_size,
            args.pipeline_parallel_size,
            args.context_parallel_size,
        )
    ):
        raise ValueError("initialize requires TP=PP=CP=1.")
    args.output.mkdir(parents=True)
    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29583",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", 0),
    )
    try:
        parallel_state.initialize_model_parallel()
        model_parallel_cuda_manual_seed(1234)
        provider, _ = model_config.get_builder_cls()(model_config).build(
            memopt_level=0,
            memopt_checkpoint_budget="memory",
            memopt_compress_inputs=False,
            resume=False,
        )
        model = provider(True, True).cuda()
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
        dist_checkpointing.save(
            {
                "model": model.sharded_state_dict(metadata=metadata),
                "recipe": recipe,
            },
            str(args.output),
        )
        return {"checkpoint": str(args.output), "recipe": recipe}
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _export_sglang(
    args: argparse.Namespace, model_config: Qwen2Config
) -> dict[str, object]:
    provider, _ = model_config.get_builder_cls()(model_config).build(
        memopt_level=0,
        memopt_checkpoint_budget="memory",
        memopt_compress_inputs=False,
        resume=True,
    )
    export_sglang(
        model_config,
        provider,
        args.checkpoint,
        args.output,
        tokenizer=args.source,
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
    parser = argparse.ArgumentParser(description="Qwen2 MCore inference workflow")
    parser.add_argument(
        "command",
        choices=(
            "train",
            "evaluate",
            "generate",
            "initialize",
            "export-sglang",
        ),
    )
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--calibration", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--pipeline-microbatches", type=int, default=1)
    parser.add_argument("--timing-warmup-batches", type=int, default=0)
    parser.add_argument("--evaluation-samples", type=int)
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
    if args.evaluation_samples is not None and args.evaluation_samples <= 0:
        parser.error("evaluation-samples must be positive")
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
