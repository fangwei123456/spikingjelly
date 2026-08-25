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
from benchmark.snn_llm.spikelm import SpikeLMConfig
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


def _export_sglang(
    args: argparse.Namespace, model_config: SpikeLMConfig
) -> dict[str, object]:
    if (
        args.tensor_parallel_size != 1
        or args.pipeline_parallel_size != 1
        or args.context_parallel_size != 1
    ):
        raise ValueError("SpikeLM SGLang export currently requires TP=PP=CP=1.")
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError(f"Export directory is not empty: {args.output}")
    try:
        from safetensors.torch import save_file
    except ImportError as error:
        raise ImportError(
            "SGLang export requires spikingjelly[sglang-export]."
        ) from error
    import torch.distributed as dist
    from megatron.core import parallel_state
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.utils import unwrap_model

    from spikingjelly.activation_based.distributed.llm.inference import (
        load_for_inference,
    )

    torch.cuda.set_device(0)
    dist.init_process_group(
        "nccl",
        init_method="tcp://127.0.0.1:29579",
        rank=0,
        world_size=1,
        device_id=torch.device("cuda", 0),
    )
    try:
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=1,
            pipeline_model_parallel_size=1,
            context_parallel_size=1,
        )
        model_parallel_cuda_manual_seed(1234)
        provider, _ = model_config.get_builder_cls()(model_config).build(
            use_snn_memopt=False, resume=True
        )
        model = load_for_inference(model_config.transformer, provider, args.checkpoint)
        source = unwrap_model(model).state_dict()
        weights = {
            "model.embedding.weight": source["embedding.word_embeddings.weight"]
            .detach()
            .cpu(),
            "model.final_norm.weight": source["decoder.final_layernorm.weight"]
            .detach()
            .cpu(),
            "model.final_norm.bias": source["decoder.final_layernorm.bias"]
            .detach()
            .cpu(),
            "lm_head.weight": source["output_layer.weight"].detach().cpu(),
        }
        head_count = model_config.transformer.num_attention_heads
        head_dim = model_config.transformer.hidden_size // head_count

        def reorder_qkv(value: torch.Tensor) -> torch.Tensor:
            shape = value.shape
            grouped = value.reshape(head_count, 3, head_dim, *shape[1:])
            return torch.cat(
                tuple(grouped[:, index].reshape(-1, *shape[1:]) for index in range(3))
            )

        for index in range(model_config.transformer.num_layers):
            source_prefix = f"decoder.layers.{index}."
            target_prefix = f"model.layers.{index}."
            mapping = {
                "input_layernorm.weight": "attn_norm.norm.weight",
                "input_layernorm.bias": "attn_norm.norm.bias",
                "input_layernorm.spike.amplitude": "attn_norm.amplitude",
                "self_attention.linear_proj.weight": "attn.proj.weight",
                "self_attention.linear_proj.bias": "attn.proj.bias",
                "pre_mlp_layernorm.weight": "mlp_norm.norm.weight",
                "pre_mlp_layernorm.bias": "mlp_norm.norm.bias",
                "pre_mlp_layernorm.spike.amplitude": "mlp_norm.amplitude",
                "mlp.linear_fc1.weight": "mlp.fc1.weight",
                "mlp.linear_fc1.bias": "mlp.fc1.bias",
                "mlp.linear_fc2.weight": "mlp.fc2.weight",
                "mlp.linear_fc2.bias": "mlp.fc2.bias",
            }
            for source_name, target_name in mapping.items():
                weights[target_prefix + target_name] = (
                    source[source_prefix + source_name].detach().cpu()
                )
            weights[target_prefix + "attn.qkv.weight"] = (
                reorder_qkv(source[source_prefix + "self_attention.linear_qkv.weight"])
                .detach()
                .cpu()
            )
            weights[target_prefix + "attn.qkv.bias"] = (
                reorder_qkv(source[source_prefix + "self_attention.linear_qkv.bias"])
                .detach()
                .cpu()
            )
        args.output.mkdir(parents=True, exist_ok=True)
        save_file(weights, args.output / "model.safetensors")
        exported_config = {
            "architectures": ["SpikingJellySpikeLMForCausalLM"],
            "model_type": "gpt2",
            "vocab_size": model_config.vocab_size,
            "n_embd": model_config.transformer.hidden_size,
            "n_layer": model_config.transformer.num_layers,
            "n_head": model_config.transformer.num_attention_heads
            * model_config.time_steps,
            "n_inner": model_config.transformer.ffn_hidden_size,
            "n_positions": model_config.max_sequence_length,
            "hidden_size": model_config.transformer.hidden_size,
            "num_hidden_layers": model_config.transformer.num_layers,
            "num_attention_heads": model_config.transformer.num_attention_heads
            * model_config.time_steps,
            "num_key_value_heads": model_config.transformer.num_attention_heads
            * model_config.time_steps,
            "head_dim": model_config.transformer.hidden_size
            // model_config.transformer.num_attention_heads,
            "intermediate_size": model_config.transformer.ffn_hidden_size,
            "max_position_embeddings": model_config.max_sequence_length,
            "layer_norm_epsilon": model_config.transformer.layernorm_epsilon,
            "rope_theta": 10000.0,
            "tie_word_embeddings": False,
            "torch_dtype": "bfloat16",
            "bos_token_id": None,
            "eos_token_id": None,
            "snn_time_steps": model_config.time_steps,
            "snn_num_attention_heads": model_config.transformer.num_attention_heads,
            "snn_spike_decay": model_config.spike_decay,
            "snn_spike_amplitude": model_config.spike_amplitude,
            "spikingjelly_artifact_schema": 1,
            "source_checkpoint": str(args.checkpoint),
        }
        (args.output / "config.json").write_text(
            json.dumps(exported_config, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        return {
            "artifact": str(args.output),
            "tensor_count": len(weights),
            "parameter_count": sum(value.numel() for value in weights.values()),
        }
    finally:
        if parallel_state.model_parallel_is_initialized():
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MCore distributed inference smoke")
    parser.add_argument(
        "command", choices=("train", "evaluate", "generate", "export-sglang")
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
    if args.command in {"train", "export-sglang"} and args.output is None:
        parser.error("train and export-sglang require --output")
    if args.command != "train" and args.checkpoint is None:
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
    else:
        result = _export_sglang(args, model)
    if int(os.environ.get("RANK", "0")) == 0:
        print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
