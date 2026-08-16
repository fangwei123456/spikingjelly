from __future__ import annotations

import argparse
import functools
import importlib.metadata
import os

import torch
import torch.distributed as dist
import torch.nn as nn

from spikingjelly.activation_based import neuron
from spikingjelly.activation_based.distributed.llm.temporal import (
    pack_time_batch,
    run_functional_sequence,
    unpack_time_batch,
)


class _SpikingLayerNorm(nn.LayerNorm):
    def __init__(
        self,
        *,
        config,
        hidden_size: int,
        eps: float,
        time_steps: int,
    ) -> None:
        del config
        super().__init__(hidden_size, eps=eps)
        self.time_steps = time_steps
        self.neuron = neuron.IFNode(step_mode="m")

    def forward(self, hidden: torch.Tensor) -> torch.Tensor:
        hidden = super().forward(hidden)
        sequence = unpack_time_batch(hidden, self.time_steps)
        spikes = run_functional_sequence(self.neuron, (sequence,))[0]
        return pack_time_batch(spikes)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="MCore public-seam smoke test for a functional spiking layer."
    )
    parser.add_argument("--tensor-parallel-size", type=int, default=1)
    parser.add_argument("--pipeline-parallel-size", type=int, default=1)
    parser.add_argument("--context-parallel-size", type=int, default=1)
    parser.add_argument("--time-steps", type=int, default=4)
    parser.add_argument("--micro-batch-size", type=int, default=2)
    parser.add_argument("--microbatches", type=int, default=4)
    parser.add_argument("--sequence-length", type=int, default=16)
    parser.add_argument("--recompute", action="store_true")
    parser.add_argument("--fp8", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    topology_size = (
        args.tensor_parallel_size
        * args.pipeline_parallel_size
        * args.context_parallel_size
    )
    if topology_size != dist.get_world_size():
        raise ValueError(
            "world size must equal tensor_parallel_size * pipeline_parallel_size * "
            "context_parallel_size"
        )
    for name in (
        "time_steps",
        "micro_batch_size",
        "microbatches",
        "sequence_length",
    ):
        if getattr(args, name) <= 0:
            raise ValueError(f"{name} must be positive")
    if args.context_parallel_size > 1 and args.sequence_length % (
        2 * args.context_parallel_size
    ):
        raise ValueError(
            "sequence_length must be divisible by 2 * context_parallel_size"
        )


def _build_batch(
    args: argparse.Namespace, device: torch.device
) -> dict[str, torch.Tensor]:
    semantic_batch = torch.arange(
        args.micro_batch_size * args.sequence_length, device=device
    ).reshape(args.micro_batch_size, args.sequence_length)
    input_ids = semantic_batch.remainder(128)
    labels = input_ids.roll(-1, dims=1)
    position_ids = torch.arange(args.sequence_length, device=device).expand_as(
        input_ids
    )
    if args.context_parallel_size > 1:
        from megatron.core import parallel_state
        from megatron.core.utils import get_batch_on_this_cp_rank

        batch = get_batch_on_this_cp_rank(
            {
                "input_ids": input_ids,
                "labels": labels,
                "position_ids": position_ids,
            },
            is_hybrid_cp=False,
            cp_group=parallel_state.get_context_parallel_group(),
        )
        input_ids = batch["input_ids"]
        labels = batch["labels"]
        position_ids = batch["position_ids"]
    input_ids = input_ids.repeat(args.time_steps, 1)
    return {
        "input_ids": input_ids,
        "labels": labels.repeat(args.time_steps, 1),
        "position_ids": position_ids.repeat(args.time_steps, 1),
        "attention_mask": None,
    }


def _main() -> None:
    if importlib.metadata.version("megatron-core") != "0.18.2":
        raise RuntimeError("mcore_smoke requires megatron-core==0.18.2")

    from megatron.core import parallel_state
    from megatron.core.models.gpt.gpt_layer_specs import (
        get_gpt_layer_local_spec,
        get_gpt_layer_with_transformer_engine_spec,
    )
    from megatron.core.models.gpt.gpt_model import GPTModel
    from megatron.core.pipeline_parallel.schedules import get_forward_backward_func
    from megatron.core.tensor_parallel.random import model_parallel_cuda_manual_seed
    from megatron.core.transformer.mlp import MLP, MLPSubmodules
    from megatron.core.transformer.module import Float16Module
    from megatron.core.transformer.transformer_config import TransformerConfig

    args = _parse_args()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    dist.init_process_group("nccl")
    model_parallel_initialized = False
    try:
        _validate_args(args)
        parallel_state.initialize_model_parallel(
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
        )
        model_parallel_initialized = True
        model_parallel_cuda_manual_seed(1234)
        config = TransformerConfig(
            num_layers=4,
            hidden_size=64,
            num_attention_heads=4,
            ffn_hidden_size=128,
            tensor_model_parallel_size=args.tensor_parallel_size,
            pipeline_model_parallel_size=args.pipeline_parallel_size,
            context_parallel_size=args.context_parallel_size,
            sequence_parallel=False,
            bf16=True,
            params_dtype=torch.bfloat16,
            pipeline_dtype=torch.bfloat16,
            hidden_dropout=0.0,
            attention_dropout=0.0,
            recompute_granularity="full" if args.recompute else None,
            recompute_method="uniform" if args.recompute else None,
            recompute_num_layers=1 if args.recompute else None,
            fp8="hybrid" if args.fp8 else None,
            fp8_recipe="delayed" if args.fp8 else None,
        )
        use_te = args.fp8 or args.context_parallel_size > 1
        layer_spec = (
            get_gpt_layer_with_transformer_engine_spec()
            if use_te
            else get_gpt_layer_local_spec()
        )
        layer_spec.submodules.pre_mlp_layernorm = functools.partial(
            _SpikingLayerNorm, time_steps=args.time_steps
        )
        if use_te:
            from megatron.core.extensions.transformer_engine import (
                TEColumnParallelLinear,
            )

            mlp_submodules = layer_spec.submodules.mlp.keywords["submodules"]
            layer_spec.submodules.mlp = functools.partial(
                MLP.as_mlp_submodule,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=mlp_submodules.linear_fc2,
                ),
            )
        model = GPTModel(
            config=config,
            transformer_layer_spec=layer_spec,
            vocab_size=128,
            max_sequence_length=args.sequence_length,
            pre_process=parallel_state.is_pipeline_first_stage(),
            post_process=parallel_state.is_pipeline_last_stage(),
            parallel_output=True,
            position_embedding_type="rope",
        ).cuda()
        if config.fp16 or config.bf16:
            model = Float16Module(config, model)
        if use_te:
            from megatron.core.extensions.transformer_engine import TELinear

            if not any(isinstance(module, TELinear) for module in model.modules()):
                raise RuntimeError("Smoke test did not build Transformer Engine GEMMs.")
        batch = _build_batch(args, torch.device("cuda", local_rank))
        data_iterator = iter([batch] * args.microbatches)

        def forward_step(data, current_model):
            current_batch = next(data)
            output = current_model(**current_batch)

            def loss_func(token_losses):
                loss = token_losses.float().mean()
                return loss, {"loss": loss.detach().clone()}

            return output, loss_func

        losses = get_forward_backward_func()(
            forward_step_func=forward_step,
            data_iterator=data_iterator,
            model=model,
            num_microbatches=args.microbatches,
            seq_length=args.sequence_length,
            micro_batch_size=args.time_steps * args.micro_batch_size,
            forward_only=False,
        )

        grads_finite = torch.ones((), device="cuda", dtype=torch.int32)
        for parameter in model.parameters():
            if parameter.grad is not None and not parameter.grad.isfinite().all():
                grads_finite.zero_()
        dist.all_reduce(grads_finite, op=dist.ReduceOp.MIN)

        seam_grad = torch.zeros((), device="cuda")
        if parallel_state.is_pipeline_first_stage():
            spiking_norm = next(
                module
                for module in model.modules()
                if isinstance(module, _SpikingLayerNorm)
            )
            seam_grad = spiking_norm.weight.grad.float().abs().sum()
            dist.all_reduce(
                seam_grad, group=parallel_state.get_tensor_model_parallel_group()
            )

        loss = torch.zeros((), device="cuda")
        if parallel_state.is_pipeline_last_stage():
            loss = torch.stack([item["loss"] for item in losses]).mean()
        dist.broadcast(loss, src=dist.get_world_size() - 1)
        if dist.get_rank() == 0:
            print(
                f"loss={loss.item():.8f} seam_grad={seam_grad.item():.8e} "
                f"grads_finite={bool(grads_finite.item())} "
                f"tp={args.tensor_parallel_size} pp={args.pipeline_parallel_size} "
                f"cp={args.context_parallel_size} fp8={args.fp8} "
                f"recompute={args.recompute}"
            )
    finally:
        if model_parallel_initialized:
            parallel_state.destroy_model_parallel()
        dist.destroy_process_group()


if __name__ == "__main__":
    _main()
