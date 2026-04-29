import argparse
import os
import time
from typing import Tuple

import torch
import torch.distributed as dist

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.distributed import (
    SNNDistributedConfig,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    apply_snn_fsdp2,
    build_snn_optimizer,
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_spikformer_distributed,
    configure_spikformer_fsdp2,
    configure_snn_distributed,
    ensure_distributed_initialized,
    materialize_dtensor_output,
    resolve_data_parallel_partition,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.examples.memopt.models import VGGBlock
from spikingjelly.activation_based.memopt import memory_optimization
from spikingjelly.activation_based.model import spikformer
from spikingjelly.activation_based.model.spikformer import SpikformerConv2dBNLIF, SpikformerMLP
from spikingjelly.activation_based.layer.attention import SpikingSelfAttention


def parse_args():
    parser = argparse.ArgumentParser(description="Benchmark distributed SNN training modes.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--steps", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument(
        "--mode",
        type=str,
        default="none",
        choices=("none", "dp", "tp", "fsdp2", "fsdp2_tp"),
    )
    parser.add_argument(
        "--model",
        type=str,
        default="cifar10dvs_vgg",
        choices=("cifar10dvs_vgg", "spikformer_ti", "spikformer_s"),
    )
    parser.add_argument("--image-size", type=int, default=224)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--memopt-level", type=int, default=0)
    parser.add_argument("--memopt-compress-x", action="store_true")
    parser.add_argument(
        "--optimizer-sharding",
        type=str,
        default="none",
        choices=("none", "zero"),
    )
    parser.add_argument("--mesh-shape", type=int, nargs="*", default=None)
    parser.add_argument("--tp-mesh-dim", type=int, default=0)
    parser.add_argument("--dp-mesh-dim", type=int, default=None)
    return parser.parse_args()


def setup_runtime(mode: str):
    if mode == "none":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    ensure_distributed_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = torch.device("cuda", local_rank) if torch.cuda.is_available() else torch.device("cpu")
    return device, rank, world_size


def maybe_apply_memopt(args, model, sample_input):
    if args.memopt_level <= 0:
        return model, 0.0
    if args.mode in ("tp", "fsdp2_tp") and args.memopt_level != 1:
        raise NotImplementedError(
            "TP-aware memopt currently supports level=1 only; split-search levels have not been integrated yet."
        )
    if args.model == "cifar10dvs_vgg":
        target_types = (VGGBlock,)
    else:
        target_types = (SpikformerConv2dBNLIF, SpikingSelfAttention, SpikformerMLP)
    start = time.time()
    model = memory_optimization(
        model,
        target_types,
        dummy_input=(sample_input,),
        compress_x=args.memopt_compress_x,
        level=args.memopt_level,
        verbose=False,
    )
    model = model.to(sample_input.device)
    return model, (time.time() - start) * 1000.0


def build_model(args, device, world_size):
    if args.model == "cifar10dvs_vgg":
        model = CIFAR10DVSVGG(dropout=0.0, backend=args.backend).to(device)
        sample_input = torch.randn(args.batch_size, args.T, 2, 48, 48, device=device)
    else:
        model = spikformer.__dict__[args.model](
            T=args.T,
            img_size_h=args.image_size,
            img_size_w=args.image_size,
            num_classes=args.num_classes,
            backend=args.backend,
        ).to(device)
        sample_input = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
    defer_memopt_until_after_tp = args.mode in ("tp", "fsdp2_tp") and args.memopt_level > 0
    optimize_ms = 0.0
    if not defer_memopt_until_after_tp:
        model, optimize_ms = maybe_apply_memopt(args, model, sample_input)
    if args.mode == "none":
        return model, None, optimize_ms
    mesh_shape = tuple(args.mesh_shape) if args.mesh_shape else None
    if args.mode == "dp":
        config = SNNDistributedConfig(
            device_type=device.type,
            mesh_shape=mesh_shape or (world_size,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else 0,
        )
        model, mesh, _ = configure_snn_distributed(model, config)
        return model, mesh, optimize_ms
    if args.mode == "tp":
        if args.model == "cifar10dvs_vgg":
            model, mesh, _ = configure_cifar10dvs_vgg_distributed(
                model,
                device_type=device.type,
                mesh_shape=mesh_shape or (world_size,),
                enable_data_parallel=False,
                tp_mesh_dim=args.tp_mesh_dim,
                dp_mesh_dim=args.dp_mesh_dim,
            )
        else:
            model, mesh, _ = configure_spikformer_distributed(
                model,
                device_type=device.type,
                mesh_shape=mesh_shape or (world_size,),
                enable_data_parallel=False,
                tp_mesh_dim=args.tp_mesh_dim,
                dp_mesh_dim=args.dp_mesh_dim,
            )
        if defer_memopt_until_after_tp:
            model, optimize_ms = maybe_apply_memopt(args, model, sample_input)
        return model, mesh, optimize_ms
    if args.mode == "fsdp2":
        if args.model == "cifar10dvs_vgg":
            model, mesh, _ = configure_cifar10dvs_vgg_fsdp2(
                model,
                device_type=device.type,
                mesh_shape=mesh_shape or (world_size,),
                enable_classifier_tensor_parallel=False,
                enable_experimental_conv_tensor_parallel=False,
                dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else 0,
            )
        else:
            model, mesh, _ = configure_spikformer_fsdp2(
                model,
                device_type=device.type,
                mesh_shape=mesh_shape or (world_size,),
                enable_head_tensor_parallel=False,
                dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else 0,
            )
        return model, mesh, optimize_ms
    if args.mode == "fsdp2_tp":
        if mesh_shape is None:
            raise ValueError("fsdp2_tp mode requires --mesh-shape.")
        tp_mesh_dim = args.tp_mesh_dim if args.tp_mesh_dim != 0 or args.dp_mesh_dim is not None else 1
        dp_mesh_dim = args.dp_mesh_dim if args.dp_mesh_dim is not None else 0
        if args.model == "cifar10dvs_vgg":
            if defer_memopt_until_after_tp:
                model, mesh, _ = configure_cifar10dvs_vgg_distributed(
                    model,
                    device_type=device.type,
                    mesh_shape=mesh_shape,
                    enable_data_parallel=False,
                    tp_mesh_dim=tp_mesh_dim,
                    dp_mesh_dim=dp_mesh_dim,
                )
                model, optimize_ms = maybe_apply_memopt(args, model, sample_input)
                model = apply_snn_fsdp2(
                    model,
                    device_mesh=mesh,
                    dp_mesh_dim=dp_mesh_dim,
                    shard_roots=["features"],
                    shard_module_root=False,
                )
            else:
                model, mesh, _ = configure_cifar10dvs_vgg_fsdp2(
                    model,
                    device_type=device.type,
                    mesh_shape=mesh_shape,
                    enable_classifier_tensor_parallel=True,
                    enable_experimental_conv_tensor_parallel=True,
                    tp_mesh_dim=tp_mesh_dim,
                    dp_mesh_dim=dp_mesh_dim,
                )
        else:
            if defer_memopt_until_after_tp:
                model, mesh, _ = configure_spikformer_distributed(
                    model,
                    device_type=device.type,
                    mesh_shape=mesh_shape,
                    enable_data_parallel=False,
                    tp_mesh_dim=tp_mesh_dim,
                    dp_mesh_dim=dp_mesh_dim,
                )
                model, optimize_ms = maybe_apply_memopt(args, model, sample_input)
                num_blocks = len(getattr(model, "blocks", ()))
                shard_roots = ["patch_embed"] + [f"blocks.{i}" for i in range(num_blocks)]
                model = apply_snn_fsdp2(
                    model,
                    device_mesh=mesh,
                    dp_mesh_dim=dp_mesh_dim,
                    shard_roots=shard_roots,
                    shard_module_root=False,
                )
            else:
                model, mesh, _ = configure_spikformer_fsdp2(
                    model,
                    device_type=device.type,
                    mesh_shape=mesh_shape,
                    enable_head_tensor_parallel=True,
                    tp_mesh_dim=tp_mesh_dim,
                    dp_mesh_dim=dp_mesh_dim,
                )
        return model, mesh, optimize_ms
    raise ValueError(args.mode)


def benchmark(args):
    device, rank, world_size = setup_runtime(args.mode)
    if args.optimizer_sharding == "zero" and not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE:
        raise RuntimeError(
            "optimizer_sharding='zero' requires torch.distributed.optim.ZeroRedundancyOptimizer."
        )
    model, mesh, optimize_ms = build_model(args, device, world_size)
    optimizer = build_snn_optimizer(
        model,
        mode=args.mode,
        lr=1e-3,
        optimizer_sharding=args.optimizer_sharding,
        foreach=False if args.mode in ("tp", "fsdp2_tp") else None,
    )
    data_replicas, data_rank = resolve_data_parallel_partition(
        mesh,
        dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else (0 if args.mode in ("dp", "fsdp2", "fsdp2_tp") and mesh is not None else None),
        sharded_by_data_parallel=args.mode in ("dp", "fsdp2", "fsdp2_tp"),
    )
    seed = 20260428 + data_rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    if args.model == "cifar10dvs_vgg":
        x = torch.randn(args.batch_size, args.T, 2, 48, 48, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
    else:
        x = torch.randn(args.batch_size, 3, args.image_size, args.image_size, device=device)
        y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        out = materialize_dtensor_output(model(x))
        out = out.mean(dim=0)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    start = time.time()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = materialize_dtensor_output(model(x))
        out = out.mean(dim=0)
        loss = torch.nn.functional.cross_entropy(out, y)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elapsed = time.time() - start

    step_ms = elapsed * 1000 / args.steps
    global_samples_per_second = args.batch_size * data_replicas * args.steps / elapsed
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024 if device.type == "cuda" else 0.0
    )

    if rank == 0:
        print(
            {
                "model": args.model,
                "mode": args.mode,
                "optimizer_sharding": args.optimizer_sharding,
                "memopt_level": args.memopt_level,
                "optimize_ms": optimize_ms,
                "batch_size": args.batch_size,
                "T": args.T,
                "steps": args.steps,
                "step_ms": step_ms,
                "global_samples_per_second": global_samples_per_second,
                "peak_allocated_mb": peak_allocated_mb,
            }
        )

    if dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        benchmark(parse_args())
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
