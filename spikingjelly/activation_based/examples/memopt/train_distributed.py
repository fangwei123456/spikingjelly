import argparse
import os
import time
from dataclasses import dataclass

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.distributed import (
    SNNDistributedConfig,
    SNN_DISTRIBUTED_PREFERENCES,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    apply_pipeline_stage_memopt,
    build_snn_optimizer,
    configure_cifar10dvs_vgg_pipeline,
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_snn_distributed,
    ensure_distributed_initialized,
    materialize_dtensor_output,
    PIPELINING_AVAILABLE,
    recommended_pipeline_microbatches,
    recommend_snn_distributed_strategy,
    resolve_data_parallel_partition,
    resolve_tensor_parallel_group_size,
)
from spikingjelly.activation_based.examples.memopt.data_module import (
    CIFAR10DVSDataModule,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.examples.memopt.models import VGGBlock
from spikingjelly.activation_based.memopt import memory_optimization


@dataclass
class DistributedRuntime:
    mode: str
    is_distributed: bool
    rank: int
    world_size: int
    local_rank: int
    device: torch.device


def parse_args():
    parser = argparse.ArgumentParser(
        description="Distributed SNN training example for CIFAR10-DVS."
    )
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--T", type=int, default=10)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument(
        "--distributed-mode",
        type=str,
        default="none",
        choices=("auto", "none", "dp", "tp", "fsdp2", "fsdp2_tp", "pp"),
    )
    parser.add_argument(
        "--prefer",
        type=str,
        default=None,
        choices=SNN_DISTRIBUTED_PREFERENCES,
    )
    parser.add_argument(
        "--mesh-shape",
        type=int,
        nargs="*",
        default=None,
        help="Explicit DeviceMesh shape. For fsdp2_tp mode, a 2D mesh such as --mesh-shape 2 4 is recommended.",
    )
    parser.add_argument("--tp-mesh-dim", type=int, default=0)
    parser.add_argument("--dp-mesh-dim", type=int, default=None)
    parser.add_argument("--disable-conv-tp", action="store_true")
    parser.add_argument("--disable-classifier-tp", action="store_true")
    parser.add_argument(
        "--optimizer-sharding",
        type=str,
        default=None,
        choices=("none", "zero"),
    )
    parser.add_argument("--memopt-level", type=int, default=None)
    parser.add_argument("--memopt-compress-x", action="store_true")
    parser.add_argument("--pp-microbatches", type=int, default=None)
    parser.add_argument("--pp-memopt-stage-budget-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pp-schedule",
        type=str,
        default="auto",
        choices=("auto", "gpipe", "1f1b", "interleaved", "zero_bubble"),
    )
    parser.add_argument("--pp-virtual-stages", type=int, default=1)
    parser.add_argument("--pp-layout", type=str, default=None)
    parser.add_argument("--pp-delay-wgrad", action="store_true")
    parser.add_argument("--print-summary", action="store_true")
    return parser.parse_args()


def _is_launched_with_torchrun() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ


def setup_runtime(args) -> DistributedRuntime:
    if args.distributed_mode == "none" or (
        args.distributed_mode == "auto" and not _is_launched_with_torchrun()
    ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return DistributedRuntime(
            mode=args.distributed_mode,
            is_distributed=False,
            rank=0,
            world_size=1,
            local_rank=0,
            device=device,
        )

    if not _is_launched_with_torchrun():
        raise RuntimeError(
            f"distributed_mode='{args.distributed_mode}' requires torchrun or environment variables "
            "RANK/WORLD_SIZE/LOCAL_RANK."
        )

    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    if device.type == "cuda":
        torch.cuda.set_device(local_rank)
    ensure_distributed_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    return DistributedRuntime(
        mode=args.distributed_mode,
        is_distributed=True,
        rank=rank,
        world_size=world_size,
        local_rank=local_rank,
        device=device,
    )


def resolve_strategy_args(args, runtime: DistributedRuntime):
    recommendation = None
    recommendation_notes = []
    if args.prefer is not None:
        recommendation = recommend_snn_distributed_strategy(
            model="cifar10dvs_vgg",
            world_size=runtime.world_size,
            prefer=args.prefer,
            batch_size=args.batch_size,
            backend=args.backend,
        )

    if args.distributed_mode == "auto":
        if recommendation is None:
            raise ValueError(
                "--distributed-mode auto requires --prefer speed|memory|capacity."
            )
        args.distributed_mode = recommendation.mode
        args.optimizer_sharding = recommendation.optimizer_sharding
        args.memopt_level = recommendation.memopt_level
        if args.mesh_shape is None and recommendation.mesh_shape is not None:
            args.mesh_shape = list(recommendation.mesh_shape)
        if args.dp_mesh_dim is None:
            args.dp_mesh_dim = recommendation.dp_mesh_dim
        if args.tp_mesh_dim == 0:
            args.tp_mesh_dim = recommendation.tp_mesh_dim
        if args.pp_microbatches is None:
            args.pp_microbatches = recommendation.pp_microbatches
        args.pp_schedule = recommendation.pp_schedule
        args.pp_virtual_stages = recommendation.pp_virtual_stages
        if args.pp_layout is None and recommendation.pp_layout is not None:
            args.pp_layout = "|".join(str(v) for v in recommendation.pp_layout)
        args.pp_delay_wgrad = recommendation.pp_delay_wgrad
        args.pp_memopt_stage_budget_ratio = recommendation.pp_memopt_stage_budget_ratio
        recommendation_notes.append(
            "Applied the full recommended distributed strategy because distributed-mode=auto."
        )
    elif recommendation is not None:
        recommendation_notes.append(
            f"Mode '{args.distributed_mode}' overrides the recommended mode '{recommendation.mode}'."
        )
        if args.memopt_level is None:
            args.memopt_level = recommendation.memopt_level
        if args.optimizer_sharding is None and args.distributed_mode == "dp":
            args.optimizer_sharding = recommendation.optimizer_sharding
        if (
            args.mesh_shape is None
            and args.distributed_mode == "fsdp2_tp"
            and recommendation.mesh_shape is not None
        ):
            args.mesh_shape = list(recommendation.mesh_shape)
        if args.pp_microbatches is None and args.distributed_mode == "pp":
            args.pp_microbatches = recommendation.pp_microbatches
        if args.distributed_mode == "pp":
            if args.pp_schedule == "auto":
                args.pp_schedule = recommendation.pp_schedule
            if args.pp_virtual_stages == 1:
                args.pp_virtual_stages = recommendation.pp_virtual_stages
            if args.pp_layout is None and recommendation.pp_layout is not None:
                args.pp_layout = "|".join(str(v) for v in recommendation.pp_layout)
            if not args.pp_delay_wgrad:
                args.pp_delay_wgrad = recommendation.pp_delay_wgrad

    if args.optimizer_sharding is None:
        args.optimizer_sharding = "none"
    if args.memopt_level is None:
        args.memopt_level = 0
    if args.distributed_mode == "pp" and args.pp_microbatches is None:
        logical_stages = runtime.world_size * max(1, args.pp_virtual_stages)
        args.pp_microbatches = recommended_pipeline_microbatches(
            args.batch_size,
            logical_stages,
        )
    return recommendation, tuple(recommendation_notes)


def build_model(args, runtime: DistributedRuntime):
    model = CIFAR10DVSVGG(dropout=0.25, backend=args.backend)
    model.to(runtime.device)
    example_input = torch.randn(
        args.batch_size, args.T, 2, 48, 48, device=runtime.device
    )
    tp_disabled = args.disable_classifier_tp and args.disable_conv_tp

    defer_memopt_until_after_pp = (
        args.distributed_mode == "pp" and args.memopt_level > 0
    )
    if args.memopt_level > 0 and not defer_memopt_until_after_pp:
        model = memory_optimization(
            model,
            (VGGBlock,),
            dummy_input=(example_input,),
            compress_x=args.memopt_compress_x,
            level=args.memopt_level,
            verbose=False,
        )
        model = model.to(runtime.device)

    if args.distributed_mode == "none":
        return model, None, None
    if args.distributed_mode == "pp":
        if not PIPELINING_AVAILABLE:
            raise RuntimeError(
                "distributed_mode='pp' requires torch.distributed.pipelining support."
            )
        pipeline_runtime = configure_cifar10dvs_vgg_pipeline(
            model,
            example_input=example_input,
            device=runtime.device,
            n_microbatches=args.pp_microbatches,
            pp_schedule=args.pp_schedule,
            pp_virtual_stages=args.pp_virtual_stages,
            pp_layout=args.pp_layout,
            pp_delay_wgrad=args.pp_delay_wgrad,
        )
        if defer_memopt_until_after_pp:
            pipeline_runtime, _, _ = apply_pipeline_stage_memopt(
                pipeline_runtime,
                memopt_level=args.memopt_level,
                compress_x=args.memopt_compress_x,
                stage_budget_ratio=args.pp_memopt_stage_budget_ratio,
                use_plan_cache=True,
            )
        return pipeline_runtime, None, None

    mesh_shape = tuple(args.mesh_shape) if args.mesh_shape else None

    if args.distributed_mode == "dp":
        config = SNNDistributedConfig(
            device_type=runtime.device.type,
            mesh_shape=mesh_shape or (runtime.world_size,),
            auto_tensor_parallel=False,
            enable_data_parallel=True,
            dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else 0,
        )
        return configure_snn_distributed(model, config)

    if args.distributed_mode == "tp":
        if tp_disabled:
            raise ValueError(
                "tp mode requires at least one tensor-parallel target. "
                "Do not disable both classifier TP and convolution TP."
            )
        return configure_cifar10dvs_vgg_distributed(
            model,
            device_type=runtime.device.type,
            mesh_shape=mesh_shape or (runtime.world_size,),
            enable_data_parallel=False,
            enable_classifier_tensor_parallel=not args.disable_classifier_tp,
            enable_experimental_conv_tensor_parallel=not args.disable_conv_tp,
            tp_mesh_dim=args.tp_mesh_dim,
            dp_mesh_dim=args.dp_mesh_dim,
        )

    if args.distributed_mode == "fsdp2":
        return configure_cifar10dvs_vgg_fsdp2(
            model,
            device_type=runtime.device.type,
            mesh_shape=mesh_shape or (runtime.world_size,),
            enable_classifier_tensor_parallel=False,
            enable_experimental_conv_tensor_parallel=False,
            dp_mesh_dim=args.dp_mesh_dim if args.dp_mesh_dim is not None else 0,
        )

    if args.distributed_mode == "fsdp2_tp":
        if mesh_shape is None or len(mesh_shape) != 2:
            raise ValueError(
                "fsdp2_tp mode requires an explicit 2D mesh, e.g. --mesh-shape 2 4."
            )
        if tp_disabled:
            raise ValueError(
                "fsdp2_tp mode requires at least one tensor-parallel target. "
                "Do not disable both classifier TP and convolution TP."
            )
        tp_mesh_dim = (
            args.tp_mesh_dim
            if args.tp_mesh_dim != 0 or args.dp_mesh_dim is not None
            else 1
        )
        dp_mesh_dim = args.dp_mesh_dim if args.dp_mesh_dim is not None else 0
        return configure_cifar10dvs_vgg_fsdp2(
            model,
            device_type=runtime.device.type,
            mesh_shape=mesh_shape,
            enable_classifier_tensor_parallel=not args.disable_classifier_tp,
            enable_experimental_conv_tensor_parallel=not args.disable_conv_tp,
            tp_mesh_dim=tp_mesh_dim,
            dp_mesh_dim=dp_mesh_dim,
        )

    raise ValueError(f"Unsupported distributed mode '{args.distributed_mode}'.")


def build_data(args, runtime: DistributedRuntime, mesh):
    dm = CIFAR10DVSDataModule(
        data_dir=args.data_dir,
        T=args.T,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )
    dm.prepare_data()
    dm.setup("fit")

    train_sampler = None
    val_sampler = None
    if runtime.is_distributed and runtime.mode == "pp":
        train_sampler = DistributedSampler(
            dm.train_set, num_replicas=1, rank=0, shuffle=True
        )
        val_sampler = DistributedSampler(
            dm.test_set, num_replicas=1, rank=0, shuffle=False
        )
    elif runtime.is_distributed:
        data_replicas, data_rank = resolve_data_parallel_partition(
            mesh,
            dp_mesh_dim=args.dp_mesh_dim
            if args.dp_mesh_dim is not None
            else (
                0
                if runtime.mode in ("dp", "fsdp2", "fsdp2_tp") and mesh is not None
                else None
            ),
            sharded_by_data_parallel=runtime.mode in ("dp", "fsdp2", "fsdp2_tp"),
        )
        train_sampler = DistributedSampler(
            dm.train_set, num_replicas=data_replicas, rank=data_rank, shuffle=True
        )
        val_sampler = DistributedSampler(
            dm.test_set, num_replicas=data_replicas, rank=data_rank, shuffle=False
        )

    train_loader = DataLoader(
        dm.train_set,
        batch_size=args.batch_size,
        sampler=train_sampler,
        shuffle=train_sampler is None,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        dm.test_set,
        batch_size=args.batch_size,
        sampler=val_sampler,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    return train_loader, val_loader, train_sampler


def reduce_mean(value: torch.Tensor) -> torch.Tensor:
    if not dist.is_initialized():
        return value
    dist.all_reduce(value)
    value /= dist.get_world_size()
    return value


def reduce_classification_output(outputs: torch.Tensor, labels: torch.Tensor):
    if outputs.ndim >= 3:
        outputs = outputs.mean(dim=0)
    if labels.ndim > 1:
        labels = labels.argmax(dim=1)
    return outputs, labels


def forward_loss(model, criterion, images, labels):
    outputs = model(images.float())
    outputs = materialize_dtensor_output(outputs)
    outputs, labels = reduce_classification_output(outputs, labels)
    loss = criterion(outputs, labels)
    return outputs, loss


def _reduce_stats_tensor(
    stat_tensor: torch.Tensor,
    runtime: DistributedRuntime,
    tp_group_size: int,
):
    if runtime.mode == "tp":
        return reduce_mean(stat_tensor)
    if runtime.is_distributed and runtime.mode in ("dp", "fsdp2", "fsdp2_tp"):
        dist.all_reduce(stat_tensor)
        if runtime.mode == "fsdp2_tp" and tp_group_size > 1:
            stat_tensor /= tp_group_size
    return stat_tensor


def train_one_epoch(
    model,
    optimizer,
    criterion,
    loader,
    runtime: DistributedRuntime,
    epoch: int,
    tp_group_size: int,
):
    model.train()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0
    start = time.time()

    for images, labels in loader:
        images = images.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs, loss = forward_loss(model, criterion, images, labels)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

        batch_size = labels.shape[0]
        preds = outputs.argmax(dim=1)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        total_loss += loss.detach() * batch_size
        total_correct += (preds == labels).sum()
        total_samples += batch_size

    loss_tensor = torch.stack(
        [
            total_loss
            if torch.is_tensor(total_loss)
            else torch.tensor(total_loss, device=runtime.device),
            total_correct
            if torch.is_tensor(total_correct)
            else torch.tensor(total_correct, device=runtime.device),
            torch.tensor(float(total_samples), device=runtime.device),
        ]
    )
    loss_tensor = _reduce_stats_tensor(loss_tensor, runtime, tp_group_size)

    denom = loss_tensor[2].item()
    avg_loss = loss_tensor[0].item() / max(denom, 1.0)
    avg_acc = loss_tensor[1].item() / max(denom, 1.0)
    throughput = denom / max(time.time() - start, 1e-6)
    return avg_loss, avg_acc, throughput


def train_one_epoch_pipeline(
    pipeline_runtime,
    optimizer,
    loader,
    runtime: DistributedRuntime,
):
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0.0
    start = time.time()

    for images, labels in loader:
        optimizer.zero_grad(set_to_none=True)
        losses = [] if pipeline_runtime.is_last else None
        step_args = ()
        step_kwargs = {}
        if pipeline_runtime.is_first:
            images = images.to(runtime.device, non_blocking=True)
            step_args = (images,)
        if pipeline_runtime.is_last:
            labels = labels.to(runtime.device, non_blocking=True)
            if labels.ndim > 1:
                labels = labels.argmax(dim=1)
            step_kwargs = {"target": labels}
        outputs = pipeline_runtime.schedule.step(
            *step_args, losses=losses, **step_kwargs
        )
        optimizer.step()
        functional.reset_net(pipeline_runtime.stage_module)

        if pipeline_runtime.is_last:
            outputs, labels = reduce_classification_output(outputs, labels)
            preds = outputs.argmax(dim=1)
            total_loss += torch.stack(losses).mean().detach() * labels.shape[0]
            total_correct += (preds == labels).sum()
            total_samples += float(labels.shape[0])

    stat_tensor = torch.tensor(
        [
            float(total_loss.item())
            if torch.is_tensor(total_loss)
            else float(total_loss),
            float(total_correct.item())
            if torch.is_tensor(total_correct)
            else float(total_correct),
            total_samples,
        ],
        device=runtime.device,
    )
    if dist.is_initialized():
        dist.all_reduce(stat_tensor)
    denom = max(stat_tensor[2].item(), 1.0)
    throughput = stat_tensor[2].item() / max(time.time() - start, 1e-6)
    return stat_tensor[0].item() / denom, stat_tensor[1].item() / denom, throughput


@torch.inference_mode()
def evaluate(model, criterion, loader, runtime: DistributedRuntime, tp_group_size: int):
    model.eval()
    total_loss = 0.0
    total_correct = 0.0
    total_samples = 0

    for images, labels in loader:
        images = images.to(runtime.device, non_blocking=True)
        labels = labels.to(runtime.device, non_blocking=True)
        outputs, loss = forward_loss(model, criterion, images, labels)
        functional.reset_net(model)
        batch_size = labels.shape[0]
        preds = outputs.argmax(dim=1)
        if labels.ndim > 1:
            labels = labels.argmax(dim=1)
        total_loss += loss.detach() * batch_size
        total_correct += (preds == labels).sum()
        total_samples += batch_size

    stat_tensor = torch.stack(
        [
            total_loss
            if torch.is_tensor(total_loss)
            else torch.tensor(total_loss, device=runtime.device),
            total_correct
            if torch.is_tensor(total_correct)
            else torch.tensor(total_correct, device=runtime.device),
            torch.tensor(float(total_samples), device=runtime.device),
        ]
    )
    stat_tensor = _reduce_stats_tensor(stat_tensor, runtime, tp_group_size)

    denom = stat_tensor[2].item()
    return stat_tensor[0].item() / max(denom, 1.0), stat_tensor[1].item() / max(
        denom, 1.0
    )


def main():
    args = parse_args()
    runtime = setup_runtime(args)
    recommendation, recommendation_notes = resolve_strategy_args(args, runtime)
    runtime.mode = args.distributed_mode
    if args.optimizer_sharding == "zero" and not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE:
        raise RuntimeError(
            "optimizer_sharding='zero' requires torch.distributed.optim.ZeroRedundancyOptimizer."
        )
    model, mesh, analysis = build_model(args, runtime)
    train_loader, val_loader, train_sampler = build_data(args, runtime, mesh)
    optimizer = build_snn_optimizer(
        model.stage_module if args.distributed_mode == "pp" else model,
        mode=args.distributed_mode,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer_sharding=args.optimizer_sharding,
        foreach=False if args.distributed_mode in ("tp", "fsdp2_tp") else None,
    )
    criterion = nn.CrossEntropyLoss()

    if runtime.rank == 0 and args.print_summary:
        print(model)
        if analysis is not None:
            print("analysis:", analysis)
        if mesh is not None:
            print("mesh:", mesh)
        if recommendation is not None:
            print("recommendation:", recommendation)
        if recommendation_notes:
            print("recommendation_notes:", recommendation_notes)

    tp_group_size = resolve_tensor_parallel_group_size(
        mesh,
        tp_mesh_dim=args.tp_mesh_dim
        if args.tp_mesh_dim != 0 or args.dp_mesh_dim is not None
        else (1 if args.distributed_mode == "fsdp2_tp" and mesh is not None else 0),
        tensor_parallel_enabled=args.distributed_mode in ("tp", "fsdp2_tp"),
    )

    for epoch in range(args.epochs):
        if train_sampler is not None:
            train_sampler.set_epoch(epoch)
        if args.distributed_mode == "pp":
            train_loss, train_acc, train_sps = train_one_epoch_pipeline(
                model, optimizer, train_loader, runtime
            )
            val_loss, val_acc = float("nan"), float("nan")
        else:
            train_loss, train_acc, train_sps = train_one_epoch(
                model, optimizer, criterion, train_loader, runtime, epoch, tp_group_size
            )
            val_loss, val_acc = evaluate(
                model, criterion, val_loader, runtime, tp_group_size
            )
        if runtime.rank == 0:
            print(
                f"epoch={epoch} mode={args.distributed_mode} optimizer_sharding={args.optimizer_sharding} "
                f"train_loss={train_loss:.4f} train_acc={train_acc * 100:.2f}% "
                f"val_loss={val_loss:.4f} val_acc={val_acc * 100:.2f}% "
                f"global_samples/s={train_sps:.2f}"
            )

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    try:
        main()
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
