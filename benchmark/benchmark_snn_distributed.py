import argparse
import codecs
import contextlib
import datetime as dt
import json
import os
import sys
import threading
import time
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.distributed as dist

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.distributed import (
    SNNDistributedConfig,
    SNN_DISTRIBUTED_PREFERENCES,
    ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE,
    apply_pipeline_stage_memopt,
    apply_snn_fsdp2,
    build_snn_optimizer,
    recommend_snn_distributed_strategy,
    recommended_pipeline_microbatches,
    configure_cifar10dvs_vgg_pipeline,
    configure_cifar10dvs_vgg_distributed,
    configure_cifar10dvs_vgg_fsdp2,
    configure_spikformer_pipeline,
    configure_spikformer_distributed,
    configure_spikformer_fsdp2,
    configure_snn_distributed,
    ensure_distributed_initialized,
    materialize_dtensor_output,
    PIPELINING_AVAILABLE,
    resolve_data_parallel_partition,
)
from spikingjelly.activation_based.examples.memopt.models import CIFAR10DVSVGG
from spikingjelly.activation_based.examples.memopt.models import VGGBlock
from spikingjelly.activation_based.memopt import memory_optimization
from spikingjelly.activation_based.model import spikformer
from spikingjelly.activation_based.model.spikformer import (
    SpikformerConv2dBNLIF,
    SpikformerMLP,
)
from spikingjelly.activation_based.layer.attention import SpikingSelfAttention


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


def _non_negative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("value must be >= 0")
    return parsed


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark distributed SNN training modes."
    )
    parser.add_argument("--batch-size", type=_positive_int, default=8)
    parser.add_argument("--T", type=_positive_int, default=10)
    parser.add_argument("--steps", type=_positive_int, default=20)
    parser.add_argument("--warmup", type=_non_negative_int, default=5)
    parser.add_argument("--backend", type=str, default="inductor")
    parser.add_argument(
        "--mode",
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
        "--model",
        type=str,
        default="cifar10dvs_vgg",
        choices=("cifar10dvs_vgg", "spikformer_ti", "spikformer_s"),
    )
    parser.add_argument("--image-size", type=_positive_int, default=224)
    parser.add_argument("--num-classes", type=_positive_int, default=1000)
    parser.add_argument("--memopt-level", type=int, default=None)
    parser.add_argument("--memopt-compress-x", action="store_true")
    parser.add_argument(
        "--optimizer-sharding",
        type=str,
        default=None,
        choices=("none", "zero"),
    )
    parser.add_argument("--mesh-shape", type=int, nargs="*", default=None)
    parser.add_argument("--tp-mesh-dim", type=int, default=0)
    parser.add_argument("--dp-mesh-dim", type=int, default=None)
    parser.add_argument("--pp-microbatches", type=_positive_int, default=None)
    parser.add_argument("--pp-memopt-stage-budget-ratio", type=float, default=0.5)
    parser.add_argument(
        "--pp-schedule",
        type=str,
        default="auto",
        choices=("auto", "gpipe", "1f1b", "interleaved", "zero_bubble"),
    )
    parser.add_argument("--pp-virtual-stages", type=_positive_int, default=1)
    parser.add_argument("--pp-layout", type=str, default=None)
    parser.add_argument("--pp-delay-wgrad", action="store_true")
    parser.add_argument(
        "--result-log",
        type=str,
        default=str(Path("benchmark") / "results" / "benchmark_snn_distributed.jsonl"),
    )
    return parser.parse_args()


def _reduce_classification_output(
    out: torch.Tensor, target: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    if out.ndim >= 3:
        out = out.mean(dim=0)
    if target.ndim > 1:
        target = target.argmax(dim=1)
    return out, target


class _LinePatternCounter:
    def __init__(self):
        self.warning_count = 0
        self.recompile_count = 0
        self.graph_break_count = 0
        self._buffer = ""

    def _consume_line(self, line: str):
        stripped = line.lstrip()
        lowered = stripped.lower()
        if "recompile" in lowered:
            self.recompile_count += 1
        if "graph break" in lowered:
            self.graph_break_count += 1
        if "warning" in lowered or (
            len(stripped) >= 5 and stripped[0] == "W" and stripped[1:5].isdigit()
        ):
            self.warning_count += 1

    def feed(self, text: str):
        self._buffer += text
        while True:
            newline_idx = self._buffer.find("\n")
            if newline_idx < 0:
                break
            line = self._buffer[:newline_idx]
            self._buffer = self._buffer[newline_idx + 1 :]
            self._consume_line(line)

    def finalize(self):
        if self._buffer:
            self._consume_line(self._buffer)
            self._buffer = ""


class _FDRedirectCapture:
    def __init__(self, fd: int, counter: _LinePatternCounter):
        self.fd = fd
        self.counter = counter
        self._saved_fd = None
        self._read_fd = None
        self._write_fd = None
        self._thread = None
        self._decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")

    def _reader_loop(self):
        try:
            while True:
                chunk = os.read(self._read_fd, 4096)
                if not chunk:
                    break
                text = self._decoder.decode(chunk)
                if text:
                    self.counter.feed(text)
                if self._saved_fd is not None:
                    os.write(self._saved_fd, chunk)
            tail = self._decoder.decode(b"", final=True)
            if tail:
                self.counter.feed(tail)
        finally:
            if self._read_fd is not None:
                os.close(self._read_fd)
                self._read_fd = None

    def __enter__(self):
        self._saved_fd = os.dup(self.fd)
        self._read_fd, self._write_fd = os.pipe()
        os.dup2(self._write_fd, self.fd)
        self._thread = threading.Thread(target=self._reader_loop, daemon=True)
        self._thread.start()
        return self

    def __exit__(self, exc_type, exc, tb):
        os.dup2(self._saved_fd, self.fd)
        os.close(self._write_fd)
        self._write_fd = None
        if self._thread is not None:
            self._thread.join()
            self._thread = None
        os.close(self._saved_fd)
        self._saved_fd = None


@contextlib.contextmanager
def capture_benchmark_events():
    counter = _LinePatternCounter()
    sys.stdout.flush()
    sys.stderr.flush()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        with _FDRedirectCapture(1, counter), _FDRedirectCapture(2, counter):
            yield counter
    counter.finalize()
    counter.warning_count += len(caught)


def resolve_strategy_args(args, world_size: int):
    recommendation = None
    recommendation_notes = []
    if args.prefer is not None:
        recommendation = recommend_snn_distributed_strategy(
            model=args.model,
            world_size=world_size,
            prefer=args.prefer,
            batch_size=args.batch_size,
            backend=args.backend,
        )

    if args.mode == "auto":
        if recommendation is None:
            raise ValueError("--mode auto requires --prefer speed|memory|capacity.")
        args.mode = recommendation.mode
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
            "Applied the full recommended distributed strategy because mode=auto."
        )
    elif recommendation is not None:
        recommendation_notes.append(
            f"Mode '{args.mode}' overrides the recommended mode '{recommendation.mode}'."
        )
        if args.memopt_level is None:
            args.memopt_level = recommendation.memopt_level
        if args.optimizer_sharding is None and args.mode == "dp":
            args.optimizer_sharding = recommendation.optimizer_sharding
        if (
            args.mesh_shape is None
            and args.mode == "fsdp2_tp"
            and recommendation.mesh_shape is not None
        ):
            args.mesh_shape = list(recommendation.mesh_shape)
        if args.pp_microbatches is None and args.mode == "pp":
            args.pp_microbatches = recommendation.pp_microbatches
        if args.mode == "pp":
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
    if args.mode == "pp" and args.pp_microbatches is None:
        logical_stages = world_size * max(1, args.pp_virtual_stages)
        args.pp_microbatches = recommended_pipeline_microbatches(
            args.batch_size,
            logical_stages,
        )
    return recommendation, tuple(recommendation_notes)


def _aggregate_event_counts(
    counter: _LinePatternCounter, device: torch.device
) -> Dict[str, int]:
    values = torch.tensor(
        [counter.warning_count, counter.recompile_count, counter.graph_break_count],
        device=device,
        dtype=torch.int64,
    )
    if dist.is_initialized():
        dist.all_reduce(values, op=dist.ReduceOp.SUM)
    return {
        "warning_count": int(values[0].item()),
        "recompile_count": int(values[1].item()),
        "graph_break_count": int(values[2].item()),
    }


def _comparison_key(record: Dict[str, object]) -> Dict[str, object]:
    def _normalize_json_value(value):
        if isinstance(value, tuple):
            return [_normalize_json_value(item) for item in value]
        if isinstance(value, list):
            return [_normalize_json_value(item) for item in value]
        if isinstance(value, dict):
            return {key: _normalize_json_value(item) for key, item in value.items()}
        return value

    keys = (
        "model",
        "mode",
        "backend",
        "world_size",
        "optimizer_sharding",
        "memopt_level",
        "batch_size",
        "num_classes",
        "T",
        "steps",
        "warmup",
        "image_size",
        "mesh_shape",
        "tp_mesh_dim",
        "dp_mesh_dim",
        "pp_microbatches",
        "pp_memopt_stage_budget_ratio",
        "pp_schedule",
        "pp_virtual_stages",
        "pp_layout",
        "pp_delay_wgrad",
        "memopt_compress_x",
        "prefer",
    )
    return {key: _normalize_json_value(record.get(key)) for key in keys}


def _append_benchmark_record(
    path: str, record: Dict[str, object]
) -> Optional[Dict[str, object]]:
    result_path = Path(path)
    result_path.parent.mkdir(parents=True, exist_ok=True)
    comparison_key = _comparison_key(record)
    previous = None
    if result_path.exists():
        with result_path.open("r", encoding="utf-8") as handle:
            for line in handle:
                if not line.strip():
                    continue
                candidate = json.loads(line)
                if candidate.get("comparison_key") == comparison_key:
                    previous = candidate
    record_to_write = dict(record)
    record_to_write["comparison_key"] = comparison_key
    with result_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record_to_write, ensure_ascii=False) + "\n")
    return previous


def _summarize_benchmark_comparison(
    current: Dict[str, object], previous: Optional[Dict[str, object]]
) -> Optional[Dict[str, Dict[str, float]]]:
    if previous is None:
        return None
    summary: Dict[str, Dict[str, float]] = {}
    for key in (
        "global_samples_per_second",
        "peak_allocated_mb",
        "optimize_ms",
        "warning_count",
        "recompile_count",
        "graph_break_count",
    ):
        previous_value = previous.get(key)
        current_value = current.get(key)
        if previous_value is None or current_value is None:
            continue
        delta = float(current_value) - float(previous_value)
        pct = None
        if float(previous_value) != 0.0:
            pct = delta / float(previous_value) * 100.0
        summary[key] = {"delta": delta, "pct": pct}
    return summary


def setup_runtime(mode: str):
    if mode == "none" or (mode == "auto" and "WORLD_SIZE" not in os.environ):
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return device, 0, 1
    rank = int(os.environ.get("RANK", 0))
    local_rank = int(os.environ.get("LOCAL_RANK", rank))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
    ensure_distributed_initialized()
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    device = (
        torch.device("cuda", local_rank)
        if torch.cuda.is_available()
        else torch.device("cpu")
    )
    return device, rank, world_size


def maybe_apply_memopt(args, model, sample_input):
    if args.memopt_level <= 0:
        return model, 0.0
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


def maybe_apply_pipeline_memopt(args, runtime):
    if args.memopt_level <= 0:
        return runtime, 0.0
    runtime, local_optimize_ms, _ = apply_pipeline_stage_memopt(
        runtime,
        memopt_level=args.memopt_level,
        compress_x=args.memopt_compress_x,
        stage_budget_ratio=args.pp_memopt_stage_budget_ratio,
        use_plan_cache=True,
    )
    optimize_tensor = torch.tensor(
        [local_optimize_ms],
        device=runtime.device,
        dtype=torch.float64,
    )
    if dist.is_initialized():
        dist.all_reduce(optimize_tensor, op=dist.ReduceOp.MAX, group=runtime.group)
    return runtime, float(optimize_tensor.item())


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
        sample_input = torch.randn(
            args.batch_size, 3, args.image_size, args.image_size, device=device
        )
    defer_memopt_until_after_tp = (
        args.mode in ("tp", "fsdp2_tp") and args.memopt_level == 1
    )
    defer_memopt_until_after_pp = args.mode == "pp" and args.memopt_level > 0
    optimize_ms = 0.0
    if not defer_memopt_until_after_tp and not defer_memopt_until_after_pp:
        model, optimize_ms = maybe_apply_memopt(args, model, sample_input)
    if args.mode == "none":
        return model, None, optimize_ms
    if args.mode == "pp":
        if not PIPELINING_AVAILABLE:
            raise RuntimeError(
                "mode='pp' requires torch.distributed.pipelining support in the current PyTorch build."
            )
        logical_stages = world_size * max(1, args.pp_virtual_stages)
        n_microbatches = args.pp_microbatches or recommended_pipeline_microbatches(
            args.batch_size,
            logical_stages,
        )
        if args.model == "cifar10dvs_vgg":
            runtime = configure_cifar10dvs_vgg_pipeline(
                model,
                example_input=sample_input,
                device=device,
                n_microbatches=n_microbatches,
                pp_schedule=args.pp_schedule,
                pp_virtual_stages=args.pp_virtual_stages,
                pp_layout=args.pp_layout,
                pp_delay_wgrad=args.pp_delay_wgrad,
            )
        else:
            runtime = configure_spikformer_pipeline(
                model,
                example_input=sample_input,
                device=device,
                n_microbatches=n_microbatches,
                pp_schedule=args.pp_schedule,
                pp_virtual_stages=args.pp_virtual_stages,
                pp_layout=args.pp_layout,
                pp_delay_wgrad=args.pp_delay_wgrad,
            )
        if defer_memopt_until_after_pp:
            runtime, optimize_ms = maybe_apply_pipeline_memopt(args, runtime)
        return runtime, None, optimize_ms
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
        if mesh_shape is None or len(mesh_shape) != 2:
            raise ValueError(
                "fsdp2_tp mode requires an explicit 2D mesh, e.g. --mesh-shape 2 4."
            )
        tp_mesh_dim = (
            args.tp_mesh_dim
            if args.tp_mesh_dim != 0 or args.dp_mesh_dim is not None
            else 1
        )
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
                shard_roots = ["patch_embed"] + [
                    f"blocks.{i}" for i in range(num_blocks)
                ]
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


def benchmark(args, counter: _LinePatternCounter):
    device, rank, world_size = setup_runtime(args.mode)
    recommendation, recommendation_notes = resolve_strategy_args(args, world_size)
    if args.optimizer_sharding == "zero" and not ZERO_REDUNDANCY_OPTIMIZER_AVAILABLE:
        raise RuntimeError(
            "optimizer_sharding='zero' requires torch.distributed.optim.ZeroRedundancyOptimizer."
        )
    model, mesh, optimize_ms = build_model(args, device, world_size)
    optimizer_target = model.stage_module if args.mode == "pp" else model
    optimizer = build_snn_optimizer(
        optimizer_target,
        mode=args.mode,
        lr=1e-3,
        optimizer_sharding=args.optimizer_sharding,
        foreach=False if args.mode in ("tp", "fsdp2_tp") else None,
    )
    data_replicas, data_rank = resolve_data_parallel_partition(
        mesh,
        dp_mesh_dim=args.dp_mesh_dim
        if args.dp_mesh_dim is not None
        else (
            0 if args.mode in ("dp", "fsdp2", "fsdp2_tp") and mesh is not None else None
        ),
        sharded_by_data_parallel=args.mode in ("dp", "fsdp2", "fsdp2_tp"),
    )
    seed = 20260428 + data_rank
    torch.manual_seed(seed)
    if device.type == "cuda":
        torch.cuda.manual_seed(seed)
    if args.mode == "pp":
        if args.model == "cifar10dvs_vgg":
            x = (
                torch.randn(args.batch_size, args.T, 2, 48, 48, device=device)
                if model.is_first
                else None
            )
            y = (
                torch.randint(0, 10, (args.batch_size,), device=device)
                if model.is_last
                else None
            )
        else:
            x = (
                torch.randn(
                    args.batch_size,
                    3,
                    args.image_size,
                    args.image_size,
                    device=device,
                )
                if model.is_first
                else None
            )
            y = (
                torch.randint(0, args.num_classes, (args.batch_size,), device=device)
                if model.is_last
                else None
            )
        return benchmark_pipeline(
            args,
            model,
            optimizer,
            x,
            y,
            device,
            rank,
            world_size,
            optimize_ms,
            recommendation,
            recommendation_notes,
            counter,
        )

    if args.model == "cifar10dvs_vgg":
        x = torch.randn(args.batch_size, args.T, 2, 48, 48, device=device)
        y = torch.randint(0, 10, (args.batch_size,), device=device)
    else:
        x = torch.randn(
            args.batch_size, 3, args.image_size, args.image_size, device=device
        )
        y = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        out = materialize_dtensor_output(model(x))
        out, target = _reduce_classification_output(out, y)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    start = time.time()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        out = materialize_dtensor_output(model(x))
        out, target = _reduce_classification_output(out, y)
        loss = torch.nn.functional.cross_entropy(out, target)
        loss.backward()
        optimizer.step()
        functional.reset_net(model)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    elapsed = time.time() - start
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )

    elapsed_tensor = torch.tensor(elapsed, device=device)
    peak_tensor = torch.tensor(float(peak_allocated_mb), device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)
    elapsed = elapsed_tensor.item()
    peak_allocated_mb = peak_tensor.item()

    step_ms = elapsed * 1000 / args.steps
    global_samples_per_second = args.batch_size * data_replicas * args.steps / elapsed

    event_counts = _aggregate_event_counts(counter, device)
    record = None
    if rank == 0:
        record = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": args.model,
            "mode": args.mode,
            "prefer": args.prefer,
            "backend": args.backend,
            "world_size": world_size,
            "optimizer_sharding": args.optimizer_sharding,
            "memopt_level": args.memopt_level,
            "optimize_ms": optimize_ms,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "T": args.T,
            "steps": args.steps,
            "warmup": args.warmup,
            "image_size": args.image_size,
            "mesh_shape": tuple(args.mesh_shape)
            if args.mesh_shape is not None
            else None,
            "tp_mesh_dim": args.tp_mesh_dim,
            "dp_mesh_dim": args.dp_mesh_dim,
            "memopt_compress_x": args.memopt_compress_x,
            "pp_microbatches": args.pp_microbatches,
            "pp_memopt_stage_budget_ratio": args.pp_memopt_stage_budget_ratio,
            "pp_schedule": args.pp_schedule,
            "pp_virtual_stages": args.pp_virtual_stages,
            "pp_layout": args.pp_layout,
            "pp_delay_wgrad": args.pp_delay_wgrad,
            "step_ms": step_ms,
            "global_samples_per_second": global_samples_per_second,
            "peak_allocated_mb": peak_allocated_mb,
            "recommendation_mode": recommendation.mode
            if recommendation is not None
            else None,
            "recommendation_rationale": recommendation.rationale
            if recommendation is not None
            else (),
            "recommendation_notes": recommendation_notes,
            **event_counts,
        }

    if dist.is_initialized():
        dist.destroy_process_group()
    return record


def benchmark_pipeline(
    args,
    runtime,
    optimizer,
    x,
    y,
    device,
    rank,
    world_size,
    optimize_ms,
    recommendation,
    recommendation_notes,
    counter: _LinePatternCounter,
):
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)

    for _ in range(args.warmup):
        optimizer.zero_grad(set_to_none=True)
        losses = [] if runtime.is_last else None
        step_args = (x,) if runtime.is_first else ()
        step_kwargs = {"target": y} if runtime.is_last else {}
        runtime.schedule.step(*step_args, losses=losses, **step_kwargs)
        optimizer.step()
        functional.reset_net(runtime.stage_module)

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()

    start = time.time()
    for _ in range(args.steps):
        optimizer.zero_grad(set_to_none=True)
        losses = [] if runtime.is_last else None
        step_args = (x,) if runtime.is_first else ()
        step_kwargs = {"target": y} if runtime.is_last else {}
        runtime.schedule.step(*step_args, losses=losses, **step_kwargs)
        optimizer.step()
        functional.reset_net(runtime.stage_module)
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    if dist.is_initialized():
        dist.barrier()
    elapsed = time.time() - start

    elapsed_tensor = torch.tensor(elapsed, device=device)
    if dist.is_initialized():
        dist.all_reduce(elapsed_tensor, op=dist.ReduceOp.MAX)
    peak_allocated_mb = (
        torch.cuda.max_memory_allocated(device) / 1024 / 1024
        if device.type == "cuda" and torch.cuda.is_available()
        else 0.0
    )
    peak_tensor = torch.tensor(float(peak_allocated_mb), device=device)
    if dist.is_initialized():
        dist.all_reduce(peak_tensor, op=dist.ReduceOp.MAX)

    elapsed = elapsed_tensor.item()
    step_ms = elapsed * 1000 / args.steps
    global_samples_per_second = args.batch_size * args.steps / elapsed

    event_counts = _aggregate_event_counts(counter, device)
    record = None
    if rank == 0:
        record = {
            "timestamp": dt.datetime.now(dt.timezone.utc).isoformat(),
            "model": args.model,
            "mode": args.mode,
            "prefer": args.prefer,
            "backend": args.backend,
            "world_size": world_size,
            "optimizer_sharding": args.optimizer_sharding,
            "memopt_level": args.memopt_level,
            "optimize_ms": optimize_ms,
            "batch_size": args.batch_size,
            "num_classes": args.num_classes,
            "T": args.T,
            "steps": args.steps,
            "warmup": args.warmup,
            "image_size": args.image_size,
            "mesh_shape": tuple(args.mesh_shape)
            if args.mesh_shape is not None
            else None,
            "tp_mesh_dim": args.tp_mesh_dim,
            "dp_mesh_dim": args.dp_mesh_dim,
            "memopt_compress_x": args.memopt_compress_x,
            "pp_microbatches": runtime.n_microbatches,
            "pp_memopt_stage_budget_ratio": args.pp_memopt_stage_budget_ratio,
            "pp_schedule": runtime.schedule_kind,
            "pp_virtual_stages": runtime.virtual_pipeline_size,
            "pp_layout": runtime.pp_layout,
            "pp_delay_wgrad": runtime.delayed_wgrad,
            "pp_memopt_stages": runtime.memopt_selected_stage_indices,
            "step_ms": step_ms,
            "global_samples_per_second": global_samples_per_second,
            "peak_allocated_mb": peak_tensor.item(),
            "recommendation_mode": recommendation.mode
            if recommendation is not None
            else None,
            "recommendation_rationale": recommendation.rationale
            if recommendation is not None
            else (),
            "recommendation_notes": recommendation_notes,
            **event_counts,
        }

    if dist.is_initialized():
        dist.destroy_process_group()
    return record


if __name__ == "__main__":
    args = parse_args()
    try:
        with capture_benchmark_events() as event_counter:
            record = benchmark(args, event_counter)
        if record is not None:
            previous = _append_benchmark_record(args.result_log, record)
            record["comparison_to_previous"] = _summarize_benchmark_comparison(
                record, previous
            )
            print(record)
    finally:
        if dist.is_available() and dist.is_initialized():
            dist.destroy_process_group()
