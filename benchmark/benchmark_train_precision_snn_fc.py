import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)

FP8_ALIGNMENT = 16


@dataclass(frozen=True)
class BenchResult:
    precision: str
    batch_size: int
    steps: int
    warmup: int
    forward_ms: float
    backward_ms: float
    optimizer_ms: float
    total_step_ms: float
    samples_per_sec: float
    peak_allocated_mb: float
    peak_reserved_mb: float
    conversion_report: dict


class TemporalSelfAttentionBlock(torch_nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.scale = self.head_dim**-0.5
        self.q_proj = layer.Linear(hidden_dim, hidden_dim, step_mode="m")
        self.k_proj = layer.Linear(hidden_dim, hidden_dim, step_mode="m")
        self.v_proj = layer.Linear(hidden_dim, hidden_dim, step_mode="m")
        self.out_proj = layer.Linear(hidden_dim, hidden_dim, step_mode="m")
        self.norm = torch_nn.LayerNorm(hidden_dim)

    def _reshape(self, x: torch.Tensor) -> torch.Tensor:
        t, b, c = x.shape
        x = x.reshape(t, b, self.num_heads, self.head_dim)
        return x.permute(1, 2, 0, 3)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        residual = x_seq
        q = self._reshape(self.q_proj(x_seq))
        k = self._reshape(self.k_proj(x_seq))
        v = self._reshape(self.v_proj(x_seq))
        attn = F.scaled_dot_product_attention(q, k, v, scale=self.scale)
        attn = attn.permute(2, 0, 1, 3).reshape_as(x_seq)
        attn = self.out_proj(attn)
        return functional.seq_to_ann_forward(residual + attn, self.norm)


class DeepFCSNN(torch_nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        tau: float,
        backend: str,
        depth: int,
        attention_every: int,
        num_heads: int,
    ):
        super().__init__()
        if depth < 2:
            raise ValueError("depth must be >= 2")

        sg = surrogate.Sigmoid(alpha=4.0)
        blocks: list[torch_nn.Module] = []
        in_dim = input_dim
        for block_idx in range(depth - 1):
            blocks.append(layer.Linear(in_dim, hidden_dim, step_mode="m"))
            blocks.append(
                neuron.LIFNode(
                    tau=tau,
                    surrogate_function=sg,
                    detach_reset=False,
                    step_mode="m",
                    backend=backend,
                )
            )
            if attention_every > 0 and (block_idx + 1) % attention_every == 0:
                blocks.append(TemporalSelfAttentionBlock(hidden_dim, num_heads))
            in_dim = hidden_dim
        blocks.append(layer.Linear(hidden_dim, num_classes, step_mode="m"))
        self.net = torch_nn.Sequential(*blocks)

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        return self.net(x_seq).mean(0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark SpikingJelly deep FC SNN training under fp32, bf16, and "
            "fp8-torchao, and fp8-te."
        )
    )
    parser.add_argument(
        "--device", default="cuda:0", help="CUDA device for the benchmark."
    )
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--input-dim", type=int, default=4096)
    parser.add_argument("--hidden-dim", type=int, default=4096)
    parser.add_argument("--num-classes", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=20)
    parser.add_argument(
        "--attention-every",
        type=int,
        default=0,
        help="Insert one native temporal self-attention block after every N hidden blocks. 0 disables attention.",
    )
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--tau", type=float, default=2.0)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--backend",
        choices=("torch", "triton"),
        default="torch",
        help="Neuron backend used by LIF nodes.",
    )
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "bf16", "fp8-torchao"],
        choices=("fp32", "bf16", "fp8-torchao", "fp8-te"),
        help="Precision modes to benchmark.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full benchmark report as JSON."
    )
    return parser.parse_args()


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _event_elapsed_ms(start: torch.cuda.Event, end: torch.cuda.Event) -> float:
    return start.elapsed_time(end)


def _time_cpu_section(fn) -> tuple[float, object]:
    t0 = time.perf_counter()
    out = fn()
    return (time.perf_counter() - t0) * 1e3, out


def make_timing_events(device: torch.device) -> dict[str, torch.cuda.Event] | None:
    if device.type != "cuda":
        return None
    return {
        "forward_start": torch.cuda.Event(enable_timing=True),
        "forward_end": torch.cuda.Event(enable_timing=True),
        "backward_start": torch.cuda.Event(enable_timing=True),
        "backward_end": torch.cuda.Event(enable_timing=True),
        "optimizer_start": torch.cuda.Event(enable_timing=True),
        "optimizer_end": torch.cuda.Event(enable_timing=True),
        "step_start": torch.cuda.Event(enable_timing=True),
        "step_end": torch.cuda.Event(enable_timing=True),
    }


def build_model(args: argparse.Namespace) -> DeepFCSNN:
    return DeepFCSNN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        tau=args.tau,
        backend=args.backend,
        depth=args.depth,
        attention_every=args.attention_every,
        num_heads=args.num_heads,
    )


def validate_precision_shape_constraints(args: argparse.Namespace) -> None:
    constrained_dims = {
        "input_dim": args.input_dim,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
    }
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"hidden_dim={args.hidden_dim} must be divisible by num_heads={args.num_heads}."
        )
    if "fp8-torchao" not in args.precisions:
        return
    invalid_dims = [
        f"{name}={value}"
        for name, value in constrained_dims.items()
        if value % FP8_ALIGNMENT != 0
    ]
    if invalid_dims:
        raise ValueError(
            "fp8-torchao currently requires every linear dimension used by this "
            f"benchmark to be divisible by {FP8_ALIGNMENT}. Invalid values: "
            + ", ".join(invalid_dims)
        )


def run_training_step(
    model: torch_nn.Module,
    artifacts,
    optimizer: torch.optim.Optimizer,
    criterion: torch_nn.Module,
    x_seq: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    functional.reset_net(model)
    optimizer.zero_grad(set_to_none=True)

    cuda_events = make_timing_events(device)
    if cuda_events is not None:
        cuda_events["step_start"].record()
        cuda_events["forward_start"].record()
        with artifacts.autocast_context():
            logits = model(x_seq)
            loss = criterion(logits, target)
        cuda_events["forward_end"].record()
        cuda_events["backward_start"].record()
        artifacts.backward(loss, optimizer, step_optimizer=False)
        cuda_events["backward_end"].record()
        cuda_events["optimizer_start"].record()
        optimizer.step()
        cuda_events["optimizer_end"].record()
        cuda_events["step_end"].record()
        sync_if_needed(device)
        return {
            "forward_ms": _event_elapsed_ms(
                cuda_events["forward_start"], cuda_events["forward_end"]
            ),
            "backward_ms": _event_elapsed_ms(
                cuda_events["backward_start"], cuda_events["backward_end"]
            ),
            "optimizer_ms": _event_elapsed_ms(
                cuda_events["optimizer_start"], cuda_events["optimizer_end"]
            ),
            "total_step_ms": _event_elapsed_ms(
                cuda_events["step_start"], cuda_events["step_end"]
            ),
        }

    def forward_section():
        with artifacts.autocast_context():
            logits = model(x_seq)
            return logits, criterion(logits, target)

    forward_ms, (logits, loss) = _time_cpu_section(forward_section)
    backward_ms, _ = _time_cpu_section(
        lambda: artifacts.backward(loss, optimizer, step_optimizer=False)
    )
    optimizer_ms, _ = _time_cpu_section(optimizer.step)
    _ = logits
    return {
        "forward_ms": forward_ms,
        "backward_ms": backward_ms,
        "optimizer_ms": optimizer_ms,
        "total_step_ms": forward_ms + backward_ms + optimizer_ms,
    }


def benchmark_one_precision(
    args: argparse.Namespace,
    precision: str,
    model_state: dict,
    x_seq: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> BenchResult:
    model = build_model(args).to(device)
    model.load_state_dict(model_state, strict=True)
    model.train()

    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(mode=precision, strictness="strict", device=str(device)),
    )
    model = artifacts.model
    optimizer = torch.optim.SGD(
        model.parameters(),
        lr=args.lr,
        momentum=args.momentum,
    )
    criterion = torch_nn.CrossEntropyLoss()

    for _ in range(args.warmup):
        run_training_step(model, artifacts, optimizer, criterion, x_seq, target, device)

    sync_if_needed(device)
    if device.type == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)

    forward_ms = 0.0
    backward_ms = 0.0
    optimizer_ms = 0.0
    total_step_ms = 0.0
    wall_start = time.perf_counter()
    for _ in range(args.steps):
        step_metrics = run_training_step(
            model, artifacts, optimizer, criterion, x_seq, target, device
        )
        forward_ms += step_metrics["forward_ms"]
        backward_ms += step_metrics["backward_ms"]
        optimizer_ms += step_metrics["optimizer_ms"]
        total_step_ms += step_metrics["total_step_ms"]
    sync_if_needed(device)
    wall_elapsed = time.perf_counter() - wall_start

    peak_allocated_mb = float("nan")
    peak_reserved_mb = float("nan")
    if device.type == "cuda":
        peak_allocated_mb = torch.cuda.max_memory_allocated(device) / 1024 / 1024
        peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024

    return BenchResult(
        precision=precision,
        batch_size=args.batch_size,
        steps=args.steps,
        warmup=args.warmup,
        forward_ms=forward_ms / args.steps,
        backward_ms=backward_ms / args.steps,
        optimizer_ms=optimizer_ms / args.steps,
        total_step_ms=total_step_ms / args.steps,
        samples_per_sec=(args.batch_size * args.steps) / wall_elapsed,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        conversion_report=artifacts.policy.conversion_report(),
    )


def print_table(results: list[BenchResult]) -> None:
    print(
        f"{'precision':<14s} {'forward_ms':>12s} {'backward_ms':>12s} "
        f"{'optim_ms':>10s} {'step_ms':>12s} {'samples/s':>12s} "
        f"{'peak_alloc_MB':>14s} {'peak_resv_MB':>14s}"
    )
    for result in results:
        print(
            f"{result.precision:<14s} "
            f"{result.forward_ms:12.3f} "
            f"{result.backward_ms:12.3f} "
            f"{result.optimizer_ms:10.3f} "
            f"{result.total_step_ms:12.3f} "
            f"{result.samples_per_sec:12.1f} "
            f"{result.peak_allocated_mb:14.1f} "
            f"{result.peak_reserved_mb:14.1f}"
        )


def main() -> None:
    args = parse_args()
    device = torch.device(args.device)

    if device.type != "cuda":
        raise RuntimeError(
            "benchmark_train_precision_snn_fc.py requires CUDA because the target "
            "comparison includes bf16/fp8 training speed and peak GPU memory."
        )
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    validate_precision_shape_constraints(args)

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    base_model = build_model(args)
    model_state = base_model.state_dict()

    x_seq = torch.randn(args.time_steps, args.batch_size, args.input_dim, device=device)
    target = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    results = []
    for precision in args.precisions:
        result = benchmark_one_precision(
            args, precision, model_state, x_seq, target, device
        )
        results.append(result)

    print_table(results)

    if args.json:
        payload = {
            "device": str(device),
            "time_steps": args.time_steps,
            "batch_size": args.batch_size,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": args.num_classes,
            "depth": args.depth,
            "attention_every": args.attention_every,
            "num_heads": args.num_heads,
            "tau": args.tau,
            "lr": args.lr,
            "momentum": args.momentum,
            "warmup": args.warmup,
            "steps": args.steps,
            "backend": args.backend,
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
