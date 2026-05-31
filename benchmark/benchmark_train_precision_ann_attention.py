import argparse
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

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


class ResidualAttentionBlock(torch_nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, mlp_ratio: int = 4):
        super().__init__()
        self.norm1 = torch_nn.LayerNorm(hidden_dim)
        self.q_proj = torch_nn.Linear(hidden_dim, hidden_dim)
        self.k_proj = torch_nn.Linear(hidden_dim, hidden_dim)
        self.v_proj = torch_nn.Linear(hidden_dim, hidden_dim)
        self.out_proj = torch_nn.Linear(hidden_dim, hidden_dim)
        self.norm2 = torch_nn.LayerNorm(hidden_dim)
        self.mlp = torch_nn.Sequential(
            torch_nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            torch_nn.GELU(),
            torch_nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads

    def _reshape_heads(self, x: torch.Tensor) -> torch.Tensor:
        b, t, c = x.shape
        x = x.reshape(b, t, self.num_heads, self.head_dim)
        return x.permute(0, 2, 1, 3)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = self.norm1(x)
        q = self._reshape_heads(self.q_proj(x))
        k = self._reshape_heads(self.k_proj(x))
        v = self._reshape_heads(self.v_proj(x))
        attn = F.scaled_dot_product_attention(q, k, v)
        attn = attn.permute(0, 2, 1, 3).reshape_as(x)
        x = residual + self.out_proj(attn)
        x = x + self.mlp(self.norm2(x))
        return x


class DeepMLPBlock(torch_nn.Module):
    def __init__(self, hidden_dim: int, mlp_ratio: int = 4):
        super().__init__()
        self.net = torch_nn.Sequential(
            torch_nn.LayerNorm(hidden_dim),
            torch_nn.Linear(hidden_dim, hidden_dim * mlp_ratio),
            torch_nn.GELU(),
            torch_nn.Linear(hidden_dim * mlp_ratio, hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class DeepANN(torch_nn.Module):
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_classes: int,
        seq_len: int,
        depth: int,
        attention_every: int,
        num_heads: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_proj = torch_nn.Linear(input_dim, hidden_dim)
        self.blocks = torch_nn.ModuleList()
        for block_idx in range(depth):
            self.blocks.append(DeepMLPBlock(hidden_dim))
            if attention_every > 0 and (block_idx + 1) % attention_every == 0:
                self.blocks.append(ResidualAttentionBlock(hidden_dim, num_heads))
        self.head_norm = torch_nn.LayerNorm(hidden_dim)
        self.head = torch_nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x.mean(dim=1))
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark ANN training under fp32, bf16, and fp8-torchao on a pure "
            "GEMM-heavy workload without spiking neurons."
        )
    )
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--seq-len", type=int, default=128)
    parser.add_argument("--input-dim", type=int, default=1024)
    parser.add_argument("--hidden-dim", type=int, default=2048)
    parser.add_argument("--num-classes", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=24)
    parser.add_argument("--attention-every", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=16)
    parser.add_argument("--lr", type=float, default=5e-2)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--steps", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20260531)
    parser.add_argument(
        "--precisions",
        nargs="+",
        default=["fp32", "bf16", "fp8-torchao"],
        choices=("fp32", "bf16", "fp8-torchao"),
    )
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.hidden_dim % args.num_heads != 0:
        raise ValueError(
            f"hidden_dim={args.hidden_dim} must be divisible by num_heads={args.num_heads}."
        )
    if "fp8-torchao" in args.precisions:
        aligned = {
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": args.num_classes,
        }
        invalid = [f"{k}={v}" for k, v in aligned.items() if v % FP8_ALIGNMENT != 0]
        if invalid:
            raise ValueError(
                f"fp8-torchao requires dimensions divisible by {FP8_ALIGNMENT}: "
                + ", ".join(invalid)
            )


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


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


def build_model(args: argparse.Namespace) -> DeepANN:
    return DeepANN(
        input_dim=args.input_dim,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
        seq_len=args.seq_len,
        depth=args.depth,
        attention_every=args.attention_every,
        num_heads=args.num_heads,
    )


def run_training_step(
    model: torch_nn.Module,
    artifacts,
    optimizer: torch.optim.Optimizer,
    criterion: torch_nn.Module,
    x: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    events = make_timing_events(device)
    if events is not None:
        events["step_start"].record()
        events["forward_start"].record()
        with artifacts.autocast_context():
            logits = model(x)
            loss = criterion(logits, target)
        events["forward_end"].record()
        events["backward_start"].record()
        artifacts.backward(loss, optimizer, step_optimizer=False)
        events["backward_end"].record()
        events["optimizer_start"].record()
        optimizer.step()
        events["optimizer_end"].record()
        events["step_end"].record()
        sync_if_needed(device)
        return {
            "forward_ms": events["forward_start"].elapsed_time(events["forward_end"]),
            "backward_ms": events["backward_start"].elapsed_time(events["backward_end"]),
            "optimizer_ms": events["optimizer_start"].elapsed_time(events["optimizer_end"]),
            "total_step_ms": events["step_start"].elapsed_time(events["step_end"]),
        }

    t0 = time.perf_counter()
    with artifacts.autocast_context():
        logits = model(x)
        loss = criterion(logits, target)
    forward_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    artifacts.backward(loss, optimizer, step_optimizer=False)
    backward_ms = (time.perf_counter() - t0) * 1e3
    t0 = time.perf_counter()
    optimizer.step()
    optimizer_ms = (time.perf_counter() - t0) * 1e3
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
    x: torch.Tensor,
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
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    criterion = torch_nn.CrossEntropyLoss()

    for _ in range(args.warmup):
        run_training_step(model, artifacts, optimizer, criterion, x, target, device)

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
        step_metrics = run_training_step(model, artifacts, optimizer, criterion, x, target, device)
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
        raise RuntimeError("benchmark_train_precision_ann_attention.py requires CUDA.")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available.")

    validate_args(args)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.benchmark = True

    base_model = build_model(args)
    model_state = base_model.state_dict()
    x = torch.randn(args.batch_size, args.seq_len, args.input_dim, device=device)
    target = torch.randint(0, args.num_classes, (args.batch_size,), device=device)

    results = []
    for precision in args.precisions:
        results.append(
            benchmark_one_precision(args, precision, model_state, x, target, device)
        )

    print_table(results)

    if args.json:
        payload = {
            "device": str(device),
            "batch_size": args.batch_size,
            "seq_len": args.seq_len,
            "input_dim": args.input_dim,
            "hidden_dim": args.hidden_dim,
            "num_classes": args.num_classes,
            "depth": args.depth,
            "attention_every": args.attention_every,
            "num_heads": args.num_heads,
            "lr": args.lr,
            "momentum": args.momentum,
            "warmup": args.warmup,
            "steps": args.steps,
            "results": [asdict(result) for result in results],
        }
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
