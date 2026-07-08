import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import Converter, STATransformerRecipe


@dataclass(frozen=True)
class BenchResult:
    name: str
    total_seconds: float
    seconds_per_epoch: float
    samples_per_second: float


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, mlp_dim: int):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=0.0,
            batch_first=True,
        )
        self.norm2 = nn.LayerNorm(embed_dim)
        self.fc1 = nn.Linear(embed_dim, mlp_dim)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x_norm = self.norm1(x)
        attn, _ = self.attn(x_norm, x_norm, x_norm, need_weights=False)
        x = residual + attn
        x = x + self.fc2(self.act(self.fc1(self.norm2(x))))
        return x


class TransformerClassifier(nn.Module):
    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        seq_len: int,
        depth: int,
        num_heads: int,
        mlp_dim: int,
        num_classes: int,
    ):
        super().__init__()
        self.seq_len = seq_len
        self.token_proj = nn.Linear(input_dim, embed_dim)
        self.blocks = nn.ModuleList(
            TransformerBlock(embed_dim, num_heads, mlp_dim) for _ in range(depth)
        )
        self.head_norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.token_proj(x)
        for block in self.blocks:
            x = block(x)
        x = self.head_norm(x.mean(dim=1))
        return self.head(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark STA Transformer converted-model inference with SpikingJelly "
            "step modes. The single-step case runs an explicit Python time loop; "
            "the multi-step case runs the full sequence in one forward."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--mode",
        choices=("equivalent", "spiking_encoder"),
        default="equivalent",
        help="STATransformerRecipe mode used during conversion.",
    )
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--batches-per-epoch", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--seq-len", type=int, default=64)
    parser.add_argument("--input-dim", type=int, default=128)
    parser.add_argument("--embed-dim", type=int, default=256)
    parser.add_argument("--mlp-dim", type=int, default=1024)
    parser.add_argument("--depth", type=int, default=4)
    parser.add_argument("--num-heads", type=int, default=8)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260630)
    parser.add_argument("--atol", type=float, default=1e-4)
    parser.add_argument("--rtol", type=float, default=1e-4)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "epochs": args.epochs,
        "seq_len": args.seq_len,
        "input_dim": args.input_dim,
        "embed_dim": args.embed_dim,
        "mlp_dim": args.mlp_dim,
        "depth": args.depth,
        "num_heads": args.num_heads,
        "num_classes": args.num_classes,
    }
    invalid = [f"{name}={value}" for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError("Arguments must be positive: " + ", ".join(invalid))
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if args.calibration_batches <= 0:
        raise ValueError("calibration_batches must be positive.")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    if args.embed_dim % args.num_heads != 0:
        raise ValueError(
            f"embed_dim={args.embed_dim} must be divisible by "
            f"num_heads={args.num_heads}."
        )


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batches(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    return [
        torch.randn(
            args.batch_size,
            args.seq_len,
            args.input_dim,
            device=device,
        )
        for _ in range(args.batches_per_epoch)
    ]


def make_calibration(
    args: argparse.Namespace,
    device: torch.device,
) -> list[tuple[torch.Tensor]]:
    return [
        (
            torch.randn(
                args.batch_size,
                args.seq_len,
                args.input_dim,
                device=device,
            ),
        )
        for _ in range(args.calibration_batches)
    ]


def make_first_real_then_zero_sequence(
    x: torch.Tensor,
    time_steps: int,
) -> torch.Tensor:
    out = torch.zeros((time_steps, *x.shape), dtype=x.dtype, device=x.device)
    out[0] = x
    return out


def run_single_step_epoch(
    model: nn.Module,
    batches: list[torch.Tensor],
    time_steps: int,
):
    functional.set_step_mode(model, "s")
    last_output = None
    for x in batches:
        functional.reset_net(model)
        x_seq = make_first_real_then_zero_sequence(x, time_steps)
        output = None
        for step in range(time_steps):
            step_output = model(x_seq[step])
            output = step_output if output is None else output + step_output
        last_output = output
    return last_output


def run_multi_step_epoch(
    model: nn.Module,
    batches: list[torch.Tensor],
    time_steps: int,
):
    functional.set_step_mode(model, "m")
    last_output = None
    for x in batches:
        functional.reset_net(model)
        x_seq = make_first_real_then_zero_sequence(x, time_steps)
        last_output = model(x_seq).sum(dim=0)
    return last_output


def measure(
    name: str,
    run_epoch,
    model: nn.Module,
    batches: list[torch.Tensor],
    time_steps: int,
    warmup_epochs: int,
    epochs: int,
    device: torch.device,
) -> BenchResult:
    with torch.inference_mode():
        for _ in range(warmup_epochs):
            run_epoch(model, batches, time_steps)
        sync_if_needed(device)
        start = time.perf_counter()
        for _ in range(epochs):
            run_epoch(model, batches, time_steps)
        sync_if_needed(device)
        elapsed = time.perf_counter() - start

    samples = epochs * len(batches) * batches[0].shape[0]
    return BenchResult(
        name=name,
        total_seconds=elapsed,
        seconds_per_epoch=elapsed / epochs,
        samples_per_second=samples / elapsed,
    )


def assert_parity(
    single_model: nn.Module,
    multi_model: nn.Module,
    batches: list[torch.Tensor],
    time_steps: int,
    atol: float,
    rtol: float,
) -> None:
    with torch.inference_mode():
        for x in batches[: min(2, len(batches))]:
            single_out = run_single_step_epoch(single_model, [x], time_steps)
            multi_out = run_multi_step_epoch(multi_model, [x], time_steps)
            if not torch.allclose(single_out, multi_out, atol=atol, rtol=rtol):
                max_diff = (single_out - multi_out).abs().max().item()
                raise AssertionError(
                    "single-step loop and multi-step outputs differ: "
                    f"max_abs_diff={max_diff:.6g}"
                )


def build_converted_model(args: argparse.Namespace, device: torch.device) -> nn.Module:
    model = (
        TransformerClassifier(
            input_dim=args.input_dim,
            embed_dim=args.embed_dim,
            seq_len=args.seq_len,
            depth=args.depth,
            num_heads=args.num_heads,
            mlp_dim=args.mlp_dim,
            num_classes=args.num_classes,
        )
        .to(device)
        .eval()
    )
    dataloader = (
        make_calibration(args, device) if args.mode == "spiking_encoder" else None
    )
    recipe = STATransformerRecipe(
        dataloader=dataloader,
        time_steps=args.time_steps,
        mode=args.mode,
    )
    return Converter(recipe=recipe, device=device).convert(model)


def result_by_name(results: list[BenchResult], name: str) -> BenchResult:
    for result in results:
        if result.name == name:
            return result
    raise ValueError(f"Missing benchmark result {name!r}.")


def print_table(
    results: list[BenchResult],
    baseline_name: str = "single_step_loop",
) -> None:
    print(
        f"{'mode':<18} {'total_s':>12} {'epoch_s':>12} {'samples/s':>12} "
        f"{'speedup':>10}"
    )
    baseline = result_by_name(results, baseline_name).seconds_per_epoch
    for result in results:
        speedup = baseline / result.seconds_per_epoch
        print(
            f"{result.name:<18} {result.total_seconds:>12.4f} "
            f"{result.seconds_per_epoch:>12.4f} "
            f"{result.samples_per_second:>12.2f} {speedup:>10.2f}x"
        )


def main() -> None:
    args = parse_args()
    validate_args(args)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device requested, but torch.cuda.is_available() is false."
        )

    torch.manual_seed(args.seed)
    batches = make_batches(args, device)
    base = build_converted_model(args, device)
    single_model = copy.deepcopy(base).eval()
    multi_model = copy.deepcopy(base).eval()

    assert_parity(
        single_model,
        multi_model,
        batches,
        args.time_steps,
        atol=args.atol,
        rtol=args.rtol,
    )

    results = [
        measure(
            "single_step_loop",
            run_single_step_epoch,
            single_model,
            batches,
            args.time_steps,
            args.warmup_epochs,
            args.epochs,
            device,
        ),
        measure(
            "multi_step",
            run_multi_step_epoch,
            multi_model,
            batches,
            args.time_steps,
            args.warmup_epochs,
            args.epochs,
            device,
        ),
    ]

    if args.json:
        single_result = result_by_name(results, "single_step_loop")
        multi_result = result_by_name(results, "multi_step")
        print(
            json.dumps(
                {
                    "config": {
                        key: value for key, value in vars(args).items() if key != "json"
                    },
                    "results": [asdict(result) for result in results],
                    "multi_step_speedup": (
                        single_result.seconds_per_epoch / multi_result.seconds_per_epoch
                    ),
                },
                indent=2,
            )
        )
    else:
        print("STA Transformer step-mode inference benchmark")
        print(
            "config: "
            f"device={device}, mode={args.mode}, time_steps={args.time_steps}, "
            f"batch_size={args.batch_size}, "
            f"batches_per_epoch={args.batches_per_epoch}, "
            f"epochs={args.epochs}, warmup_epochs={args.warmup_epochs}"
        )
        print_table(results)


if __name__ == "__main__":
    main()
