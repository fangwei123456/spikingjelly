import argparse
import copy
import json
import time
from dataclasses import asdict, dataclass

import torch
import torch.nn as nn

from spikingjelly.activation_based import functional
from spikingjelly.activation_based.ann2snn import Converter, RateCodingRecipe


@dataclass(frozen=True)
class BenchResult:
    name: str
    total_seconds: float
    seconds_per_epoch: float
    samples_per_second: float


class ConvClassifier(nn.Module):
    def __init__(
        self,
        in_channels: int,
        channels: int,
        image_size: int,
        depth: int,
        hidden_dim: int,
        num_classes: int,
    ):
        super().__init__()
        blocks = []
        current_channels = in_channels
        current_size = image_size
        for _ in range(depth):
            blocks.extend(
                [
                    nn.Conv2d(
                        current_channels,
                        channels,
                        kernel_size=3,
                        padding=1,
                        bias=False,
                    ),
                    nn.BatchNorm2d(channels),
                    nn.ReLU(),
                    nn.AvgPool2d(2),
                ]
            )
            current_channels = channels
            current_size //= 2
        self.features = nn.Sequential(*blocks)
        self.flatten = nn.Flatten()
        self.fc0 = nn.Linear(current_channels * current_size * current_size, hidden_dim)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.flatten(x)
        x = self.relu(self.fc0(x))
        return self.fc1(x)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark RateCodingRecipe converted-model inference with "
            "SpikingJelly step modes. The single-step case runs an explicit "
            "Python time loop with repeated static input; the multi-step case "
            "runs a repeated input sequence in one forward."
        )
    )
    parser.add_argument(
        "--device",
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument("--time-steps", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--batches-per-epoch", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--warmup-epochs", type=int, default=1)
    parser.add_argument("--image-size", type=int, default=32)
    parser.add_argument("--in-channels", type=int, default=3)
    parser.add_argument("--channels", type=int, default=64)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--hidden-dim", type=int, default=512)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--calibration-batches", type=int, default=4)
    parser.add_argument("--seed", type=int, default=20260703)
    parser.add_argument("--atol", type=float, default=1e-5)
    parser.add_argument("--rtol", type=float, default=1e-5)
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_ints = {
        "time_steps": args.time_steps,
        "batch_size": args.batch_size,
        "batches_per_epoch": args.batches_per_epoch,
        "epochs": args.epochs,
        "image_size": args.image_size,
        "in_channels": args.in_channels,
        "channels": args.channels,
        "depth": args.depth,
        "hidden_dim": args.hidden_dim,
        "num_classes": args.num_classes,
        "calibration_batches": args.calibration_batches,
    }
    invalid = [f"{name}={value}" for name, value in positive_ints.items() if value <= 0]
    if invalid:
        raise ValueError("Arguments must be positive: " + ", ".join(invalid))
    if args.warmup_epochs < 0:
        raise ValueError("warmup_epochs must be non-negative.")
    if args.atol < 0 or args.rtol < 0:
        raise ValueError("atol and rtol must be non-negative.")
    if args.image_size % (2**args.depth) != 0:
        raise ValueError("image_size must be divisible by 2 ** depth.")


def sync_if_needed(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def make_batches(args: argparse.Namespace, device: torch.device) -> list[torch.Tensor]:
    return [
        torch.randn(
            args.batch_size,
            args.in_channels,
            args.image_size,
            args.image_size,
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
                args.in_channels,
                args.image_size,
                args.image_size,
                device=device,
            ),
        )
        for _ in range(args.calibration_batches)
    ]


def make_repeated_sequence(x: torch.Tensor, time_steps: int) -> torch.Tensor:
    shape = (time_steps, *x.shape)
    return x.unsqueeze(0).expand(shape).contiguous()


def run_single_step_epoch(
    model: nn.Module,
    batches: list[torch.Tensor],
    time_steps: int,
):
    functional.set_step_mode(model, "s")
    last_output = None
    for x in batches:
        functional.reset_net(model)
        output = None
        for _ in range(time_steps):
            step_output = model(x)
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
        x_seq = make_repeated_sequence(x, time_steps)
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
    model = ConvClassifier(
        in_channels=args.in_channels,
        channels=args.channels,
        image_size=args.image_size,
        depth=args.depth,
        hidden_dim=args.hidden_dim,
        num_classes=args.num_classes,
    ).to(device).eval()
    recipe = RateCodingRecipe(
        dataloader=make_calibration(args, device),
        mode="Max",
        fuse_flag=True,
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
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
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
                        key: value
                        for key, value in vars(args).items()
                        if key != "json"
                    },
                    "results": [asdict(result) for result in results],
                    "multi_step_speedup": (
                        single_result.seconds_per_epoch
                        / multi_result.seconds_per_epoch
                    ),
                },
                indent=2,
            )
        )
    else:
        print("RateCodingRecipe step-mode inference benchmark")
        print(
            "config: "
            f"device={device}, time_steps={args.time_steps}, "
            f"batch_size={args.batch_size}, "
            f"batches_per_epoch={args.batches_per_epoch}, "
            f"epochs={args.epochs}, warmup_epochs={args.warmup_epochs}"
        )
        print_table(results)


if __name__ == "__main__":
    main()
