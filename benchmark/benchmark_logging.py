"""Microbenchmark disabled, filtered, and active Loguru calls."""

from __future__ import annotations

import argparse
import logging
import statistics
import time

import torch
import torch.nn as nn

from spikingjelly.activation_based.layer.misc import PrintShapeModule
from spikingjelly.logger import logger


_STDLIB_LOGGER = logging.getLogger("spikingjelly-benchmark-baseline")


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be > 0")
    return parsed


class _StdlibPrintShapeModule(nn.Module):
    def forward(self, x):
        _STDLIB_LOGGER.debug("benchmark shape=%s", x.shape)
        return x


class _Workload(nn.Module):
    def __init__(self, diagnostic_layer: nn.Module) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(256, 512),
            nn.ReLU(),
            diagnostic_layer,
            nn.Linear(512, 10),
        )

    def forward(self, x):
        return self.layers(x)


def _measure(call, calls: int) -> float:
    started = time.perf_counter()
    for index in range(calls):
        call(index)
    return time.perf_counter() - started


def _measure_baseline(calls: int) -> float:
    return _measure(lambda _: None, calls)


def _run(repeats: int, call, calls: int) -> list[float]:
    call(0)
    return [_measure(call, calls) for _ in range(repeats)]


def _print_result(name: str, samples: list[float], calls: int) -> None:
    median_seconds = statistics.median(samples)
    print(
        f"{name}: median_seconds={median_seconds:.6f} "
        f"ns_per_call={median_seconds / calls * 1e9:.1f} "
        f"samples={','.join(f'{sample:.6f}' for sample in samples)}"
    )


def _measure_workload(diagnostic_layer: nn.Module, steps: int, training: bool) -> float:
    torch.manual_seed(2025)
    model = _Workload(diagnostic_layer)
    inputs = torch.randn(128, 256)
    targets = torch.randint(0, 10, (128,))
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01) if training else None

    def step() -> None:
        if optimizer is None:
            with torch.inference_mode():
                model(inputs)
            return
        optimizer.zero_grad(set_to_none=True)
        nn.functional.cross_entropy(model(inputs), targets).backward()
        optimizer.step()

    for _ in range(5):
        step()
    started = time.perf_counter()
    for _ in range(steps):
        step()
    return time.perf_counter() - started


def _print_workload_comparison(
    name: str, baseline: list[float], migrated: list[float], steps: int
) -> None:
    batch_size = 128
    baseline_throughput = batch_size * steps / statistics.median(baseline)
    migrated_throughput = batch_size * steps / statistics.median(migrated)
    degradation = (
        (baseline_throughput - migrated_throughput) / baseline_throughput * 100
    )
    print(
        f"{name}: baseline_samples_per_second={baseline_throughput:.1f} "
        f"loguru_samples_per_second={migrated_throughput:.1f} "
        f"degradation_percent={degradation:.3f}"
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calls", type=_positive_int, default=1_000_000)
    parser.add_argument("--repeats", type=_positive_int, default=5)
    parser.add_argument("--training-steps", type=_positive_int, default=50)
    parser.add_argument("--inference-steps", type=_positive_int, default=200)
    args = parser.parse_args()

    _STDLIB_LOGGER.propagate = False
    _STDLIB_LOGGER.setLevel(logging.WARNING)
    _STDLIB_LOGGER.addHandler(logging.NullHandler())

    logger.remove()
    _print_result(
        "baseline",
        [_measure_baseline(args.calls) for _ in range(args.repeats)],
        args.calls,
    )

    logger.disable(__name__)
    _print_result(
        "namespace_disabled_info",
        _run(
            args.repeats,
            lambda index: logger.info("benchmark index={}", index),
            args.calls,
        ),
        args.calls,
    )
    _print_result(
        "namespace_disabled_debug",
        _run(
            args.repeats,
            lambda index: logger.debug("benchmark index={}", index),
            args.calls,
        ),
        args.calls,
    )

    logger.enable(__name__)
    sink_id = logger.add(lambda _: None, level="WARNING", format="{message}")
    _print_result(
        "level_filtered_info",
        _run(
            args.repeats,
            lambda index: logger.info("benchmark index={}", index),
            args.calls,
        ),
        args.calls,
    )
    logger.remove(sink_id)

    active_calls = min(args.calls, 100_000)
    sink_id = logger.add(lambda _: None, level="INFO", format="{message}")
    _print_result(
        "formatted_memory_sink",
        _run(
            args.repeats,
            lambda index: logger.info("benchmark index={}", index),
            active_calls,
        ),
        active_calls,
    )
    logger.remove(sink_id)

    logger.disable("spikingjelly")
    module = PrintShapeModule("benchmark")
    tensor = torch.zeros(1)
    forward_calls = min(args.calls, 100_000)
    _print_result(
        "disabled_debug_forward",
        _run(args.repeats, lambda _: module(tensor), forward_calls),
        forward_calls,
    )

    for name, steps, training in (
        ("training_workload", args.training_steps, True),
        ("inference_workload", args.inference_steps, False),
    ):
        baseline = [
            _measure_workload(_StdlibPrintShapeModule(), steps, training)
            for _ in range(args.repeats)
        ]
        migrated = [
            _measure_workload(PrintShapeModule("benchmark"), steps, training)
            for _ in range(args.repeats)
        ]
        _print_workload_comparison(name, baseline, migrated, steps)


if __name__ == "__main__":
    main()
