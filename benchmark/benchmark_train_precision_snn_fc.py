import argparse
from contextlib import contextmanager
import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterator

import torch
import torch.nn as torch_nn
import torch.nn.functional as F

if __package__:
    from .fp8_coverage import _FP8CoverageTracker
else:
    from fp8_coverage import _FP8CoverageTracker
from spikingjelly.activation_based import functional, layer, neuron, surrogate
from spikingjelly.activation_based.precision import (
    PrecisionArtifacts,
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
    inference_ms: float
    inference_samples_per_sec: float
    inference_peak_allocated_mb: float
    inference_peak_reserved_mb: float
    conversion_report: dict
    precision_report: dict
    coverage_report: dict


def _tensor_metadata(value: Any) -> Any:
    if torch.is_tensor(value):
        return {
            "shape": list(value.shape),
            "dtype": str(value.dtype),
            "device": str(value.device),
            "stride": list(value.stride()),
            "contiguous": value.is_contiguous(),
            "bytes": value.numel() * value.element_size(),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {str(key): _tensor_metadata(item) for key, item in value.items()}
    return type(value).__name__


@contextmanager
def _nvtx_range(name: str, enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


@contextmanager
def _cuda_profiler_capture(enabled: bool) -> Iterator[None]:
    if enabled:
        torch.cuda.profiler.start()
    try:
        yield
    finally:
        if enabled:
            torch.cuda.profiler.stop()


class _ProfileHooks:
    def __init__(self, model: torch_nn.Module, output: Path):
        self.output = output
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self.active = False
        self.metadata_modules: set[str] = set()
        self.open_ranges = 0

        for name, module in model.net.named_children():
            self.handles.append(
                module.register_forward_pre_hook(self._forward_pre(name, module))
            )
            self.handles.append(
                module.register_forward_hook(self._forward_post(name), always_call=True)
            )
            self.handles.append(
                module.register_full_backward_pre_hook(self._backward_pre(name, module))
            )
            self.handles.append(
                module.register_full_backward_hook(self._backward_post())
            )

    def _push(self, name: str) -> None:
        torch.cuda.nvtx.range_push(name)
        self.open_ranges += 1

    def _pop(self) -> None:
        torch.cuda.nvtx.range_pop()
        self.open_ranges -= 1

    def _record(
        self, event: str, name: str, module: torch_nn.Module, value: Any
    ) -> None:
        if not self.active:
            return
        record = {"event": event, "module": name, "type": type(module).__name__}
        if name not in self.metadata_modules:
            record["value"] = _tensor_metadata(value)
            self.records.append(record)
            if event == "forward_output":
                self.metadata_modules.add(name)

    def _forward_pre(self, name: str, module: torch_nn.Module):
        def hook(_module: torch_nn.Module, inputs: tuple[Any, ...]) -> None:
            if self.active:
                self._push(f"module_forward:{name}:{type(module).__name__}")
                self._record("forward_input", name, module, inputs)

        return hook

    def _forward_post(self, name: str):
        def hook(module: torch_nn.Module, _inputs: Any, output: Any) -> None:
            if self.active:
                self._record("forward_output", name, module, output)
                self._pop()

        return hook

    def _backward_pre(self, name: str, module: torch_nn.Module):
        def hook(_module: torch_nn.Module, _grad_output: tuple[Any, ...]) -> None:
            if self.active:
                self._push(f"module_backward:{name}:{type(module).__name__}")

        return hook

    def _backward_post(self):
        def hook(_module: torch_nn.Module, _grad_input: Any, _grad_output: Any) -> None:
            if self.active:
                self._pop()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()
        while self.open_ranges:
            torch.cuda.nvtx.range_pop()
            self.open_ranges -= 1
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as file:
            for record in self.records:
                file.write(json.dumps(record) + "\n")


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
            "Benchmark SpikingJelly deep FC SNN training and inference under fp32, "
            "fp16, bf16, and fp8."
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
    parser.add_argument("--inference-steps", type=int, default=100)
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
        default=["fp32", "fp16", "bf16", "fp8"],
        choices=("fp32", "fp16", "bf16", "fp8"),
        help="Precision modes to benchmark.",
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("auto", "delayed", "current", "block", "mxfp8"),
        default="auto",
        help="Transformer Engine FP8 scaling recipe.",
    )
    parser.add_argument(
        "--fp8-fallback-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
        help="Autocast dtype for non-TE CUDA operations in FP8 runs.",
    )
    parser.add_argument(
        "--triton-storage",
        choices=("none", "fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"),
        default="none",
        help="Optional Triton neuron state storage dtype.",
    )
    parser.add_argument(
        "--triton-fwd",
        choices=("fp8", "fp16", "bf16", "fp32"),
        default="fp32",
        help="Triton neuron forward compute dtype.",
    )
    parser.add_argument(
        "--triton-bwd",
        choices=("fp8", "fp16", "bf16", "fp32"),
        default="fp32",
        help="Triton neuron backward compute dtype.",
    )
    parser.add_argument(
        "--profile",
        action="store_true",
        help="Capture bounded NVTX/CUDA-profiler ranges for one precision.",
    )
    parser.add_argument(
        "--profile-module-hooks",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Add per-module forward/backward NVTX ranges during profiling.",
    )
    parser.add_argument(
        "--profile-steps",
        type=int,
        default=10,
        help="Number of training and inference steps inside a profile capture.",
    )
    parser.add_argument(
        "--tensor-metadata-output",
        type=Path,
        help="Optional JSONL path for first-step module tensor metadata.",
    )
    parser.add_argument(
        "--json", action="store_true", help="Print the full benchmark report as JSON."
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Optional path for the full JSON benchmark report.",
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


def validate_profile_args(args: argparse.Namespace) -> None:
    if not args.profile:
        return
    if args.profile_steps <= 0:
        raise ValueError("profile_steps must be positive when profiling is enabled.")
    if len(args.precisions) != 1:
        raise ValueError("profile mode requires exactly one precision.")
    if args.output is None:
        raise ValueError("profile mode requires --output for the benchmark JSON.")


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
    fp8_precisions = {"fp8"} & set(args.precisions)
    if not fp8_precisions:
        return
    invalid_dims = [
        f"{name}={value}"
        for name, value in constrained_dims.items()
        if value % FP8_ALIGNMENT != 0
    ]
    if invalid_dims:
        requested = ", ".join(sorted(fp8_precisions))
        raise ValueError(
            f"{requested} benchmark runs currently require every linear dimension "
            f"to be divisible by {FP8_ALIGNMENT}. Invalid values: "
            + ", ".join(invalid_dims)
        )


def _step_optimizer(
    artifacts: PrecisionArtifacts, optimizer: torch.optim.Optimizer
) -> None:
    if artifacts.scaler is None:
        optimizer.step()
    else:
        artifacts.scaler.step(optimizer)
        artifacts.scaler.update()


def run_training_step(
    model: torch_nn.Module,
    artifacts: PrecisionArtifacts,
    optimizer: torch.optim.Optimizer,
    criterion: torch_nn.Module,
    x_seq: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
    nvtx_step: str | None = None,
) -> dict[str, float]:
    profile = nvtx_step is not None
    with _nvtx_range(nvtx_step or "", profile):
        with _nvtx_range("reset", profile):
            functional.reset_net(model)
        with _nvtx_range("zero_grad", profile):
            optimizer.zero_grad(set_to_none=True)

        cuda_events = make_timing_events(device)
        if cuda_events is not None:
            cuda_events["step_start"].record()
            cuda_events["forward_start"].record()
            with _nvtx_range("forward", profile):
                with artifacts.autocast_context():
                    logits = model(x_seq)
            with _nvtx_range("loss", profile):
                loss = criterion(logits, target)
            cuda_events["forward_end"].record()
            cuda_events["backward_start"].record()
            with _nvtx_range("backward", profile):
                artifacts.backward(loss, optimizer, step_optimizer=False)
            cuda_events["backward_end"].record()
            cuda_events["optimizer_start"].record()
            with _nvtx_range("optimizer", profile):
                _step_optimizer(artifacts, optimizer)
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
            with _nvtx_range("forward", profile):
                with artifacts.autocast_context():
                    logits = model(x_seq)
            with _nvtx_range("loss", profile):
                return logits, criterion(logits, target)

        def backward_section() -> None:
            with _nvtx_range("backward", profile):
                artifacts.backward(loss, optimizer, step_optimizer=False)

        def optimizer_section() -> None:
            with _nvtx_range("optimizer", profile):
                _step_optimizer(artifacts, optimizer)

        forward_ms, (_, loss) = _time_cpu_section(forward_section)

        backward_ms, _ = _time_cpu_section(backward_section)
        optimizer_ms, _ = _time_cpu_section(optimizer_section)
        return {
            "forward_ms": forward_ms,
            "backward_ms": backward_ms,
            "optimizer_ms": optimizer_ms,
            "total_step_ms": forward_ms + backward_ms + optimizer_ms,
        }


def run_inference_step(
    model: torch_nn.Module,
    artifacts: PrecisionArtifacts,
    x_seq: torch.Tensor,
    device: torch.device,
    nvtx_step: str | None = None,
) -> float:
    profile = nvtx_step is not None
    with _nvtx_range(nvtx_step or "", profile):
        with _nvtx_range("reset", profile):
            functional.reset_net(model)

        if device.type == "cuda":
            start = torch.cuda.Event(enable_timing=True)
            end = torch.cuda.Event(enable_timing=True)
            start.record()
            with _nvtx_range("forward", profile):
                with torch.inference_mode(), artifacts.autocast_context():
                    output = model(x_seq)
            end.record()
            sync_if_needed(device)
            if not torch.isfinite(output).all():
                raise RuntimeError("Inference produced non-finite output.")
            return _event_elapsed_ms(start, end)

        def forward_section():
            with _nvtx_range("forward", profile):
                with torch.inference_mode(), artifacts.autocast_context():
                    return model(x_seq)

        inference_ms, output = _time_cpu_section(forward_section)
        if not torch.isfinite(output).all():
            raise RuntimeError("Inference produced non-finite output.")
        return inference_ms


def benchmark_one_precision(
    args: argparse.Namespace,
    precision: str,
    model_state: dict,
    x_seq: torch.Tensor,
    target: torch.Tensor,
    device: torch.device,
) -> BenchResult:
    profile_enabled = args.profile
    training_steps = args.profile_steps if profile_enabled else args.steps
    inference_steps = args.profile_steps if profile_enabled else args.inference_steps
    model = build_model(args).to(device)
    model.load_state_dict(model_state, strict=True)
    model.train()

    triton_storage = args.triton_storage
    if triton_storage == "none":
        triton_storage = None
    artifacts = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(
            mode=precision,
            fp8_recipe=args.fp8_recipe,
            fp8_fallback_dtype=args.fp8_fallback_dtype
            if precision == "fp8"
            else "bf16",
            triton_storage=triton_storage,
            triton_fwd=args.triton_fwd,
            triton_bwd=args.triton_bwd,
        ),
    )
    model = artifacts.model
    precision_report = artifacts.describe()
    coverage_tracker = _FP8CoverageTracker(
        model,
        precision_report["conversion_report"],
    )
    model.eval()
    try:
        functional.reset_net(model)
        with torch.inference_mode(), artifacts.autocast_context():
            model(x_seq)
    finally:
        coverage_tracker.close()
        functional.reset_net(model)
        model.train()
    coverage_report = coverage_tracker.report()
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

    profile_hooks = None
    if profile_enabled and args.profile_module_hooks:
        metadata_output = args.tensor_metadata_output
        if metadata_output is None:
            metadata_output = Path(f"{args.output}.tensors.jsonl")
        profile_hooks = _ProfileHooks(model, metadata_output)

    forward_ms = 0.0
    backward_ms = 0.0
    optimizer_ms = 0.0
    total_step_ms = 0.0
    peak_allocated_mb = float("nan")
    peak_reserved_mb = float("nan")
    inference_ms = 0.0
    inference_peak_allocated_mb = float("nan")
    inference_peak_reserved_mb = float("nan")
    try:
        with _cuda_profiler_capture(profile_enabled):
            if profile_hooks is not None:
                profile_hooks.active = True
            wall_start = time.perf_counter()
            for index in range(training_steps):
                step_metrics = run_training_step(
                    model,
                    artifacts,
                    optimizer,
                    criterion,
                    x_seq,
                    target,
                    device,
                    nvtx_step=f"benchmark_step:training:{index}"
                    if profile_enabled
                    else None,
                )
                forward_ms += step_metrics["forward_ms"]
                backward_ms += step_metrics["backward_ms"]
                optimizer_ms += step_metrics["optimizer_ms"]
                total_step_ms += step_metrics["total_step_ms"]
            sync_if_needed(device)
            wall_elapsed = time.perf_counter() - wall_start

            if device.type == "cuda":
                peak_allocated_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024 / 1024
                )
                peak_reserved_mb = torch.cuda.max_memory_reserved(device) / 1024 / 1024

            optimizer.zero_grad(set_to_none=True)
            optimizer.state.clear()
            model.eval()
            if profile_hooks is not None:
                profile_hooks.active = False
            for _ in range(args.warmup):
                run_inference_step(model, artifacts, x_seq, device)
            sync_if_needed(device)
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats(device)

            if profile_hooks is not None:
                profile_hooks.active = True
            inference_wall_start = time.perf_counter()
            for index in range(inference_steps):
                inference_ms += run_inference_step(
                    model,
                    artifacts,
                    x_seq,
                    device,
                    nvtx_step=f"benchmark_step:inference:{index}"
                    if profile_enabled
                    else None,
                )
            sync_if_needed(device)
            inference_wall_elapsed = time.perf_counter() - inference_wall_start
            if profile_hooks is not None:
                profile_hooks.active = False

            if device.type == "cuda":
                inference_peak_allocated_mb = (
                    torch.cuda.max_memory_allocated(device) / 1024 / 1024
                )
                inference_peak_reserved_mb = (
                    torch.cuda.max_memory_reserved(device) / 1024 / 1024
                )
    finally:
        if profile_hooks is not None:
            profile_hooks.close()

    return BenchResult(
        precision=precision,
        batch_size=args.batch_size,
        steps=training_steps,
        warmup=args.warmup,
        forward_ms=forward_ms / training_steps,
        backward_ms=backward_ms / training_steps,
        optimizer_ms=optimizer_ms / training_steps,
        total_step_ms=total_step_ms / training_steps,
        samples_per_sec=(args.batch_size * training_steps) / wall_elapsed,
        peak_allocated_mb=peak_allocated_mb,
        peak_reserved_mb=peak_reserved_mb,
        inference_ms=inference_ms / inference_steps,
        inference_samples_per_sec=(args.batch_size * inference_steps)
        / inference_wall_elapsed,
        inference_peak_allocated_mb=inference_peak_allocated_mb,
        inference_peak_reserved_mb=inference_peak_reserved_mb,
        conversion_report=precision_report["conversion_report"],
        precision_report=precision_report,
        coverage_report=coverage_report,
    )


def print_table(results: list[BenchResult]) -> None:
    print(
        f"{'precision':<14s} {'forward_ms':>12s} {'backward_ms':>12s} "
        f"{'optim_ms':>10s} {'step_ms':>12s} {'train_s/s':>12s} "
        f"{'train_alloc_MB':>14s} {'infer_ms':>10s} {'infer_s/s':>12s} "
        f"{'infer_alloc_MB':>14s}"
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
            f"{result.inference_ms:10.3f} "
            f"{result.inference_samples_per_sec:12.1f} "
            f"{result.inference_peak_allocated_mb:14.1f}"
        )


def main() -> None:
    args = parse_args()
    validate_profile_args(args)
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
        "inference_steps": args.inference_steps,
        "backend": args.backend,
        "fp8_recipe": args.fp8_recipe,
        "fp8_fallback_dtype": args.fp8_fallback_dtype,
        "triton_storage": args.triton_storage,
        "triton_fwd": args.triton_fwd,
        "triton_bwd": args.triton_bwd,
        "profile": args.profile,
        "profile_steps": args.profile_steps,
        "profile_module_hooks": args.profile_module_hooks,
        "results": [asdict(result) for result in results],
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2) + "\n")

    if args.json:
        print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
