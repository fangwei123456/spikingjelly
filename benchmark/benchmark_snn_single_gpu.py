from __future__ import annotations

import argparse
from contextlib import contextmanager, suppress
import json
import os
import platform
import signal
import statistics
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch

if __package__:
    from .fp8_coverage import _FP8CoverageTracker
else:
    from fp8_coverage import _FP8CoverageTracker
from spikingjelly.activation_based.precision import (
    PrecisionConfig,
    prepare_model_for_precision,
)

DEFAULT_T = 4
DEFAULT_IMAGE_SIZE = 224
DEFAULT_SEED = 20260808
MODEL_NAMES = ("spiking_vgg16_bn", "sew_resnet18", "spikformer_ti")
PHASES = ("inference", "training")
EXECUTIONS = ("eager", "compile")


@contextmanager
def _nvtx_range(name: str, enabled: bool):
    if enabled:
        torch.cuda.nvtx.range_push(name)
    try:
        yield
    finally:
        if enabled:
            torch.cuda.nvtx.range_pop()


def summarize_samples(samples_ms: list[float]) -> dict[str, float]:
    if not samples_ms:
        raise ValueError("samples_ms must not be empty")
    deciles = (
        statistics.quantiles(samples_ms, n=10, method="inclusive")
        if len(samples_ms) > 1
        else samples_ms * 9
    )
    return {
        "median_ms": statistics.median(samples_ms),
        "p10_ms": deciles[0],
        "p90_ms": deciles[8],
        "stdev_ms": statistics.stdev(samples_ms) if len(samples_ms) > 1 else 0.0,
    }


def parse_source_specs(values: list[str]) -> list[tuple[str, Path]]:
    if len(values) != 2:
        raise ValueError("matrix requires exactly two --source LABEL=PATH values")

    sources = []
    for value in values:
        if "=" not in value:
            raise ValueError(f"invalid source {value!r}; expected LABEL=PATH")
        label, raw_path = value.split("=", 1)
        path = Path(raw_path).expanduser().resolve()
        if not label:
            raise ValueError("source labels must be non-empty")
        if not (path / "spikingjelly" / "__init__.py").is_file():
            raise ValueError(f"source root does not contain spikingjelly/: {path}")
        sources.append((label, path))

    if sources[0][0] == sources[1][0]:
        raise ValueError(f"source labels must be unique: {sources[0][0]!r}")
    return sources


def case_key(record: dict[str, Any]) -> tuple[str, str, str, int, int, int]:
    case = record["case"]
    return (
        case["model"],
        case["phase"],
        case["execution"],
        case["T"],
        case["batch_size"],
        case["image_size"],
    )


def _run_isolated_case(command: list[str], env: dict[str, str], timeout: int) -> None:
    process = subprocess.Popen(
        command,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        start_new_session=True,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired as error:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        stdout, stderr = process.communicate()
        raise subprocess.TimeoutExpired(
            command, timeout, output=stdout, stderr=stderr
        ) from error
    if process.returncode:
        raise subprocess.CalledProcessError(
            process.returncode, command, output=stdout, stderr=stderr
        )


def _stop_monitor(monitor) -> None:
    if monitor is None:
        return
    try:
        if monitor.poll() is not None:
            return
        monitor.terminate()
        monitor.wait(timeout=10)
    except ProcessLookupError:
        return
    except subprocess.TimeoutExpired:
        with suppress(ProcessLookupError):
            monitor.kill()
            monitor.wait()


def aggregate_records(
    records: list[dict[str, Any]], baseline_label: str, candidate_label: str
) -> dict[str, Any]:
    grouped: dict[tuple[str, str, str, int, int, int], dict[str, list[dict]]] = {}
    failures = []
    for record in records:
        compile_metrics = record.get("dynamo", {})
        compile_metrics_invalid = record.get("case", {}).get(
            "execution"
        ) == "compile" and (
            compile_metrics.get("graph_break_count") != 0
            or compile_metrics.get("recompile_count") != 0
        )
        if record.get("status") == "error" or compile_metrics_invalid:
            failures.append(record)
            continue
        grouped.setdefault(case_key(record), {}).setdefault(
            record["source_label"], []
        ).append(record)

    comparisons = []
    for key, by_source in sorted(grouped.items()):
        baseline = by_source.get(baseline_label, [])
        candidate = by_source.get(candidate_label, [])
        if not baseline or not candidate:
            continue
        baseline_by_round = {item["round"]: item for item in baseline}
        candidate_by_round = {item["round"]: item for item in candidate}
        paired_rounds = sorted(baseline_by_round.keys() & candidate_by_round.keys())
        baseline = [baseline_by_round[round_index] for round_index in paired_rounds]
        candidate = [candidate_by_round[round_index] for round_index in paired_rounds]
        if not baseline:
            continue
        baseline_rounds = [item["timing"]["median_ms"] for item in baseline]
        candidate_rounds = [item["timing"]["median_ms"] for item in candidate]
        baseline_ms = statistics.median(baseline_rounds)
        candidate_ms = statistics.median(candidate_rounds)
        candidate_min = min(candidate_rounds)
        latency_change_pct = (
            (candidate_ms / baseline_ms - 1.0) * 100.0 if baseline_ms else None
        )
        candidate_round_spread = (
            max(candidate_rounds) / candidate_min if candidate_min else None
        )
        baseline_peak = statistics.median(
            item["memory"]["peak_allocated_bytes"] for item in baseline
        )
        candidate_peak = statistics.median(
            item["memory"]["peak_allocated_bytes"] for item in candidate
        )
        comparisons.append(
            {
                "case": {
                    "model": key[0],
                    "phase": key[1],
                    "execution": key[2],
                    "T": key[3],
                    "batch_size": key[4],
                    "image_size": key[5],
                },
                "rounds": min(len(baseline), len(candidate)),
                "baseline_round_medians_ms": baseline_rounds,
                "candidate_round_medians_ms": candidate_rounds,
                "baseline_median_ms": baseline_ms,
                "candidate_median_ms": candidate_ms,
                "latency_change_pct": latency_change_pct,
                "peak_allocated_change_pct": (
                    (candidate_peak / baseline_peak - 1.0) * 100.0
                    if baseline_peak
                    else None
                ),
                "all_candidate_rounds_faster": all(
                    candidate_item["timing"]["median_ms"]
                    < baseline_item["timing"]["median_ms"]
                    for baseline_item, candidate_item in zip(
                        baseline, candidate, strict=False
                    )
                ),
                "candidate_round_spread": candidate_round_spread,
            }
        )
    qualifying_families = {
        item["case"]["model"]
        for item in comparisons
        if item["all_candidate_rounds_faster"]
        and (
            (
                item["latency_change_pct"] is not None
                and item["latency_change_pct"] <= -5.0
            )
            or (
                item["peak_allocated_change_pct"] is not None
                and item["peak_allocated_change_pct"] <= -15.0
            )
        )
    }
    stable_rounds = all(
        item["rounds"] >= 3
        and item["candidate_round_spread"] is not None
        and item["candidate_round_spread"] <= 1.05
        for item in comparisons
    )
    no_latency_regression = all(
        item["latency_change_pct"] is not None and item["latency_change_pct"] <= 3.0
        for item in comparisons
    )
    return {
        "baseline_label": baseline_label,
        "candidate_label": candidate_label,
        "comparisons": comparisons,
        "failures": failures,
        "performance_gates": {
            "qualifying_model_families": sorted(qualifying_families),
            "at_least_two_model_families": len(qualifying_families) >= 2,
            "three_stable_rounds_per_case": stable_rounds,
            "no_case_latency_regression_over_3pct": no_latency_regression,
            "met": (
                len(qualifying_families) >= 2
                and stable_rounds
                and no_latency_regression
                and not failures
            ),
        },
    }


def _run_text(command: list[str], cwd: Path | None = None) -> str | None:
    try:
        return subprocess.check_output(
            command, cwd=cwd, stderr=subprocess.DEVNULL, text=True, timeout=15
        ).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def _git_metadata(source_root: Path) -> dict[str, Any]:
    status = _run_text(["git", "status", "--short"], cwd=source_root)
    return {
        "revision": _run_text(["git", "rev-parse", "HEAD"], cwd=source_root),
        "dirty": bool(status),
        "status": status,
    }


def _physical_gpu_selector(device: torch.device) -> str:
    logical_index = device.index or 0
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if visible_devices:
        selectors = [item.strip() for item in visible_devices.split(",")]
        if logical_index < len(selectors) and selectors[logical_index]:
            return selectors[logical_index]
    return str(logical_index)


def _nvidia_snapshot(gpu_selector: str) -> dict[str, Any] | None:
    fields = (
        "name,uuid,compute_cap,driver_version,memory.total,clocks.current.sm,"
        "clocks.current.memory,power.draw,temperature.gpu,pstate,"
        "clocks_throttle_reasons.active"
    )
    raw = _run_text(
        [
            "nvidia-smi",
            "-i",
            gpu_selector,
            f"--query-gpu={fields}",
            "--format=csv,noheader,nounits",
        ]
    )
    if raw is None:
        return None
    processes = _run_text(
        [
            "nvidia-smi",
            "-i",
            gpu_selector,
            "--query-compute-apps=pid,process_name,used_memory",
            "--format=csv,noheader,nounits",
        ]
    )
    return {
        "gpu_selector": gpu_selector,
        "query_fields": fields.split(","),
        "gpu_rows": raw.splitlines(),
        "processes": processes,
    }


def _environment_metadata(
    source_root: Path,
    device: torch.device,
    package_file: Path,
    gpu_selector: str,
) -> dict[str, Any]:
    try:
        import triton

        triton_version = triton.__version__
    except ImportError:
        triton_version = None
    props = torch.cuda.get_device_properties(device)
    tracked_env = {
        key: value
        for key, value in os.environ.items()
        if key.startswith(("CUDA", "TORCH", "TRITON", "SJ_", "NCCL"))
        and not any(
            marker in key.upper()
            for marker in ("TOKEN", "SECRET", "KEY", "PASSWORD", "CREDENTIAL")
        )
    }
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "argv": sys.argv,
        "source_root": str(source_root),
        "spikingjelly_file": str(package_file),
        "git": _git_metadata(source_root),
        "python": sys.version,
        "platform": platform.platform(),
        "torch": torch.__version__,
        "triton": triton_version,
        "torch_cuda": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(),
        "selected_device_index": device.index,
        "physical_gpu_selector": gpu_selector,
        "gpu": {
            "name": props.name,
            "compute_capability": [props.major, props.minor],
            "total_memory_bytes": props.total_memory,
        },
        "environment": tracked_env,
        "nvidia_smi_before": _nvidia_snapshot(gpu_selector),
    }


def _tensor_metadata(value: Any) -> Any:
    if isinstance(value, torch.Tensor):
        return {
            "shape": list(value.shape),
            "stride": list(value.stride()),
            "storage_offset": value.storage_offset(),
            "dtype": str(value.dtype).removeprefix("torch."),
            "device": str(value.device),
            "bytes": value.numel() * value.element_size(),
            "contiguous": value.is_contiguous(),
        }
    if isinstance(value, (tuple, list)):
        return [_tensor_metadata(item) for item in value]
    if isinstance(value, dict):
        return {key: _tensor_metadata(item) for key, item in value.items()}
    return type(value).__name__


class _ProfileHooks:
    _INTERESTING = (
        "Node",
        "SeqToANN",
        "Attention",
        "BatchNorm",
        "Conv",
        "Pool",
        "Linear",
    )

    def __init__(self, model: torch.nn.Module, output: Path):
        self.output = output
        self.records: list[dict[str, Any]] = []
        self.handles = []
        self.metadata_modules: set[str] = set()
        for name, module in model.named_modules():
            if name and any(
                token in type(module).__name__ for token in self._INTERESTING
            ):
                self.handles.append(
                    module.register_forward_pre_hook(self._pre_hook(name, module))
                )
                self.handles.append(module.register_forward_hook(self._post_hook(name)))

    def _pre_hook(self, name: str, module: torch.nn.Module):
        def hook(_module, inputs):
            torch.cuda.nvtx.range_push(f"module:{type(module).__name__}:{name}")
            if name not in self.metadata_modules:
                self.records.append(
                    {
                        "event": "input",
                        "module": name,
                        "type": type(module).__name__,
                        "value": _tensor_metadata(inputs),
                    }
                )

        return hook

    def _post_hook(self, name: str):
        def hook(module, _inputs, output):
            if name not in self.metadata_modules:
                self.records.append(
                    {
                        "event": "output",
                        "module": name,
                        "type": type(module).__name__,
                        "value": _tensor_metadata(output),
                    }
                )
                self.metadata_modules.add(name)
            torch.cuda.nvtx.range_pop()

        return hook

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.output.parent.mkdir(parents=True, exist_ok=True)
        with self.output.open("w", encoding="utf-8") as file:
            for record in self.records:
                file.write(json.dumps(record) + "\n")


def _build_model(name: str, backend: str, T: int, num_classes: int):
    from spikingjelly.activation_based import functional, neuron
    from spikingjelly.activation_based.model import sew_resnet, spikformer, spiking_vgg

    node_kwargs = {
        "spiking_neuron": neuron.LIFNode,
        "tau": 2.0,
        "detach_reset": True,
        "step_mode": "m",
        "backend": backend,
    }
    if name == "spiking_vgg16_bn":
        model = spiking_vgg.spiking_vgg16_bn(num_classes=num_classes, **node_kwargs)
    elif name == "sew_resnet18":
        model = sew_resnet.sew_resnet18(
            cnf="ADD", num_classes=num_classes, **node_kwargs
        )
    elif name == "spikformer_ti":
        model = spikformer.spikformer_ti(T=T, num_classes=num_classes, backend=backend)
    else:
        raise ValueError(name)
    functional.set_step_mode(model, "m")
    return model


def _make_batch(args: argparse.Namespace, device: torch.device):
    shape = (args.batch_size, 3, args.image_size, args.image_size)
    if args.model != "spikformer_ti":
        shape = (args.T, *shape)
    x = torch.randn(shape, device=device)
    if args.channels_last:
        if x.ndim == 5:
            x = (
                x.flatten(0, 1)
                .contiguous(memory_format=torch.channels_last)
                .view(shape)
            )
        else:
            x = x.contiguous(memory_format=torch.channels_last)
    target = torch.randint(args.num_classes, (args.batch_size,), device=device)
    return x, target


def _dynamo_metrics() -> dict[str, Any]:
    dynamo = getattr(torch, "_dynamo", None)
    utils = getattr(dynamo, "utils", None)
    counters = getattr(utils, "counters", {})
    raw = {group: dict(values) for group, values in counters.items() if values}
    return {
        "graph_count": int(raw.get("stats", {}).get("unique_graphs", 0)),
        "graph_break_count": int(sum(raw.get("graph_break", {}).values())),
        "recompile_count": int(sum(raw.get("recompiles", {}).values())),
        "raw_counters": raw,
    }


def run_case(args: argparse.Namespace) -> dict[str, Any]:
    if not torch.cuda.is_available():
        raise RuntimeError("benchmark cases require CUDA")
    source_root = Path(
        os.environ.get("SJ_BENCH_SOURCE_ROOT", Path(__file__).parents[1])
    ).resolve()
    sys.path.insert(0, str(source_root))
    import spikingjelly
    from spikingjelly.activation_based import functional

    package_file = Path(spikingjelly.__file__).resolve()
    try:
        package_file.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            f"imported spikingjelly from {package_file}, outside source root {source_root}"
        ) from error
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    device = torch.device("cuda", args.device)
    torch.cuda.set_device(device)
    gpu_selector = _physical_gpu_selector(device)
    if (
        args.require_gpu_name
        and args.require_gpu_name not in torch.cuda.get_device_name(device)
    ):
        raise RuntimeError(
            f"required GPU containing {args.require_gpu_name!r}, got {torch.cuda.get_device_name(device)!r}"
        )

    metadata = _environment_metadata(source_root, device, package_file, gpu_selector)
    model = _build_model(args.model, args.neuron_backend, args.T, args.num_classes).to(
        device
    )
    precision = prepare_model_for_precision(
        model,
        device,
        PrecisionConfig(
            mode=args.precision,
            fp8_recipe=args.fp8_recipe,
            fp8_fallback_dtype=args.fp8_fallback_dtype,
            triton_storage=args.triton_storage,
            triton_fwd=args.triton_fwd,
            triton_bwd=args.triton_bwd,
        ),
    )
    model = precision.model
    state_model = model
    model.train(args.phase == "training")
    metadata["precision"] = precision.describe()
    x, target = _make_batch(args, device)
    coverage_tracker = _FP8CoverageTracker(
        state_model,
        metadata["precision"]["conversion_report"],
    )
    optimizer = (
        torch.optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
        if args.phase == "training"
        else None
    )
    criterion = torch.nn.CrossEntropyLoss()

    def reset_net() -> None:
        functional.reset_net(state_model)

    def step() -> torch.Tensor:
        if optimizer is None:
            with _nvtx_range("forward", args.profile):
                with torch.inference_mode(), precision.autocast_context():
                    output = model(x)
            with _nvtx_range("reset", args.profile):
                reset_net()
            return output
        with _nvtx_range("zero_grad", args.profile):
            optimizer.zero_grad(set_to_none=True)
        with _nvtx_range("forward", args.profile):
            with precision.autocast_context():
                output = model(x)
        with _nvtx_range("loss", args.profile):
            loss = criterion(output.mean(0), target)
        with _nvtx_range("backward", args.profile):
            precision.backward(loss, optimizer, step_optimizer=False)
        with _nvtx_range("optimizer", args.profile):
            if precision.scaler is None:
                optimizer.step()
            else:
                precision.scaler.step(optimizer)
                precision.scaler.update()
        with _nvtx_range("reset", args.profile):
            reset_net()
        return loss

    dynamo = getattr(torch, "_dynamo", None)
    if dynamo is not None:
        dynamo.reset()
        counters = getattr(getattr(dynamo, "utils", None), "counters", None)
        if counters is not None:
            counters.clear()
    if args.execution == "compile":
        model = torch.compile(
            model,
            backend="inductor",
            options={"triton.cudagraphs": False, "triton.cudagraph_trees": False},
        )

    hooks = None
    if args.profile and args.tensor_metadata:
        hooks = _ProfileHooks(state_model, args.tensor_metadata)
    monitor = None
    if args.monitor_log:
        args.monitor_log.parent.mkdir(parents=True, exist_ok=True)
        try:
            with args.monitor_log.open("w", encoding="utf-8") as monitor_file:
                monitor = subprocess.Popen(
                    [
                        "nvidia-smi",
                        "dmon",
                        "-i",
                        gpu_selector,
                        "-s",
                        "pucvmet",
                        "-d",
                        "1",
                        "-o",
                        "DT",
                    ],
                    stdout=monitor_file,
                    stderr=subprocess.STDOUT,
                    text=True,
                )
        except OSError:
            pass

    try:
        if args.profile:
            torch.cuda.profiler.start()
        torch.cuda.synchronize(device)
        first_started = time.perf_counter()
        step()
        torch.cuda.synchronize(device)
        first_step_wall_ms = (time.perf_counter() - first_started) * 1000.0
        coverage_tracker.close()
        metadata["fp8_coverage"] = coverage_tracker.report()

        for _ in range(args.warmup):
            step()
        torch.cuda.synchronize(device)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(device)
        allocated_before = torch.cuda.memory_allocated(device)
        reserved_before = torch.cuda.memory_reserved(device)
        starts = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
        ends = [torch.cuda.Event(enable_timing=True) for _ in range(args.steps)]
        for index in range(args.steps):
            starts[index].record()
            with _nvtx_range(f"benchmark_step:{index}", args.profile):
                step()
            ends[index].record()
        ends[-1].synchronize()
        samples_ms = [
            start.elapsed_time(end) for start, end in zip(starts, ends, strict=True)
        ]
        peak_allocated_bytes = torch.cuda.max_memory_allocated(device)
        peak_reserved_bytes = torch.cuda.max_memory_reserved(device)

        if args.tensor_metadata and hooks is None:
            hooks = _ProfileHooks(state_model, args.tensor_metadata)
            with _nvtx_range("layout_metadata_step", True):
                step()
            torch.cuda.synchronize(device)
            hooks.close()
            hooks = None
        if hooks is not None:
            torch.cuda.synchronize(device)
            hooks.close()
            hooks = None
    finally:
        if args.profile:
            torch.cuda.profiler.stop()
        coverage_tracker.close()
        if hooks is not None:
            hooks.close()
        _stop_monitor(monitor)

    dynamo_metrics = _dynamo_metrics()
    result = {
        "schema_version": 1,
        "source_label": args.source_label,
        "round": args.round,
        "case": {
            "model": args.model,
            "phase": args.phase,
            "execution": args.execution,
            "backend": args.neuron_backend,
            "dtype": "float32",
            "T": args.T,
            "batch_size": args.batch_size,
            "image_size": args.image_size,
            "num_classes": args.num_classes,
            "channels_last": args.channels_last,
            "neuron_backend": args.neuron_backend,
            "precision": args.precision,
            "warmup": args.warmup,
            "steps": args.steps,
            "seed": args.seed,
            "optimizer": (
                {"name": "SGD", "lr": args.lr, "momentum": args.momentum}
                if optimizer is not None
                else None
            ),
            "compile": (
                {"backend": "inductor", "mode": "default", "cudagraphs": False}
                if args.execution == "compile"
                else None
            ),
        },
        "timing": {
            **summarize_samples(samples_ms),
            "samples_ms": samples_ms,
            "first_step_wall_ms": first_step_wall_ms,
        },
        "memory": {
            "allocated_before_bytes": allocated_before,
            "reserved_before_bytes": reserved_before,
            "peak_allocated_bytes": peak_allocated_bytes,
            "peak_reserved_bytes": peak_reserved_bytes,
        },
        "dynamo": dynamo_metrics,
        "metadata": {
            **metadata,
            "nvidia_smi_after": _nvidia_snapshot(gpu_selector),
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                "output": str(args.output),
                "case": result["case"],
                "timing": result["timing"],
            }
        )
    )
    return result


def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    sources = parse_source_specs(args.source)
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    script = Path(__file__).resolve()
    records: list[dict[str, Any]] = []
    for round_index in range(1, args.rounds + 1):
        ordered_sources = sources if round_index % 2 else list(reversed(sources))
        for model in args.models:
            for phase in args.phases:
                batch_size = (
                    args.training_batch_size
                    if phase == "training"
                    else args.inference_batch_size
                )
                warmup = (
                    args.training_warmup
                    if phase == "training"
                    else args.inference_warmup
                )
                steps = (
                    args.training_steps if phase == "training" else args.inference_steps
                )
                if args.profile:
                    steps = args.profile_steps
                for execution in args.executions:
                    for label, source_root in ordered_sources:
                        stem = f"r{round_index}-{label}-{model}-{phase}-{execution}"
                        output = output_dir / f"{stem}.json"
                        command = [
                            sys.executable,
                            str(script),
                            "case",
                            "--model",
                            model,
                            "--phase",
                            phase,
                            "--execution",
                            execution,
                            "--batch-size",
                            str(batch_size),
                            "--warmup",
                            str(warmup),
                            "--steps",
                            str(steps),
                            "--T",
                            str(args.T),
                            "--image-size",
                            str(args.image_size),
                            "--num-classes",
                            str(args.num_classes),
                            "--seed",
                            str(args.seed),
                            "--device",
                            str(args.device),
                            "--neuron-backend",
                            args.neuron_backend,
                            "--precision",
                            args.precision,
                            "--fp8-recipe",
                            args.fp8_recipe,
                            "--fp8-fallback-dtype",
                            args.fp8_fallback_dtype,
                            "--source-label",
                            label,
                            "--round",
                            str(round_index),
                            "--output",
                            str(output),
                            "--monitor-log",
                            str(output_dir / f"{stem}.dmon.log"),
                        ]
                        if args.profile:
                            command.append("--profile")
                            if execution == "eager":
                                command.extend(
                                    [
                                        "--tensor-metadata",
                                        str(output_dir / f"{stem}.tensors.jsonl"),
                                    ]
                                )
                        if args.channels_last:
                            command.append("--channels-last")
                        if args.require_gpu_name:
                            command.extend(
                                ["--require-gpu-name", args.require_gpu_name]
                            )
                        if args.triton_storage is not None:
                            command.extend(["--triton-storage", args.triton_storage])
                        command.extend(["--triton-fwd", args.triton_fwd])
                        command.extend(["--triton-bwd", args.triton_bwd])
                        env = os.environ.copy()
                        env["SJ_BENCH_SOURCE_ROOT"] = str(source_root)
                        env["PYTHONPATH"] = str(source_root)
                        env["TORCHINDUCTOR_CACHE_DIR"] = str(
                            output_dir / "cache" / stem
                        )
                        try:
                            _run_isolated_case(command, env, args.timeout)
                            record = json.loads(output.read_text(encoding="utf-8"))
                        except (
                            subprocess.CalledProcessError,
                            subprocess.TimeoutExpired,
                            OSError,
                        ) as error:
                            record = {
                                "source_label": label,
                                "round": round_index,
                                "status": "error",
                                "case": {
                                    "model": model,
                                    "phase": phase,
                                    "execution": execution,
                                    "T": args.T,
                                    "batch_size": batch_size,
                                    "image_size": args.image_size,
                                },
                                "error": {
                                    "type": type(error).__name__,
                                    "message": str(error),
                                    "stdout": str(getattr(error, "stdout", "") or "")[
                                        -4000:
                                    ],
                                    "stderr": str(getattr(error, "stderr", "") or "")[
                                        -4000:
                                    ],
                                },
                            }
                        records.append(record)

    baseline_label, candidate_label = (label for label, _ in sources)
    summary = aggregate_records(records, baseline_label, candidate_label)
    payload = {
        "schema_version": 1,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "sources": [{"label": label, "path": str(path)} for label, path in sources],
        "records": records,
        "comparison": summary,
    }
    (output_dir / "matrix.json").write_text(
        json.dumps(payload, indent=2), encoding="utf-8"
    )
    return payload


def _add_case_parser(subparsers) -> None:
    parser = subparsers.add_parser("case", help="run one isolated CUDA benchmark case")
    parser.add_argument("--model", choices=MODEL_NAMES, required=True)
    parser.add_argument("--phase", choices=PHASES, required=True)
    parser.add_argument("--execution", choices=EXECUTIONS, required=True)
    parser.add_argument("--batch-size", type=int, required=True)
    parser.add_argument("--warmup", type=int, required=True)
    parser.add_argument("--steps", type=int, required=True)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--neuron-backend", choices=("torch", "triton"), default="triton"
    )
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("auto", "delayed", "current", "block", "mxfp8"),
        default="auto",
    )
    parser.add_argument(
        "--fp8-fallback-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--triton-storage",
        choices=("fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"),
    )
    parser.add_argument(
        "--triton-fwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument(
        "--triton-bwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--momentum", type=float, default=0.9)
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--require-gpu-name", default="")
    parser.add_argument("--source-label", default="working-tree")
    parser.add_argument("--round", type=int, default=1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tensor-metadata", type=Path)
    parser.add_argument("--monitor-log", type=Path)
    parser.set_defaults(handler=run_case)


def _add_matrix_parser(subparsers) -> None:
    parser = subparsers.add_parser(
        "matrix", help="run source trees in alternating isolated processes"
    )
    parser.add_argument("--source", action="append", required=True, help="LABEL=PATH")
    parser.add_argument(
        "--models", nargs="+", choices=MODEL_NAMES, default=list(MODEL_NAMES)
    )
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument(
        "--executions", nargs="+", choices=EXECUTIONS, default=list(EXECUTIONS)
    )
    parser.add_argument("--rounds", type=int, default=3)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--inference-batch-size", type=int, default=64)
    parser.add_argument("--training-batch-size", type=int, default=32)
    parser.add_argument("--inference-warmup", type=int, default=100)
    parser.add_argument("--training-warmup", type=int, default=50)
    parser.add_argument("--inference-steps", type=int, default=500)
    parser.add_argument("--training-steps", type=int, default=100)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-steps", type=int, default=10)
    parser.add_argument("--T", type=int, default=DEFAULT_T)
    parser.add_argument("--image-size", type=int, default=DEFAULT_IMAGE_SIZE)
    parser.add_argument("--num-classes", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--device", type=int, default=0)
    parser.add_argument(
        "--neuron-backend", choices=("torch", "triton"), default="triton"
    )
    parser.add_argument(
        "--precision", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument(
        "--fp8-recipe",
        choices=("auto", "delayed", "current", "block", "mxfp8"),
        default="auto",
    )
    parser.add_argument(
        "--fp8-fallback-dtype",
        choices=("fp32", "fp16", "bf16"),
        default="bf16",
    )
    parser.add_argument(
        "--triton-storage",
        choices=("fp32", "fp16", "bf16", "float8_e4m3fn", "float8_e5m2"),
    )
    parser.add_argument(
        "--triton-fwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument(
        "--triton-bwd", choices=("fp32", "fp16", "bf16", "fp8"), default="fp32"
    )
    parser.add_argument("--channels-last", action="store_true")
    parser.add_argument("--require-gpu-name", default="")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.set_defaults(handler=run_matrix)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Single-GPU SNN benchmark runner")
    subparsers = parser.add_subparsers(dest="command", required=True)
    _add_case_parser(subparsers)
    _add_matrix_parser(subparsers)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if min(args.T, args.image_size) <= 0:
        raise ValueError("T and image size must be positive")
    if args.command == "case":
        if args.batch_size <= 0 or args.steps <= 0 or args.warmup < 0:
            raise ValueError(
                "batch size and steps must be positive; warmup must be non-negative"
            )
    else:
        positive = (
            args.rounds,
            args.inference_batch_size,
            args.training_batch_size,
            args.inference_steps,
            args.training_steps,
            args.profile_steps,
            args.timeout,
        )
        if min(positive) <= 0 or min(args.inference_warmup, args.training_warmup) < 0:
            raise ValueError("matrix counts must be positive and warmups non-negative")
    args.handler(args)


if __name__ == "__main__":
    main()
