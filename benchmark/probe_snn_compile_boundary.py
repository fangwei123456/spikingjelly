from __future__ import annotations

import argparse
import importlib.util
import json
import os
import re
import subprocess
import sys
from pathlib import Path
from typing import Any
from uuid import uuid4

import torch

CASES = (
    "torch_lif",
    "custom_lif",
    "triton_lif",
    "torch_flexsn",
    "triton_flexsn",
    "opaque_flexsn",
)
PHASES = ("inference", "training")


def _registration_env(case: str) -> dict[str, str]:
    return {
        "SJ_USE_TRITON_OP": ("1" if case in {"triton_lif", "triton_flexsn"} else "0"),
        "SJ_USE_WRAP_TRITON": (
            "1" if case in {"custom_lif", "triton_lif", "triton_flexsn"} else "0"
        ),
    }


def _graph_break_count(explain_output: Any) -> int:
    if hasattr(explain_output, "graph_break_count"):
        return int(explain_output.graph_break_count)
    if hasattr(explain_output, "break_reasons"):
        return len(explain_output.break_reasons)
    raise TypeError(f"unsupported explain output: {type(explain_output).__name__}")


def _graph_count(explain_output: Any) -> int:
    if hasattr(explain_output, "graph_count"):
        return int(explain_output.graph_count)
    graphs = getattr(explain_output, "graphs", None)
    if graphs is not None:
        return len(graphs)
    raise TypeError(f"unsupported explain output: {type(explain_output).__name__}")


def _unsupported(
    case: str, phase: str, reason_code: str, reason: str
) -> dict[str, Any]:
    return {
        "case": case,
        "phase": phase,
        "status": "unsupported",
        "reason_code": reason_code,
        "reason": reason,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "graph_count": None,
        "graph_break_count": None,
        "recompile_count": None,
        "generated_code_paths": [],
        "allocation_count": None,
        "kernel_launch_count": None,
        "physical_metrics_source": None,
        "output_close": None,
        "gradient_close": None,
    }


def _prerequisite_failure(case: str, phase: str) -> dict[str, Any] | None:
    if not hasattr(torch, "compile"):
        return _unsupported(
            case,
            phase,
            "torch_compile_unavailable",
            "torch.compile and torch._dynamo are required",
        )
    try:
        importlib.import_module("torch._dynamo")
    except ImportError:
        return _unsupported(
            case,
            phase,
            "torch_compile_unavailable",
            "torch.compile and torch._dynamo are required",
        )
    if case in {
        "custom_lif",
        "triton_lif",
        "triton_flexsn",
        "opaque_flexsn",
    }:
        if not torch.cuda.is_available():
            return _unsupported(
                case, phase, "cuda_required", "the Triton-backed boundary requires CUDA"
            )
        if importlib.util.find_spec("triton") is None:
            return _unsupported(
                case, phase, "triton_required", "the Triton package is not importable"
            )
    registration_env = _registration_env(case)
    if registration_env["SJ_USE_TRITON_OP"] == "1" and not hasattr(
        torch.library, "triton_op"
    ):
        return _unsupported(
            case,
            phase,
            "triton_op_unavailable",
            "torch.library.triton_op is required",
        )
    if registration_env["SJ_USE_WRAP_TRITON"] == "1" and not hasattr(
        torch.library, "wrap_triton"
    ):
        return _unsupported(
            case,
            phase,
            "wrap_triton_unavailable",
            "torch.library.wrap_triton is required",
        )
    return None


def _lif_core(x: torch.Tensor, v: torch.Tensor):
    h = v + (x - v) / 2.0
    output = torch.tanh(h)
    return output, h - output


def _build_model(case: str, phase: str, device: torch.device) -> torch.nn.Module:
    from spikingjelly.activation_based import neuron, surrogate

    if case.endswith("lif") and "flexsn" not in case:
        backend = "torch" if case == "torch_lif" else "triton"
        node = neuron.LIFNode(
            tau=2.0,
            decay_input=True,
            v_threshold=1.0,
            v_reset=0.0,
            surrogate_function=surrogate.Sigmoid(alpha=4.0),
            detach_reset=False,
            step_mode="m",
            backend=backend,
        )
    else:
        from spikingjelly.activation_based.neuron.flexsn import FlexSN

        backend = "torch" if case == "torch_flexsn" else "triton"
        example = torch.zeros(2, 16, device=device)
        node = FlexSN(
            core=_lif_core,
            num_inputs=1,
            num_states=1,
            num_outputs=1,
            example_inputs=(example, example),
            example_outputs=(example,),
            requires_grad=(True, True),
            step_mode="m",
            backend=backend,
        )
        if backend == "triton":
            if phase == "inference" and not (
                (
                    node._triton_scan_final_state_kernel is not None
                    and node._triton_scan_final_state_info is not None
                )
                or (
                    node._triton_scan_kernel is not None
                    and node._triton_scan_info is not None
                )
            ):
                raise RuntimeError("FlexSN Triton inference kernel construction failed")
            if phase == "training" and not (
                node._triton_fwd_kernel is not None
                and node._triton_bwd_kernel is not None
                and node._triton_train_info is not None
            ):
                raise RuntimeError("FlexSN Triton training kernel construction failed")

    class PointwiseNode(torch.nn.Module):
        def __init__(self, inner: torch.nn.Module):
            super().__init__()
            self.inner = inner

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            return (self.inner(x * 1.1) + 0.2) * 0.9

    return PointwiseNode(node).to(device)


def _run_once(model: torch.nn.Module, x: torch.Tensor, phase: str):
    from spikingjelly.activation_based import functional

    functional.reset_net(model)
    if phase == "inference":
        with torch.inference_mode():
            return model(x), None
    x.grad = None
    output = model(x)
    output.sum().backward()
    return output.detach(), x.grad.detach().clone()


def _generated_code(cache_dir: Path) -> tuple[list[str], int, int]:
    """Count same-line Inductor launcher calls; profiler counts are authoritative."""
    paths = sorted(cache_dir.rglob("output_code.py"))
    allocation_count = 0
    launch_count = 0
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="replace")
        allocation_count += text.count("empty_strided_") + text.count(
            "alloc_from_pool("
        )
        launch_count += len(
            re.findall(
                r"^\s*(?:\w+\s*=\s*)?(?:(?:triton_|cpp_)\w+|\w*kernel\w*)\.run\(",
                text,
                re.MULTILINE,
            )
        )
        launch_count += len(
            re.findall(
                r"^\s*(?:\w+\s*=\s*)?extern_kernels\.\w+\(",
                text,
                re.MULTILINE,
            )
        )
    return [str(path) for path in paths], allocation_count, launch_count


def _profile_physical_step(model, x, phase: str) -> tuple[int | None, int | None]:
    if x.device.type != "cuda":
        return None, None
    activities = [
        torch.profiler.ProfilerActivity.CPU,
        torch.profiler.ProfilerActivity.CUDA,
    ]
    with torch.profiler.profile(activities=activities, profile_memory=True) as profile:
        _run_once(model, x, phase)
    kernel_events = [
        event
        for event in profile.events()
        if event.device_type == torch.autograd.DeviceType.CUDA
        and not event.name.startswith(("Memcpy", "Memset"))
        and event.name != "[memory]"
    ]
    allocations = [
        event
        for event in profile.events()
        if event.name == "[memory]"
        and getattr(
            event,
            "device_memory_usage",
            getattr(event, "cuda_memory_usage", 0),
        )
        > 0
    ]
    return len(allocations), len(kernel_events)


def _clear_dynamo_state() -> None:
    torch._dynamo.reset()
    counters = getattr(getattr(torch._dynamo, "utils", None), "counters", None)
    if counters is not None:
        counters.clear()


def _dynamo_counts() -> dict[str, int]:
    counters = getattr(getattr(torch._dynamo, "utils", None), "counters", {})
    return {
        "graph_count": int(counters.get("stats", {}).get("unique_graphs", 0)),
        "graph_break_count": int(sum(counters.get("graph_break", {}).values())),
        "recompile_count": int(sum(counters.get("recompiles", {}).values())),
    }


def run_child(args: argparse.Namespace) -> dict[str, Any]:
    prerequisite = _prerequisite_failure(args.case, args.phase)
    if prerequisite is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(prerequisite, indent=2), encoding="utf-8")
        return prerequisite

    source_root = Path(
        os.environ.get("SJ_BENCH_SOURCE_ROOT", Path(__file__).parents[1])
    ).resolve()
    sys.path.insert(0, str(source_root))
    import spikingjelly

    package_file = Path(spikingjelly.__file__).resolve()
    try:
        package_file.relative_to(source_root)
    except ValueError as error:
        raise RuntimeError(
            f"imported spikingjelly from {package_file}, outside source root {source_root}"
        ) from error
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    x_template = torch.randn(4, 2, 16, device=device)

    explain_model = _build_model(args.case, args.phase, device).train(
        args.phase == "training"
    )
    explain_input = x_template.clone().requires_grad_(args.phase == "training")
    _clear_dynamo_state()
    try:
        explain_output = torch._dynamo.explain(explain_model)(explain_input)
        graph_count = _graph_count(explain_output)
        graph_break_count = _graph_break_count(explain_output)
    except Exception as error:
        result = _unsupported(
            args.case,
            args.phase,
            "dynamo_explain_failed",
            f"{type(error).__name__}: {error}",
        )
        result["status"] = "error"
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
        return result

    eager_model = _build_model(args.case, args.phase, device).train(
        args.phase == "training"
    )
    compiled_source = _build_model(args.case, args.phase, device).train(
        args.phase == "training"
    )
    compiled_source.load_state_dict(eager_model.state_dict(), strict=True)
    eager_x = x_template.clone().requires_grad_(args.phase == "training")
    compiled_x = x_template.clone().requires_grad_(args.phase == "training")
    eager_output, eager_grad = _run_once(eager_model, eager_x, args.phase)

    _clear_dynamo_state()
    compiled_model = torch.compile(
        compiled_source,
        backend="inductor",
        fullgraph=True,
        options={"triton.cudagraphs": False, "triton.cudagraph_trees": False},
    )
    try:
        compiled_output, compiled_grad = _run_once(
            compiled_model, compiled_x, args.phase
        )
        compiled_output_2, _ = _run_once(compiled_model, compiled_x, args.phase)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        allocation_count, kernel_launch_count = _profile_physical_step(
            compiled_model, compiled_x, args.phase
        )
        generated_paths, generated_allocations, generated_launches = _generated_code(
            Path(os.environ["TORCHINDUCTOR_CACHE_DIR"])
        )
        compile_counts = _dynamo_counts()
        result = {
            "case": args.case,
            "phase": args.phase,
            "status": "ok",
            "reason_code": None,
            "reason": None,
            "device": str(device),
            "spikingjelly_file": str(package_file),
            "registration_environment": {
                key: os.environ.get(key) for key in _registration_env(args.case)
            },
            "graph_count": compile_counts["graph_count"],
            "graph_break_count": compile_counts["graph_break_count"],
            "recompile_count": compile_counts["recompile_count"],
            "explain_graph_count": graph_count,
            "explain_graph_break_count": graph_break_count,
            "generated_code_paths": generated_paths,
            "generated_allocation_count": generated_allocations,
            "generated_launch_count": generated_launches,
            "allocation_count": allocation_count,
            "kernel_launch_count": kernel_launch_count,
            "physical_metrics_source": "torch.profiler"
            if device.type == "cuda"
            else None,
            "output_close": torch.allclose(
                eager_output, compiled_output, rtol=1e-4, atol=1e-5
            ),
            "second_output_close": torch.allclose(
                eager_output, compiled_output_2, rtol=1e-4, atol=1e-5
            ),
            "gradient_close": (
                torch.allclose(eager_grad, compiled_grad, rtol=1e-3, atol=1e-4)
                if eager_grad is not None and compiled_grad is not None
                else None
            ),
        }
    except Exception as error:
        result = _unsupported(
            args.case,
            args.phase,
            "fullgraph_compile_failed",
            f"{type(error).__name__}: {error}",
        )
        result.update(
            {
                "status": "error",
                "explain_graph_count": graph_count,
                "explain_graph_break_count": graph_break_count,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return result


def run_parent(args: argparse.Namespace) -> dict[str, Any]:
    output = args.output.resolve()
    work_dir = output.parent / f"{output.stem}-artifacts-{uuid4().hex[:8]}"
    work_dir.mkdir(parents=True, exist_ok=True)
    results = []
    for case in args.cases:
        for phase in args.phases:
            child_output = work_dir / f"{case}-{phase}.json"
            cache_dir = work_dir / "cache" / f"{case}-{phase}"
            env = os.environ.copy()
            env.update(_registration_env(case))
            env["TORCHINDUCTOR_CACHE_DIR"] = str(cache_dir)
            env["TORCH_COMPILE_DEBUG"] = "1"
            command = [
                sys.executable,
                str(Path(__file__).resolve()),
                "--child",
                "--case",
                case,
                "--phase",
                phase,
                "--seed",
                str(args.seed),
                "--output",
                str(child_output),
            ]
            try:
                completed = subprocess.run(
                    command,
                    env=env,
                    capture_output=True,
                    text=True,
                    timeout=args.timeout,
                )
            except subprocess.TimeoutExpired:
                result = _unsupported(
                    case,
                    phase,
                    "child_process_timeout",
                    f"child exceeded timeout of {args.timeout} seconds",
                )
                result["status"] = "error"
                result["registration_environment"] = _registration_env(case)
                results.append(result)
                print(json.dumps(result), flush=True)
                continue
            stderr_log = None
            if completed.stderr:
                stderr_log = work_dir / f"{case}-{phase}.stderr"
                stderr_log.write_text(
                    completed.stderr, encoding="utf-8", errors="replace"
                )
            if child_output.exists():
                result = json.loads(child_output.read_text(encoding="utf-8"))
            else:
                reason = completed.stderr[-4000:]
                if stderr_log is not None:
                    reason = f"{reason}\nfull stderr: {stderr_log}"
                result = _unsupported(case, phase, "child_process_failed", reason)
                result["status"] = "error"
                result["returncode"] = completed.returncode
            if stderr_log is not None:
                result["stderr_path"] = str(stderr_log)
            result.setdefault("registration_environment", _registration_env(case))
            results.append(result)
            print(json.dumps(result), flush=True)

    payload = {
        "schema_version": 1,
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "results": results,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Probe SNN torch.compile operator boundaries"
    )
    parser.add_argument("--child", action="store_true", help=argparse.SUPPRESS)
    parser.add_argument("--case", choices=CASES)
    parser.add_argument("--phase", choices=PHASES)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--phases", nargs="+", choices=PHASES, default=list(PHASES))
    parser.add_argument("--seed", type=int, default=20260808)
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--output", type=Path, required=True)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.child:
        if args.case is None or args.phase is None:
            raise ValueError("--child requires --case and --phase")
        result = run_child(args)
        print("JSON_RESULT=" + json.dumps(result), flush=True)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
