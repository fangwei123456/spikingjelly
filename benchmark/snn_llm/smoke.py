import argparse
import datetime as dt
import hashlib
import json
import math
import os
import platform
import socket
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Optional

import torch
from torch import nn


SCHEMA_VERSION = 1
SEED = 20260713
VOCAB_SIZE = 32
HIDDEN_SIZE = 16
BATCH_SIZE = 2
SEQUENCE_LENGTH = 16
LEARNING_RATE = 1e-2
GIT_TIMEOUT_SECONDS = 30

_SCRIPT_DIR = Path(__file__).resolve().parent
_SOURCES_PATH = _SCRIPT_DIR / "sources.json"
_EXECUTION_FILES = (_SOURCES_PATH, Path(__file__).resolve())


class _TinyDenseBigramLM(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.embedding = nn.Embedding(VOCAB_SIZE, HIDDEN_SIZE)
        self.head = nn.Linear(HIDDEN_SIZE, VOCAB_SIZE, bias=False)

    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        return self.head(self.embedding(tokens))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the deterministic SNN-LLM phase-0 smoke workload."
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--source-revision")
    return parser.parse_args()


def _run_git(repo_root: Path, *args: str) -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", *args],
            cwd=repo_root,
            capture_output=True,
            text=True,
            check=False,
            timeout=GIT_TIMEOUT_SECONDS,
        )
    except (subprocess.TimeoutExpired, OSError):
        return None
    if completed.returncode != 0:
        return None
    return completed.stdout.strip()


def _sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _collect_source(repo_root: Path, requested_revision: Optional[str]) -> dict:
    git_head = _run_git(repo_root, "rev-parse", "HEAD")
    declared_revision = requested_revision or git_head
    if declared_revision is None:
        raise RuntimeError(
            "Unable to detect a Git revision; pass --source-revision explicitly."
        )

    git_status = _run_git(repo_root, "status", "--porcelain=v1")
    git_diff = _run_git(repo_root, "diff", "--binary", "HEAD")
    locks = json.loads(_SOURCES_PATH.read_text(encoding="utf-8"))
    execution_files = {
        path.relative_to(_SCRIPT_DIR).as_posix(): _sha256_bytes(path.read_bytes())
        for path in _EXECUTION_FILES
    }
    return {
        "declared_revision": declared_revision,
        "execution_files_sha256": execution_files,
        "git": {
            "available": git_head is not None,
            "diff_sha256": _sha256_bytes((git_diff or "").encode("utf-8"))
            if git_head is not None
            else None,
            "dirty": bool(git_status) if git_head is not None else None,
            "head": git_head,
        },
        "locks": locks,
    }


def _resolve_device(name: str) -> torch.device:
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested, but torch.cuda.is_available() is false."
        )
    return torch.device(name)


def _synchronize(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def _collect_environment(device: torch.device) -> dict:
    if device.type == "cuda":
        properties = torch.cuda.get_device_properties(device)
        device_details = {
            "cuda_capability": list(torch.cuda.get_device_capability(device)),
            "name": torch.cuda.get_device_name(device),
            "total_memory_bytes": properties.total_memory,
            "type": "cuda",
        }
    else:
        device_details = {
            "cuda_capability": None,
            "name": platform.processor() or platform.machine(),
            "total_memory_bytes": None,
            "type": "cpu",
        }
    return {
        "architecture": platform.machine(),
        "device": device_details,
        "hostname": socket.gethostname(),
        "os": platform.platform(),
        "python_version": platform.python_version(),
        "torch_cuda_version": torch.version.cuda,
        "torch_version": torch.__version__,
    }


def _metric(
    value,
    unit: str,
    kind: str = "measured",
    scope: Optional[str] = None,
) -> dict:
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Metric values must be finite.")
    metric = {
        "availability": "available",
        "kind": kind,
        "unit": unit,
        "value": value,
    }
    if scope is not None:
        metric["scope"] = scope
    return metric


def _run_workload(device: torch.device) -> tuple[dict, dict]:
    torch.manual_seed(SEED)
    model = _TinyDenseBigramLM()
    tokens = torch.randint(0, VOCAB_SIZE, (BATCH_SIZE, SEQUENCE_LENGTH))
    model = model.to(device)
    tokens = tokens.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE)

    if device.type == "cuda":
        torch.cuda.manual_seed_all(SEED)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.cuda.reset_peak_memory_stats(device)

    inputs = tokens[:, :-1]
    targets = tokens[:, 1:]
    processed_tokens = targets.numel()

    _synchronize(device)
    step_started = time.perf_counter()
    optimizer.zero_grad(set_to_none=True)
    logits = model(inputs)
    loss_before = nn.functional.cross_entropy(
        logits.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
    )
    loss_before.backward()
    gradient_l2_norm = math.sqrt(
        sum(
            parameter.grad.detach().float().square().sum().item()
            for parameter in model.parameters()
            if parameter.grad is not None
        )
    )
    _synchronize(device)
    optimizer_started = time.perf_counter()
    optimizer.step()
    _synchronize(device)
    optimizer_step_ms = (time.perf_counter() - optimizer_started) * 1000.0
    training_step_seconds = time.perf_counter() - step_started

    with torch.no_grad():
        logits_after = model(inputs)
        loss_after = nn.functional.cross_entropy(
            logits_after.reshape(-1, VOCAB_SIZE), targets.reshape(-1)
        )

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    if device.type == "cuda":
        peak_memory = _metric(
            torch.cuda.max_memory_allocated(device) / (1024 * 1024),
            "MiB",
            scope="smoke",
        )
    else:
        peak_memory = {
            "availability": "unavailable",
            "kind": "measured",
            "reason": "CPU process RSS collection is outside this smoke's scope.",
            "scope": "smoke",
            "unit": "MiB",
            "value": None,
        }

    experiment = {
        "batch_size": BATCH_SIZE,
        "data": {"name": "synthetic-token-v1", "split": "smoke"},
        "distributed": {"mode": "none", "world_size": 1},
        "kind": "smoke",
        "model": {
            "hidden_size": HIDDEN_SIZE,
            "name": "tiny-dense-bigram-lm-v1",
            "vocabulary_size": VOCAB_SIZE,
        },
        "optimizer": {"learning_rate": LEARNING_RATE, "name": "SGD", "steps": 1},
        "precision": {"amp": False, "dtype": "float32"},
        "seed": SEED,
        "sequence_length": SEQUENCE_LENGTH,
        "snn": {"T": None, "enabled": False, "reset_policy": "not-applicable"},
        "tokenizer": {
            "name": "identity-tokenizer-v1",
            "vocabulary_size": VOCAB_SIZE,
        },
    }
    metrics = {
        "gradient_l2_norm": _metric(gradient_l2_norm, "L2 norm"),
        "loss_after": _metric(loss_after.item(), "cross_entropy"),
        "loss_before": _metric(loss_before.item(), "cross_entropy"),
        "optimizer_step_latency_ms": _metric(optimizer_step_ms, "ms", scope="smoke"),
        "parameter_count": _metric(parameter_count, "parameters", "derived"),
        "peak_memory": peak_memory,
        "processed_tokens": _metric(processed_tokens, "tokens", "derived"),
        "tokens_per_second": _metric(
            processed_tokens / training_step_seconds,
            "tokens/s",
            "derived",
            scope="smoke",
        ),
    }
    return experiment, metrics


def _write_manifest(output_dir: Path, manifest: dict) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {manifest_path}.")
    temporary_path = output_dir / f".{manifest_path.name}.{uuid.uuid4().hex}.tmp"
    payload = json.dumps(
        manifest,
        allow_nan=False,
        indent=2,
        sort_keys=True,
    )
    try:
        temporary_path.write_text(payload + "\n", encoding="utf-8")
        try:
            os.link(temporary_path, manifest_path)
        except FileExistsError as exc:
            raise FileExistsError(
                f"Refusing to overwrite existing {manifest_path}."
            ) from exc
    finally:
        if temporary_path.exists():
            temporary_path.unlink()
    return manifest_path.resolve()


def main() -> None:
    args = _parse_args()
    output_dir = args.output_dir.resolve()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        raise FileExistsError(f"Refusing to overwrite existing {manifest_path}.")

    started_at = dt.datetime.now(dt.timezone.utc)
    device = _resolve_device(args.device)
    source = _collect_source(Path.cwd(), args.source_revision)
    environment = _collect_environment(device)
    experiment, metrics = _run_workload(device)
    completed_at = dt.datetime.now(dt.timezone.utc)
    manifest = {
        "environment": environment,
        "experiment": experiment,
        "metrics": metrics,
        "run": {
            "argv": sys.argv,
            "completed_at_utc": completed_at.isoformat(),
            "id": str(uuid.uuid4()),
            "started_at_utc": started_at.isoformat(),
        },
        "schema_version": SCHEMA_VERSION,
        "source": source,
    }
    written_path = _write_manifest(output_dir, manifest)
    print(f"manifest_path={written_path}")


if __name__ == "__main__":
    main()
