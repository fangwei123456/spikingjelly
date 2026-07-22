import copy
import json
import math
import subprocess
import sys
from pathlib import Path

import pytest
import torch

from benchmark.snn_llm import smoke


_REPO_ROOT = Path(__file__).resolve().parents[2]
_SMOKE_SCRIPT = _REPO_ROOT / "benchmark" / "snn_llm" / "smoke.py"

# ``smoke.py`` (benchmark/snn_llm/smoke.py:_collect_source) refuses to run
# unless ``cwd`` is a Git working tree or ``--source-revision`` is passed
# explicitly. The g2/g3 NFS-shared working tree at
# ``~/CodeRepo/spikingjelly-dev`` is intentionally *not* a Git working
# tree (ENV.md §3: ``remote_code_root`` is an independent copy, not a Git
# shared directory), so the four happy-path tests below would otherwise
# fail with ``RuntimeError: Unable to detect a Git revision; pass
# --source-revision explicitly.`` They therefore skip on non-Git remote
# working trees. The no-git error path is still covered by
# ``test_smoke_requires_explicit_revision_without_git``, which constructs
# a ``tmp_path / "no-git"`` cwd and asserts the refusal.
_REQUIRES_GIT_AT_REPO_ROOT = pytest.mark.skipif(
    not (_REPO_ROOT / ".git").exists(),
    reason=(
        "smoke.py requires a Git working tree at the repo root "
        "(benchmark/snn_llm/smoke.py:_collect_source). The g2/g3 "
        "NFS-shared remote working tree is intentionally non-Git "
        "(ENV.md §3). test_smoke_requires_explicit_revision_without_git "
        "still exercises the no-git error path via tmp_path."
    ),
)


def test_git_probe_has_a_timeout(monkeypatch, tmp_path):
    def fake_run(*args, **kwargs):
        assert kwargs["timeout"] == smoke.GIT_TIMEOUT_SECONDS
        return subprocess.CompletedProcess(args[0], 0, stdout="revision\n")

    monkeypatch.setattr(smoke.subprocess, "run", fake_run)

    assert smoke._run_git(tmp_path, "rev-parse", "HEAD") == "revision"


def _smoke_command(output_dir: Path, *extra_args: str, device: str = "cpu"):
    return [
        sys.executable,
        str(_SMOKE_SCRIPT),
        "--device",
        device,
        "--output-dir",
        str(output_dir),
        *extra_args,
    ]


def _run_smoke(
    output_dir: Path,
    *extra_args: str,
    cwd: Path = _REPO_ROOT,
    device: str = "cpu",
):
    return subprocess.run(
        _smoke_command(output_dir, *extra_args, device=device),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )


@_REQUIRES_GIT_AT_REPO_ROOT
def test_cpu_smoke_writes_manifest_contract(tmp_path: Path):
    output_dir = tmp_path / "cpu-smoke"

    completed = _run_smoke(output_dir)

    assert completed.returncode == 0, completed.stderr
    manifest_path = output_dir / "manifest.json"
    assert completed.stdout.strip() == f"manifest_path={manifest_path.resolve()}"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert set(manifest) == {
        "schema_version",
        "run",
        "source",
        "environment",
        "experiment",
        "metrics",
    }
    assert manifest["schema_version"] == 1


@_REQUIRES_GIT_AT_REPO_ROOT
def test_cpu_smoke_records_reproducible_workload_and_metrics(tmp_path: Path):
    output_dir = tmp_path / "cpu-smoke"

    completed = _run_smoke(output_dir)

    assert completed.returncode == 0, completed.stderr
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert set(manifest["run"]) == {
        "argv",
        "completed_at_utc",
        "id",
        "started_at_utc",
    }
    assert set(manifest["source"]) == {
        "declared_revision",
        "execution_files_sha256",
        "git",
        "locks",
    }
    assert manifest["source"]["locks"] == {
        "sources": {
            "spikegpt": {
                "repository": "https://github.com/ridgerchu/SpikeGPT.git",
                "revision": "029f86f0536f2b2451524038fc9890cc76c2429e",
            },
            "spikingjelly": {
                "repository": "https://github.com/fangwei123456/spikingjelly.git",
                "revision": "2797c5575515b59d7f09a3d5b732d6d25e148d13",
            },
        },
    }
    assert set(manifest["source"]["execution_files_sha256"]) == {
        "smoke.py",
        "sources.json",
    }
    assert all(
        len(digest) == 64
        for digest in manifest["source"]["execution_files_sha256"].values()
    )

    assert set(manifest["environment"]) == {
        "architecture",
        "device",
        "hostname",
        "os",
        "python_version",
        "torch_cuda_version",
        "torch_version",
    }
    assert manifest["environment"]["device"]["type"] == "cpu"
    assert manifest["environment"]["torch_cuda_version"] is None

    experiment = manifest["experiment"]
    assert experiment["kind"] == "smoke"
    assert experiment["seed"] == 20260713
    assert experiment["model"] == {
        "hidden_size": 16,
        "name": "tiny-dense-bigram-lm-v1",
        "vocabulary_size": 32,
    }
    assert experiment["data"] == {"name": "synthetic-token-v1", "split": "smoke"}
    assert experiment["tokenizer"] == {
        "name": "identity-tokenizer-v1",
        "vocabulary_size": 32,
    }
    assert experiment["batch_size"] == 2
    assert experiment["sequence_length"] == 16
    assert experiment["precision"] == {"amp": False, "dtype": "float32"}
    assert experiment["distributed"] == {"mode": "none", "world_size": 1}
    assert experiment["snn"] == {
        "T": None,
        "enabled": False,
        "reset_policy": "not-applicable",
    }

    metrics = manifest["metrics"]
    assert set(metrics) == {
        "gradient_l2_norm",
        "loss_after",
        "loss_before",
        "optimizer_step_latency_ms",
        "parameter_count",
        "peak_memory",
        "processed_tokens",
        "tokens_per_second",
    }
    for name, metric in metrics.items():
        if metric["availability"] == "available":
            assert math.isfinite(metric["value"]), name
    assert metrics["parameter_count"]["value"] == 1024
    assert metrics["processed_tokens"]["value"] == 30
    assert metrics["loss_after"]["value"] < metrics["loss_before"]["value"]
    assert metrics["gradient_l2_norm"]["value"] > 0
    assert metrics["peak_memory"]["availability"] == "unavailable"
    assert metrics["peak_memory"]["value"] is None
    assert metrics["peak_memory"]["reason"]
    for name in ("optimizer_step_latency_ms", "peak_memory", "tokens_per_second"):
        assert metrics[name]["scope"] == "smoke"


@pytest.mark.skipif(torch.cuda.is_available(), reason="requires a CPU-only runtime")
def test_smoke_rejects_unavailable_cuda_without_fallback(tmp_path: Path):
    output_dir = tmp_path / "cuda-smoke"

    completed = _run_smoke(output_dir, device="cuda")

    assert completed.returncode != 0
    assert "CUDA was requested" in completed.stderr
    assert not (output_dir / "manifest.json").exists()


@_REQUIRES_GIT_AT_REPO_ROOT
def test_smoke_refuses_to_overwrite_existing_manifest(tmp_path: Path):
    output_dir = tmp_path / "cpu-smoke"
    first = _run_smoke(output_dir)
    assert first.returncode == 0, first.stderr
    manifest_path = output_dir / "manifest.json"
    original = manifest_path.read_bytes()

    second = _run_smoke(output_dir)

    assert second.returncode != 0
    assert "Refusing to overwrite existing" in second.stderr
    assert manifest_path.read_bytes() == original


@_REQUIRES_GIT_AT_REPO_ROOT
def test_concurrent_smokes_publish_exactly_one_manifest(tmp_path: Path):
    output_dir = tmp_path / "concurrent-smoke"
    command = _smoke_command(output_dir)
    processes = [
        subprocess.Popen(
            command,
            cwd=_REPO_ROOT,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        for _ in range(2)
    ]

    results = [process.communicate(timeout=30) for process in processes]

    assert sum(process.returncode == 0 for process in processes) == 1
    failed_index = next(
        index for index, process in enumerate(processes) if process.returncode != 0
    )
    assert "Refusing to overwrite existing" in results[failed_index][1]
    manifest = json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
    assert manifest["schema_version"] == 1


def test_smoke_requires_explicit_revision_without_git(tmp_path: Path):
    no_git_cwd = tmp_path / "no-git"
    no_git_cwd.mkdir()
    failed_output = tmp_path / "missing-revision"

    failed = _run_smoke(failed_output, cwd=no_git_cwd)

    assert failed.returncode != 0
    assert "pass --source-revision explicitly" in failed.stderr
    assert not (failed_output / "manifest.json").exists()

    revision = "a" * 40
    successful_output = tmp_path / "explicit-revision"
    successful = _run_smoke(
        successful_output,
        "--source-revision",
        revision,
        cwd=no_git_cwd,
    )

    assert successful.returncode == 0, successful.stderr
    manifest = json.loads(
        (successful_output / "manifest.json").read_text(encoding="utf-8")
    )
    assert manifest["source"]["declared_revision"] == revision
    assert manifest["source"]["git"] == {
        "available": False,
        "diff_sha256": None,
        "dirty": None,
        "head": None,
    }
    assert set(manifest["source"]["execution_files_sha256"]) == {
        "smoke.py",
        "sources.json",
    }


@_REQUIRES_GIT_AT_REPO_ROOT
def test_cpu_smoke_is_deterministic_except_run_and_performance_data(tmp_path: Path):
    manifests = []
    for run_number in (1, 2):
        output_dir = tmp_path / f"cpu-smoke-{run_number}"
        completed = _run_smoke(output_dir)
        assert completed.returncode == 0, completed.stderr
        manifests.append(
            json.loads((output_dir / "manifest.json").read_text(encoding="utf-8"))
        )

    normalized = []
    for manifest in manifests:
        value = copy.deepcopy(manifest)
        value.pop("run")
        for metric_name in (
            "optimizer_step_latency_ms",
            "peak_memory",
            "tokens_per_second",
        ):
            value["metrics"].pop(metric_name)
        normalized.append(value)

    assert normalized[0] == normalized[1]
