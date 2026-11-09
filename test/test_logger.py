from __future__ import annotations

import subprocess
import sys

import pytest

from spikingjelly.activation_based.op_counter.simple_energy import (
    SimpleEnergyProfiler,
)


@pytest.mark.parametrize("method", ["add", "remove", "enable", "disable", "configure"])
def test_logging_policy_rejects_library_logger_configuration(tmp_path, method):
    from tools.check_logging_policy import check

    root = tmp_path / "spikingjelly"
    root.mkdir()
    source = f'from .logger import logger\nlogger.{method}("example")\n'
    (root / "component.py").write_text(source)
    examples = root / "examples"
    examples.mkdir()
    (examples / "application.py").write_text(source)
    violations = check(root)
    assert len(violations) == 1
    assert f"logger.{method} is not allowed" in violations[0]


def test_logging_policy_allows_disable_only_in_package_bootstrap(tmp_path):
    from tools.check_logging_policy import check

    root = tmp_path / "spikingjelly"
    root.mkdir()
    bootstrap = root / "logger.py"
    source = 'from loguru import logger\nlogger.disable("spikingjelly")\n'
    bootstrap.write_text(source)
    assert check(root) == []

    bootstrap.write_text(source + "logger.configure(handlers=[])\n")
    violations = check(root)
    assert len(violations) == 1
    assert "logger.configure is not allowed" in violations[0]

    bootstrap.write_text('from loguru import logger\nlogger.disable("")\n')
    violations = check(root)
    assert len(violations) == 1
    assert "logger.disable is not allowed" in violations[0]

    operators = tmp_path / "ops"
    operators.mkdir()
    (operators / "logger.py").write_text(
        'from spikingjelly.logger import logger\nlogger.disable("spikingjelly")\n'
    )
    violations = check(operators)
    assert len(violations) == 1
    assert "logger.disable is not allowed" in violations[0]


def test_logging_policy_default_includes_private_operator_package(tmp_path):
    from pathlib import Path

    (tmp_path / "spikingjelly").mkdir()
    operators = tmp_path / "ops"
    operators.mkdir()
    (operators / "component.py").write_text(
        "from spikingjelly.logger import logger\nlogger.configure(handlers=[])\n"
    )
    script = Path(__file__).resolve().parents[1] / "tools/check_logging_policy.py"
    result = subprocess.run(
        [sys.executable, str(script)], cwd=tmp_path, capture_output=True, text=True
    )
    assert result.returncode == 1
    assert "ops/component.py" in result.stdout.replace("\\", "/")
    assert "logger.configure is not allowed" in result.stdout
    subprocess.run(
        [sys.executable, str(script), "--root", "spikingjelly"],
        cwd=tmp_path,
        check=True,
        capture_output=True,
    )


@pytest.mark.parametrize("root_kind", ["default_missing", "missing", "file"])
def test_logging_policy_rejects_invalid_scan_roots(tmp_path, root_kind):
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "tools/check_logging_policy.py"
    command = [sys.executable, str(script)]
    if root_kind == "default_missing":
        (tmp_path / "spikingjelly").mkdir()
        root = Path("ops")
    else:
        root = tmp_path / root_kind
        if root_kind == "file":
            root.write_text("pass\n")
        command.extend(["--root", str(root)])
    result = subprocess.run(command, cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 1
    assert str(root) in result.stdout
    assert "existing directory" in result.stdout
    assert "Logging policy passed." not in result.stdout


def test_import_is_silent_and_does_not_change_stdlib_root():
    script = """
import contextlib
import io
import logging
root = logging.getLogger()
before = (root.level, tuple(root.handlers))
stdout = io.StringIO()
stderr = io.StringIO()
with contextlib.redirect_stdout(stdout), contextlib.redirect_stderr(stderr):
    import spikingjelly
assert stdout.getvalue() == ""
assert stderr.getvalue() == ""
assert before == (root.level, tuple(root.handlers))
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_import_preserves_user_loguru_sink():
    script = """
import io
from loguru import logger
output = io.StringIO()
sink_id = logger.add(output, format="{message}")
import spikingjelly
logger.info("user-sink-still-active")
logger.remove(sink_id)
assert output.getvalue().strip() == "user-sink-still-active"
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_operator_binding_is_logged_once_only_when_enabled():
    script = """
import io
import torch
from loguru import logger
from spikingjelly.activation_based import functional
output = io.StringIO()
sink_id = logger.add(
    output,
    format="{message}",
    filter=lambda record: record["name"].startswith("spikingjelly"),
)
v = torch.zeros(2)
functional.if_multi_step(torch.ones(2, 2), v)
assert output.getvalue() == ""
logger.enable("spikingjelly")
for _ in range(3):
    functional.lif_multi_step(torch.ones(2, 2), v)
messages = output.getvalue().splitlines()
assert len(messages) == 1, messages
assert "operator=sj_lif" in messages[0]
assert "device=cpu" in messages[0]
assert "implementation=torch-reference" in messages[0]
logger.remove(sink_id)
"""
    subprocess.run([sys.executable, "-c", script], check=True)


def test_simple_energy_summary_is_emitted_once(loguru_records):
    import torch

    model = torch.nn.Linear(2, 2, bias=False)
    profiler = SimpleEnergyProfiler()
    profiler.bind_model(model)
    with profiler:
        model(torch.ones(1, 2))
    profiler.get_report()
    profiler.get_report()
    summaries = [
        record
        for record in loguru_records
        if record["message"].startswith("Counting completed:")
    ]
    assert len(summaries) == 1
    assert "total_operations=" in summaries[0]["message"]
    assert "memory_access_bytes=" in summaries[0]["message"]


def test_data_preprocess_summary_reports_completed_tasks(loguru_records):
    from spikingjelly.datasets.base import NeuromorphicDatasetBuilder

    processed = []
    NeuromorphicDatasetBuilder._run_preprocess_tasks(
        object(), processed.append, ((1,), (2,), (3,))
    )
    summaries = [
        record
        for record in loguru_records
        if record["message"].startswith("Dataset preprocessing completed:")
    ]
    assert sorted(processed) == [1, 2, 3]
    assert len(summaries) == 1
    assert "tasks=3" in summaries[0]["message"]


def test_precision_prepare_emits_one_lifecycle_summary(loguru_records):
    import torch

    from spikingjelly.activation_based.precision import prepare_model_for_precision

    prepare_model_for_precision(torch.nn.Linear(2, 2), "cpu", "fp32")
    summaries = [
        record
        for record in loguru_records
        if record["message"].startswith("Preparation completed:")
    ]
    assert len(summaries) == 1
