import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import pytest


@pytest.mark.skipif(shutil.which("jq") is None, reason="OCR guard requires jq")
@pytest.mark.parametrize(
    "status,cli_exit,comments,expected,warning",
    [
        ("complete", 0, [], 0, False),
        ("success", 0, [], 0, False),
        ("skipped", 0, [], 0, False),
        ("partial", 0, [], 0, True),
        ("error", 0, [], 1, True),
        ("complete", 2, [], 2, False),
        ("complete", 0, [{"body": "finding"}], 0, False),
        ("partial", 1, [{"body": "finding"}], 0, True),
        ("failed", 124, [{"body": "finding"}], 0, True),
        ("failed", 124, [], 0, True),
        ("complete", 124, [], 0, True),
    ],
)
def test_ocr_guard_result(tmp_path, status, cli_exit, comments, expected, warning):
    wrapper_dir = tmp_path / "guard"
    cli_dir = tmp_path / "cli"
    wrapper_dir.mkdir()
    cli_dir.mkdir()
    result_path = tmp_path / "result.json"
    result_path.write_text(
        json.dumps(
            {
                "status": status,
                "comments": comments,
                "manifest": {"terminal_state": status},
            }
        )
    )
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper = wrapper_dir / "ocr"
    script = source.read_text()
    assert script.count("/tmp/ocr-result.json") == 1
    wrapper.write_text(script.replace("/tmp/ocr-result.json", str(result_path)))
    cli = cli_dir / "ocr"
    cli.write_text(f"#!/bin/sh\nexit {cli_exit}\n")
    cli.chmod(0o755)
    timeout = cli_dir / "timeout"
    timeout.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
    timeout.chmod(0o755)
    env = {**os.environ, "PATH": f"{wrapper_dir}:{cli_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(wrapper), "review"], capture_output=True, text=True, env=env
    )
    assert result.returncode == expected, result.stderr
    assert bool(result.stderr) == warning
    if cli_exit == 124:
        published = json.loads(result_path.read_text())
        assert published["manifest"]["terminal_state"] != "complete"
        assert published["status"] != "complete"


@pytest.mark.skipif(shutil.which("jq") is None, reason="OCR guard requires jq")
@pytest.mark.parametrize("comments", [[], [{"body": "confirmed finding"}]])
def test_ocr_guard_budget_publishes_results(tmp_path, comments):
    cli = tmp_path / "ocr"
    cli.write_text("#!/bin/sh\nexit 1\n")
    cli.chmod(0o755)
    timeout = tmp_path / "timeout"
    timeout.write_text('#!/bin/sh\nshift 4\nexec "$@"\n')
    timeout.chmod(0o755)
    result_path = tmp_path / "result.json"
    report = {
        "status": "failed",
        "comments": comments,
        "summary": {"budget_exceeded": True},
        "manifest": {"terminal_state": "failed"},
    }
    result_path.write_text(json.dumps(report))
    wrapper = tmp_path / "guard" / "ocr"
    wrapper.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper.write_text(
        source.read_text().replace("/tmp/ocr-result.json", str(result_path))
    )
    result = subprocess.run(
        ["bash", str(wrapper), "review"],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        timeout=5,
    )
    assert result.returncode == 0, result.stderr
    assert "incomplete" in result.stderr.lower()
    published = json.loads(result_path.read_text())
    assert published["status"] == "failed"
    assert published["comments"] == comments
    assert published["manifest"]["terminal_state"] == "failed"
    assert "incomplete" in published["message"].lower()
    assert published["warnings"]


@pytest.mark.skipif(shutil.which("jq") is None, reason="OCR guard requires jq")
def test_ocr_guard_timeout_flushes_findings(tmp_path):
    wrapper_dir = tmp_path / "guard"
    wrapper_dir.mkdir()
    result_path = tmp_path / "result.json"
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper = wrapper_dir / "ocr"
    wrapper.write_text(
        source.read_text().replace("/tmp/ocr-result.json", str(result_path))
    )
    report = {
        "status": "failed",
        "comments": [{"body": "confirmed finding"}],
        "manifest": {"terminal_state": "failed"},
    }
    cli = tmp_path / "ocr"
    cli.write_text(
        f"#!{sys.executable}\n"
        "import json, signal, sys\n"
        f"report = {report!r}\n"
        "def finish(*_args):\n"
        "    print(json.dumps(report), flush=True)\n"
        "    sys.exit(1)\n"
        "signal.signal(signal.SIGTERM, finish)\n"
        "print('ready', file=sys.stderr, flush=True)\n"
        "signal.pause()\n"
    )
    cli.chmod(0o755)
    timeout = tmp_path / "timeout"
    timeout.write_text(
        f"#!{sys.executable}\n"
        "import signal, subprocess, sys\n"
        "assert sys.argv[1:5] == ['--foreground', '--signal=TERM', '--kill-after=60s', '35m']\n"
        "proc = subprocess.Popen(sys.argv[5:], stderr=subprocess.PIPE, text=True)\n"
        "assert proc.stderr.readline().strip() == 'ready'\n"
        "proc.send_signal(signal.SIGTERM)\n"
        "proc.communicate(timeout=5)\n"
        "sys.exit(124)\n"
    )
    timeout.chmod(0o755)
    with result_path.open("w") as output:
        result = subprocess.run(
            ["bash", str(wrapper), "review"],
            stdout=output,
            stderr=subprocess.PIPE,
            text=True,
            env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
            timeout=10,
        )
    assert result.returncode == 0, result.stderr
    published = json.loads(result_path.read_text())
    assert published["comments"] == report["comments"]
    assert published["manifest"]["terminal_state"] != "complete"
    assert "incomplete" in published["message"].lower()
    assert "time" in published["message"].lower()


def test_ocr_guard_non_review_passthrough(tmp_path):
    wrapper = tmp_path / "guard" / "ocr"
    wrapper.parent.mkdir()
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper.write_text(source.read_text())
    cli = tmp_path / "ocr"
    cli.write_text('#!/bin/sh\nprintf "%s\\n" "$@"\nexit 2\n')
    cli.chmod(0o755)
    result = subprocess.run(
        ["bash", str(wrapper), "config", "set", "llm.extra_body", '{"key": "value"}'],
        capture_output=True,
        text=True,
        env={**os.environ, "PATH": f"{tmp_path}:{os.environ['PATH']}"},
        timeout=5,
    )
    assert result.returncode == 2
    assert result.stdout.splitlines() == [
        "config",
        "set",
        "llm.extra_body",
        '{"key": "value"}',
    ]
    assert not result.stderr
