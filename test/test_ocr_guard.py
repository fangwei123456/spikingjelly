import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("jq") is None, reason="OCR guard requires jq")
@pytest.mark.parametrize(
    "status,cli_exit,comments,expected,warning",
    [
        ("complete", 0, [], 0, False),
        ("success", 0, [], 0, False),
        ("skipped", 0, [], 0, False),
        ("partial", 0, [], 1, True),
        ("error", 0, [], 1, True),
        ("complete", 2, [], 2, False),
        ("complete", 0, [{"body": "finding"}], 0, False),
        ("partial", 1, [{"body": "finding"}], 0, True),
    ],
)
def test_ocr_guard_result(tmp_path, status, cli_exit, comments, expected, warning):
    wrapper_dir = tmp_path / "guard"
    cli_dir = tmp_path / "cli"
    wrapper_dir.mkdir()
    cli_dir.mkdir()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"status": status, "comments": comments}))
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper = wrapper_dir / "ocr"
    script = source.read_text()
    assert script.count("/tmp/ocr-result.json") == 1
    wrapper.write_text(script.replace("/tmp/ocr-result.json", str(result_path)))
    cli = cli_dir / "ocr"
    cli.write_text(f"#!/bin/sh\nexit {cli_exit}\n")
    cli.chmod(0o755)
    env = {**os.environ, "PATH": f"{wrapper_dir}:{cli_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(wrapper), "review"], capture_output=True, text=True, env=env
    )
    assert result.returncode == expected, result.stderr
    assert bool(result.stderr) == warning
