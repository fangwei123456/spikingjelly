import json
import os
from pathlib import Path
import shutil
import subprocess

import pytest


@pytest.mark.skipif(shutil.which("jq") is None, reason="OCR guard requires jq")
@pytest.mark.parametrize(
    "status,cli_exit,expected",
    [
        ("complete", 0, 0),
        ("success", 0, 0),
        ("skipped", 0, 0),
        ("partial", 0, 1),
        ("error", 0, 1),
        ("complete", 2, 2),
    ],
)
def test_ocr_guard_zero_findings(tmp_path, status, cli_exit, expected):
    wrapper_dir = tmp_path / "guard"
    cli_dir = tmp_path / "cli"
    wrapper_dir.mkdir()
    cli_dir.mkdir()
    result_path = tmp_path / "result.json"
    result_path.write_text(json.dumps({"status": status, "comments": []}))
    source = Path(__file__).resolve().parents[1] / ".github/scripts/ocr"
    wrapper = wrapper_dir / "ocr"
    # Isolate the action's fixed output path without changing its CLI protocol.
    wrapper.write_text(
        source.read_text().replace("/tmp/ocr-result.json", str(result_path))
    )
    cli = cli_dir / "ocr"
    cli.write_text(f"#!/bin/sh\nexit {cli_exit}\n")
    cli.chmod(0o755)
    env = {**os.environ, "PATH": f"{wrapper_dir}:{cli_dir}:{os.environ['PATH']}"}
    result = subprocess.run(
        ["bash", str(wrapper), "review"], capture_output=True, text=True, env=env
    )
    assert result.returncode == expected, result.stderr
