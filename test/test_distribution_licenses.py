import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

import pytest


ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize("optimization", ["normal", "-O", "PYTHONOPTIMIZE"])
@pytest.mark.parametrize(
    ("fault", "error"),
    [
        (None, None),
        ("root-license", "Root LICENSE differs"),
        ("expression", "incorrect License-Expression"),
        ("notice", "missing or changed license material: LICENSES/NOTICE"),
    ],
)
def test_distribution_license_cli(tmp_path, optimization, fault, error):
    (tmp_path / "tools").mkdir()
    (tmp_path / "LICENSES" / "third_party").mkdir(parents=True)
    script = tmp_path / "tools" / "check_distribution_licenses.py"
    shutil.copyfile(ROOT / "tools" / script.name, script)
    license_text = (ROOT / "LICENSE").read_bytes()
    (tmp_path / "LICENSE").write_bytes(license_text)
    (tmp_path / "LICENSES" / "NOTICE").write_bytes(b"Test attribution\n")
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "spikingjelly"\nversion = "2.0.0"\n'
        'license = "Apache-2.0"\nlicense-files = ["LICENSE", "LICENSES/NOTICE"]\n'
    )
    expression = "MIT" if fault == "expression" else "Apache-2.0"
    metadata = (
        "Metadata-Version: 2.4\nName: spikingjelly\nVersion: 2.0.0\n"
        f"License-Expression: {expression}\n"
        "License-File: LICENSE\nLicense-File: LICENSES/NOTICE\n\n"
    )
    wheel = tmp_path / "spikingjelly-2.0.0-py3-none-any.whl"
    prefix = "spikingjelly-2.0.0.dist-info/"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr(prefix + "METADATA", metadata)
        archive.writestr(prefix + "licenses/LICENSE", license_text)
        if fault != "notice":
            archive.writestr(prefix + "licenses/LICENSES/NOTICE", b"Test attribution\n")
    if fault == "root-license":
        (tmp_path / "LICENSE").write_bytes(b"Truncated Apache license\n")

    env = dict(os.environ, PYTHONOPTIMIZE="0")
    command = [sys.executable]
    if optimization == "-O":
        command.append("-O")
    elif optimization == "PYTHONOPTIMIZE":
        env["PYTHONOPTIMIZE"] = "1"
    result = subprocess.run(
        [*command, str(script), str(wheel)],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=30,
    )
    if fault is None:
        assert result.returncode == 0, result.stderr
        assert "verified" in result.stdout
    else:
        assert result.returncode != 0, "Invalid distribution was accepted"
        assert error in result.stderr
        assert "verified" not in result.stdout
