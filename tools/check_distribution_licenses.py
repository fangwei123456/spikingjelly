"""Check built distributions against the repository's license materials."""

import argparse
from email.parser import BytesParser
import hashlib
from pathlib import Path
import tarfile
import tomllib
import zipfile


ROOT = Path(__file__).resolve().parents[1]
APACHE_SHA256 = "cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30"


def _check_distribution(path: Path, project: dict, notices: dict[str, bytes]) -> None:
    if path.suffix == ".whl":
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            assert len(names) == len(set(names)), f"{path}: duplicate archive entries"
            files = {
                name: archive.read(name) for name in names if not name.endswith("/")
            }
        metadata_names = [
            name
            for name in files
            if name.endswith(".dist-info/METADATA") and name.count("/") == 1
        ]
        assert len(metadata_names) == 1, f"{path}: expected one wheel METADATA"
        metadata_name = metadata_names[0]
        license_prefix = metadata_name.removesuffix("METADATA") + "licenses/"
    elif path.name.endswith(".tar.gz"):
        with tarfile.open(path) as archive:
            names = archive.getnames()
            assert len(names) == len(set(names)), f"{path}: duplicate archive entries"
            files = {
                member.name: archive.extractfile(member).read()
                for member in archive.getmembers()
                if member.isfile()
            }
        license_prefix = path.name.removesuffix(".tar.gz") + "/"
        metadata_name = license_prefix + "PKG-INFO"
    else:
        raise ValueError(f"Unsupported distribution: {path}")

    metadata = BytesParser().parsebytes(files[metadata_name])
    assert metadata["Name"] == project["name"], f"{path}: wrong project name"
    assert metadata["Version"] == project["version"], f"{path}: wrong version"
    assert tuple(map(int, metadata["Metadata-Version"].split("."))) >= (2, 4), (
        f"{path}: expected license-expression metadata support"
    )
    assert metadata.get_all("License-Expression") == [project["license"]], (
        f"{path}: incorrect License-Expression"
    )
    license_files = metadata.get_all("License-File", [])
    assert len(license_files) == len(set(license_files)), (
        f"{path}: duplicate License-File declarations"
    )
    assert set(license_files) == set(notices), f"{path}: incorrect License-File list"
    for name, content in notices.items():
        assert files.get(license_prefix + name) == content, (
            f"{path}: missing or changed license material: {name}"
        )
    assert not any("LICENSES/translations/" in name for name in names), (
        f"{path}: obsolete license translations included"
    )
    for obsolete in (
        b"SpikingJelly is distributed under the Open-Intelligence",
        b"The Chinese license is authoritative",
    ):
        assert obsolete not in files[metadata_name], (
            f"{path}: outdated license claim in package description"
        )
    print(f"{path}: license metadata and {len(notices)} notice files verified")


def _main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("distributions", nargs="+", type=Path)
    args = parser.parse_args()
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert (
        hashlib.sha256((ROOT / "LICENSE").read_bytes()).hexdigest() == APACHE_SHA256
    ), "Root LICENSE differs from the official Apache-2.0 text"
    notices = {name: (ROOT / name).read_bytes() for name in project["license-files"]}
    assert "LICENSE" in notices and "NOTICE" in notices, "Missing project notices"
    third_party = {
        path.relative_to(ROOT).as_posix()
        for path in (ROOT / "LICENSES" / "third_party").iterdir()
        if path.is_file()
    }
    assert third_party <= notices.keys(), "Unpackaged third-party license material"
    for path in args.distributions:
        _check_distribution(path, project, notices)


if __name__ == "__main__":
    _main()
