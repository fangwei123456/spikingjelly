"""Compare paired vision measurements and archive release evidence."""

import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import statistics
import subprocess
import tempfile
from urllib.parse import unquote, urlsplit
import zipfile


def _hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _read(path):
    return json.loads(
        Path(path).read_text(),
        parse_constant=lambda x: (_ for _ in ()).throw(
            ValueError(f"Non-finite JSON: {x}")
        ),
    )


def _write(output, report):
    output.mkdir(parents=True, exist_ok=True)
    with (output / "report.json").open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    lines = ["# Release validation", "", f"Status: **{report['status']}**", ""]
    if "reason" in report:
        lines.append(report["reason"])
    for row in report.get("cases", []):
        lines.append(f"- {row['id']}: {row['status']} — {row.get('reason', '')}")
    if "ratios" in report:
        lines += [
            f"Candidate/baseline ratios: {report['ratios']}",
            f"Median ratio: {report['median_ratio']:.6f}",
        ]
    with (output / "REPORT.md").open("x") as stream:
        stream.write("\n".join(lines) + "\n")


def _compare(
    candidate, baseline, metric="throughput", tolerance=0.02, development=False
):
    if len(candidate) != len(baseline) or len(candidate) < 3:
        raise ValueError("At least three matching independent pairs are required")
    if len(set(map(str, candidate + baseline))) != len(candidate) + len(baseline):
        raise ValueError("Each pair must use distinct process reports")
    import torch

    ratios, rows, signature = [], [], None
    for index, (a_path, b_path) in enumerate(zip(candidate, baseline)):
        a, b = _read(a_path), _read(b_path)
        for report in (a, b):
            if (
                not report.get("complete_dataset")
                or report.get("configuration", {}).get("mode") != "benchmark"
            ):
                raise ValueError(
                    "Comparison requires complete real-data benchmark reports"
                )
            if report.get("runner_worktree_dirty") and not development:
                raise ValueError("A benchmark runner has uncommitted changes")
        config_keys = (
            "case",
            "recipe",
            "time_steps",
            "batch_size",
            "seed",
            "phase",
            "amp",
            "warmup",
            "steps",
        )
        env_keys = (
            "python",
            "platform",
            "torch",
            "torchvision",
            "cuda",
            "gpu",
            "gpu_uuid",
            "driver_version",
            "device",
            "cudnn_deterministic",
            "cudnn_benchmark",
            "cudnn_allow_tf32",
            "matmul_allow_tf32",
        )
        signatures = []
        for report in (a, b):
            measurement = report["measurements"]
            required = (
                report["script_sha256"],
                report["checkpoint_sha256"],
                report["classes_sha256"],
                measurement["input_sha256"],
                report["parameters"],
                measurement["precision"],
                measurement["indices"],
                [report["configuration"][k] for k in config_keys],
                [report["environment"][k] for k in env_keys],
            )
            if any(not report["environment"][k] for k in ("cuda", "gpu")):
                raise ValueError("CUDA hardware metadata is required")
            signatures.append(required)
            values = measurement["wall_ms"]
            if len(values) != report["configuration"]["steps"] or not all(
                math.isfinite(x) and x > 0 for x in values
            ):
                raise ValueError("Invalid or incomplete raw timing samples")
            if (
                measurement["correctness_batch_size"]
                != report["configuration"]["batch_size"]
            ):
                raise ValueError(
                    "Correctness evidence does not cover the measured batch"
                )
            certificate = Path(report["measurements"]["correctness_file"])
            if certificate.name != str(certificate):
                raise ValueError("Correctness file must be local to its report")
            certificate = Path(a_path if report is a else b_path).parent / certificate
            if _hash(certificate) != measurement["correctness_sha256"]:
                raise ValueError("Correctness evidence hash differs")
        if signatures[0] != signatures[1] or (
            signature is not None and signatures[0] != signature
        ):
            raise ValueError(
                "Weights, inputs, workload, precision or environment differ"
            )
        signature = signatures[0]
        packets = [
            torch.load(
                Path(p).parent / r["measurements"]["correctness_file"],
                map_location="cpu",
                weights_only=True,
            )
            for p, r in ((a_path, a), (b_path, b))
        ]
        torch.testing.assert_close(packets[0], packets[1])
        # Use raw wall samples; GPU-only timing and memory remain separate diagnostics.
        a_ms = statistics.median(a["measurements"]["wall_ms"])
        b_ms = statistics.median(b["measurements"]["wall_ms"])
        ratio = b_ms / a_ms if metric == "throughput" else a_ms / b_ms
        ratios.append(ratio)
        rows.append(
            {
                "pair": index,
                "candidate": str(a_path),
                "baseline": str(b_path),
                "ratio": ratio,
                "candidate_gpu_ms": a["measurements"]["gpu_median_ms"],
                "baseline_gpu_ms": b["measurements"]["gpu_median_ms"],
                "candidate_peak_bytes": a["measurements"]["peak_allocated_bytes"],
                "baseline_peak_bytes": b["measurements"]["peak_allocated_bytes"],
            }
        )
    passed = [
        r >= 1 - tolerance if metric == "throughput" else r <= 1 + tolerance
        for r in ratios
    ]
    status = "pass" if all(passed) else "fail" if not any(passed) else "blocked"
    return {
        "status": status,
        "metric": metric,
        "tolerance": tolerance,
        "ratios": ratios,
        "median_ratio": statistics.median(ratios),
        "pairs": rows,
        "reason": "All pairs meet the gate"
        if status == "pass"
        else "All pairs miss the gate"
        if status == "fail"
        else "Pairs disagree across the gate; collect more evidence",
        "development": development,
        "scope": "Selected workload only; this is not whole-release signoff",
    }


def _links(repo):
    names = subprocess.check_output(
        ["git", "-C", str(repo), "ls-files", "-z", "*.md"], text=True
    ).split("\0")
    errors, checked = [], 0
    pattern = re.compile(r'!?\[[^\]]*\]\((<[^>]+>|[^\s)]+)(?:\s+"[^"]*")?\)')
    for name in filter(None, names):
        path = repo / name
        if not path.is_file():
            continue
        in_code = False
        for number, line in enumerate(path.read_text().splitlines(), 1):
            if line.lstrip().startswith("```"):
                in_code = not in_code
            if in_code:
                continue
            for target in pattern.findall(line):
                url = urlsplit(target.strip("<>"))
                if url.scheme or url.netloc or not url.path:
                    continue
                value = unquote(url.path)
                target_path = (
                    repo / value.lstrip("/")
                    if value.startswith("/")
                    else path.parent / value
                )
                checked += 1
                if not target_path.exists():
                    errors.append(f"{name}:{number}: {value}")
    return {
        "status": "fail" if errors else "pass",
        "paths_checked": checked,
        "errors": errors,
        "scope": "Tracked Markdown local paths; anchors/external URLs are not checked",
    }


def _archive(repo, ref, run, output):
    if subprocess.check_output(
        ["git", "-C", str(repo), "status", "--porcelain"], text=True
    ).strip():
        raise ValueError("Source archives require a clean checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", ref], text=True
    ).strip()
    manifest = _read(run / "manifest.json")
    if manifest["candidate_commit"] != commit or manifest.get("development"):
        raise ValueError("Evidence is not bound to this clean candidate commit")
    evidence_roots = [run, *[Path(p).resolve() for p in manifest.get("runs", [])]]
    files = []
    for index, root in enumerate(evidence_roots):
        if _read(root / "manifest.json")["candidate_commit"] != commit:
            raise ValueError("Referenced evidence belongs to another source commit")
        if any(p.is_symlink() for p in root.rglob("*")):
            raise ValueError("Evidence archives do not follow symlinks")
        prefix = "summary" if index == 0 else f"runs/{index:03}"
        files.extend(
            (p, prefix + "/" + str(p.relative_to(root)))
            for p in root.rglob("*")
            if p.is_file()
        )
    if any(output.resolve().is_relative_to(p.resolve()) for p in evidence_roots):
        raise ValueError("Archive output must be outside its evidence directory")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source.zip"
    subprocess.run(
        [
            "git",
            "-C",
            str(repo),
            "archive",
            "--format=zip",
            f"--output={source.resolve()}",
            commit,
        ],
        check=True,
    )
    checksums = {name: _hash(path) for path, name in files}
    archive = output / "evidence.zip"
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as z:
        for path, name in files:
            z.write(path, name)
        z.writestr("SHA256.json", json.dumps(checksums, indent=2))
    result = {
        "candidate_commit": commit,
        "source_sha256": _hash(source),
        "evidence_sha256": _hash(archive),
        "files": checksums,
        "source_directories": [str(p) for p in evidence_roots],
        "scope": "Source/evidence snapshot; no tag, publication or release signoff",
    }
    (output / "archive.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


def _verify_archive(directory):
    record = _read(directory / "archive.json")
    for name, key in (
        ("source.zip", "source_sha256"),
        ("evidence.zip", "evidence_sha256"),
    ):
        path = directory / name
        if _hash(path) != record[key]:
            raise ValueError(f"Archive hash differs: {name}")
        with zipfile.ZipFile(path) as archive:
            names = archive.namelist()
            if len(names) != len(set(names)):
                raise ValueError(f"Duplicate archive members: {name}")
            if archive.testzip() is not None:
                raise ValueError(f"Corrupt archive members: {name}")
            if name == "source.zip":
                if archive.comment.decode().strip() != record["candidate_commit"]:
                    raise ValueError("Source archive commit differs")
                continue
            checksums = json.loads(archive.read("SHA256.json"))
            if checksums != record["files"] or set(names) != {
                "SHA256.json",
                *checksums,
            }:
                raise ValueError("Evidence inventory differs")
            for member, expected in checksums.items():
                with archive.open(member) as stream:
                    actual = hashlib.file_digest(stream, "sha256").hexdigest()
                if actual != expected:
                    raise ValueError(f"Evidence member hash differs: {member}")
                if member.endswith("/manifest.json"):
                    manifest = json.loads(archive.read(member))
                    if manifest["candidate_commit"] != record[
                        "candidate_commit"
                    ] or manifest.get("development"):
                        raise ValueError(f"Evidence source differs: {member}")
    return {
        "status": "pass",
        "candidate_commit": record["candidate_commit"],
        "files_verified": len(record["files"]),
        "scope": "Archive integrity only; retain archive.json independently",
    }


def _self_check():
    import torch

    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        candidates, baselines = [], []
        for index in range(3):
            pair = []
            for side in ("candidate", "baseline"):
                folder = root / f"{side}-{index}"
                folder.mkdir()
                torch.save(
                    {"output": torch.tensor([1.0]), "gradient": torch.tensor([0.1])},
                    folder / "correctness.pt",
                )
                report = {
                    "complete_dataset": True,
                    "script_sha256": "runner",
                    "checkpoint_sha256": "weights",
                    "classes_sha256": "classes",
                    "parameters": 1,
                    "configuration": dict.fromkeys(
                        (
                            "case",
                            "recipe",
                            "time_steps",
                            "batch_size",
                            "seed",
                            "phase",
                            "amp",
                            "warmup",
                            "steps",
                        ),
                        1,
                    ),
                    "environment": dict.fromkeys(
                        (
                            "python",
                            "platform",
                            "torch",
                            "torchvision",
                            "cuda",
                            "gpu",
                            "gpu_uuid",
                            "driver_version",
                            "device",
                            "cudnn_deterministic",
                            "cudnn_benchmark",
                            "cudnn_allow_tf32",
                            "matmul_allow_tf32",
                        ),
                        "fixed",
                    ),
                    "measurements": {
                        "input_sha256": "inputs",
                        "precision": "fp32",
                        "indices": [0],
                        "wall_ms": [10.0],
                        "gpu_median_ms": 10.0,
                        "peak_allocated_bytes": 1,
                        "correctness_file": "correctness.pt",
                        "correctness_batch_size": 1,
                        "correctness_sha256": _hash(folder / "correctness.pt"),
                    },
                }
                report["configuration"]["mode"] = "benchmark"
                p = folder / "report.json"
                p.write_text(json.dumps(report))
                pair.append(p)
            candidates.append(pair[0])
            baselines.append(pair[1])
        assert _compare(candidates, baselines)["status"] == "pass"
        for path in candidates:
            r = _read(path)
            r["measurements"]["wall_ms"] = [11.0]
            path.write_text(json.dumps(r))
        assert _compare(candidates, baselines)["status"] == "fail"
        r = _read(candidates[0])
        r["measurements"]["wall_ms"] = [10.0]
        candidates[0].write_text(json.dumps(r))
        assert _compare(candidates, baselines)["status"] == "blocked"
        r["measurements"]["input_sha256"] = "different"
        candidates[0].write_text(json.dumps(r))
        try:
            _compare(candidates, baselines)
        except ValueError:
            pass
        else:
            raise AssertionError("Mismatched inputs were accepted")
        repo = root / "repo"
        repo.mkdir()
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        (repo / "source.py").write_text("value = 1\n")
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Self-check",
                "-c",
                "user.email=self-check@example.invalid",
                "commit",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        commit = subprocess.check_output(
            ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
        ).strip()
        run = root / "run"
        run.mkdir()
        raw = root / "raw"
        raw.mkdir()
        (raw / "manifest.json").write_text(
            json.dumps({"candidate_commit": commit, "development": False})
        )
        (run / "manifest.json").write_text(
            json.dumps(
                {"candidate_commit": commit, "development": False, "runs": [str(raw)]}
            )
        )
        (run / "commands.log").write_text("fixture evidence\n")
        try:
            _archive(repo, commit, run, raw / "archive")
        except ValueError:
            pass
        else:
            raise AssertionError("Archive output inside linked evidence was accepted")
        output = root / "archive"
        _archive(repo, commit, run, output)
        relocated = root / "relocated"
        output.rename(relocated)
        output = relocated
        assert _verify_archive(output)["status"] == "pass"
        record = _read(output / "archive.json")
        changed = dict(record, files={})
        (output / "archive.json").write_text(json.dumps(changed))
        try:
            _verify_archive(output)
        except ValueError:
            pass
        else:
            raise AssertionError("Changed inventory was accepted")
        (output / "archive.json").write_text(json.dumps(record))
        with (output / "evidence.zip").open("ab") as stream:
            stream.write(b"tampered")
        try:
            _verify_archive(output)
        except ValueError:
            pass
        else:
            raise AssertionError("Tampered evidence was accepted")
    print("Comparison/archive self-check passed; fixtures are not release evidence")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="mode", required=True)
    compare = commands.add_parser("compare")
    compare.add_argument("--candidate", nargs="+", type=Path, required=True)
    compare.add_argument("--baseline", nargs="+", type=Path, required=True)
    compare.add_argument(
        "--metric", choices=("throughput", "latency"), default="throughput"
    )
    compare.add_argument("--output-dir", type=Path, required=True)
    links = commands.add_parser("links")
    links.add_argument("--repo", type=Path, default=Path.cwd())
    archive = commands.add_parser("archive")
    archive.add_argument("--repo", type=Path, required=True)
    archive.add_argument("--ref", required=True)
    archive.add_argument("--run", type=Path, required=True)
    archive.add_argument("--output-dir", type=Path, required=True)
    verify = commands.add_parser("verify-archive")
    verify.add_argument("--archive-dir", type=Path, required=True)
    commands.add_parser("self-check")
    args = parser.parse_args()
    if args.mode == "self-check":
        _self_check()
        return
    if args.mode == "compare":
        try:
            report = _compare(args.candidate, args.baseline, args.metric)
        except (KeyError, ValueError, AssertionError, OSError, RuntimeError) as error:
            report = {"status": "blocked", "reason": str(error)}
        _write(args.output_dir, report)
    elif args.mode == "links":
        report = _links(args.repo.resolve())
    elif args.mode == "verify-archive":
        report = _verify_archive(args.archive_dir.resolve())
    else:
        report = _archive(
            args.repo.resolve(), args.ref, args.run.resolve(), args.output_dir
        )
    print(json.dumps(report, indent=2, allow_nan=False))
    if report.get("status") not in (None, "pass"):
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
