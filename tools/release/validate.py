"""Run selected release profiles, paired benchmarks, and aggregate evidence."""

import argparse
import ast
from datetime import datetime, timezone
import fnmatch
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import signal
import subprocess
import sys
import sysconfig
import tarfile
import tempfile
import time
import tomllib
import traceback
import xml.etree.ElementTree as ET

from evidence import _compare, _hash, _read, _write


_ROOT = Path(__file__).resolve().parents[2]
_CONFIG = Path(__file__).with_name("profiles.toml")
# Direct script execution must resolve the repo-owned benchmark validators.
sys.path.insert(0, str(_ROOT))


def _git(repo, *args):
    return subprocess.check_output(["git", "-C", str(repo), *args], text=True).strip()


def _command(argv, cwd, log, deadline, env=None, expected_error=None):
    started = time.monotonic()
    remaining = deadline - started
    record = {
        "argv": list(map(str, argv)),
        "cwd": str(cwd),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "log": str(log),
    }
    if remaining <= 0:
        return {
            **record,
            "status": "blocked",
            "reason": "Time budget exhausted",
            "ended_utc": datetime.now(timezone.utc).isoformat(),
        }
    with log.open("w") as stream:
        try:
            process = subprocess.Popen(
                record["argv"],
                cwd=cwd,
                env=env,
                stdout=stream,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )
        except OSError as error:
            stream.write(str(error) + "\n")
            return {
                **record,
                "status": "blocked",
                "reason": str(error),
                "seconds": time.monotonic() - started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            }
        try:
            code = process.wait(timeout=remaining)
        except subprocess.TimeoutExpired:
            os.killpg(process.pid, signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait()
            return {
                **record,
                "status": "blocked",
                "reason": "Time budget exhausted; process group terminated",
                "seconds": time.monotonic() - started,
                "ended_utc": datetime.now(timezone.utc).isoformat(),
            }
    passed = (
        code == 0
        if expected_error is None
        else code != 0 and expected_error in log.read_text()
    )
    return {
        **record,
        "status": "pass" if passed else "fail",
        "returncode": code,
        "expected_error": expected_error,
        "seconds": time.monotonic() - started,
        "ended_utc": datetime.now(timezone.utc).isoformat(),
    }


def _junit(path, allowed):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("JUnit contains no test cases")
    skips = []
    for case in cases:
        skipped = case.find("skipped")
        if skipped is None:
            continue
        reason = skipped.get("message", "")
        if reason == "collection skipped" and skipped.text:
            try:
                location = ast.literal_eval(skipped.text)
            except (SyntaxError, ValueError):
                location = None
            reason = (
                location[2]
                if isinstance(location, tuple)
                and len(location) == 3
                and isinstance(location[2], str)
                else reason
            )
        skips.append(
            {
                "test": case.get("classname", "") + "::" + case.get("name", ""),
                "reason": reason,
            }
        )
    failures = [
        c.get("name")
        for c in cases
        if c.find("failure") is not None or c.find("error") is not None
    ]
    unexpected = [
        s
        for s in skips
        if not any(fnmatch.fnmatchcase(s["reason"], p) for p in allowed)
    ]
    return {
        "status": "fail" if failures else "blocked" if unexpected else "pass",
        "tests": len(cases),
        "failures": failures,
        "skips": skips,
        "unexpected_skips": unexpected,
    }


def _environment():
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "triton",
        "transformer-engine",
        "spikingjelly",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "packages": versions,
        "execution_environment": {
            k: os.environ.get(k)
            for k in (
                "CUDA_VISIBLE_DEVICES",
                "CUDA_HOME",
                "CUDA_PATH",
                "CXX",
                "TORCH_CUDA_ARCH_LIST",
            )
        },
    }


def _extract(sdist, destination):
    destination.mkdir()
    with tarfile.open(sdist) as archive:
        for member in archive.getmembers():
            relative = Path(member.name)
            if (
                relative.is_absolute()
                or ".." in relative.parts
                or not (member.isdir() or member.isfile())
            ):
                raise ValueError(f"Unsafe sdist member: {member.name}")
            path = destination / relative
            if member.isdir():
                path.mkdir(parents=True, exist_ok=True)
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(archive.extractfile(member).read())
    roots = list(destination.iterdir())
    if len(roots) != 1 or not roots[0].is_dir():
        raise ValueError("sdist must have one root directory")
    return roots[0]


def _package(args, case_dir, deadline, native=False):
    commands = []

    def run(argv, cwd=_ROOT, env=None, expected_error=None):
        record = _command(
            argv,
            cwd,
            case_dir / f"command-{len(commands):02}.log",
            deadline,
            env,
            expected_error,
        )
        commands.append(record)
        (case_dir / "commands.json").write_text(json.dumps(commands, indent=2) + "\n")
        if record["status"] != "pass":
            raise RuntimeError(f"Command did not pass; see {record['log']}")

    dist = case_dir / "dist"
    run(["uv", "build", "--out-dir", str(dist)])
    (wheel,) = dist.glob("*.whl")
    (sdist,) = dist.glob("*.tar.gz")
    checker = _ROOT / "tools/release/package_check.py"
    run(
        [
            sys.executable,
            str(checker),
            "artifacts",
            "--wheel",
            str(wheel),
            "--sdist",
            str(sdist),
        ]
    )
    # An independent install and working directory, outside the source checkout.
    outside = Path(tempfile.mkdtemp(prefix="sj-release-install-"))
    run(["uv", "venv", "--python", sys.executable, str(outside / "environment")])
    python = outside / "environment/bin/python"
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("SJ_BUILD_NATIVE_CUDA", None)
    if args.reuse_dependencies:
        site = subprocess.check_output(
            [
                str(python),
                "-c",
                "import sysconfig; print(sysconfig.get_path('purelib'))",
            ],
            text=True,
        ).strip()
        env["PYTHONPATH"] = os.pathsep.join((site, sysconfig.get_path("purelib")))
        run(
            ["uv", "pip", "install", "--python", str(python), str(wheel), "--no-deps"],
            outside,
            env,
        )
    else:
        pins = case_dir / "environment.constraints"
        pins.write_text(
            "\n".join(
                sorted(
                    {
                        f"{name}=={importlib.metadata.version(name)}"
                        for d in importlib.metadata.distributions()
                        if (name := d.metadata["Name"].lower().replace("_", "-"))
                        != "spikingjelly"
                    }
                )
            )
            + "\n"
        )
        if args.torch_index:
            torch_packages = [
                f"{name}=={importlib.metadata.version(name)}"
                for name in ("torch", "torchvision", "torchaudio")
            ]
            run(
                [
                    "uv",
                    "pip",
                    "install",
                    "--python",
                    str(python),
                    *torch_packages,
                    "--index-url",
                    args.torch_index,
                ],
                outside,
                env,
            )
        install = [
            "uv",
            "pip",
            "install",
            "--python",
            str(python),
            str(wheel),
            "--constraints",
            str(pins),
        ]
        if args.constraints:
            install += ["--constraints", str(args.constraints.resolve())]
        run(install, outside, env)
    run(
        [
            str(python),
            str(checker),
            "training",
            "--expected-root",
            str(outside / "environment"),
            "--output-dir",
            str(case_dir / "wheel-training"),
        ],
        outside,
        env,
    )
    source = _extract(sdist, outside / "sdist")
    if native:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "setuptools>=77.0.3",
                "ninja",
            ],
            outside,
            env,
        )
        env["SJ_BUILD_NATIVE_CUDA"] = "1"
        # Use the current build interpreter/Torch, then install the resulting wheel.
        run(
            [
                "uv",
                "build",
                "--wheel",
                "--no-build-isolation",
                "--python",
                str(python),
                "--out-dir",
                str(case_dir / "native-dist"),
                str(source),
            ],
            outside,
            env,
        )
        (native_wheel,) = (case_dir / "native-dist").glob("*.whl")
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--force-reinstall",
                "--no-deps",
                str(native_wheel),
            ],
            outside,
            env,
        )
        run(
            [
                str(python),
                str(checker),
                "training",
                "--device",
                "cuda",
                "--provider",
                "cuda",
                "--expected-root",
                str(outside / "environment"),
                "--output-dir",
                str(case_dir / "native-training"),
            ],
            outside,
            env,
        )
    else:
        run(
            [
                "uv",
                "pip",
                "install",
                "--python",
                str(python),
                "--force-reinstall",
                "--no-deps",
                str(sdist),
            ],
            outside,
            env,
        )
        run(
            [
                str(python),
                str(checker),
                "training",
                "--expected-root",
                str(outside / "environment"),
                "--output-dir",
                str(case_dir / "sdist-training"),
            ],
            outside,
            env,
        )
    negative = os.environ.copy()
    negative.update(
        SJ_BUILD_NATIVE_CUDA="1", CUDA_HOME=str(outside / "missing-toolkit")
    )
    run(
        [sys.executable, str(source / "setup.py"), "--name"],
        source,
        negative,
        expected_error="Native CUDA build requires",
    )
    return {
        "commands": commands,
        "install_directory": str(outside),
        "wheel_sha256": _hash(wheel),
        "sdist_sha256": _hash(sdist),
        "dependency_reuse": args.reuse_dependencies,
    }


def _qwen_quick(case, args, folder, deadline):
    result = {
        "commands": [],
        "status": "blocked",
        "scope": "quick conversion/PPL; not full quality or speed acceptance",
    }
    if args.qwen_model_root is None or args.qwen_hf_cache is None:
        return {**result, "reason": "Set --qwen-model-root and --qwen-hf-cache"}
    model_root, cache = args.qwen_model_root.resolve(), args.qwen_hf_cache.resolve()
    if not model_root.is_dir() or not cache.is_dir():
        return {**result, "reason": "Local model/dataset cache directory is missing"}
    try:
        versions = {
            name: importlib.metadata.version(name)
            for name in ("transformers", "datasets", "huggingface-hub")
        }
    except importlib.metadata.PackageNotFoundError as error:
        return {**result, "reason": f"Missing Qwen evaluation dependency: {error}"}
    required = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
        "optional-dependencies"
    ]["qwen"][0].split("==")[1]
    if versions["transformers"] != required or not versions["datasets"].startswith(
        "5."
    ):
        return {**result, "reason": "Use the pinned qwen extra and datasets>=5,<6"}
    result["dependencies"] = versions
    base = _ROOT / "benchmark/snn_llm/qwen_conversion"
    lock_path = base / "artifacts.json"
    lock = _read(lock_path)
    model = lock["models"][case["model_key"]]
    result["artifact_lock_sha256"] = _hash(lock_path)
    hf = Path(sys.executable).parent / "hf"
    if not hf.is_file():
        return {
            **result,
            "reason": "Install huggingface_hub CLI in this environment for pinned model verification",
        }
    verify = _command(
        [
            str(hf),
            "cache",
            "verify",
            model["repository"],
            "--revision",
            model["revision"],
            "--local-dir",
            str(model_root),
        ],
        _ROOT,
        folder / "model-verify.log",
        deadline,
    )
    result["commands"].append(verify)
    if verify["status"] != "pass":
        return {
            **result,
            "reason": "Pinned local model checksum verification did not pass",
        }
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_DATASETS_OFFLINE="1")
    common = [
        "--model-key",
        case["model_key"],
        "--model-root",
        str(model_root),
        "--device",
        "cuda",
        "--worktree-revision",
        _git(_ROOT, "rev-parse", "HEAD"),
        "--time-steps",
        str(case["time_steps"]),
        "--calibration-levels",
        str(case["calibration_levels"]),
        "--calibration-quantile",
        str(case["calibration_quantile"]),
    ]
    quality = folder / "quality"
    command = _command(
        [
            sys.executable,
            "-m",
            "benchmark.snn_llm.qwen_conversion.quality_eval",
            *common,
            "--hf-cache",
            str(cache),
            "--output-dir",
            str(quality),
            "--precision",
            "bf16",
            "--calibration-windows",
            str(case["calibration_windows"]),
            "--calibration-reservoir-size",
            str(case["calibration_reservoir_size"]),
            "--wikitext-split",
            "test",
            "--max-ppl-windows",
            str(case["ppl_windows"]),
        ],
        _ROOT,
        folder / "quality.log",
        deadline,
        env,
    )
    result["commands"].append(command)
    if command["status"] != "pass":
        return {
            **result,
            "status": command["status"],
            "reason": "Qwen calibration/quick PPL did not complete",
        }
    calibration = quality / "calibration.pt"
    smoke = folder / "conversion"
    command = _command(
        [
            sys.executable,
            "-m",
            "benchmark.snn_llm.qwen_conversion.scaleout_smoke",
            *common,
            "--calibration-artifact",
            str(calibration),
            "--output-dir",
            str(smoke),
        ],
        _ROOT,
        folder / "conversion.log",
        deadline,
        env,
    )
    result["commands"].append(command)
    if command["status"] != "pass":
        return {
            **result,
            "status": command["status"],
            "reason": "Qwen conversion/cache/reset checks did not pass",
        }
    q, conversion = _read(quality / "report.json"), _read(smoke / "report.json")
    from benchmark.snn_llm.qwen_conversion._quality import validate_quality
    from benchmark.snn_llm.qwen_conversion.scaleout_smoke import _validate_report

    try:
        _validate_report(conversion)
        validate_quality(q)
        config, ppl = q["configuration"], q["quality"]["wikitext"]
        if (
            q["model"]["key"] != case["model_key"]
            or q["model"]["revision"] != model["revision"]
            or q["model"]["files"] != conversion["model"]["files"]
            or q["data"] != lock["datasets"]
        ):
            raise ValueError("Qwen model/data identities differ from fixed inputs")
        if q["conversion"]["calibration_sha256"] != _hash(calibration) or conversion[
            "conversion"
        ]["calibration_sha256"] != _hash(calibration):
            raise ValueError("Qwen checks did not share the same calibration")
        for key in (
            "time_steps",
            "calibration_levels",
            "calibration_quantile",
            "calibration_reservoir_size",
        ):
            if config[key] != case[key]:
                raise ValueError(f"Unexpected Qwen configuration: {key}")
        if (
            config["calibration_windows_requested"] != case["calibration_windows"]
            or config["calibration_valid_token_count"]
            != 512 * case["calibration_windows"]
            or config["wikitext_split"] != "test"
            or config["full_evaluation"]
            or config["precision"] != "bf16"
            or config["tasks"]
        ):
            raise ValueError("Quick Qwen calibration/evaluation scope differs")
        for split in ("train", "test"):
            if (
                not q["dataset_provenance"][split]["fingerprint"]
                or not q["dataset_provenance"][split]["token_sha256"]
            ):
                raise ValueError("Missing actual WikiText provenance")
        if (
            ppl["processed_window_indices"] != list(range(case["ppl_windows"]))
            or ppl["token_count"] != case["ppl_windows"] * 512 - 1
        ):
            raise ValueError("Quick PPL did not cover the declared token budget")
        import math

        if (
            not all(
                math.isfinite(ppl[k])
                for k in ("dense_ppl", "snn_ppl", "relative_degradation")
            )
            or min(ppl["dense_ppl"], ppl["snn_ppl"]) <= 0
        ):
            raise ValueError("Invalid Qwen perplexity")
    except (ValueError, KeyError) as error:
        return {**result, "status": "fail", "reason": str(error)}
    return {
        **result,
        "status": "pass",
        "metrics": {"ppl": ppl, "conversion": conversion["metrics"]},
        "calibration_sha256": _hash(calibration),
        "reason": "Quick profile met existing conversion tolerances and <=15% PPL degradation",
    }


def _run_case(name, case, args, output, deadline, environment):
    folder = output / name
    folder.mkdir()
    result = {"id": name, "group": case["group"], "status": "not-run", "commands": []}
    values = {"python": sys.executable, "repo": str(_ROOT), "case_dir": str(folder)}
    env = os.environ.copy()
    try:
        for key, actual in (
            ("python_prefix", environment["python"]),
            ("torch_prefix", environment["packages"]["torch"] or ""),
        ):
            if key in case and not actual.startswith(case[key]):
                return {
                    **result,
                    "status": "blocked",
                    "reason": f"Requires {key}={case[key]}; actual {actual}",
                }
        kind = case["kind"]
        if kind == "not-ready":
            return {**result, "status": "blocked", "reason": case["reason"]}
        if kind == "native":
            preflight = _command(
                [
                    sys.executable,
                    str(_ROOT / "tools/release/package_check.py"),
                    "prerequisites",
                ],
                _ROOT,
                folder / "prerequisites.log",
                deadline,
            )
            result["commands"].append(preflight)
            if preflight["status"] != "pass":
                return {
                    **result,
                    "status": "blocked",
                    "reason": "Native GPU/toolkit/compiler prerequisites are unavailable",
                }
        if kind in ("package", "native"):
            result.update(_package(args, folder, deadline, native=kind == "native"))
        elif kind == "pytest":
            if case["group"].startswith("cpu"):
                env["CUDA_VISIBLE_DEVICES"] = ""
            junit = folder / "pytest.xml"
            command = _command(
                [sys.executable, "-m", "pytest", *case["args"], f"--junitxml={junit}"],
                _ROOT,
                folder / "pytest.log",
                deadline,
                env,
            )
            result["commands"].append(command)
            if command["status"] != "pass":
                return {
                    **result,
                    "status": command["status"],
                    "reason": "pytest did not complete successfully",
                }
            result.update(_junit(junit, case["allow_skip_reasons"]))
            return result
        elif kind == "commands":
            for index, argv in enumerate(case["commands"]):
                command = _command(
                    [v.format_map(values) for v in argv],
                    _ROOT,
                    folder / f"command-{index}.log",
                    deadline,
                )
                result["commands"].append(command)
            statuses = [r["status"] for r in result["commands"]]
            result["status"] = (
                "fail"
                if "fail" in statuses
                else "blocked"
                if "blocked" in statuses
                else "pass"
            )
            return result
        elif kind == "qwen-quick":
            result.update(_qwen_quick(case, args, folder, deadline))
            return result
        elif kind == "vision-quality":
            if (
                args.data_root is None
                or args.minimum_top1 is None
                or (args.vision_mode == "evaluate" and args.checkpoint is None)
            ):
                return {
                    **result,
                    "status": "blocked",
                    "reason": "Set data-root, predeclared minimum-top1 and checkpoint for evaluation",
                }
            argv = [
                sys.executable,
                str(_ROOT / "benchmark/benchmark_image_classification.py"),
                "--case",
                case["task"],
                "--mode",
                args.vision_mode,
                "--data-root",
                str(args.data_root.resolve()),
                "--output-dir",
                str(folder / "task"),
            ]
            if args.checkpoint:
                argv += ["--checkpoint", str(args.checkpoint.resolve())]
            if args.epochs is not None:
                argv += ["--epochs", str(args.epochs)]
            command = _command(argv, _ROOT, folder / "task.log", deadline)
            result["commands"].append(command)
            if command["status"] != "pass":
                return {
                    **result,
                    "status": command["status"],
                    "reason": "Task did not complete",
                }
            report = _read(folder / "task/report.json")
            score = report.get("evaluation", report.get("final_evaluation"))
            if not report["complete_dataset"] or score["samples"] != 10000:
                return {
                    **result,
                    "status": "blocked",
                    "reason": "Incomplete real-data evaluation",
                }
            result["metrics"] = score
            result["status"] = "pass" if score["top1"] >= args.minimum_top1 else "fail"
            result["reason"] = (
                f"top1={score['top1']}; predeclared minimum={args.minimum_top1}"
            )
            return result
        else:
            raise ValueError(f"Unknown profile kind: {kind}")
        result["status"] = "pass"
    except (RuntimeError, ValueError, OSError, KeyError, ImportError) as error:
        (folder / "error.log").write_text(traceback.format_exc())
        if (folder / "commands.json").is_file():
            result["commands"] = _read(folder / "commands.json")
        status = result["commands"][-1]["status"] if result["commands"] else "blocked"
        result.update(
            status=status if status != "pass" else "blocked", reason=str(error)
        )
    return result


def _start(args, policy):
    commit = _git(_ROOT, "rev-parse", "HEAD")
    dirty = bool(_git(_ROOT, "status", "--porcelain"))
    environment = _environment()
    manifest = {
        "candidate_commit": commit,
        "development": args.development,
        "source_dirty": dirty,
        "source_diff_sha256": hashlib.sha256(
            _git(_ROOT, "diff", "HEAD", "--binary").encode()
        ).hexdigest(),
        "config_sha256": _hash(args.config),
        "environment": environment,
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "cost_usd": args.cost_usd,
        "cost_source": "operator-supplied; no rental/billing automation",
        "time_budget_seconds": args.time_budget,
        "campaign_manifest": str(args.campaign_manifest)
        if args.campaign_manifest
        else None,
        "argv": sys.argv,
        "inputs": {
            "qwen_model_root": str(args.qwen_model_root.resolve())
            if args.qwen_model_root
            else None,
            "qwen_hf_cache": str(args.qwen_hf_cache.resolve())
            if args.qwen_hf_cache
            else None,
            "minimum_top1": args.minimum_top1,
            "epochs": args.epochs,
            "vision_mode": args.vision_mode,
            "task": args.task,
            "phase": args.phase,
            "data_root": str(args.data_root.resolve()) if args.data_root else None,
            "checkpoint_sha256": _hash(args.checkpoint) if args.checkpoint else None,
        },
        "release_eligible": (
            platform.system() == "Linux" and not dirty and not args.development
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    if (dirty or platform.system() != "Linux") and not args.development:
        raise ValueError(
            "Formal validation requires Linux and a clean source checkout; development runs are not release evidence"
        )
    if not args.development:
        relative = args.config.resolve().relative_to(_ROOT)
        _git(_ROOT, "ls-files", "--error-unmatch", str(relative))
    if args.cost_usd > policy["max_cost_usd"]:
        raise ValueError("Reported cost exceeds the release budget")
    return environment


def _paired(args, policy, deadline):
    if any(
        v is None
        for v in (
            args.baseline_python,
            args.candidate_python,
            args.baseline_ref,
            args.baseline_version,
            args.checkpoint,
            args.data_root,
            args.task,
            args.candidate_wheel,
            args.baseline_wheel,
        )
    ):
        raise ValueError(
            "Paired runs require both installed Python environments/wheels, previous formal release ref/version, task, data and checkpoint"
        )
    import re

    version = tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
        "version"
    ]
    if not re.fullmatch(r"\d+(?:\.\d+)+", args.baseline_version):
        raise ValueError("Baseline must be a formal version, not dev/rc/alpha/beta")
    if tuple(map(int, args.baseline_version.split("."))) >= tuple(
        map(int, version.split("."))
    ):
        raise ValueError("Baseline must precede the candidate version")
    baseline_commit = _git(_ROOT, "rev-parse", args.baseline_ref + "^{commit}")
    manifest = _read(args.output_dir / "manifest.json")
    manifest.update(
        baseline_commit=baseline_commit,
        baseline_version=args.baseline_version,
        baseline_ref=args.baseline_ref,
        task=args.task,
        checkpoint_sha256=_hash(args.checkpoint),
    )
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    runner = _ROOT / "benchmark/benchmark_image_classification.py"
    reports = {"candidate": [], "baseline": []}
    commands = []
    for side in ("candidate", "baseline"):
        python = getattr(args, side + "_python").absolute()
        wheel = getattr(args, side + "_wheel").resolve()
        command = _command(
            [
                str(python),
                str(_ROOT / "tools/release/package_check.py"),
                "provenance",
                "--wheel",
                str(wheel),
                "--expected-root",
                str(python.parent.parent),
            ],
            Path(tempfile.gettempdir()),
            args.output_dir / f"{side}-provenance.log",
            deadline,
        )
        commands.append(command)
        if command["status"] != "pass":
            raise ValueError(f"{side} installed package is not the fixed wheel")
        if side == "candidate":
            import zipfile

            with zipfile.ZipFile(wheel) as archive:
                for name in archive.namelist():
                    if name.startswith("spikingjelly/") and name.endswith(
                        (".py", ".cuh")
                    ):
                        source = _ROOT / name.replace("spikingjelly/_ops/", "ops/")
                        if not source.is_file() or source.read_bytes() != archive.read(
                            name
                        ):
                            raise ValueError(
                                "Candidate wheel does not match the recorded source checkout"
                            )
        manifest[side + "_wheel_sha256"] = _hash(wheel)
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\n"
    )
    for index in range(policy["performance_pairs"]):
        order = (
            ("baseline", "candidate") if index % 2 == 0 else ("candidate", "baseline")
        )
        for side in order:
            python = getattr(args, side + "_python").absolute()
            folder = args.output_dir / f"{side}-{index}"
            argv = [
                str(python),
                str(runner),
                "--case",
                args.task,
                "--mode",
                "benchmark",
                "--phase",
                args.phase,
                "--data-root",
                str(args.data_root.resolve()),
                "--checkpoint",
                str(args.checkpoint.resolve()),
                "--output-dir",
                str(folder),
            ]
            env = os.environ.copy()
            env.pop("PYTHONPATH", None)
            command = _command(
                argv,
                Path(tempfile.gettempdir()),
                args.output_dir / f"{side}-{index}.log",
                deadline,
                env,
            )
            commands.append(command)
            if command["status"] != "pass":
                return {
                    "status": "blocked",
                    "cases": [
                        {
                            "id": "performance",
                            "group": "performance",
                            "status": "blocked",
                            "reason": f"{side} runner failed; incompatible old APIs need a tracked adapter, never automatic N/A",
                        }
                    ],
                    "commands": commands,
                }
            path = folder / "report.json"
            report = _read(path)
            expected_version = (
                args.baseline_version
                if side == "baseline"
                else tomllib.loads((_ROOT / "pyproject.toml").read_text())["project"][
                    "version"
                ]
            )
            if report["environment"]["spikingjelly"] != expected_version:
                raise ValueError(
                    f"{side} installed package version does not match fixed inputs"
                )
            reports[side].append(path)
    comparison = _compare(
        reports["candidate"],
        reports["baseline"],
        policy["performance_metric"],
        policy["performance_tolerance"],
        development=args.development,
    )
    return {
        **comparison,
        "commands": commands,
        "cases": [
            {
                "id": "performance",
                "group": "performance",
                "status": comparison["status"],
                "reason": comparison["reason"],
            }
        ],
    }


def _summary(args, config):
    manifests, attempts, selected, signatures = [], [], {}, {}
    for run in sorted(
        args.runs, key=lambda p: _read(p / "manifest.json")["started_utc"]
    ):
        manifest = _read(run / "manifest.json")
        manifests.append(manifest)
        for case in _read(run / "report.json")["cases"]:
            signature = {
                "environment": manifest.get("environment"),
                "inputs": manifest.get("inputs", {})
                if case["group"] in ("quality", "performance", "qwen")
                else None,
            }
            if case["id"] in signatures and signatures[case["id"]] != signature:
                raise ValueError("Retests changed predeclared inputs/thresholds")
            signatures[case["id"]] = signature
            attempts.append({**case, "run": str(run.resolve())})
            selected[case["id"]] = case
    if any(m["config_sha256"] != _hash(args.config) for m in manifests):
        raise ValueError("Cannot aggregate different profile revisions")
    commits = {m["candidate_commit"] for m in manifests}
    if len(commits) != 1:
        raise ValueError("Cannot combine different candidate commits")
    starts = [datetime.fromisoformat(m["started_utc"]).timestamp() for m in manifests]
    ends = [datetime.fromisoformat(m["ended_utc"]).timestamp() for m in manifests]
    budget_ok = (
        max(ends) - min(starts) <= config["policy"]["max_seconds"]
        and max(m["cost_usd"] for m in manifests) <= config["policy"]["max_cost_usd"]
    )
    cases = list(selected.values())
    groups = {c["group"] for c in cases if c["status"] == "pass"}
    missing = sorted(set(config["policy"]["required_groups"]) - groups)
    eligible = budget_ok and all(m.get("release_eligible") for m in manifests)
    status = (
        "fail"
        if any(c["status"] == "fail" for c in cases)
        else "blocked"
        if missing or not eligible or any(c["status"] != "pass" for c in cases)
        else "pass"
    )
    return {
        "status": status,
        "candidate_commit": next(iter(commits)),
        "cases": cases,
        "attempts": attempts,
        "missing_groups": missing,
        "campaign_budget_met": budget_ok,
        "release_eligible": eligible,
        "reason": "Latest matching attempts; all history retained. Automated checks do not replace human scope/support/documentation signoff",
    }


def _self_check(output):
    output.mkdir(parents=True, exist_ok=False)
    records = []
    for name, argv, seconds, expected in (
        ("failure", [sys.executable, "-c", "raise SystemExit(7)"], 5, "fail"),
        ("missing", [str(output / "missing-executable")], 5, "blocked"),
        (
            "timeout",
            [sys.executable, "-c", "import time; time.sleep(20)"],
            0.05,
            "blocked",
        ),
        ("expired", [sys.executable, "-c", "raise SystemExit(0)"], -1, "blocked"),
    ):
        record = _command(
            argv, output, output / f"{name}.log", time.monotonic() + seconds
        )
        assert record["status"] == expected and record["ended_utc"], record
        assert record["argv"] == argv, record
        records.append(record)
    junit = output / "skip.xml"
    junit.write_text(
        '<testsuites><testsuite><testcase name="cpu"/><testcase name="cuda">'
        '<skipped message="CUDA unavailable"/></testcase></testsuite></testsuites>'
    )
    assert _junit(junit, [])["status"] == "blocked"
    assert _junit(junit, ["CUDA unavailable"])["status"] == "pass"
    junit.write_text(
        '<testsuites><testsuite><testcase name="nir">'
        '<skipped message="collection skipped">'
        "('test_nir.py', 7, \"Skipped: could not import 'nir'\")"
        "</skipped></testcase></testsuite></testsuites>"
    )
    assert _junit(junit, [])["status"] == "blocked"
    assert _junit(junit, ["*could not import 'nir'*"])["status"] == "pass"
    junit.write_text(
        '<testsuites><testsuite><testcase name="unknown">'
        '<skipped message="collection skipped">CUDA/path is not a reason tuple'
        "</skipped></testcase></testsuite></testsuites>"
    )
    assert _junit(junit, ["*CUDA*"])["status"] == "blocked"
    _write(
        output,
        {
            "status": "pass",
            "commands": records,
            "scope": "Runner self-check fixtures only; not release evidence",
        },
    )
    print("Runner failure/timeout/skip self-check passed")


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=("run", "paired", "report", "self-check"))
    parser.add_argument("--config", type=Path, default=_CONFIG)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--cases", nargs="+", default=("package", "cpu", "docs", "links")
    )
    parser.add_argument("--development", action="store_true")
    parser.add_argument(
        "--reuse-dependencies",
        action="store_true",
        help="development only: reuse dependency paths, never installed SJ source",
    )
    parser.add_argument("--constraints", type=Path)
    parser.add_argument(
        "--torch-index",
        help="official wheel index matching the pinned existing Torch build",
    )
    parser.add_argument("--time-budget", type=float, default=86400)
    parser.add_argument("--cost-usd", type=float, default=0)
    parser.add_argument(
        "--campaign-manifest",
        type=Path,
        help="first run manifest; share its original deadline across environments/retests",
    )
    parser.add_argument("--data-root", type=Path)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--qwen-model-root", type=Path)
    parser.add_argument("--qwen-hf-cache", type=Path)
    parser.add_argument(
        "--vision-mode", choices=("train", "evaluate"), default="evaluate"
    )
    parser.add_argument("--minimum-top1", type=float)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--candidate-python", type=Path)
    parser.add_argument("--baseline-python", type=Path)
    parser.add_argument("--baseline-ref")
    parser.add_argument("--baseline-version")
    parser.add_argument("--candidate-wheel", type=Path)
    parser.add_argument("--baseline-wheel", type=Path)
    parser.add_argument(
        "--task",
        choices=("maxformer-cifar100", "maxresnet-cifar100", "clif-tinyimagenet"),
    )
    parser.add_argument(
        "--phase", choices=("inference", "training"), default="inference"
    )
    parser.add_argument("--runs", nargs="+", type=Path)
    args = parser.parse_args()
    if args.mode == "self-check":
        _self_check(args.output_dir.resolve())
        return
    config = tomllib.loads(args.config.read_text())
    policy = config["policy"]
    if not 0 < args.time_budget <= min(86400, policy["max_seconds"]):
        parser.error("time-budget must be finite, positive and <= policy maximum")
    if not 0 <= args.cost_usd <= policy["max_cost_usd"]:
        parser.error("cost-usd must be finite and within the policy maximum")
    if args.minimum_top1 is not None and not 0 <= args.minimum_top1 <= 100:
        parser.error("minimum-top1 must be in [0, 100]")
    if args.epochs is not None and args.epochs <= 0:
        parser.error("epochs must be positive")
    if args.reuse_dependencies and not args.development:
        parser.error("dependency reuse is only allowed for development checks")
    args.output_dir = args.output_dir.resolve()
    if args.campaign_manifest:
        campaign = _read(args.campaign_manifest)
        end = (
            datetime.fromisoformat(campaign["started_utc"]).timestamp()
            + campaign["time_budget_seconds"]
        )
        remaining = end - time.time()
        if remaining <= 0 or args.cost_usd < campaign["cost_usd"]:
            parser.error(
                "Campaign deadline exhausted or cumulative reported cost decreased"
            )
        args.time_budget = min(args.time_budget, remaining)
    deadline = time.monotonic() + args.time_budget
    if args.mode == "report":
        if not args.runs:
            parser.error("report requires --runs")
        report = _summary(args, config)
        _write(args.output_dir, report)
        manifest = {
            "candidate_commit": report["candidate_commit"],
            "config_sha256": _hash(args.config),
            "development": not report["release_eligible"],
            "release_eligible": report["release_eligible"],
            "runs": [str(p.resolve()) for p in args.runs],
        }
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
    else:
        report = {"status": "blocked", "cases": []}
        args.output_dir.mkdir(parents=True, exist_ok=False)
        try:
            environment = _start(args, policy)
            if args.mode == "paired":
                report = _paired(args, policy, deadline)
            else:
                if any(name not in config["cases"] for name in args.cases) or len(
                    set(args.cases)
                ) != len(args.cases):
                    raise ValueError("Unknown or duplicate case IDs")
                cases = [
                    _run_case(
                        name,
                        config["cases"][name],
                        args,
                        args.output_dir,
                        deadline,
                        environment,
                    )
                    for name in args.cases
                ]
                status = (
                    "fail"
                    if any(c["status"] == "fail" for c in cases)
                    else "blocked"
                    if any(c["status"] != "pass" for c in cases)
                    else "pass"
                )
                report = {
                    "status": status,
                    "cases": cases,
                    "scope": "Selected profiles only; use report to identify remaining release groups",
                }
        except (
            ValueError,
            RuntimeError,
            OSError,
            KeyError,
            AssertionError,
            subprocess.CalledProcessError,
        ) as error:
            report.update(status="blocked", reason=str(error))
            (args.output_dir / "error.log").write_text(traceback.format_exc())
        manifest = (
            _read(args.output_dir / "manifest.json")
            if (args.output_dir / "manifest.json").is_file()
            else {}
        )
        changed = _git(_ROOT, "rev-parse", "HEAD") != manifest.get(
            "candidate_commit"
        ) or (
            hashlib.sha256(_git(_ROOT, "diff", "HEAD", "--binary").encode()).hexdigest()
            != manifest.get("source_diff_sha256")
        )
        if changed:
            manifest["release_eligible"] = False
            report.update(
                status="blocked",
                reason="Source changed during the run; evidence is not release-eligible",
            )
        manifest["ended_utc"] = datetime.now(timezone.utc).isoformat()
        (args.output_dir / "manifest.json").write_text(
            json.dumps(manifest, indent=2) + "\n"
        )
        _write(args.output_dir, report)
    print(json.dumps(report, indent=2, allow_nan=False))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
