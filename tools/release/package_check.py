"""Reusable distribution, import and installed-training checks for CI/releases."""

import argparse
import hashlib
import importlib.util
import json
import os
import shutil
from pathlib import Path
import sys
import tarfile
import zipfile


def _artifacts(wheel, sdist):
    expected = sorted(
        p.parent.name
        for p in Path(__file__).resolve().parents[2].glob("ops/*/native.cu")
    )
    if not wheel.name.endswith("-py3-none-any.whl"):
        raise ValueError("The normal wheel must be pure Python")
    with tarfile.open(sdist) as archive:
        names = archive.getnames()
        roots = {name.split("/")[0] for name in names}
        if len(roots) != 1:
            raise ValueError("sdist must have one top-level directory")
        prefix = next(iter(roots)) + "/"
        models = sorted(
            Path(name).parent.name
            for name in names
            if name.startswith(prefix + "ops/") and name.endswith("/native.cu")
        )
        if not models or models != expected:
            raise ValueError("sdist native sources differ from the source checkout")
        required = {prefix + "pyproject.toml", prefix + "setup.py"}
        required.update(prefix + f"ops/{name}/kernels.cuh" for name in models)
        if not required.issubset(names):
            raise ValueError("sdist is missing build inputs")
    with zipfile.ZipFile(wheel) as archive:
        names = set(archive.namelist())
        required = {"spikingjelly/_ops/__init__.py", "spikingjelly/_ops/_cuda.cuh"}
        for name in models:
            required.update(
                {
                    f"spikingjelly/_ops/{name}/__init__.py",
                    f"spikingjelly/_ops/{name}/kernels.cuh",
                }
            )
        if not required.issubset(names):
            raise ValueError("wheel is missing operator modules/headers")
        if any(name.endswith((".so", ".pyd", "_native_build.json")) for name in names):
            raise ValueError("Pure wheel contains native build products")
    return {
        "native_families": models,
        "sha256": {
            str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (wheel, sdist)
        },
    }


def _imports(expected_root):
    import spikingjelly
    import spikingjelly._ops

    paths = [
        Path(spikingjelly.__file__).resolve(),
        Path(spikingjelly._ops.__file__).resolve(),
    ]
    if any(not p.is_relative_to(expected_root.resolve()) for p in paths):
        raise RuntimeError(f"Imported a different installation: {paths}")
    for name in ("lif", "if_", "plif"):
        if importlib.util.find_spec(f"spikingjelly._ops.{name}") is None:
            raise RuntimeError(f"Missing installed operator: {name}")
    if "torch" in sys.modules:
        raise RuntimeError("Basic package import eagerly loaded Torch")
    return {"import_paths": [str(p) for p in paths]}


def _provenance(wheel, expected_root):
    import importlib.metadata
    import spikingjelly

    package = Path(spikingjelly.__file__).resolve().parent
    if not package.is_relative_to(expected_root.resolve()):
        raise RuntimeError("The package is outside the selected installed environment")
    with zipfile.ZipFile(wheel) as archive:
        names = [
            n
            for n in archive.namelist()
            if n.startswith("spikingjelly/") and not n.endswith("/")
        ]
        for name in names:
            path = package / name.removeprefix("spikingjelly/")
            if not path.is_file() or path.read_bytes() != archive.read(name):
                raise RuntimeError(
                    f"Installed package differs from fixed wheel: {name}"
                )
        extensions = {str(p.relative_to(package)) for p in package.rglob("*.so")}
        expected = {n.removeprefix("spikingjelly/") for n in names if n.endswith(".so")}
        if extensions != expected:
            raise RuntimeError("Installed native extensions differ from fixed wheel")
    return {
        "package_version": importlib.metadata.version("spikingjelly"),
        "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
        "files_checked": len(names),
        "package_path": str(package),
    }


def _training(expected_root, device, provider, output):
    result = _imports(expected_root)
    import torch
    import torch.nn as nn
    from spikingjelly.activation_based import functional, layer, neuron, surrogate

    device = torch.device(device)
    if device.type == "cuda":
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable")
        if device.index is None:
            device = torch.device("cuda", torch.cuda.current_device())
        torch.cuda.set_device(device)
    elif device.type != "cpu":
        raise ValueError("Only CPU/CUDA are supported")
    torch.manual_seed(17)
    implementations = {}
    for kind, cell in (
        ("if", neuron.IFNode),
        ("lif", neuron.LIFNode),
        ("plif", neuron.ParametricLIFNode),
    ):
        os.environ[f"SJ_{kind.upper()}_CUDA_IMPLEMENTATION"] = provider
        net = nn.Sequential(
            layer.Linear(32, 16, step_mode="m"),
            cell(step_mode="m"),
            layer.Linear(16, 4, step_mode="m"),
        ).to(device)
        x = torch.randn(4, 3, 32, device=device)
        before = net[0].weight.detach().clone()
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1)
        logits = net(x).mean(0)
        if logits.shape != (3, 4) or not torch.isfinite(logits).all():
            raise RuntimeError("Invalid installed forward output")
        nn.functional.cross_entropy(
            logits, torch.tensor([0, 1, 2], device=device)
        ).backward()
        if any(
            p.grad is None or not torch.isfinite(p.grad).all() for p in net.parameters()
        ):
            raise RuntimeError("Missing or non-finite installed parameter gradient")
        optimizer.step()
        if torch.equal(before, net[0].weight):
            raise RuntimeError("Installed training did not update weights")
        functional.reset_net(net)
        if not isinstance(net[1].v, float) or net[1].v != 0.0:
            raise RuntimeError("Installed neuron reset failed")
        net.eval()
        with torch.no_grad():
            expected = net(x).detach()
            functional.reset_net(net)
            torch.testing.assert_close(net(x), expected, rtol=0, atol=0)
        functional.reset_net(net)
        checkpoint = output / f"{kind}.pt"
        torch.save(
            {"model": net.state_dict(), "optimizer": optimizer.state_dict()}, checkpoint
        )
        state = torch.load(checkpoint, map_location=device, weights_only=True)
        clone = (
            nn.Sequential(
                layer.Linear(32, 16, step_mode="m"),
                cell(step_mode="m"),
                layer.Linear(16, 4, step_mode="m"),
            )
            .to(device)
            .eval()
        )
        clone.load_state_dict(state["model"])
        with torch.no_grad():
            torch.testing.assert_close(clone(x), expected, rtol=0, atol=0)
        info = functional.neuron_implementation(kind, device)
        if (
            device.type == "cuda"
            and provider != "auto"
            and info["implementation"] != provider
        ):
            raise RuntimeError(f"Wrong provider for {kind}: {info}")
        implementations[kind] = info

    # Check values and input/state gradients against the independent Torch recurrence.
    from spikingjelly._ops.lif.reference import multi_step

    x = torch.randn(4, 2, 16, device=device, requires_grad=True)
    v = torch.randn(2, 16, device=device, requires_grad=True)
    y, u = x.detach().clone().requires_grad_(), v.detach().clone().requires_grad_()
    actual = functional.lif_multi_step(x, v, tau=2.0, store_v_seq=True)
    expected = multi_step(y, u, 2.0, True, 1.0, 0.0, surrogate.Sigmoid(), False, True)
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)
    sum(z.sum() for z in actual).backward()
    sum(z.sum() for z in expected).backward()
    torch.testing.assert_close(x.grad, y.grad, rtol=1e-5, atol=1e-6)
    torch.testing.assert_close(v.grad, u.grad, rtol=1e-5, atol=1e-6)
    return {
        **result,
        "torch": str(torch.__version__),
        "cuda": torch.version.cuda,
        "device": str(device),
        "implementations": implementations,
        "checks": [
            "forward",
            "gradients",
            "optimizer",
            "reset",
            "checkpoint",
            "lif-reference",
        ],
    }


def _main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=("artifacts", "imports", "training", "prerequisites", "provenance"),
    )
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--sdist", type=Path)
    parser.add_argument("--expected-root", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--device", default="cpu")
    parser.add_argument(
        "--provider", choices=("auto", "torch", "triton", "cuda"), default="auto"
    )
    args = parser.parse_args()
    if args.mode == "provenance":
        if args.wheel is None or args.expected_root is None:
            parser.error("provenance requires --wheel and --expected-root")
        result = _provenance(args.wheel, args.expected_root)
    elif args.mode == "prerequisites":
        import torch
        from torch.utils.cpp_extension import CUDA_HOME

        compiler = os.environ.get("CXX", "c++")
        result = {
            "torch": str(torch.__version__),
            "cuda": torch.version.cuda,
            "gpu_available": torch.cuda.is_available(),
            "cuda_home": CUDA_HOME,
            "nvcc_available": bool(
                CUDA_HOME and (Path(CUDA_HOME) / "bin/nvcc").is_file()
            ),
            "compiler_available": bool(shutil.which(compiler)),
        }
        result["ready"] = all(
            result[k]
            for k in ("cuda", "gpu_available", "nvcc_available", "compiler_available")
        )
    elif args.mode == "artifacts":
        if args.wheel is None or args.sdist is None:
            parser.error("artifacts requires --wheel and --sdist")
        result = _artifacts(args.wheel, args.sdist)
    else:
        if args.expected_root is None:
            parser.error("imports/training requires --expected-root")
        if args.mode == "imports":
            result = _imports(args.expected_root)
        else:
            if args.output_dir is None:
                parser.error("training requires --output-dir")
            args.output_dir.mkdir(parents=True, exist_ok=False)
            result = _training(
                args.expected_root, args.device, args.provider, args.output_dir
            )
    print(json.dumps(result, indent=2, allow_nan=False))
    if args.mode == "prerequisites" and not result["ready"]:
        raise SystemExit(1)


if __name__ == "__main__":
    _main()
