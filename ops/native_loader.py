import importlib.util
import json
import re
from pathlib import Path

import torch

from . import _NATIVE_ABI


def _load_native(package: str) -> dict:
    spec = importlib.util.find_spec(package + "._C")
    if spec is None:
        raise ImportError(
            f"Native operators for {package} were not built. Install from source with "
            "SJ_BUILD_NATIVE_CUDA=1 and --no-build-isolation, using a matching CUDA toolkit."
        )
    metadata = Path(spec.origin).with_name("_native_build.json")
    if not metadata.is_file():
        raise ImportError(
            "Native operator build metadata is missing; rebuild the extension"
        )
    info = json.loads(metadata.read_text())
    if info.get("operator_abi") != _NATIVE_ABI:
        raise ImportError("Native operator schema changed; rebuild the extensions")
    if (
        info["torch_version"] != str(torch.__version__)
        or info["cuda_version"] != torch.version.cuda
    ):
        raise ImportError(
            f"Native operators were built with Torch {info['torch_version']} / "
            f"CUDA {info['cuda_version']}, but runtime is {torch.__version__} / "
            f"{torch.version.cuda}; rebuild the extension"
        )
    torch.ops.load_library(spec.origin)
    return info


def _check_native_device(info: dict, index: int) -> None:
    major, minor = torch.cuda.get_device_capability(index)
    capability = major * 10 + minor
    targets = re.findall(
        r"code=(sm|compute)_(\d+)(?![\da-z])", " ".join(info["cuda_arch_flags"])
    )
    for kind, target in targets:
        target = int(target)
        if kind == "compute" and capability >= target:
            return
        if kind == "sm" and capability // 10 == target // 10 and capability >= target:
            return
    raise ImportError(
        f"Native operators have no supported binary/PTX target for sm_{capability}; "
        "rebuild with TORCH_CUDA_ARCH_LIST set for this GPU"
    )
