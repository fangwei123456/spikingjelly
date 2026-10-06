"""Optional PyTorch CUDA build hook; static configuration is in pyproject.toml."""

import json
import os
import runpy
import shutil
import warnings
from pathlib import Path

from setuptools import setup


def _native_extensions():
    if os.environ.get("SJ_BUILD_NATIVE_CUDA") != "1":
        return [], {}

    try:
        import torch
        from torch.utils.cpp_extension import (
            CUDA_HOME,
            BuildExtension,
            CUDAExtension,
            _get_cuda_arch_flags,
        )
    except ImportError as error:
        warnings.warn(
            f"Skipping SpikingJelly native CUDA extensions: {error}", stacklevel=2
        )
        return [], {}

    nvcc = Path(CUDA_HOME or "") / "bin" / ("nvcc.exe" if os.name == "nt" else "nvcc")
    compiler = os.environ.get("CXX", "cl" if os.name == "nt" else "c++")
    if torch.version.cuda is None or not CUDA_HOME or not nvcc.is_file():
        warnings.warn(
            "Skipping SpikingJelly native CUDA extensions: CUDA-enabled PyTorch and a CUDA "
            "toolkit with nvcc are required; set CUDA_HOME to the toolkit.",
            stacklevel=2,
        )
        return [], {}
    if shutil.which(compiler) is None:
        warnings.warn(
            f"Skipping SpikingJelly native CUDA extensions: C++ compiler {compiler!r} not found.",
            stacklevel=2,
        )
        return [], {}
    if not os.environ.get("TORCH_CUDA_ARCH_LIST") and not torch.cuda.is_available():
        warnings.warn(
            "Skipping SpikingJelly native CUDA extensions: set TORCH_CUDA_ARCH_LIST when "
            "building without a visible CUDA device.",
            stacklevel=2,
        )
        return [], {}

    arch_flags = _get_cuda_arch_flags()
    metadata = {
        "torch_version": str(torch.__version__),
        "operator_abi": runpy.run_path(str(Path(__file__).parent / "ops/__init__.py"))[
            "_NATIVE_ABI"
        ],
        "cuda_version": torch.version.cuda,
        "cuda_arch_flags": arch_flags,
        "torch_cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
    }

    class NativeBuildExtension(BuildExtension):
        def run(self):
            super().run()
            for extension in self.extensions:
                target = Path(self.get_ext_fullpath(extension.name))
                target.with_name("_native_build.json").write_text(
                    json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
                )

    return [
        CUDAExtension(
            f"spikingjelly._ops.{source.parent.name}._C",
            sources=[str(source)],
            depends=[
                str(source.with_name("kernels.cuh")),
                "ops/_cuda.cuh",
                "ops/cuda_surrogate.cuh",
            ],
            extra_compile_args={"nvcc": ["-O3", "--fmad=false", *arch_flags]},
        )
        for source in sorted(Path("ops").glob("*/native.cu"))
    ], {"build_ext": NativeBuildExtension}


ext_modules, cmdclass = _native_extensions()
setup(ext_modules=ext_modules, cmdclass=cmdclass)
