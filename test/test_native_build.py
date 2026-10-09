import runpy
import sys
from pathlib import Path

import pytest
import setuptools
import torch
from torch.utils import cpp_extension


@pytest.fixture
def native_build(monkeypatch):
    monkeypatch.delenv("SJ_BUILD_NATIVE_CUDA", raising=False)
    monkeypatch.setattr(setuptools, "setup", lambda **kwargs: None)
    namespace = runpy.run_path(str(Path(__file__).parents[1] / "setup.py"))
    return namespace["_native_extensions"]


def test_default_build_does_not_require_torch(native_build, monkeypatch):
    monkeypatch.setitem(sys.modules, "torch", None)
    assert native_build() == ([], {})


@pytest.mark.parametrize(
    "missing, message",
    [
        ("torch", "--no-build-isolation"),
        ("cuda", "CUDA-enabled PyTorch"),
        ("nvcc", "nvcc"),
        ("compiler", "C\\+\\+ compiler"),
        ("architectures", "TORCH_CUDA_ARCH_LIST"),
    ],
)
def test_explicit_native_build_requires_prerequisites(
    native_build, monkeypatch, tmp_path, missing, message
):
    monkeypatch.setenv("SJ_BUILD_NATIVE_CUDA", "1")
    monkeypatch.setenv("TORCH_CUDA_ARCH_LIST", "8.0")
    monkeypatch.setenv("CXX", "c++")
    monkeypatch.setattr(torch.version, "cuda", "12.8")
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(cpp_extension, "CUDA_HOME", str(tmp_path))
    monkeypatch.setattr("shutil.which", lambda executable: "/usr/bin/c++")
    nvcc = tmp_path / "bin" / ("nvcc.exe" if sys.platform == "win32" else "nvcc")
    nvcc.parent.mkdir()
    nvcc.touch()
    if missing == "torch":
        monkeypatch.setitem(sys.modules, "torch", None)
    elif missing == "cuda":
        monkeypatch.setattr(torch.version, "cuda", None)
    elif missing == "nvcc":
        nvcc.unlink()
    elif missing == "compiler":
        monkeypatch.setattr("shutil.which", lambda executable: None)
    else:
        monkeypatch.delenv("TORCH_CUDA_ARCH_LIST")
    with pytest.raises(RuntimeError, match=message):
        native_build()
