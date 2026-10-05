import os
import subprocess
import sys
import textwrap

import pytest
import torch


_CUDA = pytest.mark.skipif(
    not torch.cuda.is_available() or bool(torch.version.hip),
    reason="NVIDIA CUDA unavailable",
)

_EXERCISE = """
def exercise(device):
    x = torch.ones(2, 3, device=device, requires_grad=True)
    v = torch.zeros(3, device=device, requires_grad=True)
    result = lif(x, v, threshold=0.7)
    reference_x = x.detach().cpu().requires_grad_()
    reference_v = v.detach().cpu().requires_grad_()
    reference = lif(reference_x, reference_v, threshold=0.7)
    assert not result[2].requires_grad
    for actual, expected in zip(result, reference):
        torch.testing.assert_close(actual.cpu(), expected)
    actual_grad = torch.autograd.grad(result[0].sum() + result[1].sum(), (x, v))
    expected_grad = torch.autograd.grad(
        reference[0].sum() + reference[1].sum(), (reference_x, reference_v)
    )
    for actual, expected in zip(actual_grad, expected_grad):
        torch.testing.assert_close(actual.cpu(), expected)
"""


def _run(script, *, implementation="auto", missing=()):
    env = os.environ.copy()
    env["SJ_LIF_CUDA_IMPLEMENTATION"] = implementation
    source = (
        "import os, sys\nimport torch\n"
        + f"for name in {missing!r}:\n    sys.modules[name] = None\n"
        + "from spikingjelly._ops.lif import get_cuda_implementation, lif\n"
        + _EXERCISE
        + textwrap.dedent(script)
    )
    result = subprocess.run(
        [sys.executable, "-c", source],
        env=env,
        capture_output=True,
        text=True,
        timeout=180,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cpu_ignores_cuda_configuration_and_missing_dependencies():
    _run(
        """
        exercise(torch.device("cpu"))
        try:
            get_cuda_implementation(torch.device("cpu"))
        except ValueError as error:
            assert "CUDA" in str(error)
        else:
            raise AssertionError("CPU is not a CUDA diagnostic target")
        """,
        implementation="invalid",
        missing=("spikingjelly._ops.lif._C", "triton", "cupy"),
    )


def test_invalid_cuda_configuration_reports_allowed_choices():
    _run(
        """
        try:
            get_cuda_implementation(torch.device("cuda", 0))
        except ValueError as error:
            message = str(error)
            assert "SJ_LIF_CUDA_IMPLEMENTATION" in message
            assert all(name in message for name in ("auto", "cuda", "triton", "cupy"))
        else:
            raise AssertionError("invalid CUDA configuration was accepted")
        """,
        implementation="invalid",
    )


@_CUDA
@pytest.mark.parametrize(
    "missing,expected,unavailable",
    [
        ((), "cuda", ()),
        (("spikingjelly._ops.lif._C",), "triton", ("cuda",)),
        (("spikingjelly._ops.lif._C", "triton"), "cupy", ("cuda", "triton")),
    ],
)
def test_auto_priority_executes_real_implementation(missing, expected, unavailable):
    _run(
        f"""
        device = torch.device("cuda", 0)
        report = get_cuda_implementation(device)
        assert report["implementation"] == {expected!r}, report
        assert set(report["unavailable"]) == set({unavailable!r}), report
        assert all(report["unavailable"].values())
        # Restoring optional imports does not change an already bound device.
        for name in {missing!r}:
            sys.modules.pop(name)
        exercise(device)
        assert get_cuda_implementation(device) == report
        """,
        missing=missing,
    )


@_CUDA
def test_all_unavailable_reports_every_candidate():
    _run(
        """
        try:
            get_cuda_implementation(torch.device("cuda", 0))
        except RuntimeError as error:
            message = str(error)
            assert "No available" in message
            assert all(name + ":" in message for name in ("cuda", "triton", "cupy"))
        else:
            raise AssertionError("missing CUDA implementations were accepted")
        """,
        missing=("spikingjelly._ops.lif._C", "triton", "cupy"),
    )


@_CUDA
def test_forced_unavailable_implementation_does_not_fall_back():
    _run(
        """
        try:
            get_cuda_implementation(torch.device("cuda", 0))
        except RuntimeError as error:
            message = str(error)
            assert "cuda:" in message
            assert "triton:" not in message and "cupy:" not in message
        else:
            raise AssertionError("forced unavailable CUDA silently fell back")
        exercise(torch.device("cpu"))
        """,
        implementation="cuda",
        missing=("spikingjelly._ops.lif._C",),
    )


@_CUDA
def test_selection_survives_environment_changes_and_execution_errors():
    _run(
        """
        # Configuration belongs to the imported runtime, not each forward call.
        os.environ["SJ_LIF_CUDA_IMPLEMENTATION"] = "invalid"
        for index in range(min(torch.cuda.device_count(), 2)):
            device = torch.device("cuda", index)
            report = get_cuda_implementation(device)
            assert report["implementation"] == "cuda", report
            exercise(device)
            try:
                lif(torch.ones(2, 3, device=device, dtype=torch.float16),
                    torch.zeros(3, device=device, dtype=torch.float16))
            except RuntimeError:
                pass
            else:
                raise AssertionError("invalid input was accepted")
            assert get_cuda_implementation(device) == report
            exercise(device)
            # Diagnostics are snapshots, not a mutable handle to runtime state.
            report["unavailable"]["test"] = "not a real failure"
            assert "test" not in get_cuda_implementation(device)["unavailable"]
        """
    )
