import json
import os
import subprocess
import sys

import pytest


ENV_NAMES = (
    "SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS",
    "SJ_CUDA_THREADS",
    "SJ_CUDA_COMPILER_OPTIONS",
    "SJ_CUDA_COMPILER_BACKEND",
    "SJ_SAVE_DATASETS_COMPRESSED",
    "SJ_SAVE_SPIKE_AS_BOOL_IN_NEURON_KERNEL",
    "SJ_SAVE_BOOL_SPIKE_LEVEL",
    "SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T",
)

PRINT_CONFIG = """
import json
from spikingjelly import configure

print(json.dumps({
    "dataset_threads": configure.max_threads_number_for_datasets_preprocess,
    "cuda_threads": configure.cuda_threads,
    "compiler_options": configure.cuda_compiler_options,
    "compiler_backend": configure.cuda_compiler_backend,
    "compressed": configure.save_datasets_compressed,
    "bool_spike": configure.save_spike_as_bool_in_neuron_kernel,
    "bool_spike_level": configure.save_bool_spike_level,
    "triton_static_T": configure.triton_neuron_kernel_static_range_max_T,
}))
"""


def run_config(overrides, code=PRINT_CONFIG):
    env = os.environ.copy()
    for name in ENV_NAMES:
        env.pop(name, None)
    env.update(overrides)
    return subprocess.run(
        [sys.executable, "-c", code],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )


def test_defaults_and_environment_overrides():
    defaults = run_config({})
    assert defaults.returncode == 0, defaults.stderr
    assert json.loads(defaults.stdout) == {
        "dataset_threads": 16,
        "cuda_threads": 512,
        "compiler_options": ["-use_fast_math"],
        "compiler_backend": "nvrtc",
        "compressed": True,
        "bool_spike": False,
        "bool_spike_level": 0,
        "triton_static_T": 64,
    }

    configured = run_config(
        {
            "SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS": "4",
            "SJ_CUDA_THREADS": "256",
            "SJ_CUDA_COMPILER_OPTIONS": "-lineinfo, --device-debug",
            "SJ_CUDA_COMPILER_BACKEND": "nvcc",
            "SJ_SAVE_DATASETS_COMPRESSED": "0",
            "SJ_SAVE_SPIKE_AS_BOOL_IN_NEURON_KERNEL": "1",
            "SJ_SAVE_BOOL_SPIKE_LEVEL": "1",
            "SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T": "32",
        }
    )
    assert configured.returncode == 0, configured.stderr
    assert json.loads(configured.stdout) == {
        "dataset_threads": 4,
        "cuda_threads": 256,
        "compiler_options": ["-lineinfo", "--device-debug"],
        "compiler_backend": "nvcc",
        "compressed": False,
        "bool_spike": True,
        "bool_spike_level": 1,
        "triton_static_T": 32,
    }


def test_environment_is_read_once():
    result = run_config(
        {"SJ_CUDA_THREADS": "256"},
        """
import os
from spikingjelly import configure

os.environ["SJ_CUDA_THREADS"] = "128"
print(configure.cuda_threads)
""",
    )
    assert result.returncode == 0, result.stderr
    assert result.stdout.strip() == "256"


@pytest.mark.parametrize("value", ["invalid"])
def test_boolean_environment_requires_zero_or_one(value):
    result = run_config({"SJ_SAVE_DATASETS_COMPRESSED": value})
    assert result.returncode != 0
    assert "SJ_SAVE_DATASETS_COMPRESSED must be '0' or '1'" in result.stderr


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("SJ_CUDA_THREADS", "invalid"),
        ("SJ_CUDA_THREADS", "0"),
        ("SJ_MAX_THREADS_NUMBER_FOR_DATASETS_PREPROCESS", "-1"),
        ("SJ_TRITON_NEURON_KERNEL_STATIC_RANGE_MAX_T", "0"),
        ("SJ_CUDA_COMPILER_BACKEND", "clang"),
        ("SJ_SAVE_BOOL_SPIKE_LEVEL", "2"),
    ],
)
def test_invalid_environment_value_fails_at_import(name, value):
    result = run_config({name: value})
    assert result.returncode != 0
    assert name in result.stderr
    assert value in result.stderr
