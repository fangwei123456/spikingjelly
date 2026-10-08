import math

import pytest
import torch

from spikingjelly.activation_based import functional


def test_package_imports_without_cupy_and_retires_dense_spike_apis():
    import subprocess
    import sys
    import textwrap

    code = textwrap.dedent("""
        import importlib.abc
        import sys
        class BlockCuPy(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname.split('.')[0] in ('cupy', 'cupy_backends'):
                    raise AssertionError('package attempted a CuPy import')
        sys.meta_path.insert(0, BlockCuPy())
        from spikingjelly.activation_based import functional, layer, memopt
        from spikingjelly._ops.if_linear import if_linear
        from spikingjelly._ops.lif_linear import lif_linear
        from spikingjelly._ops.spike_linear.sparse import sparse_linear
        assert not hasattr(functional, 'spike_linear')
        assert not hasattr(functional, 'spike_conv2d')
        assert not hasattr(layer, 'SpikeLinear')
        assert not hasattr(layer, 'SpikeConv2d')
        assert 'cupy' not in sys.modules
    """)
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=60
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.fixture(params=["cpu", "cuda"], scope="module")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return torch.device(request.param)


@pytest.mark.parametrize(
    "shape", [(), (0,), (1,), (7,), (8,), (9,), (2, 0, 3), (2, 3, 7)]
)
@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.float32, torch.float16, torch.bfloat16]
)
def test_binary_pack_roundtrip_and_format(device, shape, dtype):
    x = (torch.arange(math.prod(shape), device=device) % 2).reshape(shape).to(dtype)
    if x.ndim > 1:
        x = x.transpose(0, -1)
    packed = functional.bit_spike_compress(x)
    assert packed.dtype == torch.uint8 and packed.ndim == 1
    expected = torch.zeros((x.numel() + 7) // 8, device=device, dtype=torch.uint8)
    flat = x.flatten().to(torch.uint8)
    for bit in range(8):
        part = flat[bit::8]
        expected[: part.numel()] |= part << bit
    torch.testing.assert_close(packed, expected, rtol=0, atol=0)
    decoded = functional.bit_spike_decompress(packed, tuple(x.shape), dtype)
    torch.testing.assert_close(decoded, x, rtol=0, atol=0)
