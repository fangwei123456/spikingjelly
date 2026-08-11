"""Tests for the experimental CUDA SpikeLinear kernels.

The public ``sparse_linear`` API exposes only the torch and sparse strategies.
The slower v3 kernel remains available as a low-level custom op that requires
a pre-packed uint8 spike tensor.
"""

import pytest
import torch

try:
    __import__("cupy")
    _HAS_CUPY = True
except (ImportError, OSError):
    _HAS_CUPY = False

from spikingjelly.activation_based.cuda_kernel.spike_linear import (
    bit_pack_spike_dense,
    cupy_spike_linear_sparse_forward,
    cupy_spike_linear_v3_dense_forward,
    sparse_linear,
)


pytestmark = pytest.mark.skipif(
    not _HAS_CUPY or not torch.cuda.is_available(),
    reason="requires cupy and CUDA",
)

_DTYPES = (torch.float32, torch.float16, torch.bfloat16)


def _assert_close(actual, expected):
    rtol, atol = {
        torch.float32: (1e-4, 1e-5),
        torch.float16: (2e-3, 2e-3),
        torch.bfloat16: (2e-2, 2e-2),
    }[expected.dtype]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("density", [0.02, 0.05, 0.10])
@pytest.mark.parametrize("M,K,N", [(64, 128, 64), (256, 512, 256), (512, 1024, 512)])
def test_sparse_linear_matches_dense(density, M, K, N):
    torch.manual_seed(0)
    s = (torch.rand(M, K, device="cuda") < density).float()
    W = torch.randn(N, K, device="cuda")
    y_ref = torch.nn.functional.linear(s, W)
    y_test = sparse_linear(s, W, strategy="sparse")
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_linear_with_bias(dtype):
    """Mixed empty/nonempty rows must be initialized before adding bias."""
    torch.manual_seed(0)
    M, K, N = 8, 128, 32
    s = torch.zeros(M, K, dtype=dtype, device="cuda")
    s[1, ::17] = 1.0
    s[6, ::11] = 1.0
    W = torch.randn(N, K, dtype=dtype, device="cuda")
    b = torch.randn(N, dtype=dtype, device="cuda")
    y_ref = torch.nn.functional.linear(s, W, b)
    y_test = sparse_linear(s, W, b, strategy="sparse")
    assert y_test.dtype == dtype
    _assert_close(y_test, y_ref)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_sparse_linear_backward(dtype):
    """Input, weight, and bias gradients must match F.linear."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).to(dtype).requires_grad_()
    W = torch.randn(N, K, dtype=dtype, device="cuda", requires_grad=True)
    b = torch.randn(N, dtype=dtype, device="cuda", requires_grad=True)
    s_ref = s.detach().clone().requires_grad_()
    W_ref = W.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    grad_output = torch.randn(M, N, dtype=dtype, device="cuda")

    sparse_linear(s, W, b, strategy="sparse").backward(grad_output)
    torch.nn.functional.linear(s_ref, W_ref, b_ref).backward(grad_output)

    _assert_close(s.grad, s_ref.grad)
    _assert_close(W.grad, W_ref.grad)
    _assert_close(b.grad, b_ref.grad)


def test_unknown_strategy_raises():
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.1).float()
    W = torch.randn(N, K, device="cuda")
    with pytest.raises(ValueError, match="Unknown strategy"):
        sparse_linear(s, W, strategy="auto")
    with pytest.raises(ValueError, match="Unknown strategy"):
        sparse_linear(s, W, strategy="v3_dense")
    with pytest.raises(ValueError, match="Unknown strategy"):
        sparse_linear(s, W, strategy="bogus")


def test_custom_ops_validate_input_contracts():
    s = torch.zeros(4, 16, device="cuda")
    W = torch.zeros(8, 16, device="cuda")

    with pytest.raises(TypeError, match="float32, float16, or bfloat16"):
        bit_pack_spike_dense(torch.zeros(4, 16, dtype=torch.int32, device="cuda"))
    with pytest.raises(TypeError, match="same dtype"):
        cupy_spike_linear_sparse_forward(s.half(), W, None)
    with pytest.raises(TypeError, match="float32, float16, or bfloat16"):
        cupy_spike_linear_v3_dense_forward(
            torch.zeros(4, 2, dtype=torch.uint8, device="cuda"),
            W.to(torch.int32),
            None,
        )
    with pytest.raises(ValueError, match=r"weight\.shape"):
        cupy_spike_linear_sparse_forward(s, W[:, :-1].contiguous(), None)
    with pytest.raises(ValueError, match="contiguous"):
        cupy_spike_linear_sparse_forward(s[:, ::2], W[:, ::2], None)
    with pytest.raises(ValueError, match=r"packed\.shape"):
        cupy_spike_linear_v3_dense_forward(
            torch.zeros(4, 1, dtype=torch.uint8, device="cuda"), W, None
        )


def test_nondefault_stream():
    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream):
        s = (torch.rand(64, 128, device="cuda") < 0.05).float()
        W = torch.randn(32, 128, device="cuda")
        y = sparse_linear(s, W, strategy="sparse")
        y_ref = torch.nn.functional.linear(s, W)
    torch.cuda.current_stream().wait_stream(stream)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_noncurrent_cuda_device(dtype):
    with torch.cuda.device(1):
        s = (torch.rand(32, 64, device="cuda:1") < 0.05).to(dtype)
        W = torch.randn(16, 64, dtype=dtype, device="cuda:1", requires_grad=True)
        y = sparse_linear(s, W, strategy="sparse")
        y_ref = torch.nn.functional.linear(s, W)
        y.sum().backward()
    assert y.device == torch.device("cuda:1")
    assert W.grad.device == torch.device("cuda:1")
    _assert_close(y, y_ref)

    W_other = torch.randn(16, 64, dtype=dtype, device="cuda:0")
    with pytest.raises(ValueError, match="same CUDA device"):
        cupy_spike_linear_sparse_forward(s, W_other, None)


@pytest.mark.parametrize("dtype", _DTYPES)
@pytest.mark.parametrize("M,K,N", [(0, 8, 4), (4, 8, 0), (4, 0, 8)])
def test_zero_dimensions(dtype, M, K, N):
    s = torch.empty(M, K, dtype=dtype, device="cuda")
    W = torch.empty(N, K, dtype=dtype, device="cuda")
    y = sparse_linear(s, W, strategy="sparse")
    y_ref = torch.nn.functional.linear(s, W)
    torch.testing.assert_close(y, y_ref, rtol=0, atol=0)


def test_flattened_grid_exceeds_legacy_y_limit():
    N = 256 * 65_535 + 1
    s = torch.empty(1, 0, device="cuda")
    W = torch.empty(N, 0, device="cuda")
    y = cupy_spike_linear_sparse_forward(s, W, None)
    assert y.shape == (1, N)
    assert torch.count_nonzero(y).item() == 0

    M = 64 * 65_535 + 1
    packed = torch.zeros(M, 1, dtype=torch.uint8, device="cuda")
    W = torch.zeros(1, 1, device="cuda")
    y = cupy_spike_linear_v3_dense_forward(packed, W, None)
    assert y.shape == (M, 1)
    assert torch.count_nonzero(y).item() == 0


@pytest.mark.parametrize("strategy", ["torch", "sparse"])
def test_sparse_linear_explicit_strategies(strategy):
    torch.manual_seed(0)
    M, K, N = 128, 256, 128
    s = (torch.rand(M, K, device="cuda") < 0.1).float()
    W = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    y_ref = torch.nn.functional.linear(s, W, b)
    y_test = sparse_linear(s, W, b, strategy=strategy)
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-5)


def test_default_strategy_is_torch():
    torch.manual_seed(0)
    s = (torch.rand(128, 256, device="cuda") < 0.1).float()
    W = torch.randn(128, 256, device="cuda")
    y_default = sparse_linear(s, W)
    y_ref = torch.nn.functional.linear(s, W)
    assert torch.equal(y_default, y_ref)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_v3_takes_prepacked_input(dtype):
    torch.manual_seed(0)
    M, K, N = 64, 130, 64
    s = (torch.rand(M, K, device="cuda") < 0.1).to(dtype)
    W = torch.randn(N, K, dtype=dtype, device="cuda", requires_grad=True)
    b = torch.randn(N, dtype=dtype, device="cuda", requires_grad=True)
    W_ref = W.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    packed = bit_pack_spike_dense(s)
    y_ref = torch.nn.functional.linear(s, W_ref, b_ref)
    y_v3 = cupy_spike_linear_v3_dense_forward(packed, W, b)
    assert y_v3.dtype == dtype
    _assert_close(y_v3, y_ref)

    grad_output = torch.randn(M, N, dtype=dtype, device="cuda")
    y_v3.backward(grad_output)
    y_ref.backward(grad_output)
    _assert_close(W.grad, W_ref.grad)
    _assert_close(b.grad, b_ref.grad)


def test_v3_user_packs_repeated_calls():
    torch.manual_seed(0)
    M, K = 1024, 1024
    s = (torch.rand(M, K, device="cuda") < 0.05).float()
    W1 = torch.randn(M, K, device="cuda")
    W2 = torch.randn(M, K, device="cuda")
    y_ref1 = torch.nn.functional.linear(s, W1)
    y_ref2 = torch.nn.functional.linear(s, W2)

    packed = bit_pack_spike_dense(s)
    y1 = cupy_spike_linear_v3_dense_forward(packed, W1, None)
    y2 = cupy_spike_linear_v3_dense_forward(packed, W2, None)
    torch.testing.assert_close(y1, y_ref1, rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(y2, y_ref2, rtol=1e-4, atol=1e-5)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_bit_pack_roundtrip(dtype):
    torch.manual_seed(0)
    M, K = 64, 130  # 130 not a multiple of 8
    s = (torch.rand(M, K, device="cuda") < 0.3).to(dtype)
    s[0, 0] = -0.0
    packed = bit_pack_spike_dense(s)
    # Unpack manually
    bits = (
        packed.unsqueeze(-1) >> torch.arange(8, dtype=torch.uint8, device="cuda")
    ) & 1
    bits = bits.reshape(M, packed.shape[1] * 8)[:, :K]
    s_back = bits.to(dtype)
    assert torch.equal(s, s_back)


@pytest.mark.parametrize("dtype", _DTYPES)
def test_fake_tensor_shape(dtype):
    from torch._subclasses.fake_tensor import FakeTensorMode

    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    # Build meta tensors to exercise both fake implementations.
    s_meta = torch.randn(M, K, dtype=dtype, device="meta")
    packed_meta = torch.empty(M, (K + 7) // 8, dtype=torch.uint8, device="meta")
    W_meta = torch.randn(N, K, dtype=dtype, device="meta")
    y_meta = torch.ops.sj.cupy_spike_linear_v3_dense_forward(packed_meta, W_meta, None)
    assert y_meta.shape == (M, N)
    assert y_meta.dtype == dtype
    y_meta = torch.ops.sj.cupy_spike_linear_sparse_forward(s_meta, W_meta, None)
    assert y_meta.shape == (M, N)
    assert y_meta.dtype == dtype

    with FakeTensorMode():
        spike_cuda0 = torch.empty(M, K, dtype=dtype, device="cuda:0")
        packed_cuda0 = torch.empty(M, (K + 7) // 8, dtype=torch.uint8, device="cuda:0")
        weight_cuda0 = torch.empty(N, K, dtype=dtype, device="cuda:0")
        weight_cpu = torch.empty(N, K, dtype=dtype, device="cpu")
        bias_cpu = torch.empty(N, dtype=dtype, device="cpu")
        with pytest.raises(ValueError, match="same device"):
            torch.ops.sj.cupy_spike_linear_sparse_forward(spike_cuda0, weight_cpu, None)
        with pytest.raises(ValueError, match="same device"):
            torch.ops.sj.cupy_spike_linear_v3_dense_forward(
                packed_cuda0, weight_cpu, None
            )
        with pytest.raises(ValueError, match="same device"):
            torch.ops.sj.cupy_spike_linear_sparse_forward(
                spike_cuda0, weight_cuda0, bias_cpu
            )

    with pytest.raises(TypeError, match="float32, float16, or bfloat16"):
        torch.ops.sj.cupy_spike_linear_sparse_forward(
            torch.empty(M, K, dtype=torch.int32, device="meta"),
            torch.empty(N, K, dtype=torch.int32, device="meta"),
            None,
        )


@pytest.mark.parametrize("dtype", _DTYPES)
def test_v3_via_torch_compile(dtype):
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).to(dtype)
    packed = bit_pack_spike_dense(s)
    W = torch.randn(N, K, dtype=dtype, device="cuda")

    def f(packed, W):
        return cupy_spike_linear_v3_dense_forward(packed, W, None)

    explanation = torch._dynamo.explain(f)(packed, W)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    compiled = torch.compile(f, fullgraph=True, dynamic=False)
    y = compiled(packed, W)
    _assert_close(y, torch.nn.functional.linear(s, W))


@pytest.mark.parametrize("dtype", _DTYPES)
def test_via_torch_compile(dtype):
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).to(dtype)
    W = torch.randn(N, K, dtype=dtype, device="cuda")

    def f(s, W):
        return sparse_linear(s, W, strategy="sparse")

    explanation = torch._dynamo.explain(f)(s, W)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    compiled = torch.compile(f, fullgraph=True, dynamic=False)
    y = compiled(s, W)
    y_ref = torch.nn.functional.linear(s, W)
    _assert_close(y, y_ref)
