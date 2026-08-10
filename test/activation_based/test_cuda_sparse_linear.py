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


@pytest.mark.parametrize("density", [0.02, 0.05, 0.10])
@pytest.mark.parametrize("M,K,N", [(64, 128, 64), (256, 512, 256), (512, 1024, 512)])
def test_sparse_linear_matches_dense(density, M, K, N):
    """sparse_linear must match F.linear for all density cells."""
    torch.manual_seed(0)
    s = (torch.rand(M, K, device="cuda") < density).float()
    W = torch.randn(N, K, device="cuda")
    y_ref = torch.nn.functional.linear(s, W)
    y_test = sparse_linear(s, W, strategy="sparse")
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-5)


def test_sparse_linear_with_bias():
    """Mixed empty/nonempty rows must be initialized before adding bias."""
    torch.manual_seed(0)
    M, K, N = 8, 128, 32
    s = torch.zeros(M, K, device="cuda")
    s[1, ::17] = 1.0
    s[6, ::11] = 1.0
    W = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    y_ref = torch.nn.functional.linear(s, W, b)
    y_test = sparse_linear(s, W, b, strategy="sparse")
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-5)


def test_sparse_linear_backward():
    """Input, weight, and bias gradients must match F.linear."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).float().requires_grad_()
    W = torch.randn(N, K, device="cuda", requires_grad=True)
    b = torch.randn(N, device="cuda", requires_grad=True)
    s_ref = s.detach().clone().requires_grad_()
    W_ref = W.detach().clone().requires_grad_()
    b_ref = b.detach().clone().requires_grad_()
    grad_output = torch.randn(M, N, device="cuda")

    sparse_linear(s, W, b, strategy="sparse").backward(grad_output)
    torch.nn.functional.linear(s_ref, W_ref, b_ref).backward(grad_output)

    torch.testing.assert_close(s.grad, s_ref.grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(W.grad, W_ref.grad, rtol=1e-3, atol=1e-4)
    torch.testing.assert_close(b.grad, b_ref.grad, rtol=1e-3, atol=1e-4)


def test_unknown_strategy_raises():
    """Unknown strategy should raise ValueError, not silently fall back."""
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
    with pytest.raises(TypeError, match="float32"):
        cupy_spike_linear_sparse_forward(s.half(), W, None)
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


@pytest.mark.skipif(torch.cuda.device_count() < 2, reason="requires two CUDA devices")
def test_noncurrent_cuda_device():
    with torch.cuda.device(1):
        s = (torch.rand(32, 64, device="cuda:1") < 0.05).float()
        W = torch.randn(16, 64, device="cuda:1", requires_grad=True)
        y = sparse_linear(s, W, strategy="sparse")
        y_ref = torch.nn.functional.linear(s, W)
        y.sum().backward()
    assert y.device == torch.device("cuda:1")
    assert W.grad.device == torch.device("cuda:1")
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-5)

    W_other = torch.randn(16, 64, device="cuda:0")
    with pytest.raises(ValueError, match="same CUDA device"):
        cupy_spike_linear_sparse_forward(s, W_other, None)


@pytest.mark.parametrize("M,K,N", [(0, 8, 4), (4, 8, 0), (4, 0, 8)])
def test_zero_dimensions(M, K, N):
    s = torch.empty(M, K, device="cuda")
    W = torch.empty(N, K, device="cuda")
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
    """Each explicit strategy must produce correct results."""
    torch.manual_seed(0)
    M, K, N = 128, 256, 128
    s = (torch.rand(M, K, device="cuda") < 0.1).float()
    W = torch.randn(N, K, device="cuda")
    b = torch.randn(N, device="cuda")
    y_ref = torch.nn.functional.linear(s, W, b)
    y_test = sparse_linear(s, W, b, strategy=strategy)
    torch.testing.assert_close(y_test, y_ref, rtol=1e-4, atol=1e-5)


def test_strategy_torch_skips_custom_op():
    """strategy='torch' should not call any custom CUDA kernel."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).float()
    W = torch.randn(N, K, device="cuda")

    # We can't easily intercept the custom op, but we can verify the
    # result is bit-exactly equal to F.linear (custom ops introduce
    # ~1e-6 float round-off in the bit-pack path).
    y_ref = torch.nn.functional.linear(s, W)
    y_test = sparse_linear(s, W, strategy="torch")
    err = (y_ref - y_test).abs().max().item()
    assert err == 0.0, f"strategy=torch should be bit-exact F.linear, err={err}"


def test_default_strategy_is_torch():
    """The safe default is the dense torch implementation."""
    torch.manual_seed(0)
    s = (torch.rand(128, 256, device="cuda") < 0.1).float()
    W = torch.randn(128, 256, device="cuda")
    y_default = sparse_linear(s, W)
    y_ref = torch.nn.functional.linear(s, W)
    assert torch.equal(y_default, y_ref)


def test_v3_takes_prepacked_input():
    """v3 custom op takes uint8 [M, K_PACKED]; no internal bit-pack.

    This is the right pattern when the user pre-packs (e.g. when
    calling v3 multiple times with the same spike)."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.1).float()
    W = torch.randn(N, K, device="cuda", requires_grad=True)
    packed = bit_pack_spike_dense(s)
    y_ref = torch.nn.functional.linear(s, W)
    y_v3 = cupy_spike_linear_v3_dense_forward(packed, W, None)
    torch.testing.assert_close(y_v3, y_ref, rtol=1e-4, atol=1e-5)

    y_v3.sum().backward()
    expected_grad = s.sum(0).expand_as(W)
    assert torch.allclose(W.grad, expected_grad, atol=1e-5)


def test_v3_user_packs_repeated_calls():
    """User can pack once and call v3 multiple times (no re-pack)."""
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


@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_bit_pack_roundtrip(dtype):
    """bit_pack_spike_dense followed by unpack should be lossless."""
    torch.manual_seed(0)
    M, K = 64, 130  # 130 not a multiple of 8
    s = (torch.rand(M, K, device="cuda") < 0.3).to(dtype)
    packed = bit_pack_spike_dense(s)
    # Unpack manually
    bits = (
        packed.unsqueeze(-1) >> torch.arange(8, dtype=torch.uint8, device="cuda")
    ) & 1
    bits = bits.reshape(M, packed.shape[1] * 8)[:, :K]
    s_back = bits.to(dtype)
    assert torch.equal(s, s_back)


def test_custom_op_registered():
    """Both forward custom ops must be registered with torch.library."""
    assert hasattr(torch.ops.sj, "cupy_spike_linear_v3_dense_forward")
    assert hasattr(torch.ops.sj, "cupy_spike_linear_sparse_forward")


def test_fake_tensor_shape():
    """The fake implementation should produce a tensor of the right shape."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    # Build meta tensors to exercise both fake implementations.
    s_meta = torch.randn(M, K, device="meta")
    packed_meta = torch.empty(M, (K + 7) // 8, dtype=torch.uint8, device="meta")
    W_meta = torch.randn(N, K, device="meta")
    y_meta = torch.ops.sj.cupy_spike_linear_v3_dense_forward(packed_meta, W_meta, None)
    assert y_meta.shape == (M, N)
    assert y_meta.dtype == W_meta.dtype
    y_meta = torch.ops.sj.cupy_spike_linear_sparse_forward(s_meta, W_meta, None)
    assert y_meta.shape == (M, N)
    assert y_meta.dtype == W_meta.dtype

    with pytest.raises(TypeError, match="float32"):
        torch.ops.sj.cupy_spike_linear_sparse_forward(s_meta.half(), W_meta, None)


def test_via_torch_compile():
    """sparse_linear should compile as one graph without graph breaks."""
    torch.manual_seed(0)
    M, K, N = 64, 128, 64
    s = (torch.rand(M, K, device="cuda") < 0.05).float()
    W = torch.randn(N, K, device="cuda")

    def f(s, W):
        return sparse_linear(s, W, strategy="sparse")

    explanation = torch._dynamo.explain(f)(s, W)
    assert explanation.graph_count == 1
    assert explanation.graph_break_count == 0

    compiled = torch.compile(f, fullgraph=True, dynamic=False)
    y = compiled(s, W)
    y_ref = torch.nn.functional.linear(s, W)
    torch.testing.assert_close(y, y_ref, rtol=1e-4, atol=1e-5)
