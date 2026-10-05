import importlib

import pytest
import torch


@pytest.mark.parametrize("family", ["if_", "lif", "plif"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_last_state_matches_trace_and_gradients(family, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    module = importlib.import_module(f"spikingjelly._ops.{family}")
    operation = getattr(module, "if_multi_step" if family == "if_" else family)
    torch.manual_seed(51)
    x = torch.randn(3, 2, 4, device=device).transpose(1, 2).requires_grad_()
    v = torch.randn(2, 4, device=device).t().requires_grad_()
    w = torch.tensor(-0.4, device=device, requires_grad=True)
    args = (x, v, w) if family == "plif" else (x, v)
    full = operation(*args)
    last = operation(*args, store_v_seq=False)
    assert last[1].shape == v.shape and last[1].is_contiguous()
    torch.testing.assert_close(last[0], full[0])
    torch.testing.assert_close(last[1], full[1][-1])
    torch.testing.assert_close(last[2], full[2])
    full_grad = torch.autograd.grad(full[0].sum() + full[1][-1].sum(), args)
    last_grad = torch.autograd.grad(last[0].sum() + last[1].sum(), args)
    torch.testing.assert_close(last_grad, full_grad)
