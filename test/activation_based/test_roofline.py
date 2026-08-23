import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import neuron, op_counter
from test.activation_based._op_counter_test_utils import TinySNN


def test_dispatch_counter_basic():
    net = TinySNN()
    x = torch.randn(2, 2, 3, 8, 8)

    counter1 = op_counter.FlopCounter()
    counter2 = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([counter1, counter2], strict=True):
        y = net(x)
        l = y.sum()
        l.backward()

    records1 = counter1.get_counts()
    records2 = counter2.get_counts()

    assert "Global" in records1
    assert any("sn" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("sn" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    assert total1 > 0
    total2 = counter2.get_total()
    assert total2 > 0


def test_dispatch_counter_ignore():
    net = TinySNN()
    x = torch.randn(2, 2, 3, 8, 8)

    counter1 = op_counter.FlopCounter(extra_ignore_modules=(neuron.LIFNode,))
    counter2 = op_counter.MemoryAccessCounter(extra_ignore_modules=(neuron.LIFNode,))

    with op_counter.DispatchCounterMode([counter1, counter2], strict=True):
        y = net(x)
        l = y.sum()
        l.backward()

    records1 = counter1.get_counts()
    records2 = counter2.get_counts()

    assert "Global" in records1
    assert any("conv" in k for k in records1.keys())
    assert any("fc" in k for k in records1.keys())
    assert "Global" in records2
    assert any("conv" in k for k in records2.keys())
    assert any("fc" in k for k in records2.keys())

    total1 = counter1.get_total()
    assert total1 > 0
    total2 = counter2.get_total()
    assert total2 > 0


def test_bmm_uses_gpu_roofline_flop_convention_and_ideal_memory_bytes():
    x = torch.randn(2, 3, 4)
    y = torch.randn(2, 4, 5)
    flop_counter = op_counter.FlopCounter()
    memory_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([flop_counter, memory_counter]):
        out = torch.bmm(x, y)

    assert flop_counter.get_total() == 2 * 2 * 3 * 4 * 5
    assert memory_counter.get_total() == (
        x.numel() * x.element_size()
        + y.numel() * y.element_size()
        + out.numel() * out.element_size()
    )


def test_dispatch_counter_reports_unsupported_ops_per_counter():
    counter = op_counter.FlopCounter()
    x = torch.ones(4)

    with op_counter.DispatchCounterMode([counter]) as mode:
        torch.sin(x)

    assert mode.get_unsupported(counter) == {"aten.sin.default": 1}


def test_scaled_dot_product_attention_uses_ideal_tensor_formulas():
    q = torch.randn(2, 4, 8, 16)
    k = torch.randn(2, 4, 8, 16)
    v = torch.randn(2, 4, 8, 16)
    flop_counter = op_counter.FlopCounter()
    memory_counter = op_counter.MemoryAccessCounter()

    with op_counter.DispatchCounterMode([flop_counter, memory_counter]):
        out = F.scaled_dot_product_attention(q, k, v)

    assert flop_counter.get_total() == 2 * 2 * 4 * 8 * 8 * 16 * 2
    assert memory_counter.get_total() == sum(
        tensor.numel() * tensor.element_size() for tensor in (q, k, v, out)
    )


def test_scaled_dot_product_attention_counts_mask_and_backward_bias_traffic():
    q = torch.randn(1, 2, 4, 8)
    mask = torch.randn(4, 4)
    counter = op_counter.MemoryAccessCounter()
    with op_counter.DispatchCounterMode([counter]):
        out = F.scaled_dot_product_attention(q, q, q, attn_mask=mask)
    assert counter.get_total() == sum(
        tensor.numel() * tensor.element_size() for tensor in (q, q, q, mask, out)
    )

    grad_out = torch.randn_like(out)
    logsumexp = torch.randn(1, 2, 4)
    grad_q, grad_k, grad_v = (torch.randn_like(q) for _ in range(3))
    grad_bias = torch.randn_like(mask)
    func = torch.ops.aten._scaled_dot_product_efficient_attention_backward.default
    args = (grad_out, q, q, q, mask, out, logsumexp)
    value = counter.count(func, args, {}, (grad_q, grad_k, grad_v, grad_bias))
    assert value == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (grad_out, q, q, q, mask, grad_q, grad_k, grad_v, grad_bias)
    )


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is unavailable")
@pytest.mark.parametrize("dtype", (torch.float32, torch.float16, torch.bfloat16))
def test_cuda_eager_bmm_and_sdpa_forward_backward(dtype):
    x = torch.randn(2, 3, 4, device="cuda", dtype=dtype)
    y = torch.randn(2, 4, 5, device="cuda", dtype=dtype)
    flop_counter = op_counter.FlopCounter()
    memory_counter = op_counter.MemoryAccessCounter()
    with op_counter.DispatchCounterMode([flop_counter, memory_counter]):
        bmm_out = torch.bmm(x, y)

    assert flop_counter.get_total() == 2 * 2 * 3 * 4 * 5
    assert memory_counter.get_total() == sum(
        tensor.numel() * tensor.element_size() for tensor in (x, y, bmm_out)
    )

    q = torch.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
    k = torch.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
    v = torch.randn(1, 2, 8, 16, device="cuda", dtype=dtype, requires_grad=True)
    reference = F.scaled_dot_product_attention(q, k, v)
    reference.sum().backward()
    reference_grads = (q.grad.clone(), k.grad.clone(), v.grad.clone())
    for tensor in (q, k, v):
        tensor.grad = None

    flop_counter = op_counter.FlopCounter()
    with op_counter.DispatchCounterMode([flop_counter]):
        actual = F.scaled_dot_product_attention(q, k, v)
        actual.sum().backward()

    assert flop_counter.get_total() == 2 * 1 * 2 * 8 * 8 * 16 * 7 + actual.numel() - 1
    assert torch.allclose(actual, reference, rtol=1e-3, atol=1e-3)
    assert all(
        torch.allclose(tensor.grad, expected, rtol=1e-3, atol=1e-3)
        for tensor, expected in zip((q, k, v), reference_grads, strict=True)
    )

    mask = torch.randn(8, 8, device="cuda", dtype=dtype)
    memory_counter = op_counter.MemoryAccessCounter()
    with op_counter.DispatchCounterMode([memory_counter]):
        masked = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    expanded_mask = mask.expand(q.shape[0], q.shape[1], q.shape[2], k.shape[2])
    assert memory_counter.get_total() == sum(
        tensor.numel() * tensor.element_size()
        for tensor in (q, k, v, expanded_mask, masked)
    )


class _LinearModuleCounter(op_counter.ModuleCounter):
    def __init__(self, *, ignore_modules=None):
        super().__init__()
        self.ignore_modules.extend(ignore_modules or [])
        self.rules = {
            ("forward", nn.Linear): lambda module, args, kwargs, out: out.numel(),
            ("backward", nn.Linear): (lambda module, args, kwargs, out: out[0].numel()),
        }


def test_module_counter_mode_counts_forward_backward_and_parent_scopes():
    model = nn.Module()
    model.block = nn.Sequential(nn.Linear(4, 3, bias=False))
    x = torch.randn(2, 4, requires_grad=True)
    counter = _LinearModuleCounter()

    with op_counter.ModuleCounterMode([counter], model=model, strict=True):
        model.block(x).sum().backward()

    counts = counter.get_counts()
    forward_key = ("forward", nn.Linear)
    backward_key = ("backward", nn.Linear)
    assert counts["Global"][forward_key] == 6
    assert counts["Global"][backward_key] == 6
    assert counts["block"][forward_key] == 6
    assert counts["block.0"][backward_key] == 6


def test_module_counter_mode_uses_most_specific_rule_and_forward_kwargs():
    class KeywordLinear(nn.Linear):
        def forward(self, input, *, scale=1.0):
            return super().forward(input) * scale

    counter = op_counter.ModuleCounter()
    counter.rules = {
        ("forward", nn.Linear): lambda module, args, kwargs, out: 1,
        ("forward", KeywordLinear): lambda module, args, kwargs, out: int(
            kwargs["scale"]
        ),
    }
    model = KeywordLinear(4, 3, bias=False)

    with op_counter.ModuleCounterMode([counter], model=model, strict=True):
        model(torch.randn(2, 4), scale=3.0)

    assert counter.get_total() == 3
    assert counter.get_counts()["Global"] == {("forward", KeywordLinear): 3}


def test_module_counter_mode_ignore_subtree_and_strict_diagnostics():
    model = nn.Module()
    model.block = nn.Sequential(nn.Linear(4, 3, bias=False))
    ignored = _LinearModuleCounter(ignore_modules=[nn.Sequential])

    with op_counter.ModuleCounterMode([ignored], model=model, strict=True):
        model.block(torch.randn(2, 4))
    assert ignored.get_counts() == {}

    forward_only = op_counter.ModuleCounter()
    forward_only.rules = {("forward", nn.Linear): lambda *args: 1}
    backward_only = op_counter.ModuleCounter()
    backward_only.rules = {("backward", nn.Linear): lambda *args: 1}
    mode = op_counter.ModuleCounterMode(
        [forward_only, backward_only], model=model, strict=True
    )
    with pytest.raises(
        NotImplementedError, match="forward:torch.nn.modules.linear.Linear"
    ):
        with mode:
            model.block(torch.randn(2, 4))
    assert mode.get_unsupported(backward_only) == {
        "forward:torch.nn.modules.linear.Linear": 1
    }


def test_module_counter_mode_cleans_hooks_and_does_not_implicitly_reset():
    model = nn.Linear(4, 3, bias=False)
    counter = _LinearModuleCounter()
    mode = op_counter.ModuleCounterMode([counter], model=model)

    for _ in range(2):
        with mode:
            model(torch.randn(2, 4))
        assert len(model._forward_hooks) == 0
        assert len(model._backward_hooks) == 0
    assert counter.get_total() == 12

    with pytest.raises(RuntimeError, match="boom"):
        with mode:
            raise RuntimeError("boom")
    assert len(model._forward_hooks) == 0
    assert len(model._backward_hooks) == 0


def test_module_counter_mode_cleans_partial_enter(monkeypatch):
    model = nn.Sequential(
        nn.Linear(4, 4, bias=False),
        nn.Linear(4, 3, bias=False),
    )
    counter = _LinearModuleCounter()

    def raising_register(*args, **kwargs):
        raise RuntimeError("register failed")

    monkeypatch.setattr(model[1], "register_forward_hook", raising_register)
    mode = op_counter.ModuleCounterMode([counter], model=model)
    with pytest.raises(RuntimeError, match="register failed"):
        mode.__enter__()
    assert len(model[0]._forward_hooks) == 0
    assert len(model[0]._backward_hooks) == 0
