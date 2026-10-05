"""Compare registered kernels against independent autograd with public surrogates."""

import importlib

import pytest
import torch

from spikingjelly.activation_based import functional, surrogate
from spikingjelly.activation_based.neuron.experimental import (
    ExperimentalIFNode,
    ExperimentalLIFNode,
    ExperimentalParametricLIFNode,
)


SURROGATES = (
    "Sigmoid",
    "ATan",
    "PiecewiseQuadratic",
    "PiecewiseExp",
    "SoftSign",
    "SuperSpike",
    "Erf",
)
DTYPES = (torch.float32, torch.float16, torch.bfloat16)
NODES = {
    "if": ExperimentalIFNode,
    "lif": ExperimentalLIFNode,
    "plif": ExperimentalParametricLIFNode,
}
_STATE_MODES = [
    (trace, reset, detach, decay)
    for trace in (False, True)
    for reset in (None, 0.2)
    for detach in (False, True)
    for decay in (False, True)
]


@pytest.fixture(params=["cpu", "cuda"], scope="module")
def device(request):
    if request.param == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("NVIDIA CUDA unavailable")
    return torch.device(request.param)


def _call(kind, x, v, w, sid, trace, reset, detach, decay):
    module = importlib.import_module(
        f"spikingjelly._ops.{'if_' if kind == 'if' else kind}"
    )
    tail = (0.7, reset, detach, 2.0, trace, sid)
    if kind == "if":
        return module.if_multi_step, (x, v, *tail), module
    if kind == "lif":
        return module.lif, (x, v, 2.3, decay, *tail), module
    return module.plif, (x, v, w, decay, *tail), module


def _reference(kind, x, v, w, sid, trace, reset, detach, decay):
    sg = getattr(surrogate, SURROGATES[sid])(alpha=2.0)
    spikes, voltages, charged = [], [], []
    base = 0.0 if reset is None else reset
    q = w.float().sigmoid()
    for current in x.float():
        if kind == "if":
            h = v + current
        elif kind == "lif":
            h = (
                v + (current - (v - base)) / 2.3
                if decay
                else v - (v - base) / 2.3 + current
            )
        else:
            h = (
                v + (current - (v - base)) * q
                if decay
                else v - (v - base) * q + current
            )
        spike = sg(h - 0.7)
        r = spike.detach() if detach else spike
        v = h - r * 0.7 if reset is None else (1 - r) * h + r * reset
        spikes.append(spike.to(x.dtype))
        voltages.append(v)
        charged.append(h)
    return (
        torch.stack(spikes),
        torch.stack(voltages) if trace else v,
        torch.stack(charged),
    )


@pytest.mark.parametrize(
    "kind,dtype,sid,trace,reset,detach,decay",
    # Cover all dtype/surrogate pairs and exhaust state modes once, not their product.
    [
        (kind, dtype, sid, *_STATE_MODES[(sid * len(DTYPES) + d) % len(_STATE_MODES)])
        for kind in NODES
        for d, dtype in enumerate(DTYPES)
        for sid in range(len(SURROGATES))
    ]
    + [
        (kind, torch.float32, 0, *mode)
        for kind in NODES
        for mode in _STATE_MODES[1:]
        if kind != "if" or not mode[-1]
    ],
)
def test_compatibility_matrix(device, kind, dtype, sid, trace, reset, detach, decay):
    torch.manual_seed(819)
    x = (
        (torch.randn(5, 3, 7, device=device) * 0.6)
        .to(dtype)
        .transpose(1, 2)
        .requires_grad_()
    )
    v = torch.randn(3, 7, device=device).t().requires_grad_()
    w = torch.tensor(-0.4, device=device, requires_grad=True)
    fn, args, _ = _call(kind, x, v, w, sid, trace, reset, detach, decay)
    actual = fn(*args)
    expected = _reference(kind, x, v, w, sid, trace, reset, detach, decay)
    assert actual[0].dtype == dtype
    assert actual[1].dtype == actual[2].dtype == torch.float32
    assert not actual[2].requires_grad
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1:], expected[1:], rtol=2e-5, atol=2e-6)
    gs, gv = torch.randn_like(actual[0]), torch.randn_like(actual[1])
    inputs = (x, v, w) if kind == "plif" else (x, v)
    gradients = [
        torch.autograd.grad(out[:2], inputs, (gs, gv)) for out in (actual, expected)
    ]
    for got, want in zip(*gradients):
        assert got.dtype == want.dtype
        rtol, atol = (
            (0.012, 0.008)
            if got.dtype == torch.bfloat16
            else (0.002, 0.0005)
            if got.dtype == torch.float16
            else (2e-4, 2e-5)
        )
        torch.testing.assert_close(got, want, rtol=rtol, atol=atol)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize("sid", range(7), ids=SURROGATES)
def test_surrogate_boundaries_and_unused_gradients(device, dtype, sid):
    from spikingjelly._ops.if_ import if_multi_step

    # Binary-exact values hit the threshold/support edges in all three dtypes.
    x = torch.tensor(
        [[0.5, 1.0, 1.5, -12.0, 12.0]], device=device, dtype=dtype
    ).requires_grad_()
    v = torch.zeros(5, device=device, requires_grad=True)
    sg = getattr(surrogate, SURROGATES[sid])(alpha=2.0)
    for voltage_only in (False, True):
        actual = if_multi_step(
            x,
            v,
            threshold=1.0,
            reset=None,
            alpha=2.0,
            store_v_seq=False,
            surrogate_id=sid,
        )
        spikes, voltage = functional.if_step(x[0].float(), v, 1.0, None, sg, False)
        expected = (spikes.to(dtype).unsqueeze(0), voltage)
        grads = [
            torch.autograd.grad(out[int(voltage_only)].sum(), (x, v))
            for out in (actual, expected)
        ]
        torch.testing.assert_close(
            *grads, rtol=0.012 if dtype == torch.bfloat16 else 0.002, atol=0.002
        )


@pytest.mark.parametrize("kind", NODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_state_chunks_and_low_precision_parameter(device, kind, dtype):
    torch.manual_seed(82)
    node = NODES[kind](surrogate_function=surrogate.ATan(), store_v_seq=True).to(
        device=device, dtype=dtype
    )
    x = torch.randn(6, 3, device=device, dtype=dtype, requires_grad=True)
    outputs, traces = [], []
    for chunk in (x[:1], x[1:3], x[3:]):
        outputs.append(node(chunk))
        traces.append(node.v_seq)
    last = node.v
    assert last.dtype == torch.float32
    node.reset()
    whole = node(x)
    torch.testing.assert_close(torch.cat(outputs), whole, rtol=0, atol=0)
    torch.testing.assert_close(torch.cat(traces), node.v_seq)
    inputs = (x, node.w) if kind == "plif" else (x,)
    split_loss = torch.cat(outputs).sum() + torch.cat(traces).sum() + last.sum()
    whole_loss = whole.sum() + node.v_seq.sum() + node.v.sum()
    # A low-precision parameter receives one rounded gradient per call.
    torch.testing.assert_close(
        torch.autograd.grad(split_loss, inputs),
        torch.autograd.grad(whole_loss, inputs),
        rtol=0.02 if dtype == torch.bfloat16 else 0.003,
        atol=0.02 if dtype == torch.bfloat16 else 0.003,
    )
    if kind == "plif":
        assert node.w.dtype == dtype
    node.reset()
    assert node.v is None and node.v_seq is None


@pytest.mark.parametrize("kind", NODES)
@pytest.mark.parametrize("dtype", DTYPES)
def test_registration_and_compile_compatibility(device, kind, dtype):
    torch._dynamo.reset()
    x = torch.full((2, 17), 0.3, device=device, dtype=dtype, requires_grad=True)
    v = torch.zeros(17, device=device, requires_grad=True)
    w = torch.tensor(-0.2, device=device, requires_grad=True)
    # Exercise different specializations as well as dtype metadata.
    sid = DTYPES.index(dtype) + 4
    trace = dtype == torch.float32
    fn, args, module = _call(kind, x, v, w, sid, trace, None, False, True)
    eager = fn(*args)
    torch.library.opcheck(module._forward, args)
    compiled = torch.compile(
        fn, backend="inductor" if device.type == "cuda" else "aot_eager", fullgraph=True
    )
    actual = compiled(*args)
    torch.testing.assert_close(actual, eager)
    inputs = (x, v, w) if kind == "plif" else (x, v)
    torch.testing.assert_close(
        *[
            torch.autograd.grad(out[0].sum() + out[1].sum(), inputs)
            for out in (actual, eager)
        ]
    )

    node = (
        NODES[kind](
            surrogate_function=getattr(surrogate, SURROGATES[sid])(), store_v_seq=trace
        )
        .to(device)
        .eval()
    )
    eager_s = node(x)
    eager_v = node.v
    eager_v_seq = node.v_seq
    node.reset()
    compiled_node = torch.compile(
        node,
        backend="inductor" if device.type == "cuda" else "aot_eager",
        fullgraph=True,
    )
    actual_s = compiled_node(x)
    torch.testing.assert_close((actual_s, node.v), (eager_s, eager_v))
    if trace:
        torch.testing.assert_close(node.v_seq, eager_v_seq)
    inputs = (x, node.w) if kind == "plif" else (x,)
    torch.testing.assert_close(
        torch.autograd.grad(actual_s.sum() + node.v.sum(), inputs),
        torch.autograd.grad(eager_s.sum() + eager_v.sum(), inputs),
    )


@pytest.mark.parametrize("kind", NODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_autocast_optimizer(device, kind, dtype):
    if device.type == "cpu" and dtype == torch.float16:
        pytest.skip("CPU autocast coverage uses BF16")
    torch.manual_seed(83)
    linear = torch.nn.Linear(7, 5).to(device)
    node = NODES[kind](surrogate_function=surrogate.Erf()).to(device)
    optimizer = torch.optim.SGD([*linear.parameters(), *node.parameters()], lr=0.01)
    before = linear.weight.detach().clone()
    scaler = torch.amp.GradScaler(
        device.type,
        init_scale=128.0,
        enabled=device.type == "cuda" and dtype == torch.float16,
    )
    for _ in range(2):
        node.reset()
        optimizer.zero_grad()
        with torch.autocast(device.type, dtype=dtype):
            y = node(linear(torch.randn(3, 2, 7, device=device)))
            loss = y.float().mean() + node.v.square().mean()
        assert y.dtype == dtype and node.v.dtype == torch.float32
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
    assert not torch.equal(before, linear.weight)
    assert torch.isfinite(linear.weight).all()
    if kind == "plif":
        assert node.w.dtype == node.w.grad.dtype == torch.float32
        assert torch.isfinite(node.w.grad)


def test_rejected_surrogates():
    for cls in NODES.values():
        with pytest.raises(TypeError, match="surrogate"):
            cls(surrogate_function=torch.nn.Identity())
        with pytest.raises(ValueError, match="spiking"):
            cls(surrogate_function=surrogate.ATan(spiking=False))
        sg = surrogate.Sigmoid()
        sg.alpha = torch.tensor(4.0, requires_grad=True)
        with pytest.raises(TypeError, match="scalar"):
            cls(surrogate_function=sg)


@pytest.mark.parametrize("kind", NODES)
def test_invalid_dtype_and_id(device, kind):
    x = torch.ones(2, 3, device=device)
    v = torch.zeros(3, device=device)
    w = torch.tensor(0.0, device=device)
    for dtype in (torch.float64, torch.int32, torch.float8_e4m3fn):
        fn, args, _ = _call(kind, x.to(dtype), v, w, 0, False, None, False, True)
        with pytest.raises(RuntimeError):
            fn(*args)
    fn, args, _ = _call(kind, x, v, w, 7, False, None, False, True)
    with pytest.raises(ValueError, match="surrogate_id"):
        fn(*args)


@pytest.mark.parametrize("kind", NODES)
@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_low_precision_stream_and_broadcast(kind, dtype):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA unavailable")
    for index in range(min(torch.cuda.device_count(), 2)):
        device = torch.device("cuda", index)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.device(device), torch.cuda.stream(stream):
            base = torch.full(
                (1, 257), 0.3, device=device, dtype=dtype, requires_grad=True
            )
            x = base.expand(4, -1)
            v = torch.zeros(257, device=device, requires_grad=True)
            w = torch.tensor(-0.4, device=device, requires_grad=True)
            fn, args, _ = _call(kind, x, v, w, 3, False, None, False, True)
            with torch.cuda.device(0):
                actual = fn(*args)
            expected = _reference(kind, x, v, w, 3, False, None, False, True)
            inputs = (base, v, w) if kind == "plif" else (base, v)
            grads = [
                torch.autograd.grad(out[0].sum() + out[1].sum(), inputs)
                for out in (actual, expected)
            ]
        stream.synchronize()
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(*grads, rtol=0.012, atol=0.002)


def test_surrogate_configuration_snapshot():
    sg = surrogate.ATan(alpha=2.0)
    node = ExperimentalIFNode(alpha=9.0, surrogate_function=sg)
    sg.alpha = 10.0
    x = torch.tensor([[0.4, 0.8]], requires_grad=True)
    expected = ExperimentalIFNode(surrogate_function=surrogate.ATan(alpha=2.0))
    torch.testing.assert_close(
        torch.autograd.grad(node(x).sum(), x),
        torch.autograd.grad(expected(x).sum(), x),
    )
    with pytest.raises(TypeError, match="scalar"):
        ExperimentalIFNode(alpha=torch.tensor(4.0))


@pytest.mark.parametrize(
    "dtype,n",
    [
        (torch.float32, 2097151),
        (torch.float32, 2097152),
        (torch.float32, 2097153),
        (torch.float16, 2097153),
        (torch.bfloat16, 2097153),
    ],
)
def test_lif_wide_sequence_gradients(dtype, n):
    if not torch.cuda.is_available() or torch.version.hip:
        pytest.skip("NVIDIA CUDA unavailable")
    device = torch.device("cuda")
    values = (torch.arange(n, device=device) % 7).float() * 0.35
    x = torch.stack((values, values.flip(0))).to(dtype).requires_grad_()
    v = torch.full((n,), 0.1, device=device, requires_grad=True)
    w = torch.tensor(0.0, device=device)
    sid = 1  # ATan is the specialization with the wide-kernel boundary.
    trace, reset, detach, decay = n % 2 == 0, None if n % 2 else 0.2, n % 3 == 0, True
    fn, args, _ = _call("lif", x, v, w, sid, trace, reset, detach, decay)
    actual = fn(*args)
    expected = _reference("lif", x, v, w, sid, trace, reset, detach, decay)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)
    torch.testing.assert_close(actual[1:], expected[1:], rtol=2e-5, atol=2e-6)
    gradients = [
        torch.autograd.grad(out[0].sum() + out[1].sum(), (x, v))
        for out in (actual, expected)
    ]
    torch.testing.assert_close(
        *gradients,
        rtol=0.012 if dtype == torch.bfloat16 else 0.002,
        atol=0.008 if dtype == torch.bfloat16 else 0.0005,
    )
