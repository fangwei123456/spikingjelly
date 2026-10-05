import importlib

import pytest
import torch

from spikingjelly.activation_based import functional, neuron, surrogate
from spikingjelly.activation_based.neuron.experimental import (
    ExperimentalIFNode,
    ExperimentalParametricLIFNode,
)


@pytest.fixture(params=["cpu", "cuda"], scope="module")
def device(request):
    if request.param == "cuda" and (not torch.cuda.is_available() or torch.version.hip):
        pytest.skip("NVIDIA CUDA unavailable")
    return torch.device(request.param)


@pytest.fixture(params=["if", "plif"])
def kind(request):
    return request.param


def _ops(kind):
    return importlib.import_module(
        f"spikingjelly._ops.{'if_' if kind == 'if' else 'plif'}"
    )


def _run(kind, x, v, w, **parameters):
    module = _ops(kind)
    if kind == "if":
        parameters.pop("decay_input", None)
        return module.if_multi_step(x, v, **parameters)
    return module.plif(x, v, w, **parameters)


def _reference(
    kind,
    x,
    v,
    w,
    *,
    decay_input=True,
    threshold=1.0,
    reset=0.0,
    detach_reset=False,
    alpha=4.0,
):
    spikes, voltages = [], []
    sg = surrogate.Sigmoid(alpha=alpha)
    for current in x:
        if kind == "if":
            spike, v = functional.if_step(
                current, v, threshold, reset, sg, detach_reset
            )
        else:
            spike, v = functional.plif_step(
                current, v, w, decay_input, threshold, reset, sg, detach_reset
            )
        spikes.append(spike)
        voltages.append(v)
    return torch.stack(spikes), torch.stack(voltages)


def _node(kind, **kwargs):
    cls = ExperimentalIFNode if kind == "if" else ExperimentalParametricLIFNode
    return cls(**kwargs)


@pytest.mark.parametrize("weight", [-80.0, 80.0])
@pytest.mark.parametrize("decay_input", [False, True])
def test_plif_saturated_weight_and_single_step(device, weight, decay_input):
    x = torch.tensor([[0.5, 1.5, -0.75]], device=device, requires_grad=True)
    v = torch.tensor([0.4, -0.5, 0.2], device=device, requires_grad=True)
    w = torch.tensor(weight, device=device, requires_grad=True)
    params = dict(decay_input=decay_input, reset=None)
    actual = _run("plif", x, v, w, **params)
    expected = _reference("plif", x, v, w, **params)
    torch.testing.assert_close(actual[:2], expected)
    gradients = [
        torch.autograd.grad(s.sum() + vs.sum(), (x, v, w))
        for s, vs in (actual[:2], expected)
    ]
    torch.testing.assert_close(*gradients)
    torch.testing.assert_close(gradients[0][2], gradients[1][2], rtol=1e-4, atol=1e-38)
    assert all(torch.isfinite(g).all() for g in gradients[0])


@pytest.mark.parametrize("output", ["spikes", "last_voltage"])
def test_broadcast_and_unused_output_gradients(kind, device, output):
    base = torch.tensor([[0.5, 1.0, 2.0]], device=device, requires_grad=True)
    x = base.expand(1 if output == "last_voltage" else 3, -1)
    v = torch.tensor([0.0, 0.5, -0.25], device=device, requires_grad=True)
    w = torch.tensor(0.0, device=device, requires_grad=True)
    actual = _run(kind, x, v, w)
    expected = _reference(kind, x, v, w)
    losses = [
        s.sum() if output == "spikes" else vs[-1].sum()
        for s, vs in (actual[:2], expected)
    ]
    inputs = (base, v) if kind == "if" else (base, v, w)
    torch.testing.assert_close(*(torch.autograd.grad(loss, inputs) for loss in losses))


def test_state_chunking_and_reset(kind, device):
    torch.manual_seed(47)
    node = _node(kind, v_reset=None, store_v_seq=True).to(device)
    x = torch.randn(6, 2, 7, device=device, requires_grad=True)
    reference_x = x.detach().clone().requires_grad_()
    w = node.w if kind == "plif" else None
    reference_w = w.detach().clone().requires_grad_() if w is not None else None
    outputs, traces = [], []
    for chunk in (x[:1], x[1:3], x[3:]):
        outputs.append(node(chunk))
        traces.append(node.v_seq)
    expected = _reference(
        kind, reference_x, torch.zeros_like(x[0]), reference_w, reset=None
    )
    torch.testing.assert_close(torch.cat(outputs), expected[0])
    torch.testing.assert_close(torch.cat(traces), expected[1])
    torch.testing.assert_close(node.v, expected[1][-1])
    loss = torch.cat(outputs).sum() + torch.cat(traces).sum() + node.v.sum()
    expected_loss = expected[0].sum() + expected[1].sum() + expected[1][-1].sum()
    inputs = (x,) if kind == "if" else (x, w)
    reference_inputs = (reference_x,) if kind == "if" else (reference_x, reference_w)
    torch.testing.assert_close(
        torch.autograd.grad(loss, inputs),
        torch.autograd.grad(expected_loss, reference_inputs),
    )
    assert set(node.state_dict()) == ({"w"} if kind == "plif" else set())
    saved_w = w.detach().clone() if w is not None else None
    node.reset()
    assert node.v is None and node.v_seq is None
    if w is not None:
        assert node.w is w
        torch.testing.assert_close(node.w, saved_w)
    node.store_v_seq = False
    node(torch.zeros(1, 5, device=device))
    assert node.v.shape == (5,) and node.v_seq is None
    node.double()
    assert node.v.dtype == torch.float64
    if w is not None:
        assert node.w.dtype == torch.float64
    node.float()
    assert node.v.dtype == torch.float32


def test_plif_optimizer_matches_production(device):
    actual = ExperimentalParametricLIFNode(init_tau=3.0, store_v_seq=True).to(device)
    expected = neuron.ParametricLIFNode(
        init_tau=3.0, step_mode="m", store_v_seq=True
    ).to(device)
    optimizers = [torch.optim.SGD(n.parameters(), lr=0.05) for n in (actual, expected)]
    x = torch.full((3, 4), 0.6, device=device)
    before = None
    initial_w = actual.w.detach().clone()
    for _ in range(2):
        traces = []
        for node, optimizer in zip((actual, expected), optimizers):
            node.reset()
            optimizer.zero_grad()
            spikes = node(x)
            traces.append(node.v_seq.detach().clone())
            (spikes.sum() + node.v_seq.sum()).backward()
            optimizer.step()
        torch.testing.assert_close(*traces)
        torch.testing.assert_close(actual.w, expected.w)
        if before is None:
            before = traces[0]
        else:
            assert not torch.equal(before, traces[0])
    assert not torch.equal(initial_w, actual.w)


def test_first_order_gradients_only(kind, device):
    x = torch.full((3, 2), 0.5, device=device, requires_grad=True)
    v = torch.zeros(2, device=device, requires_grad=True)
    w = torch.tensor(-0.4, device=device, requires_grad=True)
    spikes, voltages, _ = _run(kind, x, v, w)
    inputs = (x, v) if kind == "if" else (x, v, w)
    gradients = torch.autograd.grad(
        spikes.sum() + voltages.sum(), inputs, create_graph=True
    )
    assert all(not gradient.requires_grad for gradient in gradients)
    with pytest.raises(RuntimeError, match="does not require grad"):
        torch.autograd.grad(gradients[-1].sum(), inputs[-1])


def test_invalid_inputs(kind, device):
    x = torch.zeros(2, 3, device=device)
    v = torch.zeros(3, device=device)
    w = torch.tensor(0.0, device=device)
    for bad_x, bad_v in [
        (x[:0], v),
        (x[:, :0], v[:0]),
        (x, v[:2]),
        (x.half(), v.half()),
        (x.double(), v.double()),
    ]:
        with pytest.raises(RuntimeError):
            _run(kind, bad_x, bad_v, w)
    if kind == "plif":
        for bad_w in [w.reshape(1), w.double(), w.expand(2)]:
            with pytest.raises(RuntimeError):
                _run(kind, x, v, bad_w)
        if device.type == "cuda":
            with pytest.raises(RuntimeError):
                _run(kind, x, v, w.cpu())
    node = _node(kind).to(device)
    with pytest.raises(ValueError, match="T >= 1"):
        node(x[:0])
    assert node.v is None


@pytest.mark.parametrize("init_tau", [1.0, 0.0, float("nan"), float("inf")])
def test_plif_rejects_invalid_initial_tau(init_tau):
    with pytest.raises(ValueError, match="init_tau"):
        ExperimentalParametricLIFNode(init_tau=init_tau)


def test_nondefault_stream_and_second_gpu(kind, device):
    if device.type != "cuda":
        pytest.skip("CUDA stream test")
    for index in range(min(torch.cuda.device_count(), 2)):
        target = torch.device("cuda", index)
        stream = torch.cuda.Stream(device=target)
        with torch.cuda.device(target), torch.cuda.stream(stream):
            x = torch.full((4, 257), 0.6, device=target, requires_grad=True)
            v = torch.full((257,), 0.2, device=target, requires_grad=True)
            w = torch.tensor(-0.3, device=target, requires_grad=True)
            with torch.cuda.device(0):
                actual = _run(kind, x, v, w, reset=None)
            expected = _reference(kind, x, v, w, reset=None)
            inputs = (x, v) if kind == "if" else (x, v, w)
            gradients = [
                torch.autograd.grad(s.sum() + vs.sum(), inputs)
                for s, vs in (actual[:2], expected)
            ]
        stream.synchronize()
        torch.testing.assert_close(actual[:2], expected)
        torch.testing.assert_close(*gradients)
