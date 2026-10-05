"""Registered counterparts of every non-Torch functional neuron transition."""

import importlib
import os
import subprocess
import sys
import textwrap

import pytest
import torch

from spikingjelly.activation_based import functional, surrogate

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
_STATE_MODES = [
    (trace, reset, detach)
    for trace in (False, True)
    for reset in (None, 0.2)
    for detach in (False, True)
]


@pytest.fixture(params=["cpu", "cuda"], scope="module")
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("NVIDIA CUDA unavailable")
    return torch.device(request.param)


def _call(kind, x, v, w, sg, trace, reset, detach):
    common = dict(
        v_threshold=0.7, surrogate_function=sg, detach_reset=detach, store_v_seq=trace
    )
    if kind == "ilif":
        return functional.ilif_multi_step_registered(x, v, tau=2.3, **common)
    common["v_reset"] = reset
    if kind == "if":
        return functional.if_multi_step_registered(x, v, **common)
    if kind == "lif":
        return functional.lif_multi_step_registered(x, v, tau=2.3, **common)
    if kind == "plif":
        return functional.plif_multi_step_registered(x, v, w, **common)
    common["v_rest"] = -0.2
    if kind == "qif":
        return functional.qif_multi_step_registered(
            x, v, tau=2.3, v_c=0.8, a0=0.4, **common
        )
    if kind == "eif":
        return functional.eif_multi_step_registered(
            x, v, tau=2.3, theta_rh=0.9, delta_t=0.7, **common
        )
    common.pop("store_v_seq")
    return functional.izhikevich_multi_step_registered(
        x,
        v,
        w,
        tau=2.3,
        v_c=0.8,
        a0=0.4,
        a=0.2,
        b=0.3,
        tau_w=3.1,
        store_state_seq=trace,
        **common,
    )


def _reference(kind, x, v, w, sg, trace, reset, detach):
    spikes, voltages, recoveries = [], [], []
    for current in x.float():
        if kind == "if":
            spike, v = functional.if_step(current, v, 0.7, reset, sg, detach)
        elif kind == "lif":
            spike, v = functional.lif_step(
                current, v, 2.3, True, 0.7, reset, sg, detach
            )
        elif kind == "plif":
            spike, v = functional.plif_step(
                current, v, w.float(), True, 0.7, reset, sg, detach
            )
        elif kind == "qif":
            spike, v = functional.qif_step(
                current, v, 2.3, 0.4, -0.2, 0.8, 0.7, reset, sg, detach
            )
        elif kind == "eif":
            spike, v = functional.eif_step(
                current, v, 2.3, 0.7, 0.9, -0.2, 0.7, reset, sg, detach
            )
        elif kind == "ilif":
            # Charge is algebraically decay*v+x; preserve its specified FP32 form.
            h = (1 - 1 / 2.3) * v + current
            spike = sg(h / 0.7)
            v = h - (spike.detach() if detach else spike) * 0.7
        else:
            spike, v, w = functional.izhikevich_step(
                current,
                v,
                w,
                2.3,
                0.4,
                -0.2,
                0.8,
                3.1,
                0.2,
                0.3,
                0.7,
                reset,
                sg,
                detach,
            )
            recoveries.append(w)
        spikes.append(spike.to(x.dtype))
        voltages.append(v)
    if kind == "izhikevich":
        return (
            torch.stack(spikes),
            v,
            w,
            torch.stack(voltages) if trace else None,
            torch.stack(recoveries) if trace else None,
        )
    return torch.stack(spikes), v, torch.stack(voltages) if trace else None


def _assert_outputs(actual, expected):
    for got, want in zip(actual, expected):
        if want is None:
            assert got is None
        else:
            torch.testing.assert_close(got, want, rtol=2e-5, atol=4e-6)
    torch.testing.assert_close(actual[0], expected[0], rtol=0, atol=0)


def _assert_gradients(actual, expected, inputs):
    weights = [torch.randn_like(out) if out is not None else None for out in actual]
    actual_loss = sum(
        (out * weight).sum() for out, weight in zip(actual, weights) if out is not None
    )
    expected_loss = sum(
        (out * weight).sum()
        for out, weight in zip(expected, weights)
        if out is not None
    )
    got = torch.autograd.grad(actual_loss, inputs, retain_graph=True)
    want = torch.autograd.grad(expected_loss, inputs, retain_graph=True)
    for a, b in zip(got, want):
        assert a.dtype == b.dtype
        rtol, atol = (
            (0.012, 0.008)
            if a.dtype == torch.bfloat16
            else (0.002, 0.0005)
            if a.dtype == torch.float16
            else (2e-4, 2e-5)
        )
        torch.testing.assert_close(a, b, rtol=rtol, atol=atol)


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich"])
@pytest.mark.parametrize(
    "dtype,surrogate_name,trace,reset,detach",
    [
        (dtype, name, *_STATE_MODES[(sid * len(DTYPES) + d) % len(_STATE_MODES)])
        for d, dtype in enumerate(DTYPES)
        for sid, name in enumerate(SURROGATES)
    ]
    + [(torch.float32, "Sigmoid", *mode) for mode in _STATE_MODES[1:]],
)
def test_dynamics_matrix(device, kind, dtype, surrogate_name, trace, reset, detach):
    torch.manual_seed(834)
    x = (
        (torch.randn(5, 3, 7, device=device) * 0.5 + 0.6)
        .to(dtype)
        .transpose(1, 2)
        .requires_grad_()
    )
    v = (torch.randn(3, 7, device=device) * 0.15).t().requires_grad_()
    w = (
        torch.tensor(-0.4, device=device, requires_grad=True)
        if kind == "plif"
        else (torch.randn_like(v) * 0.1).requires_grad_()
    )
    sg = getattr(surrogate, surrogate_name)(alpha=2.0)
    actual = _call(kind, x, v, w, sg, trace, reset, detach)
    expected = _reference(kind, x, v, w, sg, trace, reset, detach)
    assert actual[0].dtype == dtype
    assert all(out.dtype == torch.float32 for out in actual[1:] if out is not None)
    _assert_outputs(actual, expected)
    _assert_gradients(
        actual, expected, (x, v, w) if kind in ("plif", "izhikevich") else (x, v)
    )


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "trace,detach", [(False, False), (True, False), (False, True), (True, True)]
)
@pytest.mark.parametrize("window", [None, (-0.5, 2.5)])
def test_integer_lif_counts_and_window(device, dtype, trace, detach, window):
    x = torch.tensor(
        [[0.0, 0.35, 1.05, 1.75, 2.8, -0.35], [0.7, 1.4, -0.7, 0.0, 2.1, -1.4]],
        device=device,
        dtype=dtype,
    ).requires_grad_()
    v = torch.zeros(6, device=device, requires_grad=True)
    sg = surrogate.MultiLevelSpikeCount(4, grad_window=window)
    actual = _call("ilif", x, v, None, sg, trace, None, detach)
    expected = _reference("ilif", x, v, None, sg, trace, None, detach)
    _assert_outputs(actual, expected)
    _assert_gradients(actual, expected, (x, v))


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich", "ilif"])
@pytest.mark.parametrize("dtype", DTYPES)
def test_chunks_broadcast_and_unused_gradients(device, kind, dtype):
    torch.manual_seed(51)
    x = (
        (torch.randn(6, 1, 7, device=device) * 0.2 + 0.3)
        .to(dtype)
        .expand(-1, 3, -1)
        .requires_grad_()
    )
    v = torch.zeros(1, 7, device=device).expand(3, -1).requires_grad_()
    w = torch.ones_like(v).mul(0.1).requires_grad_()
    sg = surrogate.MultiLevelSpikeCount(4) if kind == "ilif" else surrogate.ATan()
    whole = _call(kind, x, v, w, sg, True, None, False)
    cv, cw = v, w
    spikes, voltages, recoveries = [], [], []
    for part in (x[:1], x[1:3], x[3:]):
        out = _call(kind, part, cv, cw, sg, True, None, False)
        spikes.append(out[0])
        cv = out[1]
        if kind == "izhikevich":
            cw = out[2]
            voltages.append(out[3])
            recoveries.append(out[4])
        else:
            voltages.append(out[2])
    split = (
        (torch.cat(spikes), cv, cw, torch.cat(voltages), torch.cat(recoveries))
        if recoveries
        else (torch.cat(spikes), cv, torch.cat(voltages))
    )
    _assert_outputs(split, whole)
    inputs = (x, v, w) if kind == "izhikevich" else (x, v)
    _assert_gradients(split, whole, inputs)
    for index in range(3 if kind == "izhikevich" else 2):
        got = torch.autograd.grad(whole[index].sum(), inputs, retain_graph=True)
        reference = _reference(kind, x, v, w, sg, True, None, False)
        want = torch.autograd.grad(reference[index].sum(), inputs)
        torch.testing.assert_close(got, want, rtol=2e-4, atol=2e-5)


@pytest.mark.parametrize("dtype", DTYPES)
@pytest.mark.parametrize(
    "trace,reset,scalar", [(False, None, True), (True, 0.2, False), (True, None, False)]
)
def test_activation_aware_inference(device, dtype, trace, reset, scalar):
    torch.manual_seed(17)
    x = torch.randn(4, 2, 5, 3, device=device).to(dtype).transpose(-1, -2)
    v = torch.zeros_like(x[0], dtype=torch.float32)
    threshold = (
        torch.tensor([0.6, 0.8, 1.0], device=device)
        if not scalar
        else torch.tensor(0.8, device=device)
    )
    offset = (
        torch.tensor([0.0, 0.1, -0.2], device=device)
        if not scalar
        else torch.tensor(0.1, device=device)
    )
    actual = functional.activation_aware_if_multi_step_registered(
        x, v, threshold, offset, 3, 5, reset, trace
    )
    th = threshold.reshape(1, -1, 1) if not scalar else threshold
    off = offset.reshape(1, -1, 1) if not scalar else offset
    spikes, voltages = [], []
    for current in x.float():
        s = (v + current + off >= th).float()
        h = v + current
        v = h - s * th if reset is None else s * reset + (1 - s) * h
        spikes.append(s.to(dtype))
        voltages.append(v)
    _assert_outputs(
        actual, (torch.stack(spikes), v, torch.stack(voltages) if trace else None)
    )


@pytest.mark.parametrize("dtype", DTYPES)
def test_stbif_sequence_single_step_and_bounds(device, dtype):
    x = torch.tensor(
        [[0.5, 3.0, -0.5, -3.0], [0.5, 3.0, -0.5, -3.0], [1.0, 3.0, 2.0, -3.0]],
        device=device,
        dtype=dtype,
    )
    q = torch.tensor([0.5, 0.0, -0.5, 0.0], device=device)
    acc = torch.tensor([0.5, 1.5, -0.5, -1.5], device=device)
    scale, pos, neg = [torch.tensor(value, device=device) for value in (1.0, 2.0, -2.0)]
    actual = functional.stbif_multi_step_registered(x, q, acc, scale, pos, neg)
    outputs = []
    for current in x.float():
        out, q, acc, cur = functional.stbif_step(current, q, acc, scale, pos, neg)
        outputs.append(out.to(dtype))
    _assert_outputs(actual, (torch.stack(outputs), q, acc, cur))
    q0 = torch.tensor([0.5, 0.0, -0.5, 0.0], device=device)
    a0 = torch.tensor([0.5, 1.5, -0.5, -1.5], device=device)
    singles = []
    for current in x:
        out, q0, a0, cur0 = functional.stbif_single_step_registered(
            current, q0, a0, scale, pos, neg
        )
        singles.append(out)
    _assert_outputs((torch.stack(singles), q0, a0, cur0), actual)


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich", "ilif"])
def test_opcheck_and_fullgraph_gradients(device, kind):
    torch.manual_seed(834)
    x = (torch.rand(3, 7, device=device) * 0.3).requires_grad_()
    v = torch.zeros(7, device=device, requires_grad=True)
    w = torch.ones_like(v).mul(0.1).requires_grad_()
    sg = surrogate.MultiLevelSpikeCount(4) if kind == "ilif" else surrogate.ATan()
    package = importlib.import_module(f"spikingjelly._ops.{kind}")
    op = package._forward
    parameters = {
        "qif": (x, v, 2.3, -0.2, 0.8, 0.4, 0.7, None, False, 2.0, True, 1),
        "eif": (x, v, 2.3, -0.2, 0.9, 0.7, 0.7, None, False, 2.0, True, 1),
        "izhikevich": (
            x,
            v,
            w,
            2.3,
            -0.2,
            0.8,
            0.4,
            0.2,
            0.3,
            3.1,
            0.7,
            None,
            False,
            2.0,
            True,
            1,
        ),
        "ilif": (x, v, 2.3, 4.0, 0.0, 4.0, 0.7, False, True),
    }
    torch.library.opcheck(op, parameters[kind])

    def fn(x, v, w):
        return _call(kind, x, v, w, sg, True, None, False)

    eager = fn(x, v, w)
    compiled = torch.compile(fn, fullgraph=True)(x, v, w)
    _assert_outputs(compiled, eager)
    _assert_gradients(compiled, eager, (x, v, w) if kind == "izhikevich" else (x, v))


@pytest.mark.parametrize("kind", ["activation_aware_if", "stbif"])
def test_inference_opcheck_compile_and_grad_rejection(device, kind):
    x = torch.ones(3, 7, device=device)
    v = torch.zeros(7, device=device)
    th, off, neg = [torch.tensor(value, device=device) for value in (1.0, 3.0, -3.0)]
    package = importlib.import_module(f"spikingjelly._ops.{kind}")
    op = package._forward
    args = (
        (x, v, v.clone(), th, off, neg)
        if kind == "stbif"
        else (x, v, th, off, 1, 1, None, False)
    )
    torch.library.opcheck(op, args)
    eager = op(*args)
    compiled = torch.compile(op, fullgraph=True)(*args)
    _assert_outputs(compiled, eager)
    with pytest.raises(RuntimeError, match="autograd"):
        op(x.requires_grad_(), *args[1:])


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich", "ilif"])
def test_invalid_dynamics_inputs(device, kind):
    sg = surrogate.MultiLevelSpikeCount(4) if kind == "ilif" else surrogate.ATan()
    v = torch.zeros(7, device=device)
    for x in (
        torch.ones(0, 7, device=device),
        torch.ones(3, 7, device=device, dtype=torch.float64),
    ):
        with pytest.raises(RuntimeError):
            _call(kind, x, v, v, sg, False, None, False)
    with pytest.raises(RuntimeError):
        _call(
            kind, torch.ones(3, 7, device=device), v.half(), v, sg, False, None, False
        )
    with pytest.raises(TypeError):
        _call(
            kind,
            torch.ones(3, 7, device=device),
            v,
            v,
            surrogate.Rect(),
            False,
            None,
            False,
        )


def test_single_step_existing_families(device):
    x = torch.tensor([0.2, 1.2], device=device, requires_grad=True)
    v = torch.tensor([0.1, 0.3], device=device, requires_grad=True)
    for kind in ("if", "lif"):
        actual = getattr(functional, f"{kind}_step_registered")(x, v)
        expected = (
            functional.if_step(x, v, 1.0, 0.0, surrogate.Sigmoid())
            if kind == "if"
            else functional.lif_step(x, v, 2.0, True, 1.0, 0.0, surrogate.Sigmoid())
        )
        _assert_outputs(actual, expected)
        _assert_gradients(actual, expected, (x, v))


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_separate_cuda_selections_in_fresh_process():
    choices = {
        "if": "triton",
        "lif": "cuda",
        "plif": "cupy",
        "qif": "cuda",
        "eif": "triton",
        "izhikevich": "cupy",
        "ilif": "cuda",
        "activation_aware_if": "triton",
        "stbif": "cupy",
    }
    code = """
        import torch
        from spikingjelly.activation_based import functional, surrogate
        from test.activation_based.test_registered_neuron_kernels import _call, _reference

        choices = CHOICES
        device = torch.device("cuda", 0)
        for kind, provider in choices.items():
            info = functional.registered_neuron_implementation(kind, device)
            assert info["implementation"] == provider, (kind, info)
            x = torch.full((3, 7), 0.4, device=device, requires_grad=True)
            v = torch.zeros(7, device=device, requires_grad=True)
            w = torch.zeros(() if kind == "plif" else (7,), device=device, requires_grad=True)
            if kind in ("activation_aware_if", "stbif"):
                continue
            sg = surrogate.MultiLevelSpikeCount(4) if kind == "ilif" else surrogate.ATan()
            actual = _call(kind, x, v, w, sg, True, 0.2, True)
            expected = _reference(kind, x, v, w, sg, True, 0.2, True)
            for got, want in zip(actual, expected):
                if got is not None:
                    torch.testing.assert_close(got, want)
            inputs = (x, v, w) if kind in ("plif", "izhikevich") else (x, v)
            gradients = [torch.autograd.grad(sum(out.sum() for out in result if out is not None), inputs)
                         for result in (actual, expected)]
            torch.testing.assert_close(*gradients, rtol=2e-4, atol=2e-5)
        for kind, provider in choices.items():
            assert functional.registered_neuron_implementation(kind, device)["implementation"] == provider
    """.replace("CHOICES", repr(choices))
    env = dict(os.environ)
    env.update(
        {
            f"SJ_{kind.upper()}_CUDA_IMPLEMENTATION": provider
            for kind, provider in choices.items()
        }
    )
    subprocess.run(
        [sys.executable, "-c", textwrap.dedent(code)], env=env, check=True, timeout=180
    )


@pytest.mark.parametrize("kind", ["qif", "eif", "izhikevich", "ilif"])
def test_odd_size_long_sequence_and_second_order_rejection(device, kind):
    torch.manual_seed(982)
    x = (torch.rand(33, 513, device=device) * 0.1).requires_grad_()
    v = torch.zeros(513, device=device, requires_grad=True)
    w = torch.zeros_like(v).requires_grad_()
    sg = surrogate.MultiLevelSpikeCount(4) if kind == "ilif" else surrogate.Sigmoid()
    actual = _call(kind, x, v, w, sg, True, 0.2, True)
    reference = _reference(kind, x, v, w, sg, True, 0.2, True)
    _assert_outputs(actual, reference)
    _assert_gradients(actual, reference, (x, v, w) if kind == "izhikevich" else (x, v))
    gradient = torch.autograd.grad(actual[0].sum(), x, create_graph=True)[0]
    with pytest.raises(RuntimeError):
        torch.autograd.grad(gradient.sum(), x)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="NVIDIA CUDA unavailable")
def test_non_default_stream_and_second_device():
    for index in range(min(torch.cuda.device_count(), 2)):
        device = torch.device("cuda", index)
        stream = torch.cuda.Stream(device=device)
        with torch.cuda.stream(stream):
            x = torch.ones(4, 17, device=device, requires_grad=True)
            v = torch.zeros(17, device=device, requires_grad=True)
            w = torch.zeros_like(v).requires_grad_()
            actual = _call("izhikevich", x, v, w, surrogate.ATan(), True, 0.2, True)
            expected = _reference(
                "izhikevich", x, v, w, surrogate.ATan(), True, 0.2, True
            )
            _assert_outputs(actual, expected)
            _assert_gradients(actual, expected, (x, v, w))
            output = functional.stbif_multi_step_registered(
                x.detach(),
                v.detach(),
                w.detach(),
                torch.tensor(1.0, device=device),
                torch.tensor(3.0, device=device),
                torch.tensor(-3.0, device=device),
            )
            assert all(t.device == device for t in output)
        stream.synchronize()
