import gc

import pytest
import torch
import torch.nn as nn

from spikingjelly.activation_based import functional, layer, learning, neuron


def f_weight(x):
    return torch.clamp(x, -1.0, 1.0)


def _count_alive_tensors():
    gc.collect()
    return sum(1 for obj in gc.get_objects() if torch.is_tensor(obj))


def _build_net(step_mode):
    net = nn.Sequential(
        layer.Conv2d(1, 4, kernel_size=3, stride=1, bias=False),
        neuron.IFNode(),
        layer.Flatten(),
        layer.Linear(4 * 6 * 6, 5, bias=False),
        neuron.LIFNode(tau=2.0),
    )
    functional.set_step_mode(net, step_mode)
    return net


def _build_learners(net, step_mode):
    learners = []
    for i in range(len(net)):
        if isinstance(net[i], (layer.Conv2d, layer.Linear)):
            learners.append(
                learning.STDPLearner(
                    step_mode=step_mode,
                    synapse=net[i],
                    sn=net[i + 1],
                    tau_pre=2.0,
                    tau_post=2.0,
                    f_pre=f_weight,
                    f_post=f_weight,
                )
            )
    return learners


def test_stdp_learner_records_are_detached():
    net = _build_net("m")
    learners = _build_learners(net, "m")

    x = (torch.rand(4, 2, 1, 8, 8) > 0.5).float()
    y = net(x)

    assert y.requires_grad
    for learner in learners:
        for rec in learner.in_spike_monitor.records + learner.out_spike_monitor.records:
            assert not rec.requires_grad
            assert rec.grad_fn is None


def test_stdp_learner_step_does_not_retain_graph():
    net = _build_net("m")
    learners = _build_learners(net, "m")
    optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

    x = (torch.rand(4, 2, 1, 8, 8) > 0.5).float()
    net(x)

    optimizer.zero_grad()
    for learner in learners:
        learner.step(on_grad=True)
    optimizer.step()

    for learner in learners:
        assert learner.synapse.weight.grad is not None
        assert learner.synapse.weight.grad.grad_fn is None
        assert not learner.synapse.weight.grad.requires_grad
        if isinstance(learner.trace_pre, torch.Tensor):
            assert learner.trace_pre.grad_fn is None
        if isinstance(learner.trace_post, torch.Tensor):
            assert learner.trace_post.grad_fn is None

    functional.reset_net(net)
    for learner in learners:
        learner.reset()


def test_stdp_learner_matches_functional_update():
    torch.manual_seed(0)
    fc = layer.Linear(8, 5, bias=False)
    sn = neuron.IFNode()

    learner = learning.STDPLearner(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        f_pre=f_weight,
        f_post=f_weight,
    )

    in_spike = (torch.rand(3, 8) > 0.5).float()
    out_spike = sn(fc(in_spike))

    delta_w = learner.step(on_grad=False)

    _trace_pre, _trace_post, expected = learning.stdp_linear_single_step(
        fc, in_spike, out_spike.detach(), None, None, 2.0, 2.0, f_weight, f_weight
    )
    assert torch.allclose(delta_w, expected)
    assert delta_w.grad_fn is None
    assert learner.in_spike_monitor.records == []
    assert learner.out_spike_monitor.records == []
    assert learner.in_spike_monitor[""] == []
    assert learner.out_spike_monitor[""] == []


@pytest.mark.parametrize(
    ("learner_cls", "kwargs", "reward"),
    (
        (learning.STDPLearner, {}, None),
        (learning.MSTDPLearner, {}, torch.zeros(3)),
        (learning.MSTDPETLearner, {"tau_trace": 2.0}, torch.tensor(0.0)),
    ),
)
def test_learners_validate_monitor_record_counts(learner_cls, kwargs, reward):
    fc = layer.Linear(8, 5, bias=False)
    sn = neuron.IFNode()
    learner = learner_cls(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        **kwargs,
    )
    step_args = () if reward is None else (reward,)

    assert learner.step(*step_args) is None
    assert fc.weight.grad is None

    fc(torch.ones(3, 8))
    with pytest.raises(RuntimeError, match="record counts differ"):
        learner.step(*step_args)


def test_stdp_learner_frees_tensors_across_runs():
    # regression test for #576: recreating model + learner in a loop must not
    # accumulate tensors that survive garbage collection
    def one_run():
        net = _build_net("m")
        learners = _build_learners(net, "m")
        optimizer = torch.optim.SGD(net.parameters(), lr=0.01)

        for _ in range(2):
            x = (torch.rand(4, 2, 1, 8, 8) > 0.5).float()
            net(x)

            optimizer.zero_grad()
            for learner in learners:
                learner.step(on_grad=True)
            optimizer.step()

            functional.reset_net(net)
            for learner in learners:
                learner.reset()

    one_run()
    baseline = _count_alive_tensors()
    for _ in range(3):
        one_run()
    # a real leak accumulates hundreds of tensors per run; the tolerance only
    # absorbs unrelated interpreter/pytest noise
    assert _count_alive_tensors() <= baseline + 5


def test_mstdp_learners_records_are_detached():
    for cls, kwargs in (
        (learning.MSTDPLearner, {}),
        (learning.MSTDPETLearner, {"tau_trace": 2.0}),
    ):
        fc = layer.Linear(8, 5, bias=False)
        sn = neuron.IFNode()
        learner = cls(
            step_mode="s",
            synapse=fc,
            sn=sn,
            tau_pre=2.0,
            tau_post=2.0,
            f_pre=f_weight,
            f_post=f_weight,
            **kwargs,
        )
        in_spike = (torch.rand(3, 8) > 0.5).float()
        y = sn(fc(in_spike))

        assert y.requires_grad
        records = learner.in_spike_monitor.records + learner.out_spike_monitor.records
        assert len(records) == 2
        for rec in records:
            assert not rec.requires_grad
            assert rec.grad_fn is None

        functional.reset_net(sn)


def _run_mstdp_with_critic_reward(steps=4):
    # reward comes out of a differentiable critic, as in reward-modulated
    # training loops, and is passed to step() without an explicit detach
    fc = layer.Linear(8, 5, bias=False)
    sn = neuron.IFNode()
    critic = nn.Linear(5, 1)
    learner = learning.MSTDPLearner(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    for _ in range(steps):
        in_spike = (torch.rand(3, 8) > 0.5).float()
        out_spike = sn(fc(in_spike))
        reward = critic(out_spike).squeeze(-1)
        learner.step(reward, on_grad=True)
    return fc, sn, learner


def _run_mstdpet_with_critic_reward(steps=4):
    fc = layer.Linear(8, 5, bias=False)
    sn = neuron.IFNode()
    critic = nn.Linear(5, 1)
    learner = learning.MSTDPETLearner(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        tau_trace=2.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    for _ in range(steps):
        in_spike = (torch.rand(8) > 0.5).float()
        out_spike = sn(fc(in_spike))
        reward = critic(out_spike).mean()
        learner.step(reward, on_grad=True)
    return fc, sn, learner


def test_mstdp_learners_step_detaches_reward():
    for run in (_run_mstdp_with_critic_reward, _run_mstdpet_with_critic_reward):
        fc, sn, learner = run()
        assert fc.weight.grad is not None
        assert fc.weight.grad.grad_fn is None
        assert not fc.weight.grad.requires_grad
        functional.reset_net(sn)
        learner.reset()


def test_mstdp_learners_return_detached_delta_w():
    for cls, kwargs, batched in (
        (learning.MSTDPLearner, {}, True),
        (learning.MSTDPETLearner, {"tau_trace": 2.0}, False),
    ):
        fc = layer.Linear(8, 5, bias=False)
        sn = neuron.IFNode()
        critic = nn.Linear(5, 1)
        learner = cls(
            step_mode="s",
            synapse=fc,
            sn=sn,
            tau_pre=2.0,
            tau_post=2.0,
            f_pre=f_weight,
            f_post=f_weight,
            **kwargs,
        )
        if batched:
            in_spike = (torch.rand(3, 8) > 0.5).float()
        else:
            in_spike = (torch.rand(8) > 0.5).float()
        out_spike = sn(fc(in_spike))
        reward = critic(out_spike).squeeze(-1) if batched else critic(out_spike).mean()

        delta_w = learner.step(reward, on_grad=False)
        assert delta_w.grad_fn is None
        assert not delta_w.requires_grad
        assert learner.in_spike_monitor.records == []
        assert learner.out_spike_monitor.records == []
        assert learner.in_spike_monitor[""] == []
        assert learner.out_spike_monitor[""] == []

        functional.reset_net(sn)
        learner.reset()


def test_mstdp_learners_free_tensors_with_graph_connected_reward():
    # regression test for the reward-side counterpart of #576: a
    # graph-connected reward must not accumulate retained graphs across runs
    def one_run():
        for run in (_run_mstdp_with_critic_reward, _run_mstdpet_with_critic_reward):
            fc, sn, learner = run()
            functional.reset_net(sn)
            learner.reset()

    one_run()
    baseline = _count_alive_tensors()
    for _ in range(3):
        one_run()
    assert _count_alive_tensors() <= baseline + 5


def test_functional_learning_linear_helpers_match_legacy_helpers():
    fc = layer.Linear(4, 3, bias=False)
    in_spike = (torch.rand(2, 4) > 0.5).float()
    out_spike = (torch.rand(2, 3) > 0.5).float()

    expected = learning.stdp_linear_single_step(
        fc, in_spike, out_spike, None, None, 2.0, 3.0, f_weight, f_weight
    )
    delta_w, (trace_pre, trace_post) = functional.stdp_linear_step(
        in_spike,
        out_spike,
        (torch.zeros_like(in_spike), torch.zeros_like(out_spike)),
        fc.weight.data,
        tau_pre=2.0,
        tau_post=3.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    actual = trace_pre, trace_post, delta_w
    for a, b in zip(actual, expected, strict=True):
        assert torch.allclose(a, b)

    expected = learning.mstdp_linear_single_step(
        fc, in_spike, out_spike, None, None, 2.0, 3.0, f_weight, f_weight
    )
    eligibility, (trace_pre, trace_post) = functional.mstdp_linear_step(
        in_spike,
        out_spike,
        (torch.zeros_like(in_spike), torch.zeros_like(out_spike)),
        fc.weight.data,
        tau_pre=2.0,
        tau_post=3.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    actual = trace_pre, trace_post, eligibility
    for a, b in zip(actual, expected, strict=True):
        assert torch.allclose(a, b)

    expected = learning.mstdpet_linear_single_step(
        fc, in_spike[0], out_spike[0], None, None, 2.0, 3.0, f_weight, f_weight
    )
    eligibility, (trace_pre, trace_post) = functional.mstdpet_linear_step(
        in_spike[0],
        out_spike[0],
        (torch.zeros_like(in_spike[0]), torch.zeros_like(out_spike[0])),
        fc.weight.data,
        tau_pre=2.0,
        tau_post=3.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    actual = trace_pre, trace_post, eligibility
    for a, b in zip(actual, expected, strict=True):
        assert torch.allclose(a, b)


def test_functional_learning_conv_helpers_match_legacy_helpers():
    conv1 = layer.Conv1d(2, 3, kernel_size=3, padding=1, bias=False)
    in_spike1 = (torch.rand(2, 2, 5) > 0.5).float()
    out_spike1 = (torch.rand(2, 3, 5) > 0.5).float()
    expected1 = learning.stdp_conv1d_single_step(
        conv1, in_spike1, out_spike1, None, None, 2.0, 3.0, f_weight, f_weight
    )
    padded_in_spike1 = nn.functional.pad(in_spike1, (1, 1))
    delta_w1, (trace_pre1, trace_post1) = functional.stdp_conv1d_step(
        padded_in_spike1,
        out_spike1,
        (torch.zeros_like(padded_in_spike1), torch.zeros_like(out_spike1)),
        conv1.weight.data,
        stride=conv1.stride,
        tau_pre=2.0,
        tau_post=3.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    actual1 = trace_pre1, trace_post1, delta_w1
    for a, b in zip(actual1, expected1, strict=True):
        assert torch.allclose(a, b)

    conv2 = layer.Conv2d(2, 3, kernel_size=3, padding=1, bias=False)
    in_spike2 = (torch.rand(2, 2, 5, 5) > 0.5).float()
    out_spike2 = (torch.rand(2, 3, 5, 5) > 0.5).float()
    expected2 = learning.stdp_conv2d_single_step(
        conv2, in_spike2, out_spike2, None, None, 2.0, 3.0, f_weight, f_weight
    )
    padded_in_spike2 = nn.functional.pad(in_spike2, (1, 1, 1, 1))
    delta_w2, (trace_pre2, trace_post2) = functional.stdp_conv2d_step(
        padded_in_spike2,
        out_spike2,
        (torch.zeros_like(padded_in_spike2), torch.zeros_like(out_spike2)),
        conv2.weight.data,
        stride=conv2.stride,
        tau_pre=2.0,
        tau_post=3.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    actual2 = trace_pre2, trace_post2, delta_w2
    for a, b in zip(actual2, expected2, strict=True):
        assert torch.allclose(a, b)


def test_stdp_conv_helpers_keep_supported_boundary():
    conv1 = layer.Conv1d(2, 3, kernel_size=3, dilation=2, bias=False)
    with pytest.raises(NotImplementedError, match="dilation"):
        learning.stdp_conv1d_single_step(
            conv1,
            torch.zeros(1, 2, 5),
            torch.zeros(1, 3, 1),
            None,
            None,
            2.0,
            3.0,
        )

    conv2 = layer.Conv2d(2, 4, kernel_size=3, groups=2, bias=False)
    with pytest.raises(NotImplementedError, match="groups"):
        learning.stdp_conv2d_single_step(
            conv2,
            torch.zeros(1, 2, 5, 5),
            torch.zeros(1, 4, 3, 3),
            None,
            None,
            2.0,
            3.0,
        )


def test_mstdpet_reward_step_uses_explicit_tensor_state():
    eligibility = torch.arange(6, dtype=torch.float32).view(3, 2)
    trace_e = torch.ones(3, 2)
    dw, trace_e_next = functional.mstdpet_reward_step(
        0.25,
        eligibility,
        trace_e,
        tau_trace=2.0,
    )
    expected_trace = trace_e * torch.exp(torch.tensor(-0.5)) + eligibility / 2.0
    assert torch.allclose(trace_e_next, expected_trace)
    assert torch.allclose(dw, 0.25 * expected_trace)


def test_stdp_multi_step_preserves_empty_sequence_state():
    synapse = layer.Linear(3, 4)
    trace_pre = torch.randn(2, 3)
    trace_post = torch.randn(2, 4)

    trace_pre_next, trace_post_next, delta_w = learning.stdp_multi_step(
        synapse,
        torch.empty(0, 2, 3),
        torch.empty(0, 2, 4),
        trace_pre,
        trace_post,
        tau_pre=2.0,
        tau_post=3.0,
    )

    assert trace_pre_next is trace_pre
    assert trace_post_next is trace_post
    assert torch.equal(delta_w, torch.zeros_like(synapse.weight))


@pytest.mark.parametrize("dtype", [torch.float16, torch.float64])
def test_mstdp_learner_eligibility_follows_synapse_dtype(dtype):
    # A non-float32 synapse (AMP half, or double) must not produce a
    # mismatched-dtype eligibility / weight gradient. MSTDPLearner initialised
    # ``eligibility`` with torch.zeros(...) and no ``dtype=``, so it defaulted to
    # float32; on a half synapse the resulting ``weight.grad = -delta_w`` raised
    # a dtype-assignment RuntimeError. MSTDPETLearner already does this correctly
    # via torch.zeros_like(self.synapse.weight).
    fc = nn.Linear(8, 5, bias=False).to(dtype)
    sn = neuron.LIFNode(step_mode="s")
    learner = learning.MSTDPLearner(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        f_pre=f_weight,
        f_post=f_weight,
    )
    in_spike = (torch.rand(3, 8) > 0.5).to(dtype)
    sn(fc(in_spike))
    learner.step(torch.ones(3))

    assert learner.eligibility.dtype == dtype
    assert fc.weight.grad is not None
    assert fc.weight.grad.dtype == fc.weight.dtype


@pytest.mark.parametrize(
    "events,expected",
    [
        ([(1.0, 0.0), (0.0, 1.0)], [0.0, 0.5]),
        ([(0.0, 1.0), (1.0, 0.0)], [0.0, -0.5]),
        ([(1.0, 1.0), (0.0, 0.0)], [0.0, 0.0]),
    ],
)
def test_stdp_scalar_timing_oracle(events, expected):
    fc = nn.Linear(1, 1, bias=False).double()
    trace = (torch.zeros(1, 1, dtype=torch.float64),) * 2
    for (pre, post), value in zip(events, expected):
        dw, trace = functional.stdp_linear_step(
            torch.tensor([[pre]], dtype=torch.float64),
            torch.tensor([[post]], dtype=torch.float64),
            trace,
            fc.weight,
            tau_pre=2.0,
            tau_post=2.0,
            f_pre=torch.ones_like,
            f_post=torch.ones_like,
        )
        torch.testing.assert_close(
            dw, torch.tensor([[value]], dtype=torch.float64), rtol=1e-10, atol=1e-12
        )


def test_stdp_nonzero_state_and_asymmetric_simultaneous_spikes():
    weight = torch.tensor([[0.4]], dtype=torch.float64, requires_grad=True)
    pre = post = torch.ones(1, 1, dtype=torch.float64)
    trace = (
        torch.tensor([[2.0]], dtype=torch.float64),
        torch.tensor([[4.0]], dtype=torch.float64),
    )
    dw, states = functional.stdp_linear_step(
        pre,
        post,
        trace,
        weight,
        tau_pre=2.0,
        tau_post=2.0,
        f_pre=lambda w: w,
        f_post=lambda w: 2 * w,
    )
    # Updated traces are 2 and 3: .8*2 - .4*3 = .4.
    torch.testing.assert_close(
        dw, torch.tensor([[0.4]], dtype=torch.float64), rtol=1e-10, atol=1e-12
    )
    assert dw.requires_grad
    assert trace[0].item() == 2 and trace[1].item() == 4
    assert states[0].item() == 2 and states[1].item() == 3


@pytest.mark.parametrize(
    "cls,kwargs,batched",
    [
        (learning.MSTDPLearner, {}, True),
        (learning.MSTDPETLearner, {"tau_trace": 2.0}, False),
    ],
)
def test_reward_delayed_timing_oracle(cls, kwargs, batched):
    import math

    fc = nn.Linear(1, 1, bias=False).double()
    sn = nn.Identity()
    learner = cls(
        "s", fc, sn, 2.0, 2.0, f_pre=torch.ones_like, f_post=torch.ones_like, **kwargs
    )

    def emit(pre, post, reward):
        x = torch.tensor([[pre]] if batched else [pre], dtype=torch.float64)
        y = torch.tensor([[post]] if batched else [post], dtype=torch.float64)
        fc(x)
        sn(y)
        return learner.step(reward, on_grad=False).item()

    try:
        assert emit(1.0, 0.0, 1.0) == 0.0
        assert emit(0.0, 1.0, 1.0) == 0.0
        value = math.exp(-0.5) * (1.0 if batched else 0.5)
        assert emit(0.0, 0.0, 1.0) == pytest.approx(value)
        # MSTDP loses pair eligibility on the quiet step; ET retains its memory.
        negative = 0.0 if batched else -value * math.exp(-0.5)
        assert emit(0.0, 0.0, -1.0) == pytest.approx(negative)
        assert emit(0.0, 0.0, 0.0) == 0.0
        learner.reset()
        assert emit(0.0, 0.0, 1.0) == 0.0
    finally:
        learner.close()


def test_reward_batch_modulation_and_reset_size():
    import math

    fc = nn.Linear(1, 1, bias=False).double()
    sn = nn.Identity()
    learner = learning.MSTDPLearner(
        "s", fc, sn, 2.0, 2.0, f_pre=torch.ones_like, f_post=torch.ones_like
    )
    try:
        for pre, post in (
            ([[1.0], [0.0]], [[0.0], [1.0]]),
            ([[0.0], [1.0]], [[1.0], [0.0]]),
        ):
            fc(torch.tensor(pre, dtype=torch.float64))
            sn(torch.tensor(post, dtype=torch.float64))
            learner.step(0.0)
        fc(torch.zeros(2, 1, dtype=torch.float64))
        sn(torch.zeros(2, 1, dtype=torch.float64))
        dw = learner.step(torch.tensor([2.0, -3.0]), on_grad=False)
        assert dw.item() == pytest.approx(5 * math.exp(-0.5))
        fc(torch.zeros(1, 1, dtype=torch.float64))
        sn(torch.zeros(1, 1, dtype=torch.float64))
        with pytest.raises(ValueError, match="reset"):
            learner.step(1.0)
        learner.reset()
        fc(torch.zeros(1, 1, dtype=torch.float64))
        sn(torch.zeros(1, 1, dtype=torch.float64))
        assert learner.step(1.0, on_grad=False).item() == 0
    finally:
        learner.close()


@pytest.mark.parametrize("ndim", [1, 2])
def test_conv_scalar_oracle_with_padding_and_stride(ndim):
    import itertools

    conv_cls = nn.Conv1d if ndim == 1 else nn.Conv2d
    conv = conv_cls(2, 2, 2, stride=2, padding=1, bias=False).double()
    shape = (2, 2, *((3,) * ndim))
    pre = (torch.arange(torch.tensor(shape).prod()).reshape(shape) % 3 == 0).double()
    output_shape = (2, 2, *((2,) * ndim))
    post = (
        torch.arange(torch.tensor(output_shape).prod()).reshape(output_shape) % 2 == 0
    ).double()
    padded = nn.functional.pad(pre, (1, 1) * ndim)
    trace_pre = torch.full_like(padded, 2.0)
    trace_post = torch.full_like(post, 3.0)
    expected = torch.zeros_like(conv.weight)
    for n, out_c, in_c in itertools.product(range(2), repeat=3):
        for kernel in itertools.product(range(2), repeat=ndim):
            for location in itertools.product(range(2), repeat=ndim):
                source = tuple(2 * loc + k for loc, k in zip(location, kernel))
                a = padded[(n, in_c, *source)].item()
                b = post[(n, out_c, *location)].item()
                expected[(out_c, in_c, *kernel)] += (1 + a) * b - (1.5 + b) * a
    func = (
        learning.stdp_conv1d_single_step
        if ndim == 1
        else learning.stdp_conv2d_single_step
    )
    _, _, dw = func(
        conv,
        pre,
        post,
        trace_pre,
        trace_post,
        2.0,
        2.0,
        torch.ones_like,
        torch.ones_like,
    )
    torch.testing.assert_close(dw, expected, rtol=1e-10, atol=1e-12)


def test_stdp_sequence_sum_and_gradient_injection():
    fc = nn.Linear(2, 2, bias=False).double()
    sn = nn.Identity()
    pre = torch.tensor([[[1.0, 0.0]], [[0.0, 1.0]], [[1.0, 1.0]]], dtype=torch.float64)
    post = pre.flip(-1)
    tr_pre, tr_post, expected = learning.stdp_multi_step(
        fc, pre, post, None, None, 2.0, 3.0
    )
    learner = learning.STDPLearner("s", fc, sn, 2.0, 3.0)
    try:
        for x, y in zip(pre, post):
            fc(x)
            sn(y)
        fc.weight.grad = torch.ones_like(fc.weight)
        weight = fc.weight.detach().clone()
        learner.step(scale=0.25)
        torch.testing.assert_close(fc.weight.grad, 1 - 0.25 * expected)
        torch.testing.assert_close(learner.trace_pre, tr_pre)
        torch.testing.assert_close(learner.trace_post, tr_post)
        torch.optim.SGD(fc.parameters(), lr=0.1).step()
        torch.testing.assert_close(fc.weight, weight - 0.1 * (1 - 0.25 * expected))
        assert not learner.in_spike_monitor.records
    finally:
        learner.close()


@pytest.mark.parametrize(
    "cls,kwargs,batched",
    [
        (learning.STDPLearner, {}, True),
        (learning.MSTDPLearner, {}, True),
        (learning.MSTDPETLearner, {"tau_trace": 2.0}, False),
    ],
)
def test_learner_lifecycle_and_callback_graph(cls, kwargs, batched):
    fc = nn.Linear(2, 2, bias=False)
    sn = nn.Identity()
    coefficient = nn.Parameter(torch.tensor(2.0))
    learner = cls("s", fc, sn, 2.0, 2.0, f_pre=lambda w: coefficient * w, **kwargs)
    step_args = (
        () if cls is learning.STDPLearner else (torch.tensor(1.0, requires_grad=True),)
    )
    shape = (1, 2) if batched else (2,)
    try:
        fc(torch.ones(shape))
        sn(torch.ones(shape))
        dw = learner.step(*step_args, on_grad=False)
        assert not dw.requires_grad
        assert learner.trace_pre.grad_fn is None
        trace = learner.trace_pre.clone()
        learner.disable()
        fc(torch.ones(shape))
        sn(torch.ones(shape))
        assert not learner.in_spike_monitor.records
        learner.reset()
        assert learner.trace_pre is None
        assert not learner.in_spike_monitor.is_enable()
        learner.enable()
        fc(torch.ones(shape))
        sn(torch.ones(shape))
        learner.step(*step_args)
        torch.testing.assert_close(learner.trace_pre, trace)
    finally:
        learner.close()
    learner.close()
    assert not fc._forward_pre_hooks and not sn._forward_hooks
    assert learner.trace_pre is None
    for action in (learner.enable, lambda: learner.step(*step_args)):
        with pytest.raises(RuntimeError, match="closed"):
            action()


@pytest.mark.parametrize(
    "cls,kwargs",
    [(learning.MSTDPLearner, {}), (learning.MSTDPETLearner, {"tau_trace": 2.0})],
)
def test_reward_records_and_shapes_fail_before_state_update(cls, kwargs):
    fc, sn = nn.Linear(2, 2), nn.Identity()
    learner = cls("s", fc, sn, 2.0, 2.0, **kwargs)
    shape = (1, 2) if cls is learning.MSTDPLearner else (2,)
    try:
        for _ in range(2):
            fc(torch.ones(shape))
            sn(torch.ones(shape))
        with pytest.raises(RuntimeError, match="exactly one"):
            learner.step(1.0)
        assert learner.trace_pre is None
        learner.reset()
        fc(torch.ones(shape))
        sn(torch.ones(shape))
        with pytest.raises(ValueError, match="reward"):
            learner.step(torch.ones(2, 1))
        assert learner.trace_pre is None
        assert len(learner.in_spike_monitor.records) == 1
    finally:
        learner.close()


@pytest.mark.parametrize("tau", [0.0, -0.5, 0.5, float("inf"), float("nan")])
def test_invalid_stdp_constructor_installs_no_hooks(tau):
    fc, sn = nn.Linear(2, 2), nn.Identity()
    with pytest.raises(ValueError, match="tau"):
        learning.STDPLearner("s", fc, sn, tau, 2.0)
    assert not fc._forward_pre_hooks and not sn._forward_hooks


def test_unsupported_constructors_and_time_lengths():
    fc, sn = nn.Conv2d(2, 2, 3, groups=2), nn.Identity()
    with pytest.raises(NotImplementedError, match="groups"):
        learning.STDPLearner("s", fc, sn, 2.0, 2.0)
    with pytest.raises(ValueError, match="step_mode"):
        learning.MSTDPLearner("m", nn.Linear(2, 2), sn, 2.0, 2.0)
    with pytest.raises(ValueError, match="time lengths"):
        learning.stdp_multi_step(
            nn.Linear(2, 2),
            torch.zeros(2, 1, 2),
            torch.zeros(1, 1, 2),
            None,
            None,
            2.0,
            2.0,
        )


def test_mnist_competition_mapping_and_checkpoint(tmp_path):
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _assign_labels,
        _predict,
        _run_batch,
        _save_checkpoint,
        _load_models,
    )
    import copy

    model = _SNN(4, 0.05, 0.5, steps=100, covariance=torch.eye(784), device="cpu")
    baseline = copy.deepcopy(model)
    node = model.sn.sub_module[1]
    node.v = torch.tensor([[0.03, 0.03, 0.005, 0.001], [0.001, 0.002, 0.003, 0.004]])
    spike = node.neuronal_fire()
    assert spike.tolist() == [[1.0, 1.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]]
    node.neuronal_reset(spike)
    torch.testing.assert_close(node.v[0], torch.tensor([0.0, 0.0, 0.005, 0.001]))
    bias = node.bias_current.clone()
    functional.reset_net(model.sn)
    assert model.sn.y is None and node.adaptation == 0
    torch.testing.assert_close(node.bias_current, bias)
    model.eval()
    model(torch.ones(1, 784))
    torch.testing.assert_close(node.bias_current, bias)
    functional.reset_net(model.sn)
    counts = torch.tensor([[2.0, 0.0, 0.0, 0.0], [0.0, 3.0, 0.0, 0.0]])
    labels = torch.tensor([1, 2])
    assignments = _assign_labels(counts, labels)
    assert assignments.tolist() == [1, 2, -1, -1]
    assert _predict(counts, assignments).tolist() == [1, 2]
    assert _predict(torch.zeros_like(counts), assignments).tolist() == [-1, -1]
    path = tmp_path / "model.pt"
    _save_checkpoint(
        path,
        model,
        baseline,
        assignments,
        assignments,
        {
            "neurons": 4,
            "steps": 100,
            "model_kind": "current_stdp_mnist",
            "input_encoding": "regular_if_uniform_phase",
            "homeostasis_gain": 0.05,
            "inhibition": 0.5,
            "v_threshold": 0.01,
            "inhibition_tau": 5.0,
            "refractory_steps": 5,
            "bias_initial": 0.01,
        },
        1,
    )
    loaded, _ = _load_models(torch.load(path, weights_only=True), "cpu")
    torch.testing.assert_close(loaded.synapse.weight, model.synapse.weight)
    torch.testing.assert_close(
        loaded.sn.sub_module[1].bias_current, model.sn.sub_module[1].bias_current
    )
    assert loaded.sn.sub_module[1].v_threshold == model.sn.sub_module[1].v_threshold
    learner = learning.STDPLearner("s", model.synapse, model.sn, 20.0, 20.0)
    try:
        _run_batch(
            model, torch.rand(2, 28, 28), 4, torch.Generator().manual_seed(0), learner
        )
        assert all(p.grad is None for p in model.parameters())
        assert not learner.in_spike_monitor.records
    finally:
        learner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_learning_reference_and_stable_allocations():
    torch.manual_seed(0)
    cpu_fc = nn.Linear(4, 3, bias=False).double()
    pre, post = (
        torch.rand(2, 4, dtype=torch.float64),
        torch.rand(2, 3, dtype=torch.float64),
    )
    expected = learning.stdp_linear_single_step(cpu_fc, pre, post, None, None, 2.0, 3.0)
    gpu_fc = nn.Linear(4, 3, bias=False).cuda().double()
    gpu_fc.weight.data.copy_(cpu_fc.weight)
    actual = learning.stdp_linear_single_step(
        gpu_fc, pre.cuda(), post.cuda(), None, None, 2.0, 3.0
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=1e-10, atol=1e-12)
    sn = nn.Identity()
    learner = learning.MSTDPLearner("s", gpu_fc, sn, 2.0, 3.0)
    x, y = pre.cuda(), post.cuda()
    samples = []
    try:
        for step in range(1000):
            gpu_fc(x)
            sn(y)
            learner.step(torch.tensor(0.25, device="cuda", requires_grad=True))
            assert not learner.in_spike_monitor.records
            assert learner.eligibility.grad_fn is None
            if step in (100, 500, 999):
                torch.cuda.synchronize()
                samples.append(torch.cuda.memory_allocated())
        assert max(samples) - min(samples) <= 4096
    finally:
        learner.close()


@pytest.mark.parametrize(
    "padding,mode,kernel,expected",
    [
        ("same", "zeros", 2, [2.0, 1.0]),
        ("valid", "zeros", 2, [1.0, 1.0]),
        (1, "reflect", 3, [1.0, 2.0, 1.0]),
    ],
)
def test_conv_native_padding_metadata(padding, mode, kernel, expected):
    conv = nn.Conv1d(
        1, 1, kernel, padding=padding, padding_mode=mode, bias=False
    ).double()
    x = torch.tensor([[[1.0, 0.0, 1.0]]], dtype=torch.float64)
    post = torch.ones_like(conv(x))
    _, _, dw = learning.stdp_conv1d_single_step(
        conv, x, post, None, None, 2.0, 2.0, torch.zeros_like, torch.ones_like
    )
    torch.testing.assert_close(
        dw.flatten(), torch.tensor(expected, dtype=torch.float64)
    )
    learner = learning.STDPLearner("s", conv, nn.Identity(), 2.0, 2.0)
    learner.close()


def test_stdp_pending_shape_error_is_atomic():
    fc, sn = nn.Linear(2, 2), nn.Identity()
    learner = learning.STDPLearner("s", fc, sn, 2.0, 2.0)
    try:
        for batch in (2, 1):
            fc(torch.ones(batch, 2))
            sn(torch.ones(batch, 2))
        with pytest.raises(ValueError, match="reset"):
            learner.step()
        assert learner.trace_pre is None
        assert fc.weight.grad is None
        assert len(learner.in_spike_monitor.records) == 2
    finally:
        learner.close()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_conv_and_eligibility_trace_match_cpu():
    conv = nn.Conv2d(1, 2, 3, padding=1, bias=False).double()
    pre = torch.tensor([[[[1.0, 0.0], [0.0, 1.0]]]], dtype=torch.float64)
    post = torch.ones(1, 2, 2, 2, dtype=torch.float64)
    expected = learning.stdp_conv2d_single_step(conv, pre, post, None, None, 2.0, 3.0)
    conv = conv.cuda()
    actual = learning.stdp_conv2d_single_step(
        conv, pre.cuda(), post.cuda(), None, None, 2.0, 3.0
    )
    for a, b in zip(actual, expected):
        torch.testing.assert_close(a.cpu(), b, rtol=1e-10, atol=1e-12)
    learners = []
    for device in ("cpu", "cuda"):
        fc = nn.Linear(2, 2, bias=False).to(device).double()
        sn = nn.Identity()
        learner = learning.MSTDPETLearner(
            "s",
            fc,
            sn,
            2.0,
            3.0,
            2.0,
            f_pre=torch.ones_like,
            f_post=torch.ones_like,
        )
        learners.append((fc, sn, learner, device))
    try:
        for x, y in (
            ([1.0, 0.0], [0.0, 0.0]),
            ([0.0, 0.0], [1.0, 0.0]),
            ([0.0, 0.0], [0.0, 0.0]),
        ):
            updates = []
            for fc, sn, learner, device in learners:
                fc(torch.tensor(x, dtype=torch.float64, device=device))
                sn(torch.tensor(y, dtype=torch.float64, device=device))
                reward = torch.tensor(0.3, device=device, requires_grad=True)
                updates.append(learner.step(reward, on_grad=False).cpu())
                assert learner.trace_e.grad_fn is None
            torch.testing.assert_close(updates[0], updates[1], rtol=1e-10, atol=1e-12)
    finally:
        for _, _, learner, _ in learners:
            learner.close()


def test_empty_stdp_sequence_still_validates_layout():
    fc = nn.Linear(2, 3)
    with pytest.raises(ValueError, match="time dimension"):
        learning.stdp_multi_step(
            fc, torch.empty(0, 2), torch.empty(0, 3), None, None, 2.0, 2.0
        )
    with pytest.raises(ValueError, match="Sequence shapes"):
        learning.stdp_multi_step(
            fc, torch.empty(0, 1, 2), torch.empty(0, 2, 3), None, None, 2.0, 2.0
        )


def test_stdp_batch_sum_independent_oracle():
    zeros = torch.zeros(2, 2, dtype=torch.float64)
    pre_trace = torch.tensor([[2.0, 0.0], [0.0, 4.0]], dtype=torch.float64)
    dw, _ = functional.stdp_linear_step(
        zeros,
        torch.eye(2, dtype=torch.float64),
        (pre_trace, zeros),
        torch.ones(2, 2, dtype=torch.float64),
        tau_pre=2.0,
        tau_post=2.0,
        f_pre=torch.ones_like,
        f_post=torch.ones_like,
    )
    torch.testing.assert_close(
        dw,
        torch.tensor([[1.0, 0.0], [0.0, 2.0]], dtype=torch.float64),
        rtol=1e-10,
        atol=1e-12,
    )


def test_mnist_recurrent_inhibition_delay_and_reset():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import _SNN

    model = _SNN(
        3,
        0.05,
        0.2,
        bias_initial=0.0,
        steps=100,
        covariance=torch.eye(784),
        device="cpu",
    ).eval()
    node = model.sn.sub_module[1]
    with torch.no_grad():
        first = model.sn(torch.tensor([[12.0, 6.0, 2.0]]))
        torch.testing.assert_close(first, torch.tensor([[1.0, 0.0, 0.0]]))
        torch.testing.assert_close(
            node.v, torch.tensor([[0.0, -0.0013333333, -0.0093333333]])
        )
        second = model.sn(torch.zeros(1, 3))
        torch.testing.assert_close(second, torch.zeros(1, 3))
        torch.testing.assert_close(
            model.sn.sub_module[0].current,
            torch.tensor([[0.8533333, -0.1466667, -0.7866667]]),
        )
        torch.testing.assert_close(
            node.v, torch.tensor([[0.0, -0.0027866667, -0.0171066667]])
        )
        functional.reset_net(model.sn)
        assert model.sn.y is None and node.v == 0
        assert node.adaptation == 0 and node.refractory == 0
        assert model.sn.sub_module[0].current == 0
        torch.testing.assert_close(model.sn(torch.tensor([[12.0, 6.0, 2.0]])), first)
    assert model.sn.element_wise_function.inhibition == 0.2
    assert not any(p.requires_grad for p in model.parameters())


def test_recurrent_mnist_preserves_feedforward_seed_initialization():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import _SNN

    torch.manual_seed(0)
    expected = nn.Linear(784, 4, bias=False)
    with torch.no_grad():
        expected.weight.uniform_(0, 1)
        expected.weight.mul_(78.4 / expected.weight.sum(1, keepdim=True))
    torch.manual_seed(0)
    actual = _SNN(4, 0.05, 0.5, steps=100, covariance=torch.eye(784), device="cpu")
    torch.testing.assert_close(actual.synapse.weight, expected.weight, rtol=0, atol=0)


def test_mnist_cli_inhibition_and_refractory_contract():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _parser,
    )

    args = _parser().parse_args(
        ["--inhibition", "5", "--inhibition-tau", "2", "--bias-initial", "0"]
    )
    model = _SNN(
        2,
        args.homeostasis_gain,
        args.inhibition,
        args.v_threshold,
        args.inhibition_tau,
        args.refractory_steps,
        args.bias_initial,
        steps=100,
        covariance=torch.eye(784),
        device="cpu",
    ).eval()
    node = model.sn.sub_module[1]
    assert model.sn.element_wise_function.inhibition == 5
    assert model.sn.sub_module[0].tau == 2
    with torch.no_grad():
        first = node(torch.tensor([[1.2, 1.2]]))
        assert first.tolist() == [[1.0, 1.0]]
        for _ in range(args.refractory_steps):
            assert not node(torch.tensor([[100.0, 100.0]])).any()
        assert node(torch.tensor([[100.0, 100.0]])).all()


def test_mnist_frozen_evaluation_preserves_model_and_records():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _run_batch,
    )

    model = _SNN(4, 0.5, 1.0, steps=100, covariance=torch.eye(784), device="cpu").eval()
    learner = learning.STDPLearner("s", model.synapse, model.sn, 20.0, 20.0)
    learner.disable()
    before = {key: value.clone() for key, value in model.state_dict().items()}
    try:
        _run_batch(model, torch.ones(2, 28, 28), 20, torch.Generator().manual_seed(3))
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)
        assert not learner.in_spike_monitor.records
        assert not learner.out_spike_monitor.records
        assert model.sn.y is None
        assert model.sn.sub_module[0].current == 0
        assert model.sn.sub_module[1].adaptation == 0
        assert model.sn.sub_module[1].refractory == 0
        assert all(parameter.grad is None for parameter in model.parameters())
    finally:
        learner.close()


def test_mnist_control_keeps_weights_and_gain_while_bias_adapts():
    import copy

    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _run_batch,
    )

    torch.manual_seed(3)
    model = _SNN(8, 0.01, 0.1, steps=100, covariance=torch.eye(784), device="cpu")
    baseline = copy.deepcopy(model)
    weights = baseline.synapse.weight.clone()
    gain = baseline.sn.element_wise_function.input_gain.clone()
    bias = baseline.sn.sub_module[1].bias_current.clone()
    learner = learning.STDPLearner(
        "s",
        model.synapse,
        model.sn,
        20.0,
        20.0,
        f_pre=lambda w: 0.001 * w,
        f_post=lambda w: 0.001 * (1 - w),
    )
    try:
        _run_batch(
            model,
            torch.rand(2, 28, 28),
            100,
            torch.Generator().manual_seed(4),
            learner,
            baseline,
        )
        assert torch.equal(baseline.synapse.weight, weights)
        assert torch.equal(baseline.sn.element_wise_function.input_gain, gain)
        assert not torch.equal(baseline.sn.sub_module[1].bias_current, bias)
        assert baseline.sn.y is None
        assert all(parameter.grad is None for parameter in baseline.parameters())
    finally:
        learner.close()


def test_mnist_intrinsic_current_independent_state_oracle():
    import math
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _AdaptiveLIFNode,
    )

    node = _AdaptiveLIFNode(3, 0.0004, 0.01, 5, 0.0, 700).double()
    voltage = torch.zeros(2, 3, dtype=torch.float64)
    adaptation, refractory = voltage.clone(), voltage.clone()
    bias = torch.zeros(3, dtype=torch.float64)
    rng = torch.Generator().manual_seed(42)
    with torch.no_grad():
        for _ in range(100):
            current = torch.randn(2, 3, generator=rng, dtype=torch.float64) * 0.2 + 0.1
            voltage = voltage + (current + bias - adaptation - voltage) / 100
            voltage = torch.where(refractory > 0, 0.0, voltage)
            spike = ((voltage >= 0.01) & (refractory <= 0)).to(voltage)
            voltage = voltage * (1 - spike)
            refractory = torch.where(spike.bool(), 5, (refractory - 1).clamp_min(0))
            adaptation = math.exp(-1 / 200) * adaptation + 0.01 * spike
            bias = bias + 0.0004 * (120 / (3 * 700) - spike.mean(0))
            actual = node(current)
            torch.testing.assert_close(actual, spike, rtol=0, atol=0)
            for a, b in (
                (node.v, voltage),
                (node.adaptation, adaptation),
                (node.refractory, refractory),
                (node.bias_current, bias),
            ):
                torch.testing.assert_close(a, b, rtol=1e-10, atol=1e-12)
                assert a.grad_fn is None
    saved = node.bias_current.clone()
    functional.reset_net(node)
    torch.testing.assert_close(node.bias_current, saved, rtol=0, atol=0)
    assert node.v == 0 and node.adaptation == 0 and node.refractory == 0


def test_mnist_covariance_and_gain_constraint():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _training_covariance,
    )

    pixels = torch.zeros(4, 28, 28, dtype=torch.uint8)
    pixels[:, 0, 0] = torch.tensor([0, 85, 170, 255], dtype=torch.uint8)
    pixels[:, 0, 1] = torch.tensor([255, 0, 85, 170], dtype=torch.uint8)
    indices = torch.tensor([1, 3, 0])
    actual = _training_covariance(pixels, indices)
    x = pixels[indices].flatten(1).double() / 255
    centered = x - sum(x) / len(x)
    expected = sum(row[:, None] * row[None, :] for row in centered) / (len(x) - 1)
    torch.testing.assert_close(actual.double(), expected, rtol=1e-5, atol=1e-6)
    model = _SNN(4, 0.05, 0.2, steps=100, covariance=torch.eye(784), device="cpu")
    initial = model.synapse.weight.detach().double()
    reference = (initial - initial.mean(0)).square().sum(1)
    with torch.no_grad():
        model.synapse.weight[:, :10].mul_(1.2)
        model.refresh_gain()
    weights = model.synapse.weight.detach().double()
    variance = (weights - weights.mean(0)).square().sum(1)
    gain = model.sn.element_wise_function.input_gain.double()
    torch.testing.assert_close(
        gain.square() * variance, reference, rtol=1e-5, atol=1e-6
    )


def test_mnist_regular_input_counts_and_legacy_checkpoint_rejection():
    from spikingjelly.activation_based.examples.plasticity.stdp_mnist import (
        _SNN,
        _run_batch,
        _load_models,
    )

    model = _SNN(
        4, 0.05, 0.2, steps=700, covariance=torch.eye(784), device="cpu"
    ).eval()
    images = torch.rand(2, 28, 28)
    images[:, 0, 0] = 0
    seen = []
    handle = model.synapse.register_forward_pre_hook(
        lambda _, args: seen.append(args[0].detach().clone())
    )
    try:
        _run_batch(model, images, 700, torch.Generator().manual_seed(42))
    finally:
        handle.remove()
    incoming = sum(seen)
    assert ((incoming - images.flatten(1) * 0.064 * 700).abs() <= 1.0001).all()
    assert incoming[:, 0].eq(0).all()
    with pytest.raises(ValueError, match="archived"):
        _load_models({"config": {"inhibition_kind": "conductance"}}, "cpu")
