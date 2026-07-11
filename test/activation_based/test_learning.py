import gc

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
        (learning.MSTDPLearner, {"batch_size": 3}),
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
        batch_size=3,
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


def test_mstdp_learner_returns_detached_delta_w():
    fc = layer.Linear(8, 5, bias=False)
    sn = neuron.IFNode()
    critic = nn.Linear(5, 1)
    learner = learning.MSTDPLearner(
        step_mode="s",
        synapse=fc,
        sn=sn,
        tau_pre=2.0,
        tau_post=2.0,
        batch_size=3,
        f_pre=f_weight,
        f_post=f_weight,
    )
    in_spike = (torch.rand(3, 8) > 0.5).float()
    out_spike = sn(fc(in_spike))
    reward = critic(out_spike).squeeze(-1)

    delta_w = learner.step(reward, on_grad=False)
    assert delta_w.grad_fn is None
    assert not delta_w.requires_grad

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


if __name__ == "__main__":
    test_stdp_learner_records_are_detached()
    test_stdp_learner_step_does_not_retain_graph()
    test_stdp_learner_matches_functional_update()
    test_stdp_learner_frees_tensors_across_runs()
    test_mstdp_learners_records_are_detached()
    test_mstdp_learners_step_detaches_reward()
    test_mstdp_learner_returns_detached_delta_w()
    test_mstdp_learners_free_tensors_with_graph_connected_reward()
    print("Done!")
