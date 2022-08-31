import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning
from matplotlib import pyplot as plt

if __name__ == '__main__':
    fig = plt.figure(dpi=200, figsize=(8, 6))

    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    tau_pre = 20.
    tau_post = 100.
    T = 64
    N = 1
    lr = 0.1
    net = nn.Sequential(
        layer.Linear(1, 1, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.2)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

    in_spike = (torch.rand([T, N, 1]) > 0.8).float()
    stdp_learner = learning.STDPLearner(step_mode='s', synapse=net[0], sn=net[1], tau_pre=tau_pre, tau_post=tau_post,
                                        f_pre=f_weight, f_post=f_weight)

    out_spike = []
    trace_pre = []
    trace_post = []
    weight = []
    with torch.no_grad():
        for t in range(T):
            optimizer.zero_grad()
            out_spike.append(net(in_spike[t]).squeeze())
            stdp_learner.step(on_grad=True)
            optimizer.step()
            weight.append(net[0].weight.data.clone().squeeze())
            trace_pre.append(stdp_learner.trace_pre.squeeze())
            trace_post.append(stdp_learner.trace_post.squeeze())

    in_spike = in_spike.squeeze()
    out_spike = torch.stack(out_spike)
    trace_pre = torch.stack(trace_pre)
    trace_post = torch.stack(trace_post)
    weight = torch.stack(weight)

    t = torch.arange(0, T).float()

    plt.subplot(5, 1, 1)
    plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0, colors='r')
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[i]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 2)
    plt.plot(t, trace_pre)
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$tr_{pre}$', rotation=0)
    plt.yticks([trace_pre.min().item(), trace_pre.max().item()])
    plt.xticks([])

    plt.subplot(5, 1, 3)
    plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0, colors='r')
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[j]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 4)
    plt.plot(t, trace_post)
    plt.ylabel('$tr_{post}$', rotation=0)
    plt.yticks([trace_post.min().item(), trace_post.max().item()])

    plt.xlim(-0.5, T + 0.5)
    plt.xticks([])

    plt.subplot(5, 1, 5)
    plt.plot(t, weight)
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$w[i][j]$', rotation=0)
    plt.yticks([weight.min().item(), weight.max().item()])
    plt.xlabel('time-step')
    plt.show()



