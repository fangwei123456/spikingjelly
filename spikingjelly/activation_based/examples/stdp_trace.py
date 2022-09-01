import torch
import torch.nn as nn
from spikingjelly.activation_based import neuron, layer, learning
from matplotlib import pyplot as plt
torch.manual_seed(0)
# plt.style.use(['science'])

if __name__ == '__main__':
    fig = plt.figure(dpi=200, figsize=(8, 6))

    def f_weight(x):
        return torch.clamp(x, -1, 1.)

    tau_pre = 2.
    tau_post = 100.
    T = 128
    N = 1
    lr = 0.01
    net = nn.Sequential(
        layer.Linear(1, 1, bias=False),
        neuron.IFNode()
    )
    nn.init.constant_(net[0].weight.data, 0.4)
    optimizer = torch.optim.SGD(net.parameters(), lr=lr, momentum=0.)

    in_spike = (torch.rand([T, N, 1]) > 0.7).float()
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

    cmap = plt.get_cmap('tab10')
    plt.subplot(5, 1, 1)
    plt.eventplot((in_spike * t)[in_spike == 1], lineoffsets=0, colors=cmap(0))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[i]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 2)
    plt.plot(t, trace_pre, c=cmap(1))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$tr_{pre}$', rotation=0)
    plt.yticks([trace_pre.min().item(), trace_pre.max().item()])
    plt.xticks([])

    plt.subplot(5, 1, 3)
    plt.eventplot((out_spike * t)[out_spike == 1], lineoffsets=0, colors=cmap(2))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$s[j]$', rotation=0, labelpad=10)
    plt.xticks([])
    plt.yticks([])

    plt.subplot(5, 1, 4)
    plt.plot(t, trace_post, c=cmap(3))
    plt.ylabel('$tr_{post}$', rotation=0)
    plt.yticks([trace_post.min().item(), trace_post.max().item()])

    plt.xlim(-0.5, T + 0.5)
    plt.xticks([])

    plt.subplot(5, 1, 5)
    plt.plot(t, weight, c=cmap(4))
    plt.xlim(-0.5, T + 0.5)
    plt.ylabel('$w[i][j]$', rotation=0)
    plt.yticks([weight.min().item(), weight.max().item()])
    plt.xlabel('time-step')
    plt.show()
    # plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/trace.png')
    # plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/trace.svg')
    # plt.savefig('./docs/source/_static/tutorials/activation_based/stdp/trace.pdf')



