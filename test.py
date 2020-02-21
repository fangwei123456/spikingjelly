import SpikingFlow.simulating as simulating
import SpikingFlow.learning as learning
import SpikingFlow.connection as connection
import SpikingFlow.connection.transform as tf
import SpikingFlow.neuron as neuron
import torch
from matplotlib import pyplot

# 定义权值函数f_w
def f_w(x: torch.Tensor):
    x_abs = x.abs()
    return x_abs / (x_abs.sum() + 1e-6)

# 新建一个仿真器
sim = simulating.Simulator()

# 放入脉冲电流转换器、突触、LIF神经元
sim.append(tf.SpikeCurrent(amplitude=0.5))
sim.append(connection.Linear(2, 1))
sim.append(neuron.LIFNode(shape=[1], r=10.0, v_threshold=1.0, tau=100.0))

# 新建一个STDPUpdater
updater = learning.STDPUpdater(tau_pre=50.0,
                               tau_post=100.0,
                               learning_rate=1e-1,
                               f_w=f_w)

# 新建list，保存pre脉冲、post脉冲、突触权重w_00, w_01
pre_spike_list0 = []
pre_spike_list1 = []
post_spike_list = []
w_list0 = []
w_list1 = []

T = 500
for t in range(T):
    if t < 250:
        if t % 2 == 0:
            pre_spike = torch.ones(size=[2], dtype=torch.bool)
        else:
            pre_spike = torch.randint(low=0, high=2, size=[2]).bool()
    else:
        pre_spike = torch.zeros(size=[2], dtype=torch.bool)
        if t % 2 == 0:
            pre_spike[1] = True




    pre_spike_list0.append(pre_spike[0].float().item())
    pre_spike_list1.append(pre_spike[1].float().item())

    post_spike = sim.step(pre_spike)

    updater.update(sim.module_list[1], pre_spike, post_spike)

    post_spike_list.append(post_spike.float().item())

    w_list0.append(sim.module_list[1].w[:, 0].item())
    w_list1.append(sim.module_list[1].w[:, 1].item())

pyplot.figure(figsize=(8, 16))
pyplot.subplot(4, 1, 1)
pyplot.bar(torch.arange(0, T).tolist(), pre_spike_list0, width=0.1, label='pre_spike[0]')
pyplot.legend()

pyplot.subplot(4, 1, 2)
pyplot.bar(torch.arange(0, T).tolist(), pre_spike_list1, width=0.1, label='pre_spike[1]')
pyplot.legend()

pyplot.subplot(4, 1, 3)
pyplot.bar(torch.arange(0, T).tolist(), post_spike_list, width=0.1, label='post_spike')
pyplot.legend()

pyplot.subplot(4, 1, 4)
pyplot.plot(w_list0, c='r', label='w[0]')
pyplot.plot(w_list1, c='g', label='w[1]')
pyplot.legend()
pyplot.show()