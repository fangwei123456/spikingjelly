import SpikingFlow.simulating as simulating
import SpikingFlow.learning as learning
import SpikingFlow.connection as connection
import SpikingFlow.connection.transform as tf
import SpikingFlow.neuron as neuron
import torch
from matplotlib import pyplot

# 新建一个仿真器
sim = simulating.Simulator()

# 添加各个模块。为了更明显的观察到脉冲，我们使用IF神经元，而且把膜电阻设置的很大
# 突触的pre是2个输入，而post是1个输出，连接权重是shape=[1, 2]的tensor
sim.append(learning.STDPModule(tf.SpikeCurrent(amplitude=0.5),
                               connection.Linear(2, 1),
                               neuron.IFNode(shape=[1], r=50.0, v_threshold=1.0),
                               tau_pre=10.0,
                               tau_post=10.0,
                               learning_rate=1e-3
                               ))
# 新建list，分别保存pre的2个输入脉冲、post的1个输出脉冲，以及对应的连接权重
pre_spike_list0 = []
pre_spike_list1 = []
post_spike_list = []
w_list0 = []
w_list1 = []
T = 200

for t in range(T):
    if t < 100:
        # 前100步仿真，pre_spike[0]和pre_spike[1]都是发放一次1再发放一次0
        if t % 2 == 0:
            pre_spike = torch.ones(size=[2], dtype=torch.bool)
        else:
            pre_spike = torch.zeros(size=[2], dtype=torch.bool)
    else:
        # 后100步仿真，pre_spike[0]一直为0，而pre_spike[1]一直为1
        pre_spike = torch.zeros(size=[2], dtype=torch.bool)
        pre_spike[1] = True

    post_spike = sim.step(pre_spike)
    pre_spike_list0.append(pre_spike[0].float().item())
    pre_spike_list1.append(pre_spike[1].float().item())

    post_spike_list.append(post_spike.float().item())

    w_list0.append(sim.module_list[-1].module_list[2].w[:, 0].item())
    w_list1.append(sim.module_list[-1].module_list[2].w[:, 1].item())

# 画出pre_spike[0]
pyplot.bar(torch.arange(0, T).tolist(), pre_spike_list0, width=0.1, label='pre_spike[0]')
pyplot.legend()
pyplot.show()

# 画出pre_spike[1]
pyplot.bar(torch.arange(0, T).tolist(), pre_spike_list1, width=0.1, label='pre_spike[1]')
pyplot.legend()
pyplot.show()

# 画出post_spike
pyplot.bar(torch.arange(0, T).tolist(), post_spike_list, width=0.1, label='post_spike')
pyplot.legend()
pyplot.show()

# 画出2个输入与1个输出的连接权重w_0和w_1
pyplot.plot(w_list0, c='r', label='w[0]')
pyplot.plot(w_list1, c='g', label='w[1]')
pyplot.legend()
pyplot.show()