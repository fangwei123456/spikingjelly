import SpikingFlow
import SpikingFlow.neuron as neuron
# 导入绘图模块
from matplotlib import pyplot
import torch

# 新建一个LIF神经元
lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
# 新建一个空list，保存仿真过程中神经元的电压值
v_list = []
# 新建一个空list，保存神经元的输出脉冲
spike_list = []

T = 200
# 运行200次
for t in range(T):
    # 前150次，输入电流都是0.12
    if t < 150:
        spike_list.append(lif_node(0.12).float().item())
    # 后50次，不输入，也就是输入0
    else:
        spike_list.append(lif_node(0).float().item())

    # 记录每一次输入后，神经元的电压
    v_list.append(lif_node.v.item())

# 画出电压的变化
pyplot.subplot(2, 1, 1)
pyplot.plot(v_list, label='v')
pyplot.xlabel('t')
pyplot.ylabel('voltage')
pyplot.legend()

# 画出脉冲
pyplot.subplot(2, 1, 2)
pyplot.bar(torch.arange(0, T).tolist(), spike_list, label='spike')
pyplot.xlabel('t')
pyplot.ylabel('spike')
pyplot.legend()
pyplot.show()

print('t', 'v', 'spike')
for t in range(T):
    print(t, v_list[t], spike_list[t])