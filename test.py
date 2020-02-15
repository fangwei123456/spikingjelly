import SpikingFlow
import SpikingFlow.neuron as neuron
# 导入绘图模块
from matplotlib import pyplot

# 新建一个LIF神经元
lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
# 新建一个空list，保存仿真过程中神经元的电压值
v = []
# 运行1000次
for i in range(1000):
    # 前500次，输入电流都是0.1
    if i < 500:
        lif_node(0.12)
    # 后500次，不输入，也就是输入0
    else:
        lif_node(0)
    # 记录每一次输入后，神经元的电压
    v.append(lif_node.v.item())

# 画出电压的变化
pyplot.plot(v, label='v')
pyplot.legend()
pyplot.show()