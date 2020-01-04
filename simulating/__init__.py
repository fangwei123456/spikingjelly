import torch
import torch.nn as nn
import torch.nn.functional as F

class Simulator:
    '''
    仿真器，内含多个module组成的list

    测试代码如下
    sim = simulating.Simulator()
    sim.append(encoding.ConstantEncoder(shape=[1]))
    sim.append(tf.SpikeCurrent(amplitude=1.0))
    sim.append(neuron.IFNode(shape=[1], r=0.5, v_threshold=1.0))
    sim.append(tf.SpikeCurrent(amplitude=0.4))
    sim.append(neuron.IFNode(shape=[1], r=2.0, v_threshold=1.0))
    sim.append(tf.ExpDecayCurrent(tau=5.0, amplitude=1.0))
    sim.append(neuron.LIFNode(shape=[1], r=5.0, v_threshold=1.0, tau=10.0))
    v = []
    v.extend(([], [], []))
    for i in range(1000):

        output_data = sim.step(0.01)

        #print(i, sim.pipeline)
        for j in range(3):
            v[j].append(sim.module_list[2 * j + 2].v.item())

    pyplot.plot(v[0])
    pyplot.show()
    pyplot.plot(v[1])
    pyplot.show()
    pyplot.plot(v[2])
    pyplot.show()
    '''
    def __init__(self):
        self.module_list = []  # 保存各个module
        self.pipeline = []  # 保存各个module在当前时刻的输出
        self.simulated_steps = 0  # 已经运行仿真的步数

        '''        
        当前时刻启动仿真前 数据和模型如下
        input_data -> module[0] -> x[0] -> module[1] -> ... -> x[n-2] -> module[n-1] -> x[n-1]
        pipeline = [ x[0], x[1], ..., x[n-2], x[n-1] ]
        启动仿真后 应该按照如下顺序计算
        x[n-1] = module[n-1](x[n-2])
        x[n-2] = module[n-2](x[n-3])
        ...
        x[1] = module[1](x[0])
        x[0] = module[0](input_data)
        
        需要注意的是，只有运行module_list.__len__()步仿真后，才能得到第一个输出x[n-1]，此时pipeline才完全充满
        运行了simulated_steps步时，只得到了x[0], x[1], ..., x[simulated_steps-1]
        因此，当simulated_steps < module_list.__len__()时，每进行一步仿真，实际上只能进行如下运算
        
        x[simulated_steps] = module[simulated_steps](x[simulated_steps-1])
        x[simulated_steps-1] = module[simulated_steps-1](x[simulated_steps-2])
        ...
        x[1] = module[1](x[0])
        x[0] = module[0](input_data)
        '''


    def append(self, new_module):
        '''
        向Simulator中添加神经元、突触等模型
        :param new_module: 新添加的模型
        :return: 无返回值
        '''
        self.module_list.append(new_module)
        self.pipeline.append(None)
        self.simulated_steps = 0  # 添加新module后，之前的仿真运行就不算数了


    def step(self, input_data):
        '''
        输入数据input_data，仿真一步
        :param input_data: 输入数据
        :return: 输出值
        '''
        if self.simulated_steps < self.module_list.__len__():
            for i in range(self.simulated_steps, 0, -1):
                self.pipeline[i] = self.module_list[i](self.pipeline[i - 1])
            self.pipeline[0] = self.module_list[0](input_data)
            self.simulated_steps += 1
            if self.simulated_steps == self.module_list.__len__():
                return self.pipeline[-1]
            else:
                return None
        else:
            for i in range(self.module_list.__len__() - 1, 0, -1):
                self.pipeline[i] = self.module_list[i](self.pipeline[i - 1])
            self.pipeline[0] = self.module_list[0](input_data)
            self.simulated_steps += 1
            return self.pipeline[-1]


