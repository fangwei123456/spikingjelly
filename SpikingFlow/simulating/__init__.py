import torch
import torch.nn as nn
import torch.nn.functional as F

class Simulator:
    def __init__(self):
        '''
        仿真器，内含多个module组成的list

        当前时刻启动仿真前，数据和模型如下

        x[0] -> module[0] -> x[1] -> module[1] -> ... -> x[n-1] -> module[n-1] -> x[n]

        pipeline = [ x[0], x[1], ..., x[n-2], x[n-1], x[n] ]

        启动仿真后 应该按照如下顺序计算

        X[0] = input_data

        x[n] = module[n-1](x[n-1])

        x[n-1] = module[n-2](x[n-2])

        ...

        x[1] = module[0](x[0])


        测试代码

        .. code-block:: python

            sim = simulating.Simulator()
            sim.append(encoding.ConstantEncoder())
            sim.append(tf.SpikeCurrent(amplitude=0.01))
            sim.append(neuron.IFNode(shape=[1], r=0.5, v_threshold=1.0))
            sim.append(tf.SpikeCurrent(amplitude=0.4))
            sim.append(neuron.IFNode(shape=[1], r=2.0, v_threshold=1.0))
            sim.append(tf.ExpDecayCurrent(tau=5.0, amplitude=1.0))
            sim.append(neuron.LIFNode(shape=[1], r=5.0, v_threshold=1.0, tau=10.0))
            v = []
            v.extend(([], [], []))
            for i in range(1000):
                if i < 800:
                    output_data = sim.step(torch.ones(size=[1], dtype=torch.bool))
                else:
                    output_data = sim.step(torch.zeros(size=[1], dtype=torch.bool))


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
        self.module_list = []  # 保存各个module
        self.pipeline = []  # 保存各个module在当前时刻的输出
        self.simulated_steps = 0  # 已经运行仿真的步数
        self.pipeline.append(None)



    def append(self, new_module):
        '''
        :param new_module: 新添加的模型
        :return: None

        向Simulator的module_list中添加module

        只要是torch.nn.Module的子类并定义了forward()方法，就可以添加=
        '''
        self.module_list.append(new_module)
        self.pipeline.append(None)



    def step(self, input_data):
        '''
        :param input_data: 输入数据
        :return: 输出值

        输入数据input_data，仿真一步

        '''
        self.pipeline[0] = input_data

        # 首次运行时跑满pipeline
        # x[0] -> module[0] -> x[1] -> module[1] -> ... -> x[n-1] -> module[n-1] -> x[n]
        if self.simulated_steps == 0:
            for i in range(self.module_list.__len__()):
                # i = 0, 1, ..., n-1
                for j in range(i + 1, 0, -1):
                    # j = i+1, i, ..., 1
                    self.pipeline[j] = self.module_list[j - 1](self.pipeline[j - 1])

        else:
            for i in range(self.module_list.__len__(), 0, -1):
                #  x[n] = module[n-1](x[n-1])
                #  x[n-1] = module[n-2](x[n-2])
                #  ...
                #  x[1] = module[0](x[0])
                self.pipeline[i] = self.module_list[i - 1](self.pipeline[i - 1])

        self.simulated_steps += 1
        return self.pipeline[-1]

    def reset(self):
        '''
        :return: None

        重置仿真器到开始仿真前的状态，已经添加的module并不会被清除
        '''
        self.simulated_steps = 0

        for i in range(self.pipeline.__len__()):
            self.pipeline[i] = None

        for i in range(self.module_list.__len__()):
            self.module_list[i].reset()
