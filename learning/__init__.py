import torch
import torch.nn as nn
import torch.nn.functional as F
import neuron
import encoding
import connection
import simulating


class STDPModule(nn.Module):
    '''
    测试代码如下
    sim = simulating.Simulator()
    sim.append(learning.STDPModule(tf.SpikeCurrent(amplitude=0.2),
                                   connection.Linear(2, 1),
                                   neuron.IFNode(shape=[1], r=1.0, v_threshold=1.0),
                                   tau_a=10.0,
                                   tau_b=10.0,
                                   learning_rate=1e-3
                                   ))

    in_spike_list0 = []
    in_spike_list1 = []
    out_spike_list = []
    w_list0 = []
    w_list1 = []

    for i in range(600):
        if i < 400:
            in_spike = torch.ones(size=[2], dtype=torch.bool)
        else:
            in_spike = torch.zeros(size=[2], dtype=torch.bool)
            in_spike[1] = True

        out_spike = sim.step(in_spike)
        in_spike_list0.append(in_spike[0].float().item())
        in_spike_list1.append(in_spike[1].float().item())

        out_spike_list.append(out_spike.float().item())

        w_list0.append(sim.module_list[-1].module_list[2].w[:, 0].item())
        w_list1.append(sim.module_list[-1].module_list[2].w[:, 1].item())

    pyplot.plot(in_spike_list0, c='r', label='in_spike[0]')
    pyplot.plot(in_spike_list1, c='g', label='in_spike[1]')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(out_spike_list, label='out_spike')
    pyplot.legend()
    pyplot.show()
    pyplot.plot(w_list0, c='r', label='w[0]')
    pyplot.plot(w_list1, c='g', label='w[1]')
    pyplot.legend()
    pyplot.show()
    '''
    def __init__(self, tf_module, connection_module, neuron_module,
                 tau_a, tau_b, learning_rate, f_w=lambda x: torch.abs(x) + 1e-6):
        '''
        由tf_module，connection_module，neuron_module构成的STDP学习的基本单元
        利用迹的方式实现STDP学习，更新connection_module中的参数
        pre脉冲到达时，权重增加trace_a * f_w(w) * learning_rate
        post脉冲到达时，权重减少trace_b * f_w(w) * learning_rate
        :param tf_module: connection.transform中的脉冲-电流转换器
        :param connection_module: 突触
        :param neuron_module: 神经元
        :param tau_a: pre脉冲的迹的时间常数
        :param tau_b: post脉冲的迹的时间常数
        :param learning_rate: 学习率
        :param f_w: 权值函数，输入是权重w，输出是权重更新量delta_w

        '''
        super().__init__()

        self.module_list = nn.Sequential(tf_module,
                                         connection.ConstantDelay(delay_time=1),
                                         connection_module,
                                         connection.ConstantDelay(delay_time=1),
                                         neuron_module
                                         )
        '''
        如果不增加ConstantDelay，则一次forward会使数据直接通过3个module
        但实际上调用一次forward，应该只能通过1个module
        因此通过添加ConstantDelay的方式来实现
        '''

        self.tau_a = tau_a
        self.tau_b = tau_b
        self.learning_rate = learning_rate
        self.trace_a = 0
        self.trace_b = 0
        self.f_w = f_w

    def update_param(self, pre=True):
        if isinstance(self.module_list[2], connection.Linear):
            '''
            connection.Linear的更新规则
            w.shape = [out_num, in_num]
            trace_a.shape = [batch_size, *, in_num]
            trace_b.shape = [batch_size, *, out_num]
            '''
            if pre:
                self.module_list[2].w += self.learning_rate * self.f_w(self.module_list[2].w) \
                                         * self.trace_a.view(-1, self.module_list[2].w.shape[1]).mean(0)
            else:
                self.module_list[2].w = (self.module_list[2].w.t() -
                                         self.learning_rate * self.f_w(self.module_list[2].w).t() *
                                         self.trace_b.view(-1, self.module_list[2].w.shape[0]).mean(0)).t()

        else:
            raise NotImplementedError

    def forward(self, in_spike):
        self.trace_a += - self.trace_a / self.tau_a + in_spike.float()
        self.update_param(True)

        out_spike = self.module_list(in_spike)

        self.trace_b += - self.trace_b / self.tau_b + out_spike.float()
        self.update_param(False)
        return out_spike

    def reset(self):
        for i in range(self.module_list.__len__()):
            self.module_list[i].reset()
        self.trace_a = 0
        self.trace_b = 0




