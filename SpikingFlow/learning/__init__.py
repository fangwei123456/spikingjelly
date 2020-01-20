import torch
import torch.nn as nn
import torch.nn.functional as F
import neuron
import encoding
import connection
import simulating

class STDPModule(nn.Module):

    def __init__(self, tf_module, connection_module, neuron_module,
                 tau_pre, tau_post, learning_rate, f_w=lambda x: torch.abs(x) + 1e-6):
        '''
        :param tf_module: connection.transform中的脉冲-电流转换器
        :param connection_module: 突触
        :param neuron_module: 神经元
        :param tau_pre: pre脉冲的迹的时间常数
        :param tau_post: post脉冲的迹的时间常数
        :param learning_rate: 学习率
        :param f_w: 权值函数，输入是权重w，输出一个float或者是与权重w相同shape的tensor

        由tf_module，connection_module，neuron_module构成的STDP学习的基本单元

        利用迹的方式（Morrison A, Diesmann M, Gerstner W. Phenomenological models of synaptic plasticity based on spike
        timing[J]. Biological cybernetics, 2008, 98(6): 459-478.）实现STDP学习，更新connection_module中的参数

        pre脉冲到达时，权重减少trace_post * f_w(w) * learning_rate

        post脉冲到达时，权重增加trace_pre * f_w(w) * learning_rate

        示例代码

        .. code-block:: python

            sim = simulating.Simulator()
            sim.append(learning.STDPModule(tf.SpikeCurrent(amplitude=0.2),
                                           connection.Linear(2, 1),
                                           neuron.IFNode(shape=[1], r=1.0, v_threshold=1.0),
                                           tau_pre=10.0,
                                           tau_post=10.0,
                                           learning_rate=1e-3
                                           ))

            pre_spike_list0 = []
            pre_spike_list1 = []
            post_spike_list = []
            w_list0 = []
            w_list1 = []

            for i in range(600):
                if i < 400:
                    pre_spike = torch.ones(size=[2], dtype=torch.bool)
                else:
                    pre_spike = torch.zeros(size=[2], dtype=torch.bool)
                    pre_spike[1] = True

                post_spike = sim.step(pre_spike)
                pre_spike_list0.append(pre_spike[0].float().item())
                pre_spike_list1.append(pre_spike[1].float().item())

                post_spike_list.append(post_spike.float().item())

                w_list0.append(sim.module_list[-1].module_list[2].w[:, 0].item())
                w_list1.append(sim.module_list[-1].module_list[2].w[:, 1].item())

            pyplot.plot(pre_spike_list0, c='r', label='pre_spike[0]')
            pyplot.plot(pre_spike_list1, c='g', label='pre_spike[1]')
            pyplot.legend()
            pyplot.show()
            pyplot.plot(post_spike_list, label='post_spike')
            pyplot.legend()
            pyplot.show()
            pyplot.plot(w_list0, c='r', label='w[0]')
            pyplot.plot(w_list1, c='g', label='w[1]')
            pyplot.legend()
            pyplot.show()
        '''
        super().__init__()

        self.module_list = nn.Sequential(tf_module,
                                         connection.ConstantDelay(delay_time=1),
                                         connection_module,
                                         connection.ConstantDelay(delay_time=1),
                                         neuron_module
                                         )

        #  如果不增加ConstantDelay，则一次forward会使数据直接通过3个module
        #  但实际上调用一次forward，应该只能通过1个module
        #  因此通过添加ConstantDelay的方式来实现

        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.learning_rate = learning_rate
        self.trace_pre = 0
        self.trace_post = 0
        self.f_w = f_w

    def update_param(self, pre=True):
        if isinstance(self.module_list[2], connection.Linear):
            dw = 0
            if pre:
                if isinstance(self.trace_post, torch.Tensor):
                    dw = - (self.learning_rate * self.f_w(self.module_list[2].w).t() *
                          self.trace_post.view(-1, self.module_list[2].w.shape[0]).mean(0)).t()

            else:
                if isinstance(self.trace_pre, torch.Tensor):
                    dw = self.learning_rate * self.f_w(self.module_list[2].w) \
                                             * self.trace_pre.view(-1, self.module_list[2].w.shape[1]).mean(0)


        else:
            # 其他类型的尚未实现
            raise NotImplementedError

        self.module_list[2].w += dw

    def forward(self, pre_spike):
        '''
        :param pre_spike: 输入脉冲
        :return: 经过本module后的输出脉冲

        需要注意的时，由于本module含有tf_module, connection_module, neuron_module三个module

        因此在t时刻的输入，到t+3dt才能得到其输出
        '''
        self.trace_pre += - self.trace_pre / self.tau_pre + pre_spike.float()
        self.update_param(True)

        post_spike = self.module_list(pre_spike)

        self.trace_post += - self.trace_post / self.tau_post + post_spike.float()
        self.update_param(False)
        return post_spike

    def reset(self):
        '''
        :return: None

        将module自身，以及tf_module, connection_module, neuron_module的状态变量重置为初始值
        '''
        for i in range(self.module_list.__len__()):
            self.module_list[i].reset()
        self.trace_pre = 0
        self.trace_post = 0




class STDPUpdater:
    def __init__(self, tau_pre, tau_post, learning_rate, f_w=lambda x: torch.abs(x) + 1e-6):
        '''
        :param neuron_module: 神经元
        :param tau_pre: pre脉冲的迹的时间常数
        :param tau_post: post脉冲的迹的时间常数
        :param learning_rate: 学习率
        :param f_w: 权值函数，输入是权重w，输出一个float或者是与权重w相同shape的tensor

        利用迹的方式（Morrison A, Diesmann M, Gerstner W. Phenomenological models of synaptic plasticity based on spike
        timing[J]. Biological cybernetics, 2008, 98(6): 459-478.）实现STDP学习，更新connection_module中的参数

        pre脉冲到达时，权重减少trace_post * f_w(w) * learning_rate

        post脉冲到达时，权重增加trace_pre * f_w(w) * learning_rate

        与STDPModule类似，但需要手动给定前后脉冲，这也带来了更为灵活的使用方式，例如
        不使用突触实际连接的前后神经元的脉冲，而是使用其他脉冲来指导某个突触的学习

        示例代码

        .. code-block:: python

            def f_w(x: torch.Tensor):
                x_abs = x.abs()
                return x_abs / (x_abs.sum() + 1e-6)

            sim = simulating.Simulator()
            sim.append(tf.SpikeCurrent(amplitude=0.5))
            sim.append(connection.Linear(2, 1))
            sim.append(neuron.LIFNode(shape=[1], r=10.0, v_threshold=1.0, tau=100.0))

            updater = learning.STDPUpdater(tau_pre=10.0,
                                           tau_post=10.0,
                                           learning_rate=1e-3,
                                           f_w=f_w)

            post_spike_list = []
            w_list0 = []
            w_list1 = []
            for i in range(500):
                if i < 400:
                    pre_spike = torch.ones(size=[2], dtype=torch.bool)
                else:
                    pre_spike = torch.zeros(size=[2], dtype=torch.bool)

                post_spike = sim.step(pre_spike)

                updater.update(sim.module_list[1], pre_spike, post_spike)

                post_spike_list.append(post_spike.float().item())

                w_list0.append(sim.module_list[1].w[:, 0].item())
                w_list1.append(sim.module_list[1].w[:, 1].item())

            pyplot.plot(post_spike_list, label='post_spike')
            pyplot.legend()
            pyplot.show()
            pyplot.plot(w_list0, c='r', label='w[0]')
            pyplot.plot(w_list1, c='g', label='w[1]')
            pyplot.legend()
            pyplot.show()
        '''
        self.tau_pre = tau_pre
        self.tau_post = tau_post
        self.learning_rate = learning_rate
        self.trace_pre = 0
        self.trace_post = 0
        self.f_w = f_w


    def reset(self):
        '''
        :return: None

        将前后脉冲的迹设置为0
        '''
        self.trace_pre = 0
        self.trace_post = 0

    def update(self, connection_module, pre_spike, post_spike, inverse=False):
        '''
        :param connection_module: 突触
        :param pre_spike: 输入脉冲
        :param post_spike: 输出脉冲
        :param inverse: 为True则进行Anti-STDP

        指定突触的前后脉冲，进行STDP学习

        '''

        self.trace_pre += - self.trace_pre / self.tau_pre + pre_spike.float()
        self.trace_post += - self.trace_post / self.tau_post + post_spike.float()

        if isinstance(connection_module, connection.Linear):
            if inverse:
                connection_module.w -= self.learning_rate * self.f_w(connection_module.w) \
                                       * self.trace_pre.view(-1, connection_module.w.shape[1]).mean(0)
                connection_module.w = (connection_module.w.t() +
                                       self.learning_rate * self.f_w(connection_module.w).t() *
                                       self.trace_post.view(-1, connection_module.w.shape[0]).mean(0)).t()
            else:
                connection_module.w += self.learning_rate * self.f_w(connection_module.w) \
                                       * self.trace_pre.view(-1, connection_module.w.shape[1]).mean(0)
                connection_module.w = (connection_module.w.t() -
                                       self.learning_rate * self.f_w(connection_module.w).t() *
                                       self.trace_post.view(-1, connection_module.w.shape[0]).mean(0)).t()

        else:
            # 其他类型的突触尚未实现
            raise NotImplementedError
