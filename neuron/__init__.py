import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNode(nn.Module):
    def __init__(self, shape, r, v_threshold, v_reset=0.0, device='cpu'):
        '''
        时钟驱动（逐步仿真）的神经元基本模型
        这些神经元都是在t时刻接收电流i作为输入，与膜电阻r相乘，得到dv=i * r
        之后self.v += dv，然后根据神经元自身的属性，决定是否发放脉冲
        :param shape: 任意
        :param r: 膜电阻，可以是一个float，也可以是tensor
        :param v_threshold: 阈值电压，可以是一个float，也可以是tensor
        :param v_reset: 重置电压，可以是一个float，也可以是tensor
        注意，更新过程中会确保电压不低于v_reset
        :param device: 数据所在的设备
        '''
        super().__init__()
        self.shape = shape


        assert isinstance(r, float) or isinstance(r, torch.Tensor)

        if isinstance(r, torch.Tensor):
            assert r.shape == shape
            self.r = r.to(device)
        else:
            self.r = r


        assert isinstance(v_threshold, float) or isinstance(v_threshold, torch.Tensor)

        if isinstance(v_threshold, torch.Tensor):
            assert v_threshold.shape == shape
            self.v_threshold = v_threshold.to(device)
        else:
            self.v_threshold = v_threshold

        assert isinstance(v_reset, float) or isinstance(v_reset, torch.Tensor)
        if isinstance(v_reset, torch.Tensor):
            assert v_reset.shape == shape
            self.v_reset = v_reset.to(device)
        else:
            self.v_reset = v_reset

        self.v = torch.ones(size=shape, dtype=torch.float, device=device) * v_reset

    def __str__(self):

        return 'shape ' + str(self.shape) + '\nr ' + str(self.r) + '\nv_threshold ' + str(self.v_threshold) + '\nv_reset ' + str(self.v_reset)

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''
        raise NotImplementedError

    def reset(self):
        self.v = self.v_reset


class IFNode(BaseNode):
    '''
    IF神经元模型
    电压一旦达到阈值v_threshold则下一个时刻放出脉冲，同时电压归位到重置电压v_reset

    测试代码
    if_node = neuron.IFNode([1], r=1.0, v_threshold=1.0)
    v = []
    for i in range(1000):
        if_node(0.01)
        v.append(if_node.v.item())

    pyplot.plot(v)
    pyplot.show()
    '''

    def __init__(self, shape, r, v_threshold, v_reset=0.0, device='cpu'):
        super().__init__(shape, r, v_threshold, v_reset, device)

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''
        out_spike = (self.v >= self.v_threshold)

        self.v += self.r * i
        self.v[self.v < self.v_reset] = self.v_reset

        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset
        return out_spike



class LIFNode(BaseNode):
    '''
    LIF神经元模型
    电压一旦达到阈值v_threshold则下一个时刻放出脉冲，同时电压归位到重置电压v_reset
    电压在不为v_reset时，会指数衰减
    v_decay = -(self.v - self.v_reset)
    self.v += (self.r * i + v_decay) / self.tau

    测试代码
    lif_node = neuron.LIFNode([1], r=9.0, v_threshold=1.0, tau=20.0)
    v = []

    for i in range(1000):
        if i < 500:
            lif_node(0.1)
        else:
            lif_node(0)
        v.append(lif_node.v.item())

    pyplot.plot(v)
    pyplot.show()

    '''

    def __init__(self, shape, r, v_threshold, v_reset=0.0, tau=1.0, device='cpu'):
        super().__init__(shape, r, v_threshold, v_reset, device)
        self.tau = tau

    def forward(self, i):
        '''
        :param i: 当前时刻的输入电流，可以是一个float，也可以是tensor
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''
        out_spike = (self.v >= self.v_threshold)

        v_decay = -(self.v - self.v_reset)
        self.v += (self.r * i + v_decay) / self.tau
        self.v[self.v < self.v_reset] = self.v_reset

        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset

        return out_spike

    def __str__(self):
        return super().__str__() + '\ntau ' + str(self.tau)


