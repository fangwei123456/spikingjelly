import torch
import torch.nn as nn
import torch.nn.functional as F


class BaseNode(nn.Module):
    def __init__(self, shape, v_threshold, v_reset=0.0, device='cpu'):
        '''
        时钟驱动（逐步仿真）的神经元基本模型
        这些神经元都是在t时刻接收电压增量dv作为输入，之后self.v += dv，然后根据神经元自身的属性，决定是否发放脉冲
        :param shape: 任意
        :param v_threshold: 阈值电压，可以是一个float，也可以是tensor
        :param v_reset: 重置电压，可以是一个float，也可以是tensor
        注意，更新过程中会确保电压不低于v_reset
        :param device: 数据所在的设备
        '''
        super().__init__()
        self.shape = shape

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
        ret = []
        ret.append('shape')
        ret.append(self.shape)
        ret.append('v_threshold')
        ret.append(self.v_threshold)
        ret.append('v_reset')
        ret.append(self.v_reset)
        return 'shape ' + str(self.shape) + '\nv_threshold ' + str(self.v_threshold) + '\nv_reset ' + str(self.v_reset)

    def forward(self, dv):
        '''
        :param dv: shape与self.shape相同的电压增量，叠加到self.v上
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''
        raise NotImplementedError

    def reset(self):
        self.v = self.v_reset


class IFNode(BaseNode):
    '''
    IF神经元模型
    电压一旦达到阈值v_threshold则放出脉冲，同时电压归位到重置电压v_reset

    测试代码
    if_node = neuron.IFNode([1], 1.0)
    v = []
    for i in range(1000):
        if_node(0.01)
        v.append(if_node.v.item())

    pyplot.plot(v)
    pyplot.show()
    '''

    def __init__(self, shape, v_threshold, v_reset=0.0, device='cpu'):
        super().__init__(shape, v_threshold, v_reset, device)

    def forward(self, dv):
        '''
        :param dv: shape与self.shape相同的电压增量，叠加到self.v上
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''
        self.v += dv
        self.v[self.v < self.v_reset] = self.v_reset
        out_spike = (self.v >= self.v_threshold)

        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset
        return out_spike



class LIFNode(BaseNode):
    '''
    LIF神经元模型
    电压一旦达到阈值v_threshold则放出脉冲，同时电压归位到重置电压v_reset
    电压在不为v_reset时，会指数衰减，衰减速度由tau决定
    delta_v = -(self.v - self.v_reset) / self.tau
    self.v += (dv + delta_v)

    测试代码
    lif_node = neuron.LIFNode([1], v_threshold=1.0, tau=25)
    v = []

    for i in range(100):
        if i < 20:
            lif_node(0.1)
        else:
            lif_node(0)
        v.append(lif_node.v.item())
        print(v[-1])

    pyplot.plot(v)
    pyplot.show()
    '''

    def __init__(self, shape, v_threshold, v_reset=0.0, tau=1.0, device='cpu'):
        super().__init__(shape, v_threshold, v_reset, device)
        self.tau = tau

    def forward(self, dv):
        '''
        :param dv: shape与self.shape相同的电压增量，叠加到self.v上
        :return:out_spike: shape与self.shape相同，输出脉冲
        '''

        delta_v = -(self.v - self.v_reset) / self.tau

        self.v += (dv + delta_v)
        self.v[self.v < self.v_reset] = self.v_reset

        out_spike = (self.v >= self.v_threshold)
        if isinstance(self.v_reset, torch.Tensor):
            self.v[out_spike] = self.v_reset[out_spike]
        else:
            self.v[out_spike] = self.v_reset

        return out_spike

    def __str__(self):
        return super().__str__() + '\ntau ' + str(self.tau)

