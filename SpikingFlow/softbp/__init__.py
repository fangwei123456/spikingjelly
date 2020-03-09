import torch
import torch.nn as nn
import torch.nn.functional as F

class ModelPipeline(nn.Module):
    def __init__(self):
        '''
        用于解决显存不足的模型流水线。将一个模型分散到各个GPU上，流水线式的进行训练。设计思路与仿真器非常类似。

        运行时建议先取一个很小的batch_size，然后观察各个GPU的显存占用，并调整每个module_list中包含的模型比例。
        '''
        super().__init__()
        self.module_list = nn.ModuleList()
        self.gpu_list = []


    def append(self, nn_module, gpu_id):
        '''
        :param nn_module: 新添加的module
        :param gpu_id:  该模型所在的GPU，不需要带“cuda:”的前缀。例如“2”
        :return: None

        将nn_module添加到流水线中，nn_module会运行在设备gpu_id上。添加的nn_module会按照它们的添加顺序运行。例如首先添加了\
        fc1，又添加了fc2，则实际运行是按照input_data->fc1->fc2->output_data的顺序运行。
        '''
        self.module_list.append(nn_module.to('cuda:' + gpu_id))
        self.gpu_list.append('cuda:' + gpu_id)


    def forward(self, x, split_sizes):
        '''
        :param x: 输入数据
        :param split_sizes: 输入数据x会在维度0上被拆分成每spilit_size一组，得到[x0, x1, ...]，这些数据会被串行的送入\
        module_list中的各个模块进行计算
        :return: 输出数据

        例如将模型分成4部分，因而 ``module_list`` 中有4个子模型；将输入分割为3部分，则每次调用 ``forward(x, split_sizes)`` ，函数内部的\
        计算过程如下：

        .. code-block:: python

                step=0     x0, x1, x2  |m0|    |m1|    |m2|    |m3|

                step=1     x0, x1      |m0| x2 |m1|    |m2|    |m3|

                step=2     x0          |m0| x1 |m1| x2 |m2|    |m3|

                step=3                 |m0| x0 |m1| x1 |m2| x2 |m3|

                step=4                 |m0|    |m1| x0 |m2| x1 |m3| x2

                step=5                 |m0|    |m1|    |m2| x0 |m3| x1, x2

                step=6                 |m0|    |m1|    |m2|    |m3| x0, x1, x2

        不使用流水线，则任何时刻只有一个GPU在运行，而其他GPU则在等待这个GPU的数据；而使用流水线，例如上面计算过程中的 ``step=3`` 到\
        ``step=4``，尽管在代码的写法为顺序执行：

        .. code-block:: python

            x0 = m1(x0)
            x1 = m2(x1)
            x2 = m3(x2)

        但由于PyTorch优秀的特性，上面的3行代码实际上是并行执行的，因为这3个在CUDA上的计算使用各自的数据，互不影响的

        '''

        assert x.shape[0] % split_sizes == 0, print('x.shape[0]不能被split_sizes整除！')
        x = list(x.split(split_sizes, dim=0))
        x_pos = []  # x_pos[i]记录x[i]应该通过哪个module

        for i in range(x.__len__()):
            x_pos.append(i + 1 - x.__len__())

        while True:
            for i in range(x_pos.__len__() - 1, -1, -1):
                if 0 <= x_pos[i] <= self.gpu_list.__len__() - 1:
                    x[i] = self.module_list[x_pos[i]](x[i].to(self.gpu_list[x_pos[i]]))
                x_pos[i] += 1
            if x_pos[0] == self.gpu_list.__len__():
                break

        return torch.cat(x, dim=0)

class BaseNode(nn.Module):
    def __init__(self, v_threshold, v_reset):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压

        softbp包中，可微分SNN神经元的基类神经元

        可微分SNN神经元，在前向传播时输出真正的脉冲（离散的0和1）。脉冲的产生过程可以可以看作是一个\
        阶跃函数：

        .. math::
            S = \\Theta(V - V_{threshold})

            其中\\Theta(x) =
            \\begin{cases}
            1, & x \\geq 0 \\\\
            0, & x < 0
            \\end{cases}

        :math:`\\Theta(x)` 是一个不可微的函数，用一个形状类似的函数 :math:`\\sigma(x)` 去近似它，在反向传播时\
        用 :math:`\\sigma'(x)` 来近似 :math:`\\Theta'(x)`，这样就可以使用梯度下降法来更新SNN了

        前向传播使用 :math:`\\Theta(x)`，反向传播时按前向传播为 :math:`\\sigma(x)` 来计算梯度，在PyTorch中很容易实现，参见\
        这个类的spiking()函数
        '''
        super().__init__()
        self.v_threshold = v_threshold
        self.v_reset = v_reset
        self.v = v_reset

    @staticmethod
    def pulse_soft(x):
        '''
        :param x: 输入，tensor
        :return: :math:`\\sigma(x)`

        此函数即为 :math:`\\sigma(x)`。默认是用sigmoid函数。如果想使用其他函数，继承后重写pulse_soft()函数即可
        '''
        return torch.sigmoid(x)

    def spiking(self):
        '''
        :return: 神经元的输出脉冲

        根据当前神经元的电压、阈值、重置电压，计算输出脉冲，并更新神经元的电压

        前向传播使用 :math:`\\Theta(x)`，反向传播时按前向传播为 ``self.pulse_soft()`` 来计算梯度的脉冲发放函数
        '''
        if self.training:
            spike_hard = (self.v >= self.v_threshold).float()
            spike_soft = self.pulse_soft(self.v - self.v_threshold)
            v_hard = self.v_reset * spike_hard + self.v * (1 - spike_hard)
            v_soft = self.v_reset * spike_soft + self.v * (1 - spike_soft)
            self.v = v_soft + (v_hard - v_soft).detach_()
            return spike_soft + (spike_hard - spike_soft).detach_()
        else:
            spike_hard = (self.v >= self.v_threshold).float()
            self.v = self.v_reset * spike_hard + self.v * (1 - spike_hard)
            return spike_hard

    def forward(self, dv: torch.Tensor):
        '''
        :param dv: 输入到神经元的电压增量
        :return: 神经元的输出脉冲

        子类需要实现这一函数
        '''
        raise NotImplementedError

    def reset(self):
        '''
        :return: None

        重置神经元为初始状态，也就是将电压设置为v_reset

        如果子类的神经元还含有其他状态变量，需要在此函数中将这些状态变量全部重置
        '''
        self.v = self.v_reset

class IFNode(BaseNode):
    def __init__(self, v_threshold=1.0, v_reset=0.0):
        '''
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压

        IF神经元模型，可以看作理想积分器，无输入时电压保持恒定，不会像LIF神经元那样衰减

        .. math::
            \\frac{\\mathrm{d}V(t)}{\\mathrm{d} t} = R_{m}I(t)

        电压一旦达到阈值v_threshold则放出脉冲，同时电压归位到重置电压v_reset
        '''
        super().__init__(v_threshold, v_reset)

    def forward(self, dv: torch.Tensor):
        self.v += dv
        return self.spiking()



class LIFNode(BaseNode):
    def __init__(self, tau=100.0, v_threshold=1.0, v_reset=0.0):
        '''
        :param tau: 膜电位时间常数，越大则充电越慢
        :param v_threshold: 神经元的阈值电压
        :param v_reset: 神经元的重置电压

        LIF神经元模型，可以看作是带漏电的积分器

        .. math::
            \\tau_{m} \\frac{\\mathrm{d}V(t)}{\\mathrm{d}t} = -(V(t) - V_{reset}) + R_{m}I(t)

        电压在不为v_reset时，会指数衰减
        '''
        super().__init__(v_threshold, v_reset)
        self.tau = tau

    def forward(self, dv: torch.Tensor):
        self.v += (dv + -(self.v - self.v_reset)) / self.tau
        return self.spiking()

