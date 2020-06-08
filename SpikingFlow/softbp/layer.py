import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NeuNorm(nn.Module):
    def __init__(self, in_channels, k=0.9):
        '''
        .. warning::
            可能是错误的实现。测试的结果表明，增加NeuNorm后的收敛速度和正确率反而下降了。

        :param in_channels: 输入数据的通道数
        :param k: 动量项系数

        Wu Y, Deng L, Li G, et al. Direct Training for Spiking Neural Networks: Faster, Larger, Better[C]. national conference on artificial intelligence, 2019, 33(01): 1311-1318.
        中提出的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如：

        Conv2d -> LIF -> NeuNorm

        要求输入的尺寸是[batch_size, in_channels, W, H]。

        in_channels是输入到NeuNorm层的通道数，也就是论文中的 :math:`F`。

        k是动量项系数，相当于论文中的 :math:`k_{\\tau 2}`。

        论文中的 :math:`\\frac{v}{F}` 会根据 :math:`k_{\\tau 2} + vF = 1` 自动算出。
        '''
        super().__init__()
        self.x = 0
        self.k0 = k
        self.k1 = (1 - self.k0) / in_channels**2
        self.w = nn.Parameter(torch.Tensor(in_channels, 1, 1))
        nn.init.kaiming_uniform_(self.w, a=math.sqrt(5))
    def forward(self, in_spikes: torch.Tensor):
        '''
        :param in_spikes: 来自上一个卷积层的输出脉冲，shape=[batch_size, in_channels, W, H]
        :return: 正则化后的脉冲，shape=[batch_size, in_channels, W, H]
        '''
        self.x = self.k0 * self.x + (self.k1 * in_spikes.sum(dim=1).unsqueeze(1))

        return in_spikes - self.w * self.x

    def reset(self):
        '''
        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。
        '''
        self.x = 0

class DCT(nn.Module):
    def __init__(self, kernel_size):
        '''
        :param kernel_size: 进行分块DCT变换的块大小

        将输入的shape=[*, W, H]的数据进行分块DCT变换的层，*表示任意额外添加的维度。变换只在最后2维进行，要求W和H都能\\
        整除kernel_size。

        DCT是AXAT的一种特例。
        '''
        super().__init__()
        self.kernel = torch.zeros(size=[kernel_size, kernel_size])
        for i in range(0, kernel_size):
            for j in range(kernel_size):
                if i == 0:
                    self.kernel[i][j] = math.sqrt(1 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)
                else:
                    self.kernel[i][j] = math.sqrt(2 / kernel_size) * math.cos((j + 0.5) * math.pi * i / kernel_size)

    def forward(self, x: torch.Tensor):
        '''
        :param x: shape=[*, W, H]，*表示任意额外添加的维度
        :return: 对x进行分块DCT变换后得到的tensor
        '''
        if self.kernel.device != x.device:
            self.kernel = self.kernel.to(x.device)
        x_shape = x.shape
        x = x.view(-1, x_shape[-2], x_shape[-1])
        ret = torch.zeros_like(x)
        for i in range(0, x_shape[-2], self.kernel.shape[0]):
            for j in range(0, x_shape[-1], self.kernel.shape[0]):
                ret[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]] \
                    = self.kernel.matmul(x[:, i:i + self.kernel.shape[0], j:j + self.kernel.shape[0]]).matmul(self.kernel.t())
        return ret.view(x_shape)

class AXAT(nn.Module):
    def __init__(self, in_features, out_features):
        '''
        :param in_features: 输入数据的最后2维的尺寸
        :param out_features: 输出数据的最后2维的尺寸

        对输入数据 :math:`X` 进行线性变换 :math:`AXA^{T}` 的操作。

        要求输入数据的shape=[*, in_features, in_features]，*表示任意额外添加的维度。

        将输入的数据看作是批量个shape=[in_features, in_features]的矩阵，而 :math:`A` 是shape=[out_features, in_features]的矩阵。
        '''
        super().__init__()
        self.A = nn.Parameter(torch.Tensor(out_features, in_features))
        nn.init.kaiming_uniform_(self.A, a=math.sqrt(5))


    def forward(self, x: torch.Tensor):
        '''
        :param x: 输入数据，shape=[*, in_features, in_features]，*表示任意额外添加的维度
        :return: 输出数据，shape=[*, out_features, out_features]
        '''
        x_shape = list(x.shape)
        x = x.view(-1, x_shape[-2], x_shape[-1])
        x = self.A.matmul(x).matmul(self.A.t())
        x_shape[-1] = x.shape[-1]
        x_shape[-2] = x.shape[-2]
        return x.view(x_shape)

class Dropout(nn.Module):
    def __init__(self, p=0.5):
        '''
        :param p: 设置为0的概率

        与torch.nn.Dropout的操作相同，但是在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        torch.nn.Dropout在SNN中使用时，由于SNN需要运行一定的步长，每一步运行（t=0,1,...,T-1）时都会有不同的dropout，导致网络的结构\\
        实际上是在持续变化：例如可能出现t=0时刻，i到j的连接被断开，但t=1时刻，i到j的连接又被保持。

        在SNN中的dropout应该是，当前这一轮的运行中，t=0时若i到j的连接被断开，则之后t=1,2,...,T-1时刻，i到j的连接应该一直被\\
        断开；而到了下一轮运行时，重新按照概率去决定i到j的连接是否断开，因此重写了适用于SNN的Dropout。

        .. tip::
            从之前的实验结果可以看出，当使用LIF神经元，损失函数或分类结果被设置成时间上累计输出的值，torch.nn.Dropout几乎对SNN没有\\
            影响，即便dropout的概率被设置成高达0.9。可能是LIF神经元的积分行为，对某一个时刻输入的缺失并不敏感。
        '''
        super().__init__()
        assert 0 < p < 1
        self.mask = None
        self.p = p
    def forward(self, x:torch.Tensor):
        '''
        :param x: shape=[*]的tensor
        :return: shape与x.shape相同的tensor
        '''
        if self.training:
            if self.mask is None:
                self.mask = (torch.rand_like(x) > self.p).float()
            return self.mask * x / (1 - self.p)
        else:
            return x

    def reset(self):
        '''
        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。
        '''
        self.mask = None

class Dropout2d(nn.Module):
    def __init__(self, p=0.2):
        '''
        :param p: 设置为0的概率

        与torch.nn.Dropout2d的操作相同，但是在每一轮的仿真中，被设置成0的位置不会发生改变；直到下一轮运行，即网络调用reset()函\\
        数后，才会按照概率去重新决定，哪些位置被置0。

        '''
        super().__init__()
        assert 0 < p < 1
        self.mask = None
        self.p = p
    def forward(self, x:torch.Tensor):
        '''
        :param x: shape=[N, C, W, H]的tensor
        :return: shape=[N, C, W, H]，与x.shape相同的tensor
        '''
        if self.training:
            if self.mask is None:
                self.mask = (torch.rand(size=[x.shape[0], x.shape[1], 1, 1], device=x.device) > self.p).float()
            return self.mask * x / (1 - self.p)
        else:
            return x

    def reset(self):
        '''
        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。
        '''
        self.mask = None

class LowPassSynapse(nn.Module):
    def __init__(self, tau=100.0, learnable=False):
        '''
        :param tau: 突触上电流衰减的时间常数
        :param learnable: 时间常数是否设置成可以学习的参数。当设置为可学习参数时，函数参数中的tau是该参数的初始值

        具有低通滤波性质的突触。突触的输出电流满足，当没有脉冲输入时，输出电流指数衰减：

        .. math::
            \\tau \\frac{\\mathrm{d} I(t)}{\\mathrm{d} t} = - I(t)

        当有新脉冲输入时，输出电流自增1：

        .. math::
            I(t) = I(t) + 1
        ..

        记输入脉冲为 :math:`S(t)`，则离散化后，统一的电流更新方程为：

        .. math::
            I(t) = I(t-1) - (1 - S(t)) \\frac{1}{\\tau} I(t-1) + S(t)

        这种突触能将输入脉冲进行“平滑”，简单的示例代码和输出结果：

        .. code-block:: python

            T = 50
            in_spikes = (torch.rand(size=[T]) >= 0.95).float()
            lp_syn = LowPassSynapse(tau=10.0)
            pyplot.subplot(2, 1, 1)
            pyplot.bar(torch.arange(0, T).tolist(), in_spikes, label='in spike')
            pyplot.xlabel('t')
            pyplot.ylabel('spike')
            pyplot.legend()

            out_i = []
            for i in range(T):
                out_i.append(lp_syn(in_spikes[i]))
            pyplot.subplot(2, 1, 2)
            pyplot.plot(out_i, label='out i')
            pyplot.xlabel('t')
            pyplot.ylabel('i')
            pyplot.legend()
            pyplot.show()

        .. image:: ./_static/API/LowPassSynapseFilter.png

        输出电流不仅取决于当前时刻的输入，还取决于之前的输入，使得该突触具有了一定的记忆能力。

        这种突触偶有使用，例如：

        Diehl P U, Cook M. Unsupervised learning of digit recognition using spike-timing-dependent plasticity.[J]. Frontiers in Computational Neuroscience, 2015: 99-99.

        Fang H, Shrestha A, Zhao Z, et al. Exploiting Neuron and Synapse Filter Dynamics in Spatial Temporal Learning of Deep Spiking Neural Network[J]. arXiv: Neural and Evolutionary Computing, 2020.
        '''
        super().__init__()
        if learnable:
            self.tau = nn.Parameter(torch.ones(size=[1]) / tau)
        else:
            self.tau = 1 / tau
        self.out_i = 0

    def forward(self, in_spikes: torch.Tensor):
        '''
        :param in_spikes: shape任意的输入脉冲
        :return: shape与in_spikes.shape相同的输出电流
        '''
        self.out_i = self.out_i - (1 - in_spikes) * self.out_i * self.tau + in_spikes
        return self.out_i

    def reset(self):
        '''
        :return: None

        本层是一个有状态的层。此函数重置本层的状态变量。将电流重置为0。
        '''
        self.out_i = 0

class ChannelsMaxPool(nn.Module):
    def __init__(self, pool:nn.MaxPool1d):
        '''
        :param pool: nn.Maxpool1d的池化层

        在通道所在的维度，第1维，进行池化的层。
        '''
        super().__init__()
        self.pool = pool
    def forward(self, x:torch.Tensor):
        '''
        :param x: shape=[batch_size, C_in, *]的tensor，C_in是输入数据的通道数，*表示任意维度
        :return: shape=[batch_size, C_out, *]的tensor，C_out是池化后的通道数
        '''
        x_shape = x.shape
        return self.pool(x.flatten(2).permute(0, 2, 1)).permute(0, 2, 1).view((x_shape[0], -1) + x_shape[2:])
