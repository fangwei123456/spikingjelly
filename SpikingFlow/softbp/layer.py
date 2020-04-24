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
        中提出的NeuNorm层。NeuNorm层必须放在二维卷积层后的脉冲神经元后，例如

        Conv2d -> LIF -> NeuNorm

        要求输入的尺寸是[batch_size, in_channels, W, H]。

        in_channels是输入到NeuNorm层的通道数，也就是论文中\
        的 :math:`F`。

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

        将输入的shape=[*, W, H]的数据进行分块DCT变换的层，*表示任意额外添加的维度。变换只在最后2维进行，要求W和H都能\
        整除kernel_size

        DCT是AXAT的一种特例
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

        要求输入数据的shape=[*, in_features, in_features]，*表示任意额外添加的维度

        将输入的数据看作是批量个shape=[in_features, in_features]的矩阵，而 :math:`A` 是shape=[out_features, in_features]的矩阵
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
