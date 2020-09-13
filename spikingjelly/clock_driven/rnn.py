import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate, accelerating, layer
import math

def bidirectional_rnn_cell_forward(cell: nn.Module, cell_reverse: nn.Module, x: torch.Tensor,
                                   states: torch.Tensor, states_reverse: torch.Tensor):
    '''
    :param cell: 正向RNN cell，输入是正向序列
    :type cell: nn.Module
    :param cell_reverse: 反向的RNN cell，输入是反向序列
    :type cell_reverse: nn.Module
    :param x: ``shape = [T, batch_size, input_size]`` 的输入
    :type x: torch.Tensor
    :param states: 正向RNN cell的起始状态
        若RNN cell只有单个隐藏状态，则 ``shape = [batch_size, hidden_size]`` ；
        否则 ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor
    :param states_reverse: 反向RNN cell的起始状态
        若RNN cell只有单个隐藏状态，则 ``shape = [batch_size, hidden_size]`` ；
        否则 ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor
    :return: y, ss, ss_r
        y: torch.Tensor
            ``shape = [T, batch_size, 2 * hidden_size]`` 的输出。``y[t]`` 由正向cell在 ``t`` 时刻和反向cell在 ``T - t - 1``
            时刻的输出拼接而来
        ss: torch.Tensor
            ``shape`` 与 ``states`` 相同，正向cell在 ``T-1`` 时刻的状态
        ss_r: torch.Tensor
            ``shape`` 与 ``states_reverse`` 相同，反向cell在 ``0`` 时刻的状态

    计算单个正向和反向RNN cell沿着时间维度的循环并输出结果和两个cell的最终状态。
    '''
    T = x.shape[0]
    ss = states
    ss_r = states_reverse
    output = []
    output_r = []
    for t in range(T):
        ss = cell(x[t], ss)
        ss_r = cell_reverse(x[T - t - 1], ss_r)
        if states.dim() == 2:
            output.append(ss)
            output_r.append(ss_r)
        elif states.dim() == 3:
            output.append(ss[0])
            output_r.append(ss_r[0])
            # 当RNN cell具有多个隐藏状态时，通常第0个隐藏状态是其输出

    ret = []
    for t in range(T):
        ret.append(torch.cat((output[t], output_r[T - t - 1]), dim=-1))
    return torch.stack(ret), ss, ss_r
class SpikingRNNCellBase(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, bias=True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        return self.linear_ih.weight

    def weight_hh(self):
        return self.linear_hh.weight

    def bias_ih(self):
        return self.linear_ih.bias

    def bias_hh(self):
        return self.linear_hh.bias

class SpikingRNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False, *args, **kwargs):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.dropout_p = dropout_p
        self.invariant_dropout_mask = invariant_dropout_mask
        self.bidirectional = bidirectional

        if self.bidirectional:
            # 双向LSTM的结构可以参考 https://cedar.buffalo.edu/~srihari/CSE676/10.3%20BidirectionalRNN.pdf
            # https://cs224d.stanford.edu/lecture_notes/LectureNotes4.pdf
            self.cells, self.cells_reverse = self.create_cells(*args, **kwargs)

        else:
            self.cells = self.create_cells(*args, **kwargs)

    def create_cells(self, *args, **kwargs):
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            cells_reverse.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
                cells_reverse.append(self.base_cell()(self.hidden_size * 2, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)

        else:
            cells = []
            cells.append(SpikingLSTMCell(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(SpikingLSTMCell(self.hidden_size, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        raise NotImplementedError

    @staticmethod
    def states_num():
        # LSTM: 2
        # GRU: 1
        # RNN: 1
        raise NotImplementedError

    def forward(self, x: torch.Tensor, states=None):
        # x.shape=[T, batch_size, input_size]
        # states states_num 个 [num_layers * num_directions, batch, hidden_size]
        T = x.shape[0]
        batch_size = x.shape[1]

        if isinstance(states, tuple):
            # states非None且为tuple，则合并成tensor
            states_list = torch.stack(states)
            # shape = [self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]
        elif isinstance(states, torch.Tensor):
            # states非None且不为tuple时，它本身就是一个tensor，例如普通RNN的状态
            states_list = states
        elif states is None:
            # squeeze()的作用是，若states_num() == 1则去掉多余的维度
            if self.bidirectional:
                states_list = torch.zeros(
                    size=[self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]).to(x).squeeze()
            else:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers, batch_size, self.hidden_size]).to(
                    x).squeeze()
        else:
            raise TypeError

        if self.bidirectional:
            # y 表示第i层的输出。初始化时，y即为输入
            y = x
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                    cell_init_states_reverse = states_list[i + self.num_layers]
                else:
                    cell_init_states = states_list[:, i]
                    cell_init_states_reverse = states_list[:, i + self.num_layers]

                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss, ss_r = bidirectional_rnn_cell_forward(
                    self.cells[i], self.cells_reverse[i], y, cell_init_states, cell_init_states_reverse)
                # 更新states_list[i]
                if self.states_num() == 1:
                    states_list[i] = ss
                    states_list[i + self.num_layers] = ss_r
                else:
                    states_list[:, i] = torch.stack(ss)
                    states_list[:, i + self.num_layers] = torch.stack(ss_r)
            if self.states_num() == 1:
                return y, states_list
            else:
                # split使得返回值是tuple
                return y, torch.split(states_list, 1, dim=0)

        else:
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)

            output = []
            for t in range(T):
                if self.states_num() == 1:
                    states_list[0] = self.cells[0](x[t], states_list[0])
                else:
                    states_list[:, 0] = torch.stack(self.cells[0](x[t], states_list[:, 0]))
                for i in range(1, self.num_layers):
                    y = states_list[0, i - 1]
                    if self.training and self.dropout_p > 0:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                    if self.states_num() == 1:
                        states_list[i] = self.cells[i](y, states_list[i])
                    else:
                        states_list[:, i] = torch.stack(self.cells[i](y, states_list[:, i]))
                if self.states_num() == 1:
                    output.append(states_list[-1].clone().unsqueeze(0))
                else:
                    output.append(states_list[0, -1].clone().unsqueeze(0))
            if self.states_num() == 1:
                return torch.cat(output, dim=0), states_list
            else:
                # split使得返回值是tuple
                return torch.cat(output, dim=0), torch.split(states_list, 1, dim=0)

class SpikingLSTMCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        '''
        A `spiking` long short-term memory (LSTM) cell, which is firstly proposed in
        `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_.

        .. math::

            i &= \\Theta(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\\\
            f &= \\Theta(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\\\
            g &= \\Theta(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\\\
            o &= \\Theta(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\\\
            c' &= f * c + i * g \\\\
            h' &= o * c'

        where :math:`\\Theta` is the heaviside function, and :math:`*` is the Hadamard product.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int

        :param hidden_size: int
        :type hidden_size: The number of features in the hidden state ``h``

        :param bias: If ``False``, then the layer does not use bias weights ``b_ih`` and
            ``b_hh``. Default: ``True``
        :type bias: bool

        :param surrogate_function1: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``i``, ``f``, ``o``

        :param surrogate_function2: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``g``. If ``None``, the surrogate function for generating ``g``
            will be set as ``surrogate_function1``. Default: ``None``

        .. admonition:: Note
            :class: note

            All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
            where :math:`k = \\frac{1}{\\text{hidden_size}}`.

        Examples:

        .. code-block:: python

            T = 6
            batch_size = 2
            input_size = 3
            hidden_size = 4
            rnn = rnn.SpikingLSTMCell(input_size, hidden_size)
            input = torch.randn(T, batch_size, input_size) * 50
            h = torch.randn(batch_size, hidden_size)
            c = torch.randn(batch_size, hidden_size)

            output = []
            for t in range(T):
                h, c = rnn(input[t], (h, c))
                output.append(h)
            print(output)
        '''

        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()

    def forward(self, x: torch.Tensor, hc=None):
        '''
        :param x: the input tensor with ``shape = [batch_size, input_size]``
        :type x: torch.Tensor

        :param hc: (h_0, c_0)
                h_0 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``, tensor containing the initial hidden state for each element in the batch
                c_0 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``, tensor containing the initial cell state for each element in the batch
                If (h_0, c_0) is not provided, both ``h_0`` and ``c_0`` default to zero
        :type hc: tuple or None
        :return: (h_1, c_1) :
                h_1 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``, tensor containing the next hidden state for each element in the batch
                c_1 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``, tensor containing the next cell state for each element in the batch
        :rtype: tuple
        '''
        if hc is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]
        if self.surrogate_function2 is None:
            i, f, g, o = torch.split(self.surrogate_function1(self.linear_ih(x) + self.linear_hh(h)),
                                     self.hidden_size, dim=1)
        else:
            i, f, g, o = torch.split(self.linear_ih(x) + self.linear_hh(h), self.hidden_size, dim=1)
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking
        if self.surrogate_function1.spiking:
            # 可以使用针对脉冲的加速
            # c = f * c + i * g
            c = accelerating.mul(c, f) + accelerating.mul(i, g, True)
            # h = o * c
            h = accelerating.mul(c, o)
        else:
            c = c * f + i * g
            h = c * o
        return h, c

class SpikingLSTM(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout_p, invariant_dropout_mask, bidirectional,
                         surrogate_function1, surrogate_function2)
    @staticmethod
    def base_cell():
        return SpikingLSTMCell

    @staticmethod
    def states_num():
        return 2

class SpikingVanillaRNNCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function=surrogate.Erf()):
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.surrogate_function = surrogate_function

        self.reset_parameters()

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
        return self.surrogate_function(self.linear_ih(x) + self.linear_hh(h))

class SpikingVanillaRNN(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False, surrogate_function=surrogate.Erf()):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout_p, invariant_dropout_mask, bidirectional,
                         surrogate_function)

    @staticmethod
    def base_cell():
        return SpikingVanillaRNNCell

    @staticmethod
    def states_num():
        return 1

class SpikingGRUCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()
    def forward(self, x: torch.Tensor, hc=None):
        if hc is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]

        y_ih = torch.split(self.linear_ih(x), self.hidden_size, dim=1)
        y_hh = torch.split(self.linear_hh(x), self.hidden_size, dim=1)
        r = self.surrogate_function1(y_ih[0] + y_hh[0])
        z = self.surrogate_function1(y_ih[1] + y_hh[1])

        if self.surrogate_function2 is None:
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
        else:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking
            n = self.surrogate_function2(y_ih[2] + r * y_hh[2])

        if self.surrogate_function1.spiking:
            # 可以使用针对脉冲的加速
            h = accelerating.mul(accelerating.sub(torch.ones_like(z.data), z), n, True) + accelerating.mul(h, z)
            # h不一定是脉冲数据，因此没有使用 accelerating.mul(h, True)
        else:
            h = (1 - z) * n + z * h
        return h

class SpikingGRU(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        super().__init__(input_size, hidden_size, num_layers, bias, dropout_p, invariant_dropout_mask, bidirectional,
                         surrogate_function1, surrogate_function2)
    @staticmethod
    def base_cell():
        return SpikingGRUCell

    @staticmethod
    def states_num():
        return 1

