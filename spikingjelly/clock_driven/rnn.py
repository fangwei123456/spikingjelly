import torch
import torch.nn as nn
import torch.nn.functional as F
from spikingjelly.clock_driven import surrogate, layer
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
        '''
        * :ref:`API in English <SpikingRNNCellBase.__init__-en>`

        .. _SpikingRNNCellBase.__init__-cn:

        Spiking RNN Cell 的基类。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int

        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int

        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool

        .. note::

            所有权重和偏置项都会按照 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 进行初始化。
            其中 :math:`k = \\frac{1}{\\text{hidden_size}}`.

        * :ref:`中文API <SpikingRNNCellBase.__init__-cn>`

        .. _SpikingRNNCellBase.__init__-en:

        The base class of Spiking RNN Cell.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int

        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int

        :param bias: If ``False``, then the layer does not use bias weights ``b_ih`` and
            ``b_hh``. Default: ``True``
        :type bias: bool

        .. admonition:: Note
            :class: note

            All the weights and biases are initialized from :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})`
            where :math:`k = \\frac{1}{\\text{hidden_size}}`.

        '''
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        '''
        * :ref:`API in English <SpikingRNNCellBase.reset_parameters-en>`

        .. _SpikingRNNCellBase.reset_parameters-cn:

        初始化所有可学习参数。

        * :ref:`中文API <SpikingRNNCellBase.reset_parameters-cn>`

        .. _SpikingRNNCellBase.reset_parameters-en:

        Initialize all learnable parameters.
        '''
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        '''
        * :ref:`API in English <SpikingRNNCellBase.weight_ih-en>`

        .. _SpikingRNNCellBase.weight_ih-cn:

        :return: 输入到隐藏状态的连接权重
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.weight_ih-cn>`

        .. _SpikingRNNCellBase.weight_ih-en:

        :return: the learnable input-hidden weights
        :rtype: torch.Tensor
        '''
        return self.linear_ih.weight

    def weight_hh(self):
        '''
        * :ref:`API in English <SpikingRNNCellBase.weight_hh-en>`

        .. _SpikingRNNCellBase.weight_hh-cn:

        :return: 隐藏状态到隐藏状态的连接权重
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.weight_hh-cn>`

        .. _SpikingRNNCellBase.weight_hh-en:

        :return: the learnable hidden-hidden weights
        :rtype: torch.Tensor
        '''
        return self.linear_hh.weight

    def bias_ih(self):
        '''
        * :ref:`API in English <SpikingRNNCellBase.bias_ih-en>`

        .. _SpikingRNNCellBase.bias_ih-cn:

        :return: 输入到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.bias_ih-cn>`

        .. _SpikingRNNCellBase.bias_ih-en:

        :return: the learnable input-hidden bias
        :rtype: torch.Tensor
        '''
        return self.linear_ih.bias

    def bias_hh(self):
        '''
        * :ref:`API in English <SpikingRNNCellBase.bias_hh-en>`

        .. _SpikingRNNCellBase.bias_hh-cn:

        :return: 隐藏状态到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        * :ref:`中文API <SpikingRNNCellBase.bias_hh-cn>`

        .. _SpikingRNNCellBase.bias_hh-en:

        :return: the learnable hidden-hidden bias
        :rtype: torch.Tensor
        '''
        return self.linear_hh.bias

class SpikingRNNBase(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False, *args, **kwargs):
        '''
        * :ref:`API in English <SpikingRNNBase.__init__-en>`

        .. _SpikingRNNBase.__init__-cn:

        多层 `脉冲` RNN的基类。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param num_layers: 内部RNN的层数，例如 ``num_layers = 2`` 将会创建堆栈式的两层RNN，第1层接收第0层的输出作为输入，
            并计算最终输出
        :type num_layers: int
        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool
        :param dropout_p: 若非 ``0``，则除了最后一层，每个RNN层后会增加一个丢弃概率为 ``dropout_p`` 的 `Dropout` 层。
            默认为 ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: 若为 ``False``，则使用普通的 `Dropout`；若为 ``True``，则使用SNN中特有的，`mask` 不
            随着时间变化的 `Dropout``，参见 :class:`~spikingjelly.clock_driven.layer.Dropout`。默认为 ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则使用双向RNN。默认为 ``False``
        :type bidirectional: bool
        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数

        * :ref:`中文API <SpikingRNNBase.__init__-cn>`

        .. _SpikingRNNBase.__init__-en:

        The base-class of a multi-layer `spiking` RNN.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int
        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int
        :param num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would mean stacking two LSTMs
            together to form a `stacked RNN`, with the second RNN taking in outputs of the first RNN and computing the
            final results
        :type num_layers: int
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        :type bias: bool
        :param dropout_p: If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last
            layer, with dropout probability equal to :attr:`dropout`. Default: 0
        :type dropout_p: float
        :param invariant_dropout_mask: If ``False``，use the naive `Dropout`；If ``True``，use the dropout in SNN that
            `mask` doesn't change in different time steps, see :class:`~spikingjelly.clock_driven.layer.Dropout` for more
            information. Defaule: ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        :type bidirectional: bool
        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        '''
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
        '''
        * :ref:`API in English <SpikingRNNBase.create_cells-en>`

        .. _SpikingRNNBase.create_cells-cn:

        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数
        :return: 若 ``self.bidirectional == True`` 则会返回正反两个堆栈式RNN；否则返回单个堆栈式RNN
        :rtype: nn.Sequential

        * :ref:`中文API <SpikingRNNBase.create_cells-cn>`

        .. _SpikingRNNBase.create_cells-en:

        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        :return: If ``self.bidirectional == True``, return a RNN for forward direction and a RNN for reverse direction;
            else, return a single stacking RNN
        :rtype: nn.Sequential
        '''
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
            cells.append(self.base_cell()(self.input_size, self.hidden_size, self.bias, *args, **kwargs))
            for i in range(self.num_layers - 1):
                cells.append(self.base_cell()(self.hidden_size, self.hidden_size, self.bias, *args, **kwargs))
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        '''
        * :ref:`API in English <SpikingRNNBase.base_cell-en>`

        .. _SpikingRNNBase.base_cell-cn:

        :return: 构成该RNN的基本RNN Cell。例如对于 :class:`~spikingjelly.clock_driven.rnn.SpikingLSTM`，
            返回的是 :class:`~spikingjelly.clock_driven.rnn.SpikingLSTMCell`
        :rtype: nn.Module

        * :ref:`中文API <SpikingRNNBase.base_cell-cn>`

        .. _SpikingRNNBase.base_cell-en:

        :return: The base cell of this RNN. E.g., in :class:`~spikingjelly.clock_driven.rnn.SpikingLSTM` this function
            will return :class:`~spikingjelly.clock_driven.rnn.SpikingLSTMCell`
        :rtype: nn.Module
        '''
        raise NotImplementedError

    @staticmethod
    def states_num():
        '''
        * :ref:`API in English <SpikingRNNBase.states_num-en>`

        .. _SpikingRNNBase.states_num-cn:

        :return: 状态变量的数量。例如对于 :class:`~spikingjelly.clock_driven.rnn.SpikingLSTM`，由于其输出是 ``h`` 和 ``c``，
            因此返回 ``2``；而对于 :class:`~spikingjelly.clock_driven.rnn.SpikingGRU`，由于其输出是 ``h``，因此返回 ``1``
        :rtype: int

        * :ref:`中文API <SpikingRNNBase.states_num-cn>`

        .. _SpikingRNNBase.states_num-en:

        :return: The states number. E.g., for :class:`~spikingjelly.clock_driven.rnn.SpikingLSTM` the output are ``h``
            and ``c``, this function will return ``2``; for :class:`~spikingjelly.clock_driven.rnn.SpikingGRU` the output
            is ``h``, this function will return ``1``
        :rtype: int
        '''
        # LSTM: 2
        # GRU: 1
        # RNN: 1
        raise NotImplementedError

    def forward(self, x: torch.Tensor, states=None):
        '''
        * :ref:`API in English <SpikingRNNBase.forward-en>`

        .. _SpikingRNNBase.forward-cn:

        :param x: ``shape = [T, batch_size, input_size]``，输入序列
        :type x: torch.Tensor
        :param states: ``self.states_num()`` 为 ``1`` 时是单个tensor, 否则是一个tuple，包含 ``self.states_num()`` 个tensors。
            所有的tensor的尺寸均为 ``shape = [num_layers * num_directions, batch, hidden_size]``, 包含 ``self.states_num()``
            个初始状态
            如果RNN是双向的, ``num_directions`` 为 ``2``, 否则为 ``1``
        :type states: torch.Tensor or tuple
        :return: output, output_states
            output: torch.Tensor
                ``shape = [T, batch, num_directions * hidden_size]``，最后一层在所有时刻的输出
            output_states: torch.Tensor or tuple
                ``self.states_num()`` 为 ``1`` 时是单个tensor, 否则是一个tuple，包含 ``self.states_num()`` 个tensors。
                所有的tensor的尺寸均为 ``shape = [num_layers * num_directions, batch, hidden_size]``, 包含 ``self.states_num()``
                个最后时刻的状态

        * :ref:`中文API <SpikingRNNBase.forward-cn>`

        .. _SpikingRNNBase.forward-en:

        :param x: ``shape = [T, batch_size, input_size]``, tensor containing the features of the input sequence
        :type x: torch.Tensor
        :param states: a single tensor when ``self.states_num()`` is ``1``, otherwise a tuple with ``self.states_num()``
            tensors.
            ``shape = [num_layers * num_directions, batch, hidden_size]`` for all tensors, containing the ``self.states_num()``
            initial states for each element in the batch.
            If the RNN is bidirectional, ``num_directions`` should be ``2``, else it should be ``1``
        :type states: torch.Tensor or tuple
        :return: output, output_states
            output: torch.Tensor
                ``shape = [T, batch, num_directions * hidden_size]``, tensor containing the output features from the last
                layer of the RNN, for each ``t``
            output_states: torch.Tensor or tuple
                a single tensor when ``self.states_num()`` is ``1``, otherwise a tuple with ``self.states_num()``
                tensors.
                ``shape = [num_layers * num_directions, batch, hidden_size]`` for all tensors, containing the ``self.states_num()``
                states for ``t = T - 1``
        '''
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
            # squeeze(0)的作用是，若states_num() == 1则去掉多余的维度
            if self.bidirectional:
                states_list = torch.zeros(
                    size=[self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]).to(x).squeeze(0)
            else:
                states_list = torch.zeros(size=[self.states_num(), self.num_layers, batch_size, self.hidden_size]).to(
                    x).squeeze(0)
        else:
            raise TypeError

        if self.bidirectional:
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size * 2]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                new_states_list = torch.zeros_like(states_list.data)
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
                    new_states_list[i] = ss
                    new_states_list[i + self.num_layers] = ss_r
                else:
                    new_states_list[:, i] = torch.stack(ss)
                    new_states_list[:, i + self.num_layers] = torch.stack(ss_r)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                # split使得返回值是tuple
                return y, torch.split(new_states_list, 1, dim=0)

        else:
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(torch.ones(size=[self.num_layers - 1, batch_size, self.hidden_size]),
                                 p=self.dropout_p, training=True, inplace=True).to(x)

            output = []

            for t in range(T):
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    new_states_list[0] = self.cells[0](x[t], states_list[0])
                else:
                    new_states_list[:, 0] = torch.stack(self.cells[0](x[t], states_list[:, 0]))
                for i in range(1, self.num_layers):
                    y = states_list[0, i - 1]
                    if self.training and self.dropout_p > 0:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                    if self.states_num() == 1:
                        new_states_list[i] = self.cells[i](y, states_list[i])
                    else:
                        new_states_list[:, i] = torch.stack(self.cells[i](y, states_list[:, i]))
                if self.states_num() == 1:
                    output.append(new_states_list[-1].clone().unsqueeze(0))
                else:
                    output.append(new_states_list[0, -1].clone().unsqueeze(0))
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return torch.cat(output, dim=0), new_states_list
            else:
                # split使得返回值是tuple
                return torch.cat(output, dim=0), torch.split(new_states_list, 1, dim=0)

class SpikingLSTMCell(SpikingRNNCellBase):
    def __init__(self, input_size: int, hidden_size: int, bias=True,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        '''
        * :ref:`API in English <SpikingLSTMCell.__init__-en>`

        .. _SpikingLSTMCell.__init__-cn:

        `脉冲` 长短时记忆 (LSTM) cell, 最先由 `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_
        一文提出。

        .. math::

            i &= \\Theta(W_{ii} x + b_{ii} + W_{hi} h + b_{hi}) \\\\
            f &= \\Theta(W_{if} x + b_{if} + W_{hf} h + b_{hf}) \\\\
            g &= \\Theta(W_{ig} x + b_{ig} + W_{hg} h + b_{hg}) \\\\
            o &= \\Theta(W_{io} x + b_{io} + W_{ho} h + b_{ho}) \\\\
            c' &= f * c + i * g \\\\
            h' &= o * c'

        其中 :math:`\\Theta` 是heaviside阶跃函数（脉冲函数）, and :math:`*` 是Hadamard点积，即逐元素相乘。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int

        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int

        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool

        :param surrogate_function1: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``i``, ``f``, ``o`` 反向传播时使用
        :type surrogate_function1: spikingjelly.clock_driven.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``g`` 反向传播时使用。 若为 ``None``, 则设置成
            ``surrogate_function1``。默认为 ``None``
        :type surrogate_function2: None or spikingjelly.clock_driven.surrogate.SurrogateFunctionBase


        .. note::

            所有权重和偏置项都会按照 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 进行初始化。
            其中 :math:`k = \\frac{1}{\\text{hidden_size}}`.

        示例代码：

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

        * :ref:`中文API <SpikingLSTMCell.__init__-cn>`

        .. _SpikingLSTMCell.__init__-en:

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
        :type surrogate_function1: spikingjelly.clock_driven.surrogate.SurrogateFunctionBase

        :param surrogate_function2: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``g``. If ``None``, the surrogate function for generating ``g``
            will be set as ``surrogate_function1``. Default: ``None``
        :type surrogate_function2: None or spikingjelly.clock_driven.surrogate.SurrogateFunctionBase

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
        * :ref:`API in English <SpikingLSTMCell.forward-en>`

        .. _SpikingLSTMCell.forward-cn:

        :param x: ``shape = [batch_size, input_size]`` 的输入
        :type x: torch.Tensor

        :param hc: (h_0, c_0)
                h_0 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``，起始隐藏状态
                c_0 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``，起始细胞状态
                如果不提供(h_0, c_0)，``h_0`` 默认 ``c_0`` 默认为0
        :type hc: tuple or None
        :return: (h_1, c_1) :
                h_1 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``，下一个时刻的隐藏状态
                c_1 : torch.Tensor
                    ``shape = [batch_size, hidden_size]``，下一个时刻的细胞状态
        :rtype: tuple

        * :ref:`中文API <SpikingLSTMCell.forward-cn>`

        .. _SpikingLSTMCell.forward-en:

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


        c = c * f + i * g
        h = c * o
            
        return h, c

class SpikingLSTM(SpikingRNNBase):
    def __init__(self, input_size, hidden_size, num_layers, bias=True, dropout_p=0,
                 invariant_dropout_mask=False, bidirectional=False,
                 surrogate_function1=surrogate.Erf(), surrogate_function2=None):
        '''
        * :ref:`API in English <SpikingLSTM.__init__-en>`

        .. _SpikingLSTM.__init__-cn:

        多层`脉冲` 长短时记忆LSTM, 最先由 `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_
        一文提出。

        每一层的计算按照

        .. math::

            i_{t} &= \\Theta(W_{ii} x_{t} + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\
            f_{t} &= \\Theta(W_{if} x_{t} + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\
            g_{t} &= \\Theta(W_{ig} x_{t} + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\
            o_{t} &= \\Theta(W_{io} x_{t} + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\
            c_{t} &= f_{t} * c_{t-1} + i_{t} * g_{t} \\\\
            h_{t} &= o_{t} * c_{t-1}'

        其中 :math:`h_{t}` 是 :math:`t` 时刻的隐藏状态，:math:`c_{t}` 是 :math:`t` 时刻的细胞状态，:math:`h_{t-1}` 是该层 :math:`t-1`
        时刻的隐藏状态或起始状态，:math:`i_{t}`，:math:`f_{t}`，:math:`g_{t}`，:math:`o_{t}` 分别是输入，遗忘，细胞，输出门，
        :math:`\\Theta` 是heaviside阶跃函数（脉冲函数）, and :math:`*` 是Hadamard点积，即逐元素相乘。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param num_layers: 内部RNN的层数，例如 ``num_layers = 2`` 将会创建堆栈式的两层RNN，第1层接收第0层的输出作为输入，
            并计算最终输出
        :type num_layers: int
        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool
        :param dropout_p: 若非 ``0``，则除了最后一层，每个RNN层后会增加一个丢弃概率为 ``dropout_p`` 的 `Dropout` 层。
            默认为 ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: 若为 ``False``，则使用普通的 `Dropout`；若为 ``True``，则使用SNN中特有的，`mask` 不
            随着时间变化的 `Dropout``，参见 :class:`~spikingjelly.clock_driven.layer.Dropout`。默认为 ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则使用双向RNN。默认为 ``False``
        :type bidirectional: bool
        :param surrogate_function1: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``i``, ``f``, ``o`` 反向传播时使用
        :type surrogate_function1: spikingjelly.clock_driven.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``g`` 反向传播时使用。 若为 ``None``, 则设置成
            ``surrogate_function1``。默认为 ``None``
        :type surrogate_function2: None or spikingjelly.clock_driven.surrogate.SurrogateFunctionBase


        * :ref:`中文API <SpikingLSTM.__init__-cn>`

        .. _SpikingLSTM.__init__-en:

        The `spiking` multi-layer long short-term memory (LSTM), which is firstly proposed in
        `Long Short-Term Memory Spiking Networks and Their Applications <https://arxiv.org/abs/2007.04779>`_.

        For each element in the input sequence, each layer computes the following
        function:

        .. math::

            i_{t} &= \\Theta(W_{ii} x_{t} + b_{ii} + W_{hi} h_{t-1} + b_{hi}) \\\\
            f_{t} &= \\Theta(W_{if} x_{t} + b_{if} + W_{hf} h_{t-1} + b_{hf}) \\\\
            g_{t} &= \\Theta(W_{ig} x_{t} + b_{ig} + W_{hg} h_{t-1} + b_{hg}) \\\\
            o_{t} &= \\Theta(W_{io} x_{t} + b_{io} + W_{ho} h_{t-1} + b_{ho}) \\\\
            c_{t} &= f_{t} * c_{t-1} + i_{t} * g_{t} \\\\
            h_{t} &= o_{t} * c_{t-1}'

        where :math:`h_t` is the hidden state at time `t`, :math:`c_t` is the cell
        state at time `t`, :math:`x_t` is the input at time `t`, :math:`h_{t-1}`
        is the hidden state of the layer at time `t-1` or the initial hidden
        state at time `0`, and :math:`i_t`, :math:`f_t`, :math:`g_t`,
        :math:`o_t` are the input, forget, cell, and output gates, respectively.
        :math:`\\Theta` is the heaviside function, and :math:`*` is the Hadamard product.

        :param input_size: The number of expected features in the input ``x``
        :type input_size: int
        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int
        :param num_layers: Number of recurrent layers. E.g., setting ``num_layers=2`` would mean stacking two LSTMs
            together to form a `stacked RNN`, with the second RNN taking in outputs of the first RNN and computing the
            final results
        :type num_layers: int
        :param bias: If ``False``, then the layer does not use bias weights `b_ih` and `b_hh`. Default: ``True``
        :type bias: bool
        :param dropout_p: If non-zero, introduces a `Dropout` layer on the outputs of each RNN layer except the last
            layer, with dropout probability equal to :attr:`dropout`. Default: 0
        :type dropout_p: float
        :param invariant_dropout_mask: If ``False``，use the naive `Dropout`；If ``True``，use the dropout in SNN that
            `mask` doesn't change in different time steps, see :class:`~spikingjelly.clock_driven.layer.Dropout` for more
            information. Defaule: ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        :type bidirectional: bool
        :param surrogate_function1: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``i``, ``f``, ``o``
        :type surrogate_function1: spikingjelly.clock_driven.surrogate.SurrogateFunctionBase
        :param surrogate_function2: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``g``. If ``None``, the surrogate function for generating ``g``
            will be set as ``surrogate_function1``. Default: ``None``
        :type surrogate_function2: None or spikingjelly.clock_driven.surrogate.SurrogateFunctionBase
        '''
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

    def forward(self, x: torch.Tensor, h=None):
        if h is None:
            h = torch.zeros(size=[x.shape[0], self.hidden_size], dtype=torch.float, device=x.device)

        y_ih = torch.split(self.linear_ih(x), self.hidden_size, dim=1)
        y_hh = torch.split(self.linear_hh(h), self.hidden_size, dim=1)
        r = self.surrogate_function1(y_ih[0] + y_hh[0])
        z = self.surrogate_function1(y_ih[1] + y_hh[1])

        if self.surrogate_function2 is None:
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
        else:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking
            n = self.surrogate_function2(y_ih[2] + r * y_hh[2])


        h = (1. - z) * n + z * h
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
