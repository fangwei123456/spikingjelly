import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from spikingjelly.activation_based import surrogate


def directional_rnn_cell_forward(
    cell: nn.Module, x: torch.Tensor, states: torch.Tensor
):
    r"""
    **API Language:**
    :ref:`中文 <directional_rnn_cell_forward-cn>` | :ref:`English <directional_rnn_cell_forward-en>`

    ----

    .. _directional_rnn_cell_forward-cn:

    * **中文**

    沿时间维对单个 RNN cell 执行循环，返回整段输出序列和最终状态。

    ``x`` 的时间维位于第 0 维。该函数会逐个时间步调用 ``cell(x[t], ss)``，
    其中 ``ss`` 是当前状态。若 ``states`` 是二维张量，则 ``cell`` 被视为只有一个
    状态张量；若 ``states`` 是三维张量，则其第 0 维表示状态数量，且默认将第 0 个
    状态视为当前时间步的输出。

    :param cell: RNN cell
    :type cell: nn.Module

    :param x: ``shape = [T, batch_size, input_size]`` 的输入
    :type x: torch.Tensor

    :param states: RNN cell的起始状态。
        若RNN cell只有单个隐藏状态，则 ``shape = [batch_size, hidden_size]`` ；
        否则 ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor

    :return: ``(output, final_states)`` ，其中 ``output`` 的形状为
        ``[T, batch_size, hidden_size]``，``final_states`` 的形状与 ``states`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor]

    :raises ValueError: 若 ``states.dim()`` 既不是 ``2`` 也不是 ``3``，则行为未定义，调用方应保证输入合法

    ----

    .. _directional_rnn_cell_forward-en:

    * **English**

    Run a single RNN cell along the time dimension and return the full output
    sequence together with the final state.

    The time axis of ``x`` is the first dimension. The function repeatedly
    evaluates ``cell(x[t], ss)`` where ``ss`` is the current state. If
    ``states`` is a 2-D tensor, the cell is treated as having a single state
    tensor. If ``states`` is a 3-D tensor, its first dimension is interpreted
    as the number of states, and the first state is treated as the output at
    each time step.

    :param cell: the RNN cell
    :type cell: nn.Module

    :param x: input with ``shape = [T, batch_size, input_size]``
    :type x: torch.Tensor

    :param states: initial state of the RNN cell.
        If the RNN cell has a single hidden state, ``shape = [batch_size, hidden_size]``;
        otherwise ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor

    :return: ``(output, final_states)``, where ``output`` has shape
        ``[T, batch_size, hidden_size]`` and ``final_states`` has the same
        shape as ``states``
    :rtype: tuple[torch.Tensor, torch.Tensor]

    :raises ValueError: Callers are expected to provide ``states`` with rank 2 or 3
    """
    T = x.shape[0]
    ss = states

    output = []
    for t in range(T):
        ss = cell(x[t], ss)
        if states.dim() == 2:
            output.append(ss)
        elif states.dim() == 3:
            output.append(ss[0])
            # 当RNN cell具有多个隐藏状态时，通常第0个隐藏状态是其输出
    return torch.stack(output), ss


def bidirectional_rnn_cell_forward(
    cell: nn.Module,
    cell_reverse: nn.Module,
    x: torch.Tensor,
    states: torch.Tensor,
    states_reverse: torch.Tensor,
):
    r"""
    **API Language:**
    :ref:`中文 <bidirectional_rnn_cell_forward-cn>` | :ref:`English <bidirectional_rnn_cell_forward-en>`

    ----

    .. _bidirectional_rnn_cell_forward-cn:

    * **中文**

    沿时间维同时执行正向与反向 RNN cell，返回拼接后的输出序列以及两个方向的最终状态。

    对于每个时间步 ``t``，正向 cell 接收 ``x[t]``，反向 cell 接收
    ``x[T - t - 1]``。若状态张量为三维，则默认将第 0 个状态视为该时间步输出。

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
    :type states_reverse: torch.Tensor

    :return: ``(output, final_states, final_states_reverse)``。其中 ``output``
        的形状为 ``[T, batch_size, 2 * hidden_size]``，由正向与反向输出拼接而成；
        其余两个返回值的形状分别与 ``states``、``states_reverse`` 相同
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]

    ----

    .. _bidirectional_rnn_cell_forward-en:

    * **English**

    Run a forward RNN cell and a reverse RNN cell along the time dimension,
    returning the concatenated output sequence and the final states of both
    directions.

    At each time step ``t``, the forward cell consumes ``x[t]`` and the reverse
    cell consumes ``x[T - t - 1]``. If the state tensor is 3-D, its first state
    is treated as the output at that step.

    :param cell: forward RNN cell that takes the forward sequence as input
    :type cell: nn.Module

    :param cell_reverse: reverse RNN cell that takes the reverse sequence as input
    :type cell_reverse: nn.Module

    :param x: input with ``shape = [T, batch_size, input_size]``
    :type x: torch.Tensor

    :param states: initial state of the forward RNN cell.
        If the RNN cell has a single hidden state, ``shape = [batch_size, hidden_size]``;
        otherwise ``shape = [states_num, batch_size, hidden_size]``
    :type states: torch.Tensor

    :param states_reverse: initial state of the reverse RNN cell.
        If the RNN cell has a single hidden state, ``shape = [batch_size, hidden_size]``;
        otherwise ``shape = [states_num, batch_size, hidden_size]``
    :type states_reverse: torch.Tensor

    :return: ``(output, final_states, final_states_reverse)``. ``output`` has
        shape ``[T, batch_size, 2 * hidden_size]`` and concatenates the forward
        and reverse outputs. The other two return values have the same shapes as
        ``states`` and ``states_reverse``, respectively
    :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
    """
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
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.__init__-cn>` | :ref:`English <SpikingRNNCellBase.__init__-en>`

        ----

        .. _SpikingRNNCellBase.__init__-cn:

        * **中文**

        Spiking RNN cell 的基类。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int

        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int

        :param bias: 若为 ``False``, 则内部的隐藏层不会带有偏置项 ``b_ih`` 和 ``b_hh``。 默认为 ``True``
        :type bias: bool

        .. note::

            所有权重和偏置项都会按照 :math:`\\mathcal{U}(-\\sqrt{k}, \\sqrt{k})` 进行初始化。
            其中 :math:`k = \\frac{1}{\\text{hidden_size}}`.

        ----

        .. _SpikingRNNCellBase.__init__-en:

        * **English**

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
        """
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.bias = bias

    def reset_parameters(self):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.reset_parameters-cn>` | :ref:`English <SpikingRNNCellBase.reset_parameters-en>`

        ----

        .. _SpikingRNNCellBase.reset_parameters-cn:

        * **中文**

        初始化所有可学习参数。


        ----

        .. _SpikingRNNCellBase.reset_parameters-en:

        * **English**

        Initialize all learnable parameters.
        """
        sqrt_k = math.sqrt(1 / self.hidden_size)
        for param in self.parameters():
            nn.init.uniform_(param, -sqrt_k, sqrt_k)

    def weight_ih(self):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.weight_ih-cn>` | :ref:`English <SpikingRNNCellBase.weight_ih-en>`

        ----

        .. _SpikingRNNCellBase.weight_ih-cn:

        * **中文**

        :return: 输入到隐藏状态的连接权重
        :rtype: torch.Tensor

        ----

        .. _SpikingRNNCellBase.weight_ih-en:

        * **English**

        :return: the learnable input-hidden weights
        :rtype: torch.Tensor
        """
        return self.linear_ih.weight

    def weight_hh(self):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.weight_hh-cn>` | :ref:`English <SpikingRNNCellBase.weight_hh-en>`

        ----

        .. _SpikingRNNCellBase.weight_hh-cn:

        * **中文**

        :return: 隐藏状态到隐藏状态的连接权重
        :rtype: torch.Tensor

        ----

        .. _SpikingRNNCellBase.weight_hh-en:

        * **English**

        :return: the learnable hidden-hidden weights
        :rtype: torch.Tensor
        """
        return self.linear_hh.weight

    def bias_ih(self):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.bias_ih-cn>` | :ref:`English <SpikingRNNCellBase.bias_ih-en>`

        ----

        .. _SpikingRNNCellBase.bias_ih-cn:

        * **中文**

        :return: 输入到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        ----

        .. _SpikingRNNCellBase.bias_ih-en:

        * **English**

        :return: the learnable input-hidden bias
        :rtype: torch.Tensor
        """
        return self.linear_ih.bias

    def bias_hh(self):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNCellBase.bias_hh-cn>` | :ref:`English <SpikingRNNCellBase.bias_hh-en>`

        ----

        .. _SpikingRNNCellBase.bias_hh-cn:

        * **中文**

        :return: 隐藏状态到隐藏状态的连接偏置项
        :rtype: torch.Tensor

        ----

        .. _SpikingRNNCellBase.bias_hh-en:

        * **English**

        :return: the learnable hidden-hidden bias
        :rtype: torch.Tensor
        """
        return self.linear_hh.bias


class SpikingRNNBase(nn.Module):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        dropout_p=0,
        invariant_dropout_mask=False,
        bidirectional=False,
        *args,
        **kwargs,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNBase.__init__-cn>` | :ref:`English <SpikingRNNBase.__init__-en>`

        ----

        .. _SpikingRNNBase.__init__-cn:

        * **中文**

        多层（可选双向）脉冲 RNN 的基类。

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
            随着时间变化的 `Dropout``，参见 :class:`~spikingjelly.activation_based.layer.Dropout`。默认为 ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则使用双向RNN。默认为 ``False``
        :type bidirectional: bool
        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数

        ----

        .. _SpikingRNNBase.__init__-en:

        * **English**

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
            `mask` doesn't change in different time steps, see :class:`~spikingjelly.activation_based.layer.Dropout` for more
            information. Default: ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        :type bidirectional: bool
        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        """
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
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNBase.create_cells-cn>` | :ref:`English <SpikingRNNBase.create_cells-en>`

        ----

        .. _SpikingRNNBase.create_cells-cn:

        * **中文**

        :param args: 子类使用的额外参数
        :param kwargs: 子类使用的额外参数
        :return: 若 ``self.bidirectional == True`` 则会返回正反两个堆栈式RNN；否则返回单个堆栈式RNN
        :rtype: nn.Sequential | tuple[nn.Sequential, nn.Sequential]

        ----

        .. _SpikingRNNBase.create_cells-en:

        * **English**

        :param args: additional arguments for sub-class
        :param kwargs: additional arguments for sub-class
        :return: If ``self.bidirectional == True``, return a RNN for forward direction and a RNN for reverse direction;
            else, return a single stacking RNN
        :rtype: nn.Sequential | tuple[nn.Sequential, nn.Sequential]
        """
        if self.bidirectional:
            cells = []
            cells_reverse = []
            cells.append(
                self.base_cell()(
                    self.input_size, self.hidden_size, self.bias, *args, **kwargs
                )
            )
            cells_reverse.append(
                self.base_cell()(
                    self.input_size, self.hidden_size, self.bias, *args, **kwargs
                )
            )
            for i in range(self.num_layers - 1):
                cells.append(
                    self.base_cell()(
                        self.hidden_size * 2,
                        self.hidden_size,
                        self.bias,
                        *args,
                        **kwargs,
                    )
                )
                cells_reverse.append(
                    self.base_cell()(
                        self.hidden_size * 2,
                        self.hidden_size,
                        self.bias,
                        *args,
                        **kwargs,
                    )
                )
            return nn.Sequential(*cells), nn.Sequential(*cells_reverse)

        else:
            cells = []
            cells.append(
                self.base_cell()(
                    self.input_size, self.hidden_size, self.bias, *args, **kwargs
                )
            )
            for i in range(self.num_layers - 1):
                cells.append(
                    self.base_cell()(
                        self.hidden_size, self.hidden_size, self.bias, *args, **kwargs
                    )
                )
            return nn.Sequential(*cells)

    @staticmethod
    def base_cell():
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNBase.base_cell-cn>` | :ref:`English <SpikingRNNBase.base_cell-en>`

        ----

        .. _SpikingRNNBase.base_cell-cn:

        * **中文**

        :return: 构成该RNN的基本RNN Cell。例如对于 :class:`~spikingjelly.activation_based.rnn.SpikingLSTM`，
            返回的是 :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module

        ----

        .. _SpikingRNNBase.base_cell-en:

        * **English**

        :return: The base cell of this RNN. E.g., in :class:`~spikingjelly.activation_based.rnn.SpikingLSTM` this function
            will return :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module
        """
        raise NotImplementedError

    @staticmethod
    def states_num():
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNBase.states_num-cn>` | :ref:`English <SpikingRNNBase.states_num-en>`

        ----

        .. _SpikingRNNBase.states_num-cn:

        * **中文**

        :return: 状态变量的数量。例如对于 :class:`~spikingjelly.activation_based.rnn.SpikingLSTM`，由于其输出是 ``h`` 和 ``c``，
            因此返回 ``2``；而对于 :class:`~spikingjelly.activation_based.rnn.SpikingGRU`，由于其输出是 ``h``，因此返回 ``1``
        :rtype: int

        ----

        .. _SpikingRNNBase.states_num-en:

        * **English**

        :return: The states number. E.g., for :class:`~spikingjelly.activation_based.rnn.SpikingLSTM` the output are ``h``
            and ``c``, this function will return ``2``; for :class:`~spikingjelly.activation_based.rnn.SpikingGRU` the output
            is ``h``, this function will return ``1``
        :rtype: int
        """
        # LSTM: 2
        # GRU: 1
        # RNN: 1
        raise NotImplementedError

    def forward(self, x: torch.Tensor, states=None):
        r"""
        **API Language:**
        :ref:`中文 <SpikingRNNBase.forward-cn>` | :ref:`English <SpikingRNNBase.forward-en>`

        ----

        .. _SpikingRNNBase.forward-cn:

        * **中文**

        执行多层（可选双向）脉冲 RNN 的前向传播。

        输入 ``x`` 的时间维位于第 0 维。若 ``states`` 为 ``None``，则会在 ``x.device``
        上自动创建零初始状态。对双向 RNN，状态张量的第 0 维按照
        ``[layer_0_forward, ..., layer_{L-1}_forward, layer_0_reverse, ..., layer_{L-1}_reverse]``
        排列；对单向 RNN，则仅包含各层正向状态。

        :param x: ``shape = [T, batch_size, input_size]``，输入序列
        :type x: torch.Tensor
        :param states: ``self.states_num()`` 为 ``1`` 时是单个tensor, 否则是一个tuple，包含 ``self.states_num()`` 个tensors。
            所有的tensor的尺寸均为 ``shape = [num_layers * num_directions, batch, hidden_size]``, 包含 ``self.states_num()``
            个初始状态
            如果RNN是双向的, ``num_directions`` 为 ``2``, 否则为 ``1``
        :type states: Union[torch.Tensor, tuple]
        :return: ``(output, output_states)``。``output`` 的形状为
            ``[T, batch, num_directions * hidden_size]``；``output_states`` 在
            ``self.states_num() == 1`` 时为单个张量，否则为包含多个状态张量的 tuple
        :rtype: tuple[torch.Tensor, Union[torch.Tensor, tuple]]

        :raises TypeError: 若 ``states`` 既不是 ``None``、``torch.Tensor`` 也不是 ``tuple``
        :raises ValueError: 若 ``states`` 的层数/方向数维度与当前 RNN 配置不匹配

        ----

        .. _SpikingRNNBase.forward-en:

        * **English**

        Run the forward pass of a stacked spiking RNN, optionally bidirectional.

        The time axis of ``x`` is the first dimension. If ``states`` is ``None``,
        zero initial states are created on ``x.device``. For bidirectional RNNs,
        the first state dimension is ordered as
        ``[layer_0_forward, ..., layer_{L-1}_forward, layer_0_reverse, ..., layer_{L-1}_reverse]``.

        :param x: ``shape = [T, batch_size, input_size]``, tensor containing the features of the input sequence
        :type x: torch.Tensor
        :param states: a single tensor when ``self.states_num()`` is ``1``, otherwise a tuple with ``self.states_num()``
            tensors.
            ``shape = [num_layers * num_directions, batch, hidden_size]`` for all tensors, containing the ``self.states_num()``
            initial states for each element in the batch.
            If the RNN is bidirectional, ``num_directions`` should be ``2``, else it should be ``1``
        :type states: Union[torch.Tensor, tuple]
        :return: ``(output, output_states)``. ``output`` has shape
            ``[T, batch, num_directions * hidden_size]``. ``output_states`` is a
            single tensor when ``self.states_num() == 1``; otherwise it is a tuple
            of state tensors
        :rtype: tuple[torch.Tensor, Union[torch.Tensor, tuple]]

        :raises TypeError: If ``states`` is neither ``None``, ``torch.Tensor`` nor ``tuple``
        :raises ValueError: If the layer/direction dimension of ``states`` does not match this RNN configuration
        """
        # x.shape=[T, batch_size, input_size]
        # states states_num 个 [num_layers * num_directions, batch, hidden_size]
        batch_size = x.shape[1]

        if isinstance(states, tuple):
            # states非None且为tuple，则合并成tensor
            states_list = torch.stack(states)
            # shape = [self.states_num(), self.num_layers * 2, batch_size, self.hidden_size]
        elif isinstance(states, torch.Tensor):
            if states.dim() == 3:
                states_list = states
            else:
                raise TypeError
        elif states is None:
            if self.bidirectional:
                states_list = torch.zeros(
                    size=[
                        self.states_num(),
                        self.num_layers * 2,
                        x.shape[1],
                        self.hidden_size,
                    ],
                    dtype=x.dtype,
                    device=x.device,
                ).squeeze(0)
            else:
                states_list = torch.zeros(
                    size=[
                        self.states_num(),
                        self.num_layers,
                        x.shape[1],
                        self.hidden_size,
                    ],
                    dtype=x.dtype,
                    device=x.device,
                ).squeeze(0)

        else:
            raise TypeError

        # print(states_list.shape) [state_num num_direction*num_layer, B, H] or [num_direction*num_layer, B, H]

        if self.bidirectional:
            # 判断 num_direction*num_layers 是否符合要求，否则 new_states_list 会存在额外的0矩阵
            if (
                states_list.dim() == 4 and states_list.shape[1] != 2 * self.num_layers
            ) or (
                states_list.dim() == 3 and states_list.shape[0] != 2 * self.num_layers
            ):
                raise ValueError
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(
                    torch.ones(
                        size=[self.num_layers - 1, batch_size, self.hidden_size * 2]
                    ),
                    p=self.dropout_p,
                    training=True,
                    inplace=True,
                ).to(x)
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
                    self.cells[i],
                    self.cells_reverse[i],
                    y,
                    cell_init_states,
                    cell_init_states_reverse,
                )
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
                return y, tuple(new_states_list)

        else:
            # 判断 num_direction*num_layers 是否符合要求，否则 new_states_list 会存在额外的0矩阵
            if (states_list.dim() == 4 and states_list.shape[1] != self.num_layers) or (
                states_list.dim() == 3 and states_list.shape[0] != self.num_layers
            ):
                raise ValueError
            # y 表示第i层的输出。初始化时，y即为输入
            y = x.clone()
            if self.training and self.dropout_p > 0 and self.invariant_dropout_mask:
                mask = F.dropout(
                    torch.ones(
                        size=[self.num_layers - 1, batch_size, self.hidden_size * 2]
                    ),
                    p=self.dropout_p,
                    training=True,
                    inplace=True,
                ).to(x)
            for i in range(self.num_layers):
                # 第i层神经元的起始状态从输入states_list获取
                new_states_list = torch.zeros_like(states_list.data)
                if self.states_num() == 1:
                    cell_init_states = states_list[i]
                else:
                    cell_init_states = states_list[:, i]

                if self.training and self.dropout_p > 0:
                    if i > 1:
                        if self.invariant_dropout_mask:
                            y = y * mask[i - 1]
                        else:
                            y = F.dropout(y, p=self.dropout_p, training=True)
                y, ss = directional_rnn_cell_forward(self.cells[i], y, cell_init_states)
                # 更新states_list[i]
                if self.states_num() == 1:
                    new_states_list[i] = ss
                else:
                    new_states_list[:, i] = torch.stack(ss)
                states_list = new_states_list.clone()
            if self.states_num() == 1:
                return y, new_states_list
            else:
                return y, tuple(new_states_list)


class SpikingLSTMCell(SpikingRNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias=True,
        surrogate_function1=surrogate.Erf(),
        surrogate_function2=None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingLSTMCell.__init__-cn>` | :ref:`English <SpikingLSTMCell.__init__-en>`

        ----

        .. _SpikingLSTMCell.__init__-cn:

        * **中文**

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
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``g`` 反向传播时使用。 若为 ``None``, 则设置成
            ``surrogate_function1``。默认为 ``None``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]


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

        ----

        .. _SpikingLSTMCell.__init__-en:

        * **English**

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

        :param hidden_size: The number of features in the hidden state ``h``
        :type hidden_size: int

        :param bias: If ``False``, then the layer does not use bias weights ``b_ih`` and
            ``b_hh``. Default: ``True``
        :type bias: bool

        :param surrogate_function1: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``i``, ``f``, ``o``
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase

        :param surrogate_function2: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``g``. If ``None``, the surrogate function for generating ``g``
            will be set as ``surrogate_function1``. Default: ``None``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]

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
        """

        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 4 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 4 * hidden_size, bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()

    def forward(self, x: torch.Tensor, hc=None):
        r"""
        **API Language:**
        :ref:`中文 <SpikingLSTMCell.forward-cn>` | :ref:`English <SpikingLSTMCell.forward-en>`

        ----

        .. _SpikingLSTMCell.forward-cn:

        * **中文**

        执行单步 Spiking LSTM cell 前向传播。

        若 ``hc`` 为 ``None``，则会在 ``x.device`` 上构造零初始状态 ``h_0`` 和
        ``c_0``。返回值中的 ``h_1``、``c_1`` 分别表示当前时间步的隐藏状态与细胞状态。

        :param x: ``shape = [batch_size, input_size]`` 的输入
        :type x: torch.Tensor

        :param hc: ``(h_0, c_0)``，其中两个张量的形状均为
            ``[batch_size, hidden_size]``。若为 ``None``，则 ``h_0`` 和 ``c_0``
            默认为零张量
        :type hc: Optional[tuple[torch.Tensor, torch.Tensor]]
        :return: ``(h_1, c_1)``，两个张量的形状均为 ``[batch_size, hidden_size]``
        :rtype: tuple[torch.Tensor, torch.Tensor]

        ----

        .. _SpikingLSTMCell.forward-en:

        * **English**

        Run the forward pass of a single-step Spiking LSTM cell.

        If ``hc`` is ``None``, zero initial states ``h_0`` and ``c_0`` are created
        on ``x.device``. The returned ``h_1`` and ``c_1`` are the hidden state and
        cell state of the current time step.

        :param x: the input tensor with ``shape = [batch_size, input_size]``
        :type x: torch.Tensor

        :param hc: ``(h_0, c_0)``, where both tensors have shape
            ``[batch_size, hidden_size]``. If ``None``, both states default to zeros
        :type hc: Optional[tuple[torch.Tensor, torch.Tensor]]
        :return: ``(h_1, c_1)``, both tensors have shape ``[batch_size, hidden_size]``
        :rtype: tuple[torch.Tensor, torch.Tensor]
        """
        if hc is None:
            h = torch.zeros(
                size=[x.shape[0], self.hidden_size], dtype=x.dtype, device=x.device
            )
            c = torch.zeros_like(h)
        else:
            h = hc[0]
            c = hc[1]

        if self.surrogate_function2 is None:
            i, f, g, o = torch.split(
                self.surrogate_function1(self.linear_ih(x) + self.linear_hh(h)),
                self.hidden_size,
                dim=1,
            )
        else:
            i, f, g, o = torch.split(
                self.linear_ih(x) + self.linear_hh(h), self.hidden_size, dim=1
            )
            i = self.surrogate_function1(i)
            f = self.surrogate_function1(f)
            g = self.surrogate_function2(g)
            o = self.surrogate_function1(o)

        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        c = c * f + i * g
        # According to the original paper, ``c`` can take values 0, 1, or 2.
        # To keep the state bounded, the implementation clamps it to 1 after
        # the update, which matches the intended binary cell-state behavior.

        with torch.no_grad():
            torch.clamp_max_(c, 1.0)

        h = c * o

        return h, c


class SpikingLSTM(SpikingRNNBase):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        dropout_p=0,
        invariant_dropout_mask=False,
        bidirectional=False,
        surrogate_function1=surrogate.Erf(),
        surrogate_function2=None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingLSTM.__init__-cn>` | :ref:`English <SpikingLSTM.__init__-en>`

        ----

        .. _SpikingLSTM.__init__-cn:

        * **中文**

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
            随着时间变化的 `Dropout``，参见 :class:`~spikingjelly.activation_based.layer.Dropout`。默认为 ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则使用双向RNN。默认为 ``False``
        :type bidirectional: bool
        :param surrogate_function1: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``i``, ``f``, ``o`` 反向传播时使用
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 反向传播时用来计算脉冲函数梯度的替代函数, 计算 ``g`` 反向传播时使用。 若为 ``None``, 则设置成
            ``surrogate_function1``。默认为 ``None``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]

        ----

        .. _SpikingLSTM.__init__-en:

        * **English**

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
            `mask` doesn't change in different time steps, see :class:`~spikingjelly.activation_based.layer.Dropout` for more
            information. Default: ``False``
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, becomes a bidirectional LSTM. Default: ``False``
        :type bidirectional: bool
        :param surrogate_function1: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``i``, ``f``, ``o``
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: surrogate function for replacing gradient of spiking functions during
            back-propagation, which is used for generating ``g``. If ``None``, the surrogate function for generating ``g``
            will be set as ``surrogate_function1``. Default: ``None``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        """
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            dropout_p,
            invariant_dropout_mask,
            bidirectional,
            surrogate_function1,
            surrogate_function2,
        )

    @staticmethod
    def base_cell():
        r"""
        **API Language:**
        :ref:`中文 <SpikingLSTM.base_cell-cn>` | :ref:`English <SpikingLSTM.base_cell-en>`

        ----

        .. _SpikingLSTM.base_cell-cn:

        * **中文**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module

        ----

        .. _SpikingLSTM.base_cell-en:

        * **English**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingLSTMCell`
        :rtype: nn.Module
        """
        return SpikingLSTMCell

    @staticmethod
    def states_num():
        r"""
        **API Language:**
        :ref:`中文 <SpikingLSTM.states_num-cn>` | :ref:`English <SpikingLSTM.states_num-en>`

        ----

        .. _SpikingLSTM.states_num-cn:

        * **中文**

        :return: ``2`` （隐藏状态 ``h`` 和细胞状态 ``c``）
        :rtype: int

        ----

        .. _SpikingLSTM.states_num-en:

        * **English**

        :return: ``2`` (hidden state ``h`` and cell state ``c``)
        :rtype: int
        """
        return 2


class SpikingVanillaRNNCell(SpikingRNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias=True,
        surrogate_function=surrogate.Erf(),
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingVanillaRNNCell.__init__-cn>` | :ref:`English <SpikingVanillaRNNCell.__init__-en>`

        ----

        .. _SpikingVanillaRNNCell.__init__-cn:

        * **中文**

        单步脉冲 Vanilla RNN cell。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param bias: 若为 ``False``，则内部线性层不使用偏置项。默认为 ``True``
        :type bias: bool
        :param surrogate_function: 反向传播时用于替代脉冲函数梯度的替代函数
        :type surrogate_function: spikingjelly.activation_based.surrogate.SurrogateFunctionBase

        ----

        .. _SpikingVanillaRNNCell.__init__-en:

        * **English**

        Single-step spiking Vanilla RNN cell.

        :param input_size: Number of input features in ``x``
        :type input_size: int
        :param hidden_size: Number of features in the hidden state ``h``
        :type hidden_size: int
        :param bias: If ``False``, the internal linear layers do not use bias. Default: ``True``
        :type bias: bool
        :param surrogate_function: Surrogate function used to replace the spike gradient during back-propagation
        :type surrogate_function: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        """
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, hidden_size, bias=bias)

        self.surrogate_function = surrogate_function

        self.reset_parameters()

    def forward(self, x: torch.Tensor, h=None):
        r"""
        **API Language:**
        :ref:`中文 <SpikingVanillaRNNCell.forward-cn>` | :ref:`English <SpikingVanillaRNNCell.forward-en>`

        ----

        .. _SpikingVanillaRNNCell.forward-cn:

        * **中文**

        执行单步 Spiking Vanilla RNN cell 前向传播。

        :param x: ``shape = [batch_size, input_size]`` 的输入
        :type x: torch.Tensor
        :param h: ``shape = [batch_size, hidden_size]`` 的起始隐藏状态。若为 ``None``，则使用零初始状态
        :type h: Optional[torch.Tensor]
        :return: ``shape = [batch_size, hidden_size]`` 的下一时刻隐藏状态
        :rtype: torch.Tensor

        ----

        .. _SpikingVanillaRNNCell.forward-en:

        * **English**

        Run the forward pass of a single-step Spiking Vanilla RNN cell.

        :param x: Input tensor with ``shape = [batch_size, input_size]``
        :type x: torch.Tensor
        :param h: Initial hidden state with ``shape = [batch_size, hidden_size]``. If ``None``, a zero state is used
        :type h: Optional[torch.Tensor]
        :return: Next hidden state with ``shape = [batch_size, hidden_size]``
        :rtype: torch.Tensor
        """
        if h is None:
            h = torch.zeros(
                size=[x.shape[0], self.hidden_size], dtype=x.dtype, device=x.device
            )
        return self.surrogate_function(self.linear_ih(x) + self.linear_hh(h))


class SpikingVanillaRNN(SpikingRNNBase):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        dropout_p=0,
        invariant_dropout_mask=False,
        bidirectional=False,
        surrogate_function=surrogate.Erf(),
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingVanillaRNN.__init__-cn>` | :ref:`English <SpikingVanillaRNN.__init__-en>`

        ----

        .. _SpikingVanillaRNN.__init__-cn:

        * **中文**

        多层（可选双向）脉冲 Vanilla RNN。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param num_layers: 循环层数
        :type num_layers: int
        :param bias: 若为 ``False``，内部线性层不使用偏置项。默认为 ``True``
        :type bias: bool
        :param dropout_p: 除最后一层外的层间 dropout 概率。默认为 ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: 若为 ``True``，则训练时在时间维共享 dropout mask
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则构造双向 RNN。默认为 ``False``
        :type bidirectional: bool
        :param surrogate_function: 反向传播时用于替代脉冲函数梯度的替代函数
        :type surrogate_function: spikingjelly.activation_based.surrogate.SurrogateFunctionBase

        ----

        .. _SpikingVanillaRNN.__init__-en:

        * **English**

        Stacked spiking Vanilla RNN, optionally bidirectional.

        :param input_size: Number of input features in ``x``
        :type input_size: int
        :param hidden_size: Number of features in the hidden state ``h``
        :type hidden_size: int
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        :param bias: If ``False``, the internal linear layers do not use bias. Default: ``True``
        :type bias: bool
        :param dropout_p: Inter-layer dropout probability except for the last layer. Default: ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: If ``True``, the dropout mask is shared across time steps during training
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, build a bidirectional RNN. Default: ``False``
        :type bidirectional: bool
        :param surrogate_function: Surrogate function used to replace the spike gradient during back-propagation
        :type surrogate_function: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        """
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            dropout_p,
            invariant_dropout_mask,
            bidirectional,
            surrogate_function,
        )

    @staticmethod
    def base_cell():
        r"""
        **API Language:**
        :ref:`中文 <SpikingVanillaRNN.base_cell-cn>` | :ref:`English <SpikingVanillaRNN.base_cell-en>`

        ----

        .. _SpikingVanillaRNN.base_cell-cn:

        * **中文**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingVanillaRNNCell`
        :rtype: nn.Module

        ----

        .. _SpikingVanillaRNN.base_cell-en:

        * **English**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingVanillaRNNCell`
        :rtype: nn.Module
        """
        return SpikingVanillaRNNCell

    @staticmethod
    def states_num():
        r"""
        **API Language:**
        :ref:`中文 <SpikingVanillaRNN.states_num-cn>` | :ref:`English <SpikingVanillaRNN.states_num-en>`

        ----

        .. _SpikingVanillaRNN.states_num-cn:

        * **中文**

        :return: ``1`` （仅隐藏状态 ``h``）
        :rtype: int

        ----

        .. _SpikingVanillaRNN.states_num-en:

        * **English**

        :return: ``1`` (hidden state ``h`` only)
        :rtype: int
        """
        return 1


class SpikingGRUCell(SpikingRNNCellBase):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        bias=True,
        surrogate_function1=surrogate.Erf(),
        surrogate_function2=None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingGRUCell.__init__-cn>` | :ref:`English <SpikingGRUCell.__init__-en>`

        ----

        .. _SpikingGRUCell.__init__-cn:

        * **中文**

        单步脉冲 GRU cell。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param bias: 若为 ``False``，则内部线性层不使用偏置项。默认为 ``True``
        :type bias: bool
        :param surrogate_function1: 用于 ``r``、``z`` 门及默认候选态梯度近似的替代函数
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 候选态 ``n`` 的替代函数。若为 ``None``，则复用 ``surrogate_function1``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]

        ----

        .. _SpikingGRUCell.__init__-en:

        * **English**

        Single-step spiking GRU cell.

        :param input_size: Number of input features in ``x``
        :type input_size: int
        :param hidden_size: Number of features in the hidden state ``h``
        :type hidden_size: int
        :param bias: If ``False``, the internal linear layers do not use bias. Default: ``True``
        :type bias: bool
        :param surrogate_function1: Surrogate function used for the ``r`` and ``z`` gates and, by default, the candidate state
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: Surrogate function for the candidate state ``n``. If ``None``, ``surrogate_function1`` is reused
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        """
        super().__init__(input_size, hidden_size, bias)

        self.linear_ih = nn.Linear(input_size, 3 * hidden_size, bias=bias)
        self.linear_hh = nn.Linear(hidden_size, 3 * hidden_size, bias=bias)

        self.surrogate_function1 = surrogate_function1
        self.surrogate_function2 = surrogate_function2
        if self.surrogate_function2 is not None:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking

        self.reset_parameters()

    def forward(self, x: torch.Tensor, h=None):
        r"""
        **API Language:**
        :ref:`中文 <SpikingGRUCell.forward-cn>` | :ref:`English <SpikingGRUCell.forward-en>`

        ----

        .. _SpikingGRUCell.forward-cn:

        * **中文**

        执行单步 Spiking GRU cell 前向传播。

        :param x: ``shape = [batch_size, input_size]`` 的输入
        :type x: torch.Tensor
        :param h: ``shape = [batch_size, hidden_size]`` 的起始隐藏状态。若为 ``None``，则使用零初始状态
        :type h: Optional[torch.Tensor]
        :return: ``shape = [batch_size, hidden_size]`` 的下一时刻隐藏状态
        :rtype: torch.Tensor

        ----

        .. _SpikingGRUCell.forward-en:

        * **English**

        Run the forward pass of a single-step Spiking GRU cell.

        :param x: Input tensor with ``shape = [batch_size, input_size]``
        :type x: torch.Tensor
        :param h: Initial hidden state with ``shape = [batch_size, hidden_size]``. If ``None``, a zero state is used
        :type h: Optional[torch.Tensor]
        :return: Next hidden state with ``shape = [batch_size, hidden_size]``
        :rtype: torch.Tensor
        """
        if h is None:
            h = torch.zeros(
                size=[x.shape[0], self.hidden_size], dtype=x.dtype, device=x.device
            )

        y_ih = torch.split(self.linear_ih(x), self.hidden_size, dim=1)
        y_hh = torch.split(self.linear_hh(h), self.hidden_size, dim=1)
        r = self.surrogate_function1(y_ih[0] + y_hh[0])
        z = self.surrogate_function1(y_ih[1] + y_hh[1])

        if self.surrogate_function2 is None:
            n = self.surrogate_function1(y_ih[2] + r * y_hh[2])
        else:
            assert self.surrogate_function1.spiking == self.surrogate_function2.spiking
            n = self.surrogate_function2(y_ih[2] + r * y_hh[2])

        h = (1.0 - z) * n + z * h
        return h


class SpikingGRU(SpikingRNNBase):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers,
        bias=True,
        dropout_p=0,
        invariant_dropout_mask=False,
        bidirectional=False,
        surrogate_function1=surrogate.Erf(),
        surrogate_function2=None,
    ):
        r"""
        **API Language:**
        :ref:`中文 <SpikingGRU.__init__-cn>` | :ref:`English <SpikingGRU.__init__-en>`

        ----

        .. _SpikingGRU.__init__-cn:

        * **中文**

        多层（可选双向）脉冲 GRU。

        :param input_size: 输入 ``x`` 的特征数
        :type input_size: int
        :param hidden_size: 隐藏状态 ``h`` 的特征数
        :type hidden_size: int
        :param num_layers: 循环层数
        :type num_layers: int
        :param bias: 若为 ``False``，内部线性层不使用偏置项。默认为 ``True``
        :type bias: bool
        :param dropout_p: 除最后一层外的层间 dropout 概率。默认为 ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: 若为 ``True``，则训练时在时间维共享 dropout mask
        :type invariant_dropout_mask: bool
        :param bidirectional: 若为 ``True``，则构造双向 RNN。默认为 ``False``
        :type bidirectional: bool
        :param surrogate_function1: 用于 ``r``、``z`` 门及默认候选态梯度近似的替代函数
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: 候选态 ``n`` 的替代函数。若为 ``None``，则复用 ``surrogate_function1``
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]

        ----

        .. _SpikingGRU.__init__-en:

        * **English**

        Stacked spiking GRU, optionally bidirectional.

        :param input_size: Number of input features in ``x``
        :type input_size: int
        :param hidden_size: Number of features in the hidden state ``h``
        :type hidden_size: int
        :param num_layers: Number of recurrent layers
        :type num_layers: int
        :param bias: If ``False``, the internal linear layers do not use bias. Default: ``True``
        :type bias: bool
        :param dropout_p: Inter-layer dropout probability except for the last layer. Default: ``0``
        :type dropout_p: float
        :param invariant_dropout_mask: If ``True``, the dropout mask is shared across time steps during training
        :type invariant_dropout_mask: bool
        :param bidirectional: If ``True``, build a bidirectional RNN. Default: ``False``
        :type bidirectional: bool
        :param surrogate_function1: Surrogate function used for the ``r`` and ``z`` gates and, by default, the candidate state
        :type surrogate_function1: spikingjelly.activation_based.surrogate.SurrogateFunctionBase
        :param surrogate_function2: Surrogate function for the candidate state ``n``. If ``None``, ``surrogate_function1`` is reused
        :type surrogate_function2: Optional[spikingjelly.activation_based.surrogate.SurrogateFunctionBase]
        """
        super().__init__(
            input_size,
            hidden_size,
            num_layers,
            bias,
            dropout_p,
            invariant_dropout_mask,
            bidirectional,
            surrogate_function1,
            surrogate_function2,
        )

    @staticmethod
    def base_cell():
        r"""
        **API Language:**
        :ref:`中文 <SpikingGRU.base_cell-cn>` | :ref:`English <SpikingGRU.base_cell-en>`

        ----

        .. _SpikingGRU.base_cell-cn:

        * **中文**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingGRUCell`
        :rtype: nn.Module

        ----

        .. _SpikingGRU.base_cell-en:

        * **English**

        :return: :class:`~spikingjelly.activation_based.rnn.SpikingGRUCell`
        :rtype: nn.Module
        """
        return SpikingGRUCell

    @staticmethod
    def states_num():
        r"""
        **API Language:**
        :ref:`中文 <SpikingGRU.states_num-cn>` | :ref:`English <SpikingGRU.states_num-en>`

        ----

        .. _SpikingGRU.states_num-cn:

        * **中文**

        :return: ``1`` （仅隐藏状态 ``h``）
        :rtype: int

        ----

        .. _SpikingGRU.states_num-en:

        * **English**

        :return: ``1`` (hidden state ``h`` only)
        :rtype: int
        """
        return 1
