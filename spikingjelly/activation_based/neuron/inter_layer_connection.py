from abc import abstractmethod
from typing import Optional

import torch
import torch.nn as nn

from .. import surrogate, base


__all__ = ["ILCBaseNode", "ILCIFNode", "ILCLIFNode", "ILCCUBALIFNode"]


class ILCBaseNode(nn.Module, base.MultiStepModule):
    r"""
    **API Language** - :ref:`中文 <ILCBaseNode-cn>` | :ref:`English <ILCBaseNode-en>`

    ----

    .. _ILCBaseNode-cn:

    * **中文**

    层间连接（Inter-Layer Connection，ILC）神经元基类，是构建跨层脉冲传播连接的抽象基类。

    ILC 神经元与普通神经元类似，在每个时间步按顺序执行充电、放电、重置三个步骤。
    与普通神经元的区别在于，它将输出脉冲通过一个可学习的一维卷积连接
    （:class:`torch.nn.Conv1d`）传递给下一层的输入，从而实现层间的结构化连接。

    各子类通过重写 :meth:`neuronal_charge` 实现不同的充电动力学：

    - :class:`ILCIFNode` — 积分发放（IF）动力学
    - :class:`ILCLIFNode` — 漏电积分发放（LIF）动力学，增加膜电位衰减
    - :class:`ILCCUBALIFNode` — 基于电流的 LIF 动力学，增加电流衰减

    :param act_dim: 输入激活的特征维度
    :type act_dim: int
    :param dec_pop_dim: 解码种群维度，每个特征对应的神经元数量
    :type dec_pop_dim: int
    :param v_threshold: 神经元的阈值电压，膜电位超过该值时发放脉冲
    :type v_threshold: float
    :param v_reset: 重置电压。若为 ``None``，采用软重置（减去阈值）；否则硬重置到此值
    :type v_reset: Optional[float]
    :param surrogate_function: 替代梯度函数，用于在反向传播中计算脉冲函数的近似梯度
    :type surrogate_function: surrogate.SurrogateFunctionBase

    ----

    .. _ILCBaseNode-en:

    * **English**

    Inter-Layer Connection (ILC) neuron base class. An abstract base class for
    building cross-layer spike-propagation connections.

    At each time step, the ILC neuron performs charge-fire-reset dynamics.
    Unlike standard neurons, it passes the output spike through a learnable 1D
    convolution (:class:`torch.nn.Conv1d`) to the next layer's input, enabling
    structured inter-layer connections.

    Subclasses override :meth:`neuronal_charge` to implement different charging dynamics:

    - :class:`ILCIFNode` — Integrate-and-Fire dynamics
    - :class:`ILCLIFNode` — Leaky Integrate-and-Fire dynamics with membrane decay
    - :class:`ILCCUBALIFNode` — Current-Based LIF dynamics with current decay

    :param act_dim: Feature dimension of the input activation
    :type act_dim: int
    :param dec_pop_dim: Decoding population dimension, number of neurons per feature
    :type dec_pop_dim: int
    :param v_threshold: Threshold voltage. A spike is emitted when membrane potential exceeds this
    :type v_threshold: float
    :param v_reset: Reset voltage. If ``None``, soft reset (subtract threshold); otherwise hard reset
    :type v_reset: Optional[float]
    :param surrogate_function: Surrogate gradient function for approximating the spike function gradient
    :type surrogate_function: surrogate.SurrogateFunctionBase
    """

    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.out_pop_dim = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.out_pop_dim, dec_pop_dim, groups=act_dim)

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <neuronal_charge-cn>` | :ref:`English <neuronal_charge-en>`

        ----

        .. _neuronal_charge-cn:

        * **中文**

        神经元充电动力学方程（抽象方法）。根据输入 ``x`` 更新膜电位 :attr:`self.v`。

        充电是神经元动力学的第一步。对于 IF 和 LIF 神经元，这一步将输入电流
        累积到膜电位上；对于 CUBALIF 神经元，这一步会同时更新突触电流和膜电位。
        具体的充电方式由子类实现。

        :param x: 当前时间步的输入张量
        :type x: torch.Tensor

        ----

        .. _neuronal_charge-en:

        * **English**

        Neuronal charging dynamics (abstract method). Updates membrane potential
        :attr:`self.v` based on the input ``x``.

        This is the first step of the charge-fire-reset cycle. IF and LIF neurons
        accumulate the input current into the membrane potential, while CUBALIF
        updates both synaptic current and membrane potential. The specific charging
        behavior is defined by the subclass implementation.

        :param x: Input tensor for the current time step
        :type x: torch.Tensor
        """

        raise NotImplementedError

    def neuronal_fire(self):
        r"""
        **API Language** - :ref:`中文 <neuronal_fire-cn>` | :ref:`English <neuronal_fire-en>`

        ----

        .. _neuronal_fire-cn:

        * **中文**

        神经元放电函数。根据当前膜电位与阈值的差值生成脉冲。

        通过替代梯度函数 :attr:`self.surrogate_function` 计算脉冲：
        当膜电位 :attr:`self.v` 超过阈值 :attr:`self.v_threshold` 时输出 ``1``，
        否则输出 ``0``。在反向传播时，替代梯度函数会用一个平滑的近似梯度
        替代脉冲函数的不可导部分。

        这是充电-放电-重置循环的第二步。

        :return: 脉冲张量，元素为 0 或 1
        :rtype: torch.Tensor

        ----

        .. _neuronal_fire-en:

        * **English**

        Neuronal fire function. Generates a spike based on the difference
        between the membrane potential and the threshold.

        Uses the surrogate function :attr:`self.surrogate_function` to compute
        the spike: outputs ``1`` when the membrane potential :attr:`self.v`
        exceeds the threshold :attr:`self.v_threshold`, and ``0`` otherwise.
        During backpropagation, the surrogate function provides a smooth
        approximation of the spike function's gradient.

        This is the second step of the charge-fire-reset cycle.

        :return: Spike tensor with elements 0 or 1
        :rtype: torch.Tensor
        """

        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        r"""
        **API Language** - :ref:`中文 <neuronal_reset-cn>` | :ref:`English <neuronal_reset-en>`

        ----

        .. _neuronal_reset-cn:

        * **中文**

        神经元重置函数。在脉冲发放后重置膜电位。

        支持两种重置模式：

        - **软重置** （当 ``v_reset`` 为 ``None`` 时）：膜电位减去阈值，即
          :math:`V = V - S \\cdot V_{th}`
        - **硬重置** （当 ``v_reset`` 为数值时）：膜电位重置为固定值，即
          :math:`V = V_{reset}` 或 :math:`V = (1 - S) \\cdot V + S \\cdot V_{reset}`

        这是充电-放电-重置循环的第三步。

        :param spike: 由 :meth:`neuronal_fire` 生成的脉冲张量
        :type spike: torch.Tensor

        ----

        .. _neuronal_reset-en:

        * **English**

        Neuronal reset function. Resets the membrane potential after spike emission.

        Supports two reset modes:

        - **Soft reset** (when ``v_reset`` is ``None``): subtracts the threshold,
          i.e., :math:`V = V - S \\cdot V_{th}`
        - **Hard reset** (when ``v_reset`` is a float): resets the membrane
          potential to a fixed value, i.e.,
          :math:`V = V_{reset}` or :math:`V = (1 - S) \\cdot V + S \\cdot V_{reset}`

        This is the third step of the charge-fire-reset cycle.

        :param spike: Spike tensor generated by :meth:`neuronal_fire`
        :type spike: torch.Tensor
        """

        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1.0 - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <init_tensor-cn>` | :ref:`English <init_tensor-en>`

        ----

        .. _init_tensor-cn:

        * **中文**

        初始化膜电位张量。根据输入数据 ``data`` 的形状创建初始膜电位。

        将膜电位 :attr:`self.v` 初始化为形状与 ``data`` 相同、所有元素为
        :attr:`self.v_reset` 的张量。在 :meth:`forward` 开始时调用。

        :param data: 用于确定形状的参考张量
        :type data: torch.Tensor

        ----

        .. _init_tensor-en:

        * **English**

        Initialize the membrane potential tensor. Creates the initial membrane
        potential based on the shape of the input ``data``.

        Initializes :attr:`self.v` to a tensor with the same shape as ``data``,
        filled with :attr:`self.v_reset`. Called at the beginning of :meth:`forward`.

        :param data: Reference tensor used to determine shape
        :type data: torch.Tensor
        """

        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <forward-cn>` | :ref:`English <forward-en>`

        ----

        .. _forward-cn:

        * **中文**

        多步前向传播函数。对输入的时间序列逐时间步执行充电-放电-重置循环。

        在每个时间步 :math:`t`：

        1. 调用 :meth:`neuronal_charge` 更新膜电位
        2. 调用 :meth:`neuronal_fire` 生成脉冲（判断是否放电）
        3. 调用 :meth:`neuronal_reset` 重置膜电位
        4. 将当前步输出的脉冲通过可学习卷积连接 :attr:`self.conn` 传递到下一时间步的输入

        :param x_seq: 输入序列，形状 ``[T, N, *]``，其中 ``T`` 为时间步数，``N`` 为 batch 大小
        :type x_seq: torch.Tensor
        :return: 脉冲序列，形状与 ``x_seq`` 相同
        :rtype: torch.Tensor

        ----

        .. _forward-en:

        * **English**

        Multi-step forward function. Performs the charge-fire-reset cycle at
        each time step on the input sequence.

        At each time step :math:`t`:

        1. Call :meth:`neuronal_charge` to update the membrane potential
        2. Call :meth:`neuronal_fire` to generate a spike
        3. Call :meth:`neuronal_reset` to reset the membrane potential
        4. Pass the current output spike through the learnable convolution
           connection :attr:`self.conn` to the next time step's input

        :param x_seq: Input sequence, shape ``[T, N, *]``, where ``T`` is the
            number of time steps and ``N`` is the batch size
        :type x_seq: torch.Tensor
        :return: Spike sequence with the same shape as ``x_seq``
        :rtype: torch.Tensor
        """

        self.init_tensor(x_seq[0].data)

        T = x_seq.shape[0]
        spike_seq = []

        for t in range(T):
            self.neuronal_charge(x_seq[t])
            spike = self.neuronal_fire()
            self.neuronal_reset(spike)
            spike_seq.append(spike)
            if t < T - 1:
                x_seq[t + 1] = x_seq[t + 1] + self.conn(
                    spike.view(-1, self.act_dim, self.dec_pop_dim)
                ).view(-1, self.out_pop_dim)

        return torch.stack(spike_seq)


class ILCIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        """
        **API Language** - :ref:`中文 <ILCIFNode-cn>` | :ref:`English <ILCIFNode-en>`

        ----

        .. _ILCIFNode-cn:

        * **中文**

        ILC-IF 神经元。使用积分发放（Integrate-and-Fire）充电动力学的
        层间连接神经元子类。

        充电公式为 :math:`V[t] = V[t-1] + X[t]`，即直接将输入累加到膜电位上，
        不引入漏电衰减。适合对输入信号进行简单累加的场景。

        ----

        .. _ILCIFNode-en:

        * **English**

        ILC Integrate-and-Fire neuron. An ILC neuron subclass that uses
        Integrate-and-Fire charging dynamics.

        The charging equation is :math:`V[t] = V[t-1] + X[t]`, where the input
        is directly accumulated into the membrane potential without leakage.
        Suitable for simple accumulation of input signals.
        """
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <ILCIFNode.neuronal_charge-cn>` | :ref:`English <ILCIFNode.neuronal_charge-en>`

        ----

        .. _ILCIFNode.neuronal_charge-cn:

        * **中文**

        IF 充电动力学。直接将输入 ``x`` 累加到膜电位上。

        实现公式 :math:`V = V + x`。无漏电项，输入完全累积到膜电位中。

        :param x: 当前时间步的输入
        :type x: torch.Tensor

        ----

        .. _ILCIFNode.neuronal_charge-en:

        * **English**

        IF charging dynamics. Directly accumulates the input ``x`` into the
        membrane potential.

        Implements :math:`V = V + x`. No leakage — the input is fully
        accumulated into the membrane potential.

        :param x: Input at the current time step
        :type x: torch.Tensor
        """

        self.v = self.v + x


class ILCLIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        v_decay: float = 0.75,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        """
        **API Language** - :ref:`中文 <ILCLIFNode-cn>` | :ref:`English <ILCLIFNode-en>`

        ----

        .. _ILCLIFNode-cn:

        * **中文**

        ILC-LIF 神经元。使用漏电积分发放（Leaky Integrate-and-Fire）充电动力学的
        层间连接神经元子类。

        充电公式为 :math:`V[t] = V[t-1] \\cdot v_{decay} + X[t]`，其中 :math:`v_{decay}`
        是膜电位衰减系数（默认 0.75）。每一步膜电位会先按比例衰减，再累加输入。
        适合需要短期记忆的场景。

        ----

        .. _ILCLIFNode-en:

        * **English**

        ILC Leaky Integrate-and-Fire neuron. An ILC neuron subclass that uses
        Leaky Integrate-and-Fire charging dynamics.

        The charging equation is :math:`V[t] = V[t-1] \\cdot v_{decay} + X[t]`,
        where :math:`v_{decay}` is the membrane decay factor (default 0.75).
        The membrane potential decays proportionally at each step before
        accumulating the input. Suitable for tasks requiring short-term memory.
        """
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <ILCLIFNode.neuronal_charge-cn>` | :ref:`English <ILCLIFNode.neuronal_charge-en>`

        ----

        .. _ILCLIFNode.neuronal_charge-cn:

        * **中文**

        LIF 充电动力学。膜电位先按因子衰减，再累加输入。

        实现公式 :math:`V = V \\cdot v_{decay} + x`。其中 ``v_decay`` 控制
        膜电位在每一步的保留比例，值越接近 1 衰减越慢，记忆越长。

        :param x: 当前时间步的输入
        :type x: torch.Tensor

        ----

        .. _ILCLIFNode.neuronal_charge-en:

        * **English**

        LIF charging dynamics. The membrane potential decays by a factor
        before accumulating the input.

        Implements :math:`V = V \\cdot v_{decay} + x`. The ``v_decay``
        parameter controls how much of the membrane potential is retained
        at each step — values closer to 1 mean slower decay and longer memory.

        :param x: Input at the current time step
        :type x: torch.Tensor
        """

        self.v = self.v * self.v_decay + x


class ILCCUBALIFNode(ILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Rect(),
    ):
        """
        **API Language** - :ref:`中文 <ILCCUBALIFNode-cn>` | :ref:`English <ILCCUBALIFNode-en>`

        ----

        .. _ILCCUBALIFNode-cn:

        * **中文**

        ILC-CUBALIF 神经元。使用基于电流的 CUBALIF（Current-Based LIF）充电动力学的
        层间连接神经元子类。

        充电过程分为两步：

        1. 突触电流衰减：:math:`C[t] = C[t-1] \\cdot c_{decay} + X[t]`
        2. 膜电位衰减并累加电流：:math:`V[t] = V[t-1] \\cdot v_{decay} + C[t]`

        与 ILCLIFNode 相比，此处多了一层电流衰减，使神经元能对输入的时间模式
        进行更丰富的建模。``c_decay`` 控制电流衰减速度，``v_decay`` 控制膜电位衰减速度。

        ----

        .. _ILCCUBALIFNode-en:

        * **English**

        ILC Current-Based LIF neuron. An ILC neuron subclass that uses
        Current-Based LIF (CUBALIF) charging dynamics.

        The charging process has two steps:

        1. Synaptic current decay: :math:`C[t] = C[t-1] \\cdot c_{decay} + X[t]`
        2. Membrane potential decay with current accumulation:
           :math:`V[t] = V[t-1] \\cdot v_{decay} + C[t]`

        Compared to ILCLIFNode, the additional current decay layer enables richer
        temporal modeling. ``c_decay`` controls the current decay rate, while
        ``v_decay`` controls the membrane potential decay rate.
        """
        super().__init__(act_dim, dec_pop_dim, v_threshold, v_reset, surrogate_function)

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <ILCCUBALIFNode.neuronal_charge-cn>` | :ref:`English <ILCCUBALIFNode.neuronal_charge-en>`

        ----

        .. _ILCCUBALIFNode.neuronal_charge-cn:

        * **中文**

        CUBALIF 充电动力学。先更新突触电流，再更新膜电位。

        实现两步更新：

        - 电流衰减：:math:`C = C \\cdot c_{decay} + x`
        - 膜电位更新：:math:`V = V \\cdot v_{decay} + C`

        :param x: 当前时间步的输入
        :type x: torch.Tensor

        ----

        .. _ILCCUBALIFNode.neuronal_charge-en:

        * **English**

        CUBALIF charging dynamics. First updates the synaptic current, then
        updates the membrane potential.

        Two-step update:

        - Current decay: :math:`C = C \\cdot c_{decay} + x`
        - Membrane update: :math:`V = V \\cdot v_{decay} + C`

        :param x: Input at the current time step
        :type x: torch.Tensor
        """

        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        r"""
        **API Language** - :ref:`中文 <ILCCUBALIFNode.init_tensor-cn>` | :ref:`English <ILCCUBALIFNode.init_tensor-en>`

        ----

        .. _ILCCUBALIFNode.init_tensor-cn:

        * **中文**

        初始化膜电位和突触电流张量。

        与基类的 :meth:`ILCBaseNode.init_tensor` 不同，此处还需要额外初始化
        突触电流 :attr:`self.c` 为全零张量。这是因为 CUBALIF 维护了独立的
        电流状态，而不仅仅是膜电位。

        :param data: 用于确定形状的参考张量
        :type data: torch.Tensor

        ----

        .. _ILCCUBALIFNode.init_tensor-en:

        * **English**

        Initialize the membrane potential and synaptic current tensors.

        Unlike the base class :meth:`ILCBaseNode.init_tensor`, this also
        initializes the synaptic current :attr:`self.c` to zero, because
        CUBALIF maintains an independent current state in addition to the
        membrane potential.

        :param data: Reference tensor used to determine shape
        :type data: torch.Tensor
        """

        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)
