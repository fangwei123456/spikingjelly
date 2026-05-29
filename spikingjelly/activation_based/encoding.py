import torch
import torch.nn as nn
import torch.nn.functional as F
from . import functional
import math
from . import base, neuron, surrogate
from abc import abstractmethod


class StatelessEncoder(nn.Module, base.StepModule):
    def __init__(self, step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <StatelessEncoder.__init__-cn>` | :ref:`English <StatelessEncoder.__init__-en>`

        ----

        .. _StatelessEncoder.__init__-cn:

        * **中文**

        无状态编码器的基类。
        无状态编码器 ``encoder = StatelessEncoder()``，直接调用 ``encoder(x)`` 即可将 ``x`` 编码为 ``spike`` 。

        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _StatelessEncoder.__init__-en:

        * **English**

        The base class of stateless encoder.
        The stateless encoder ``encoder = StatelessEncoder()`` can encode ``x`` to
        ``spike`` by ``encoder(x)``.

        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        """
        super().__init__()
        self.step_mode = step_mode

    @abstractmethod
    def forward(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <StatelessEncoder.forward-cn>` | :ref:`English <StatelessEncoder.forward-en>`

        ----

        .. _StatelessEncoder.forward-cn:

        * **中文**

        编码函数，将输入 ``x`` 编码为脉冲 ``spike`` 。

        :param x: 输入数据
        :type x: torch.Tensor

        :return: 脉冲张量，形状与 ``x.shape`` 相同
        :rtype: torch.Tensor

        ----

        .. _StatelessEncoder.forward-en:

        * **English**

        Encode function that converts input ``x`` to spike ``spike`` .

        :param x: input data
        :type x: torch.Tensor

        :return: spike, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class StatefulEncoder(base.MemoryModule):
    def __init__(self, T: int, step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <StatefulEncoder.__init__-cn>` | :ref:`English <StatefulEncoder.__init__-en>`

        ----

        .. _StatefulEncoder.__init__-cn:

        * **中文**

        有状态编码器的基类。
        有状态编码器 ``encoder = StatefulEncoder(T)`` ，编码器会在首次调用 ``encoder(x)`` 时对 ``x`` 进行编码。
        在第 ``t`` 次调用 ``encoder(x)`` 时会输出 ``spike[t % T]`` 。

        :param T: 编码周期。通常情况下，与SNN的仿真周期（总步长一致）
        :type T: int

        ----

        .. _StatefulEncoder.__init__-en:

        * **English**

        The base class of stateful encoder.
        The stateful encoder ``encoder = StatefulEncoder(T)`` will encode ``x`` to
        ``spike`` at the first time of calling ``encoder(x)``.
        It will output ``spike[t % T]``  at the ``t`` -th calling

        :param T: the encoding period. It is usually same with the total simulation time-steps of SNN
        :type T: int

        ----

        * **代码示例 | Example**

        .. code-block:: python

            encoder = StatefulEncoder(T)
            s_list = []
            for t in range(T):
                s_list.append(encoder(x))  # s_list[t] == spike[t]
        """
        super().__init__()
        self.step_mode = step_mode
        assert isinstance(T, int) and T >= 1
        self.T = T
        self.register_memory("spike", None)
        self.register_memory("t", 0)

    def single_step_forward(self, x: torch.Tensor = None):
        r"""
        **API Language:**
        :ref:`中文 <StatefulEncoder.forward-cn>` | :ref:`English <StatefulEncoder.forward-en>`

        ----

        .. _StatefulEncoder.forward-cn:

        * **中文**

        编码函数，返回第 ``t`` 次调用的编码结果 ``spike[t % T]`` 。

        :param x: 输入数据
        :type x: torch.Tensor

        :return: 脉冲，shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        ----

        .. _StatefulEncoder.forward-en:

        * **English**

        Encode function that returns the encoding result at the ``t`` -th call, ``spike[t % T]``.

        :param x: input data
        :type x: torch.Tensor

        :return: spike, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """

        if self.spike is None:
            self.single_step_encode(x)

        t = self.t
        self.t += 1
        if self.t >= self.T:
            self.t = 0
        return self.spike[t]

    @abstractmethod
    def single_step_encode(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <StatefulEncoder.single_step_encode-cn>` | :ref:`English <StatefulEncoder.single_step_encode-en>`

        ----

        .. _StatefulEncoder.single_step_encode-cn:

        * **中文**

        :param x: 输入数据
        :type x: torch.Tensor

        :return: None
        :rtype: None

        ----

        .. _StatefulEncoder.single_step_encode-en:

        * **English**

        :param x: input data
        :type x: torch.Tensor

        :return: None
        :rtype: None
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        r"""
        **API Language:**
        :ref:`中文 <StatefulEncoder.extra_repr-cn>` | :ref:`English <StatefulEncoder.extra_repr-en>`

        ----

        .. _StatefulEncoder.extra_repr-cn:

        * **中文**

        返回编码器的额外表示信息，即编码周期 ``T``。

        :return: 表示字符串 ``"T=T"``
        :rtype: str

        ----

        .. _StatefulEncoder.extra_repr-en:

        * **English**

        Returns the extra representation string of the encoder, i.e., the encoding period ``T``.

        :return: the representation string ``"T=T"``
        :rtype: str
        """
        return f"T={self.T}"


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: torch.Tensor, step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <PeriodicEncoder.__init__-cn>` | :ref:`English <PeriodicEncoder.__init__-en>`

        ----

        .. _PeriodicEncoder.__init__-cn:

        * **中文**

        周期性编码器，在第 ``t`` 次调用时输出 ``spike[t % T]``，其中 ``T = spike.shape[0]``

        .. warning::

            不要忘记调用 :meth:`reset` ，因为这个编码器是有状态的。

        :param spike: 输入脉冲
        :type spike: torch.Tensor

        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _PeriodicEncoder.__init__-en:

        * **English**

        The periodic encoder that outputs ``spike[t % T]`` at ``t`` -th calling, where ``T = spike.shape[0]``

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        :param spike: the input spike
        :type spike: torch.Tensor

        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        """
        super().__init__(spike.shape[0], step_mode)

    def single_step_encode(self, spike: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <PeriodicEncoder.single_step_encode-cn>` | :ref:`English <PeriodicEncoder.single_step_encode-en>`

        ----

        .. _PeriodicEncoder.single_step_encode-cn:

        * **中文**

        设置周期性编码器的脉冲序列。将输入的 ``spike`` 作为编码器的脉冲序列，并将编码周期 ``T`` 设置为 ``spike.shape[0]``。

        :param spike: 脉冲张量，其第0维为编码周期
        :type spike: torch.Tensor
        :return: None
        :rtype: None

        ----

        .. _PeriodicEncoder.single_step_encode-en:

        * **English**

        Sets the spike sequence for the periodic encoder. The input ``spike`` is used as the encoder's
        spike sequence, and the encoding period ``T`` is set to ``spike.shape[0]``.

        :param spike: the spike tensor, whose 0-th dimension is the encoding period
        :type spike: torch.Tensor
        :return: None
        :rtype: None
        """
        self.spike = spike
        self.T = spike.shape[0]


class LatencyEncoder(StatefulEncoder):
    def __init__(self, T: int, enc_function="linear", step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <LatencyEncoder.__init__-cn>` | :ref:`English <LatencyEncoder.__init__-en>`

        ----

        .. _LatencyEncoder.__init__-cn:

        * **中文**

        延迟编码器，将 ``0 <= x <= 1`` 的输入转化为在 ``0 <= t_f <= T-1`` 时刻发放的脉冲。输入的强度越大，发放越早。

        当 ``enc_function == 'linear'``:
            .. math::
                t_f(x) = (T - 1)(1 - x)

        当 ``enc_function == 'log'``:
            .. math::
                t_f(x) = (T - 1) - \text{ln}(\alpha * x + 1)

        其中 :math:`\alpha` 满足 :math:`t_f(1) = T - 1`

        .. warning::

            必须确保 ``0 <= x <= 1``。

        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。

        :param T: 最大（最晚）脉冲发放时刻
        :type T: int

        :param enc_function: 定义使用哪个函数将输入强度转化为脉冲发放时刻，可以为 `linear` 或 `log`
        :type enc_function: str

        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _LatencyEncoder.__init__-en:

        * **English**

        The latency encoder will encode ``0 <= x <= 1`` to spike whose firing time is ``0 <= t_f <= T-1``. A larger
        ``x`` will cause a earlier firing time.

        If ``enc_function == 'linear'``:
            .. math::
                t_f(x) = (T - 1)(1 - x)

        If ``enc_function == 'log'``:
            .. math::
                t_f(x) = (T - 1) - \text{ln}(\alpha * x + 1)

        where :math:`\alpha` satisfies :math:`t_f(1) = T - 1`

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        :param T: the maximum (latest) firing time
        :type T: int

        :param enc_function: how to convert intensity to firing time. `linear` or `log`
        :type enc_function: str

        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str

        ----

        * **代码示例 | Example**

        .. code-block:: python

            x = torch.rand(size=[8, 2])
            print("x", x)
            T = 20
            encoder = LatencyEncoder(T)
            for t in range(T):
                print(encoder(x))
        """
        super().__init__(T, step_mode)
        if enc_function == "log":
            self.alpha = math.exp(T - 1.0) - 1.0
        elif enc_function != "linear":
            raise NotImplementedError

        self.enc_function = enc_function

    def single_step_encode(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <LatencyEncoder.single_step_encode-cn>` | :ref:`English <LatencyEncoder.single_step_encode-en>`

        ----

        .. _LatencyEncoder.single_step_encode-cn:

        * **中文**

        单步编码函数。根据输入的 ``x`` 计算脉冲发放时刻 :math:`t_f`，并生成对应的 one-hot 脉冲张量。

        :param x: 输入数据，取值范围应为 ``[0, 1]``
        :type x: torch.Tensor
        :return: None
        :rtype: None

        ----

        .. _LatencyEncoder.single_step_encode-en:

        * **English**

        Single-step encoding function. Computes the firing time :math:`t_f` from input ``x`` and
        generates the corresponding one-hot spike tensor.

        :param x: input data, which should be in the range ``[0, 1]``
        :type x: torch.Tensor
        :return: None
        :rtype: None
        """
        if self.enc_function == "log":
            t_f = (self.T - 1.0 - torch.log(self.alpha * x + 1.0)).round().long()
        else:
            t_f = ((self.T - 1.0) * (1.0 - x)).round().long()

        self.spike = F.one_hot(t_f, num_classes=self.T).to(x)
        # [*, T] -> [T, *]
        d_seq = list(range(self.spike.ndim - 1))
        d_seq.insert(0, self.spike.ndim - 1)
        self.spike = self.spike.permute(d_seq)


class PoissonEncoder(StatelessEncoder):
    def __init__(self, step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <PoissonEncoder.__init__-cn>` | :ref:`English <PoissonEncoder.__init__-en>`

        ----

        .. _PoissonEncoder.__init__-cn:

        * **中文**

        无状态的泊松编码器。输出脉冲的发放概率与输入 ``x`` 相同。

        .. warning::

            必须确保 ``0 <= x <= 1``。

        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _PoissonEncoder.__init__-en:

        * **English**

        The poisson encoder will output spike whose firing probability is ``x``。

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.

        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        """
        super().__init__(step_mode)

    def forward(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <PoissonEncoder.forward-cn>` | :ref:`English <PoissonEncoder.forward-en>`

        ----

        .. _PoissonEncoder.forward-cn:

        * **中文**

        前向传播。根据输入 ``x`` 中的概率值生成泊松脉冲序列。每个元素以概率 ``x`` 独立地发放脉冲。

        :param x: 输入数据，取值范围应为 ``[0, 1]``，表示脉冲发放概率
        :type x: torch.Tensor
        :return: 脉冲张量，形状与 ``x`` 相同
        :rtype: torch.Tensor

        ----

        .. _PoissonEncoder.forward-en:

        * **English**

        Forward pass. Generates Poisson spike trains based on the probability values in ``x``.
        Each element independently fires a spike with probability ``x``.

        :param x: input data in the range ``[0, 1]``, representing firing probabilities
        :type x: torch.Tensor
        :return: spike tensor with the same shape as ``x``
        :rtype: torch.Tensor
        """
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike


class WeightedPhaseEncoder(StatefulEncoder):
    def __init__(self, K: int, step_mode="s"):
        r"""
        **API Language:**
        :ref:`中文 <WeightedPhaseEncoder.__init__-cn>` | :ref:`English <WeightedPhaseEncoder.__init__-en>`

        ----

        .. _WeightedPhaseEncoder.__init__-cn:

        * **中文**

        带权的相位编码，一种基于二进制表示的编码方法。

        将输入按照二进制各位展开，从高位到低位遍历输入进行脉冲编码。相比于频率编码，每一位携带的信息量更多。编码相位数为 :math:`K` 时，
        可以对于处于区间 :math:`[0, 1-2^{-K}]` 的数进行编码。以下为原始论文中的示例：

        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | Phase (K=8)                      | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              |
        +==================================+================+================+================+================+================+================+================+================+
        | Spike weight :math:`\omega(t)`   | 2\ :sup:`-1`   | 2\ :sup:`-2`   | 2\ :sup:`-3`   | 2\ :sup:`-4`   | 2\ :sup:`-5`   | 2\ :sup:`-6`   | 2\ :sup:`-7`   | 2\ :sup:`-8`   |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 192/256                          | 1              | 1              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 1/256                            | 0              | 0              | 0              | 0              | 0              | 0              | 0              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 128/256                          | 1              | 0              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 255/256                          | 1              | 1              | 1              | 1              | 1              | 1              | 1              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+

        参考文献：
        Kim J, Kim H, Huh S, et al. Deep neural networks with weighted spikes[J]. Neurocomputing, 2018, 311: 373-386.

        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。

        :param K: 编码周期。通常情况下，与SNN的仿真周期（总步长一致）
        :type K: int

        :param step_mode: 步进模式，可以为 ``'s'`` (单步) 或 ``'m'`` (多步)
        :type step_mode: str

        ----

        .. _WeightedPhaseEncoder.__init__-en:

        * **English**

        The weighted phase encoder, which is based on binary system. It will flatten ``x`` as a binary number. When
        ``T=K``, it can encode :math:`x \in [0, 1-2^{-K}]` to different spikes. Here is the example from the origin paper:

        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | Phase (K=8)                      | 1              | 2              | 3              | 4              | 5              | 6              | 7              | 8              |
        +==================================+================+================+================+================+================+================+================+================+
        | Spike weight :math:`\omega(t)`   | 2\ :sup:`-1`   | 2\ :sup:`-2`   | 2\ :sup:`-3`   | 2\ :sup:`-4`   | 2\ :sup:`-5`   | 2\ :sup:`-6`   | 2\ :sup:`-7`   | 2\ :sup:`-8`   |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 192/256                          | 1              | 1              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 1/256                            | 0              | 0              | 0              | 0              | 0              | 0              | 0              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 128/256                          | 1              | 0              | 0              | 0              | 0              | 0              | 0              | 0              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+
        | 255/256                          | 1              | 1              | 1              | 1              | 1              | 1              | 1              | 1              |
        +----------------------------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+----------------+

        Reference:
        Kim J, Kim H, Huh S, et al. Deep neural networks with weighted spikes[J]. Neurocomputing, 2018, 311: 373-386.

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        :param K: the encoding period. It is usually same with the total simulation time-steps of SNN
        :type K: int

        :param step_mode: the step mode, which can be ``'s'`` (single-step) or ``'m'`` (multi-step)
        :type step_mode: str
        """
        super().__init__(K, step_mode)

    def single_step_encode(self, x: torch.Tensor):
        r"""
        **API Language:**
        :ref:`中文 <WeightedPhaseEncoder.single_step_encode-cn>` | :ref:`English <WeightedPhaseEncoder.single_step_encode-en>`

        ----

        .. _WeightedPhaseEncoder.single_step_encode-cn:

        * **中文**

        单步编码函数。将输入 ``x`` 按照二进制加权相位编码方式展开为脉冲序列。

        :param x: 输入数据，取值范围应为 ``[0, 1 - 2^{-T}]``
        :type x: torch.Tensor
        :raises AssertionError: 当 ``x`` 不在 ``[0, 1 - 2^{-T}]`` 范围内时

        ----

        .. _WeightedPhaseEncoder.single_step_encode-en:

        * **English**

        Single-step encoding function. Encodes the input ``x`` into spike trains using the
        weighted phase encoding scheme based on binary decomposition.

        :param x: input data, which should be in the range ``[0, 1 - 2^{-T}]``
        :type x: torch.Tensor
        :raises AssertionError: if ``x`` is not in the range ``[0, 1 - 2^{-T}]``
        """
        assert (x >= 0).all() and (x <= 1 - 2 ** (-self.T)).all()
        inputs = x.clone()
        self.spike = torch.empty(
            (self.T,) + x.shape, device=x.device
        )  # Encoding to [T, batch_size, *]
        w = 0.5
        for i in range(self.T):
            self.spike[i] = inputs >= w
            inputs -= w * self.spike[i]
            w *= 0.5


class PopSpikeEncoderDeterministic(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        r"""
        **API Language:**
        :ref:`中文 <PopSpikeEncoderDeterministic.__init__-cn>` | :ref:`English <PopSpikeEncoderDeterministic.__init__-en>`

        ----

        .. _PopSpikeEncoderDeterministic.__init__-cn:

        * **中文**

        可学习的群体编码脉冲编码器，使用确定性脉冲序列。编码器使用高斯函数作为感受野，将输入观测映射到群体输出。

        :param obs_dim: 观测空间的维度
        :type obs_dim: int

        :param pop_dim: 每个观测维度的编码器数量
        :type pop_dim: int

        :param spike_ts: 脉冲时间步数
        :type spike_ts: int

        :param mean_range: 均值范围 [min, max]
        :type mean_range: tuple

        :param std: 标准差
        :type std: float

        ----

        .. _PopSpikeEncoderDeterministic.__init__-en:

        * **English**

        Learnable population coding spike encoder with deterministic spike trains. The encoder uses Gaussian functions
        as receptive fields to map input observations to population outputs.

        :param obs_dim: dimension of observation space
        :type obs_dim: int

        :param pop_dim: number of encoders per observation dimension
        :type pop_dim: int

        :param spike_ts: number of spike time steps
        :type spike_ts: int

        :param mean_range: mean range [min, max]
        :type mean_range: tuple

        :param std: standard deviation
        :type std: float
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std

        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

        self.neurons = neuron.IFNode(
            v_threshold=0.999,
            v_reset=None,
            surrogate_function=surrogate.DeterministicPass(),
            detach_reset=True,
        )

        functional.set_step_mode(self, step_mode="m")
        functional.set_backend(self, backend="torch")

    def forward(self, obs):
        r"""
        **API Language:**
        :ref:`中文 <PopSpikeEncoderDeterministic.forward-cn>` | :ref:`English <PopSpikeEncoderDeterministic.forward-en>`

        ----

        .. _PopSpikeEncoderDeterministic.forward-cn:

        * **中文**

        前向传播。将输入观测映射为高斯群体激活，并在 ``spike_ts`` 个时间步上重复后输入脉冲神经元，输出确定性脉冲序列。

        :param obs: 输入观测张量，最后一维大小应与 ``obs_dim`` 一致
        :type obs: torch.Tensor
        :return: 编码后的脉冲序列，形状为 ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor

        ----

        .. _PopSpikeEncoderDeterministic.forward-en:

        * **English**

        Forward pass. The input observations are mapped to Gaussian population activations, repeated for
        ``spike_ts`` time steps, and fed into spiking neurons to produce deterministic spike trains.

        :param obs: input observation tensor whose last dimension should match ``obs_dim``
        :type obs: torch.Tensor
        :return: encoded spike sequence with shape ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor
        """
        obs = obs.view(-1, self.obs_dim, 1)

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(
            -(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)
        ).view(-1, self.encoder_neuron_num)
        pop_act = pop_act.unsqueeze(0).repeat(self.spike_ts, 1, 1)

        return self.neurons(pop_act)


class PopSpikeEncoderRandom(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        r"""
        **API Language:**
        :ref:`中文 <PopSpikeEncoderRandom.__init__-cn>` | :ref:`English <PopSpikeEncoderRandom.__init__-en>`

        ----

        .. _PopSpikeEncoderRandom.__init__-cn:

        * **中文**

        可学习的群体编码脉冲编码器，使用随机脉冲序列。编码器使用高斯函数作为感受野，将输入观测映射到群体输出。

        :param obs_dim: 观测空间的维度
        :type obs_dim: int

        :param pop_dim: 每个观测维度的编码器数量
        :type pop_dim: int

        :param spike_ts: 脉冲时间步数
        :type spike_ts: int

        :param mean_range: 均值范围 [min, max]
        :type mean_range: tuple

        :param std: 标准差
        :type std: float

        ----

        .. _PopSpikeEncoderRandom.__init__-en:

        * **English**

        Learnable population coding spike encoder with random spike trains. The encoder uses Gaussian functions
        as receptive fields to map input observations to population outputs.

        :param obs_dim: dimension of observation space
        :type obs_dim: int

        :param pop_dim: number of encoders per observation dimension
        :type pop_dim: int

        :param spike_ts: number of spike time steps
        :type spike_ts: int

        :param mean_range: mean range [min, max]
        :type mean_range: tuple

        :param std: standard deviation
        :type std: float
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std

        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

        self.pseudo_spike = surrogate.poisson_pass.apply

    def forward(self, obs):
        r"""
        **API Language:**
        :ref:`中文 <PopSpikeEncoderRandom.forward-cn>` | :ref:`English <PopSpikeEncoderRandom.forward-en>`

        ----

        .. _PopSpikeEncoderRandom.forward-cn:

        * **中文**

        前向传播。将输入观测映射为高斯群体激活，并在每个时间步通过随机采样生成脉冲序列。

        :param obs: 输入观测张量，最后一维大小应与 ``obs_dim`` 一致
        :type obs: torch.Tensor
        :return: 随机脉冲序列，形状为 ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor

        ----

        .. _PopSpikeEncoderRandom.forward-en:

        * **English**

        Forward pass. The input observations are mapped to Gaussian population activations, and random spikes
        are sampled at each time step.

        :param obs: input observation tensor whose last dimension should match ``obs_dim``
        :type obs: torch.Tensor
        :return: random spike sequence with shape ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor
        """
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(
            -(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)
        ).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(
            self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device
        )

        # Generate Random Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[step, :, :] = self.pseudo_spike(pop_act)

        return pop_spikes


class PopEncoder(nn.Module):
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
        r"""
        **API Language:**
        :ref:`中文 <PopEncoder.__init__-cn>` | :ref:`English <PopEncoder.__init__-en>`

        ----

        .. _PopEncoder.__init__-cn:

        * **中文**

        可学习的群体编码器，输出编码后的脉冲输入序列，用于SNN训练。

        :param obs_dim: 观测空间的维度
        :type obs_dim: int

        :param pop_dim: 每个观测维度的编码器数量
        :type pop_dim: int

        :param spike_ts: 脉冲时间步数
        :type spike_ts: int

        :param mean_range: 均值范围 [min, max]
        :type mean_range: tuple

        :param std: 标准差
        :type std: float

        ----

        .. _PopEncoder.__init__-en:

        * **English**

        Learnable population coding encoder that outputs encoded spike input sequences for SNN training.

        :param obs_dim: dimension of observation space
        :type obs_dim: int

        :param pop_dim: number of encoders per observation dimension
        :type pop_dim: int

        :param spike_ts: number of spike time steps
        :type spike_ts: int

        :param mean_range: mean range [min, max]
        :type mean_range: tuple

        :param std: standard deviation
        :type std: float
        """
        super().__init__()
        self.obs_dim = obs_dim
        self.pop_dim = pop_dim
        self.encoder_neuron_num = obs_dim * pop_dim
        self.spike_ts = spike_ts

        # Compute evenly distributed mean and variance
        tmp_mean = torch.zeros(1, obs_dim, pop_dim)
        delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
        for num in range(pop_dim):
            tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
        tmp_std = torch.zeros(1, obs_dim, pop_dim) + std

        self.mean = nn.Parameter(tmp_mean)
        self.std = nn.Parameter(tmp_std)

    def forward(self, obs):
        r"""
        **API Language:**
        :ref:`中文 <PopEncoder.forward-cn>` | :ref:`English <PopEncoder.forward-en>`

        ----

        .. _PopEncoder.forward-cn:

        * **中文**

        前向传播。将输入观测映射为高斯群体激活，并在 ``spike_ts`` 个时间步上复制该激活，返回连续输入序列。

        :param obs: 输入观测张量，最后一维大小应与 ``obs_dim`` 一致
        :type obs: torch.Tensor
        :return: 编码输入序列，形状为 ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor

        ----

        .. _PopEncoder.forward-en:

        * **English**

        Forward pass. The input observations are mapped to Gaussian population activations and repeated for
        ``spike_ts`` time steps as a continuous input sequence.

        :param obs: input observation tensor whose last dimension should match ``obs_dim``
        :type obs: torch.Tensor
        :return: encoded input sequence with shape ``[spike_ts, batch_size, obs_dim * pop_dim]``
        :rtype: torch.Tensor
        """
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(
            -(1.0 / 2.0) * (obs - self.mean).pow(2) / self.std.pow(2)
        ).view(-1, self.encoder_neuron_num)
        pop_inputs = torch.zeros(
            self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device
        )

        # Generate Input Trains
        for step in range(self.spike_ts):
            pop_inputs[step, :, :] = pop_act

        return pop_inputs
