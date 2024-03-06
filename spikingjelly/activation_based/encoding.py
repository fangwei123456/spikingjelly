import torch
import torch.nn as nn
import torch.nn.functional as F
from . import functional
import math
from . import base, neuron, surrogate
from abc import abstractmethod


class StatelessEncoder(nn.Module, base.StepModule):
    def __init__(self, step_mode='s'):
        """
        * :ref:`API in English <StatelessEncoder.__init__-en>`

        .. _StatelessEncoder.__init__-cn:

        无状态编码器的基类。无状态编码器 ``encoder = StatelessEncoder()``，直接调用 ``encoder(x)`` 即可将 ``x`` 编码为 ``spike``。

        * :ref:`中文API <StatelessEncoder.__init__-cn>`

        .. _StatelessEncoder.__init__-en:

        The base class of stateless encoder. The stateless encoder ``encoder = StatelessEncoder()`` can encode ``x`` to
        ``spike`` by ``encoder(x)``.

        """
        super().__init__()
        self.step_mode = step_mode

    @abstractmethod
    def forward(self, x: torch.Tensor):
        """
        * :ref:`API in English <StatelessEncoder.forward-en>`

        .. _StatelessEncoder.forward-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatelessEncoder.forward-cn>`

        .. _StatelessEncoder.forward-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """
        raise NotImplementedError


class StatefulEncoder(base.MemoryModule):
    def __init__(self, T: int, step_mode='s'):
        """
        * :ref:`API in English <StatefulEncoder.__init__-en>`

        .. _StatefulEncoder.__init__-cn:

        :param T: 编码周期。通常情况下，与SNN的仿真周期（总步长一致）
        :type T: int

        有状态编码器的基类。有状态编码器 ``encoder = StatefulEncoder(T)``，编码器会在首次调用 ``encoder(x)`` 时对 ``x`` 进行编码。在第 ``t`` 次调用 ``encoder(x)`` 时会输出 ``spike[t % T]``

        .. code-block:: python

            encoder = StatefulEncoder(T)
            s_list = []
            for t in range(T):
                s_list.append(encoder(x))  # s_list[t] == spike[t]

        * :ref:`中文API <StatefulEncoder.__init__-cn>`

        .. _StatefulEncoder.__init__-en:

        :param T: the encoding period. It is usually same with the total simulation time-steps of SNN
        :type T: int

        The base class of stateful encoder. The stateful encoder ``encoder = StatefulEncoder(T)`` will encode ``x`` to
        ``spike`` at the first time of calling ``encoder(x)``. It will output ``spike[t % T]``  at the ``t`` -th calling

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
        self.register_memory('spike', None)
        self.register_memory('t', 0)


    def single_step_forward(self, x: torch.Tensor = None):
        """
        * :ref:`API in English <StatefulEncoder.forward-en>`

        .. _StatefulEncoder.forward-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatefulEncoder.forward-cn>`

        .. _StatefulEncoder.forward-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
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
        """
        * :ref:`API in English <StatefulEncoder.single_step_encode-en>`

        .. _StatefulEncoder.single_step_encode-cn:

        :param x: 输入数据
        :type x: torch.Tensor
        :return: ``spike``, shape 与 ``x.shape`` 相同
        :rtype: torch.Tensor

        * :ref:`中文API <StatefulEncoder.single_step_encode-cn>`

        .. _StatefulEncoder.single_step_encode-en:

        :param x: input data
        :type x: torch.Tensor
        :return: ``spike``, whose shape is same with ``x.shape``
        :rtype: torch.Tensor
        """
        raise NotImplementedError

    def extra_repr(self) -> str:
        return f'T={self.T}'


class PeriodicEncoder(StatefulEncoder):
    def __init__(self, spike: torch.Tensor, step_mode='s'):
        """
        * :ref:`API in English <PeriodicEncoder.__init__-en>`

        .. _PeriodicEncoder.__init__-cn:

        :param spike: 输入脉冲
        :type spike: torch.Tensor

        周期性编码器，在第 ``t`` 次调用时输出 ``spike[t % T]``，其中 ``T = spike.shape[0]``


        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。


        * :ref:`中文API <PeriodicEncoder.__init__-cn>`

        .. _PeriodicEncoder.__init__-en:

        :param spike: the input spike
        :type spike: torch.Tensor

        The periodic encoder that outputs ``spike[t % T]`` at ``t`` -th calling, where ``T = spike.shape[0]``

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        """
        super().__init__(spike.shape[0], step_mode)

    def single_step_encode(self, spike: torch.Tensor):
        self.spike = spike
        self.T = spike.shape[0]


class LatencyEncoder(StatefulEncoder):
    def __init__(self, T: int, enc_function='linear', step_mode='s'):
        """
        * :ref:`API in English <LatencyEncoder.__init__-en>`

        .. _LatencyEncoder.__init__-cn:

        :param T: 最大（最晚）脉冲发放时刻
        :type T: int
        :param enc_function: 定义使用哪个函数将输入强度转化为脉冲发放时刻，可以为 `linear` 或 `log`
        :type enc_function: str

        延迟编码器，将 ``0 <= x <= 1`` 的输入转化为在 ``0 <= t_f <= T-1`` 时刻发放的脉冲。输入的强度越大，发放越早。
        当 ``enc_function == 'linear'``
            .. math::
                t_f(x) = (T - 1)(1 - x)

        当 ``enc_function == 'log'``
            .. math::
                t_f(x) = (T - 1) - ln(\\alpha * x + 1)

        其中 :math:`\alpha` 满足 :math:`t_f(1) = T - 1`


        实例代码：

        .. code-block:: python

            x = torch.rand(size=[8, 2])
            print('x', x)
            T = 20
            encoder = LatencyEncoder(T)
            for t om range(T):
                print(encoder(x))

        .. warning::

            必须确保 ``0 <= x <= 1``。

        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。


        * :ref:`中文API <LatencyEncoder.__init__-cn>`

        .. _LatencyEncoder.__init__-en:

        :param T: the maximum (latest) firing time
        :type T: int
        :param enc_function: how to convert intensity to firing time. `linear` or `log`
        :type enc_function: str

        The latency encoder will encode ``0 <= x <= 1`` to spike whose firing time is ``0 <= t_f <= T-1``. A larger
        ``x`` will cause a earlier firing time.

        If ``enc_function == 'linear'``
            .. math::
                t_f(x) = (T - 1)(1 - x)

        If ``enc_function == 'log'``
            .. math::
                t_f(x) = (T - 1) - ln(\\alpha * x + 1)

        where :math:`\alpha` satisfies :math:`t_f(1) = T - 1`


        Example:
        .. code-block:: python

            x = torch.rand(size=[8, 2])
            print('x', x)
            T = 20
            encoder = LatencyEncoder(T)
            for t in range(T):
                print(encoder(x))

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        """
        super().__init__(T, step_mode)
        if enc_function == 'log':
            self.alpha = math.exp(T - 1.) - 1.
        elif enc_function != 'linear':
            raise NotImplementedError

        self.enc_function = enc_function

    def single_step_encode(self, x: torch.Tensor):
        if self.enc_function == 'log':
            t_f = (self.T - 1. - torch.log(self.alpha * x + 1.)).round().long()
        else:
            t_f = ((self.T - 1.) * (1. - x)).round().long()

        self.spike = F.one_hot(t_f, num_classes=self.T).to(x)
        # [*, T] -> [T, *]
        d_seq = list(range(self.spike.ndim - 1))
        d_seq.insert(0, self.spike.ndim - 1)
        self.spike = self.spike.permute(d_seq)


class PoissonEncoder(StatelessEncoder):
    def __init__(self, step_mode='s'):
        """
        * :ref:`API in English <PoissonEncoder.__init__-en>`

        .. _PoissonEncoder.__init__-cn:

        无状态的泊松编码器。输出脉冲的发放概率与输入 ``x`` 相同。

        .. warning::

            必须确保 ``0 <= x <= 1``。

        * :ref:`中文API <PoissonEncoder.__init__-cn>`

        .. _PoissonEncoder.__init__-en:

        The poisson encoder will output spike whose firing probability is ``x``。

        .. admonition:: Warning
            :class: warning

            The user must assert ``0 <= x <= 1``.
        """
        super().__init__(step_mode)

    def forward(self, x: torch.Tensor):
        out_spike = torch.rand_like(x).le(x).to(x)
        return out_spike


class WeightedPhaseEncoder(StatefulEncoder):
    def __init__(self, K: int, step_mode='s'):
        """
        * :ref:`API in English <WeightedPhaseEncoder.__init__-en>`

        .. _WeightedPhaseEncoder.__init__-cn:

        :param K: 编码周期。通常情况下，与SNN的仿真周期（总步长一致）
        :type K: int

        Kim J, Kim H, Huh S, et al. Deep neural networks with weighted spikes[J]. Neurocomputing, 2018, 311: 373-386.

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


        .. warning::

            不要忘记调用reset，因为这个编码器是有状态的。


        * :ref:`中文API <WeightedPhaseEncoder.__init__-cn>`

        .. _WeightedPhaseEncoder.__init__-en:

        :param K: the encoding period. It is usually same with the total simulation time-steps of SNN
        :type K: int

        The weighted phase encoder, which is based on binary system. It will flatten ``x`` as a binary number. When
        ``T=k``, it can encode :math:`x \in [0, 1-2^{-K}]` to different spikes. Here is the example from the origin paper:

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

        .. admonition:: Warning
            :class: warning

            Do not forget to reset the encoder because the encoder is stateful!

        """
        super().__init__(K, step_mode)

    def single_step_encode(self, x: torch.Tensor):
        assert (x >= 0).all() and (x <= 1 - 2 ** (-self.T)).all()
        inputs = x.clone()
        self.spike = torch.empty((self.T,) + x.shape, device=x.device)  # Encoding to [T, batch_size, *]
        w = 0.5
        for i in range(self.T):
            self.spike[i] = inputs >= w
            inputs -= w * self.spike[i]
            w *= 0.5


class PopSpikeEncoderDeterministic(nn.Module):
    """ Learnable Population Coding Spike Encoder with Deterministic Spike Trains"""

    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
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

        self.neurons = neuron.IFNode(v_threshold=0.999, v_reset=None, surrogate_function=surrogate.DeterministicPass(), detach_reset=True)

        functional.set_step_mode(self, step_mode='m')
        functional.set_backend(self, backend='torch')

    def forward(self, obs):
        obs = obs.view(-1, self.obs_dim, 1)

        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_act = pop_act.unsqueeze(0).repeat(self.spike_ts, 1, 1)

        return self.neurons(pop_act)


class PopSpikeEncoderRandom(nn.Module):
    """ Learnable Population Coding Spike Encoder with Random Spike Trains """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
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
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]
        
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_spikes = torch.zeros(self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device)
        
        # Generate Random Spike Trains
        for step in range(self.spike_ts):
            pop_spikes[step, :, :] = self.pseudo_spike(pop_act)
        
        return pop_spikes


class PopEncoder(nn.Module):
    """ Learnable Population Coding Encoder """
    def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
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
        obs = obs.view(-1, self.obs_dim, 1)
        batch_size = obs.shape[0]
        
        # Receptive Field of encoder population has Gaussian Shape
        pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
        pop_inputs = torch.zeros(self.spike_ts, batch_size, self.encoder_neuron_num, device=obs.device)
        
        # Generate Input Trains
        for step in range(self.spike_ts):
            pop_inputs[step, :, :] = pop_act
        
        return pop_inputs
