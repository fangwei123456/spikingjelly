from abc import abstractmethod
from typing import Union, Iterable, Optional, Callable
import math

import torch
import torch.nn as nn
from numpy import sqrt, newaxis, integer
from numpy.fft import irfft, rfftfreq
from numpy.random import default_rng, Generator, RandomState
from numpy import sum as npsum

from .. import surrogate, base


__all__ = [
    "powerlaw_psd_gaussian",
    "NoisyBaseNode",
    "NoisyCUBALIFNode",
    "NoisyILCBaseNode",
    "NoisyILCCUBALIFNode",
    "NoisyNonSpikingBaseNode",
    "NoisyNonSpikingIFNode",
]


def powerlaw_psd_gaussian(
    exponent: float,
    size: Union[int, Iterable[int]],
    fmin: float = 0.0,
    random_state: Optional[Union[int, Generator, RandomState]] = None,
):
    r"""
    **API Language:**
    :ref:`中文 <powerlaw_psd_gaussian-cn>` | :ref:`English <powerlaw_psd_gaussian-en>`

    ----

    .. _powerlaw_psd_gaussian-cn:

    * **中文**

    生成具有 :math:`(1/f)^\beta` 功率谱的高斯噪声。生成的噪声满足

    .. math::
        S(f) = (1 / f)^\beta

    Flicker / pink noise:

    .. math::
        \beta = 1

    Brown noise:

    .. math::
        \beta = 2

    自相关衰减比例为 :math:`\text{lag}^{-\gamma}`，其中 :math:`\gamma = 1 - \beta (0 < \beta < 1)`。
    对于接近 1 的 :math:`\beta` 值可能存在有限大小效应。该算法基于文章
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995).

    :param exponent: 噪声的功率谱指数 :math:`\beta`
    :type exponent: float

    :param size: 输出样本的形状，最后一个维度作为时间轴，其余维度独立。
    :type size: Union[int, Iterable[int]]

    :param fmin: 低频截止，默认为 0，对应原始论文。低于 fmin 的频率功率谱平坦。fmin 定义为
        相对于单位采样率。内部会映射为 max(fmin, 1/samples)。最大值为 fmin = 0.5，
        即 Nyquist 频率，此时输出为白噪声。
    :type fmin: float, optional

    :param random_state: 可选，设置 NumPy 随机数生成器状态。支持整数、None、
        np.random.Generator 或 np.random.RandomState。
    :type random_state: int, numpy.integer, numpy.random.Generator,
        numpy.random.RandomState, optional

    :return: 生成的噪声样本
    :rtype: array

    ----

    .. _powerlaw_psd_gaussian-en:

    * **English**

    Generate Gaussian noise with a power spectrum proportional to :math:`(1/f)^\beta`.
    The generated noise satisfies

    .. math::
        S(f) = (1 / f)^\beta

    Flicker / pink noise:

    .. math::
        \beta = 1

    Brown noise:

    .. math::
        \beta = 2

    The autocorrelation decays proportional to :math:`\text{lag}^{-\gamma}`,
    where :math:`\gamma = 1 - \beta` for :math:`0 < \beta < 1`. Finite-size effects may
    occur when :math:`\beta` is close to 1. The algorithm is based on:
    Timmer, J. and Koenig, M.:
    On generating power law noise.
    Astron. Astrophys. 300, 707-710 (1995).

    :param exponent: the power spectrum exponent $\beta$.
    :type exponent: float

    :param size: shape of the output samples. The last axis is taken as time,
        and all other axes are independent.
    :type size: Union[int, Iterable[int]]

    :param fmin: low-frequency cutoff. Default 0 corresponds to the original paper.
        Frequencies below fmin are flat. fmin is defined relative to unit sampling rate.
        Internally mapped to max(fmin, 1/samples). The maximum allowed value
        is 0.5 (Nyquist frequency), producing white noise.
    :type fmin: float, optional

    :param random_state: optional, sets the state of NumPy's underlying random number generator.
        Supports int, None, np.random.Generator, or np.random.RandomState.
    :type random_state: int, numpy.integer, numpy.random.Generator,
        numpy.random.RandomState, optional

    :return: generated Gaussian noise samples
    :rtype: array

    ----

    **Examples:**

    .. code-block:: python

        # generate 1/f noise == pink noise == flicker noise
        >>> import colorednoise as cn
        >>> y = cn.powerlaw_psd_gaussian(1, 5)
    """
    # Make sure size is a list so we can iterate it and assign to it.
    if isinstance(size, (integer, int)):
        size = [size]
    elif isinstance(size, Iterable):
        size = list(size)
    else:
        raise ValueError("Size must be of type int or Iterable[int]")

    # The number of samples in each time series
    samples = size[-1]

    # Calculate Frequencies (we asume a sample rate of one)
    # Use fft functions for real output (-> hermitian spectrum)
    f = rfftfreq(samples)  # type: ignore # mypy 1.5.1 has problems here

    # Validate / normalise fmin
    if 0 <= fmin <= 0.5:
        fmin = max(fmin, 1.0 / samples)  # Low frequency cutoff
    else:
        raise ValueError("fmin must be chosen between 0 and 0.5.")

    # Build scaling factors for all frequencies
    s_scale = f
    ix = npsum(s_scale < fmin)  # Index of the cutoff
    if ix and ix < len(s_scale):
        s_scale[:ix] = s_scale[ix]
    s_scale = s_scale ** (-exponent / 2.0)

    # Calculate theoretical output standard deviation from scaling
    w = s_scale[1:].copy()
    w[-1] *= (1 + (samples % 2)) / 2.0  # correct f = +-0.5
    sigma = 2 * sqrt(npsum(w**2)) / samples

    # Adjust size to generate one Fourier component per frequency
    size[-1] = len(f)

    # Add empty dimension(s) to broadcast s_scale along last
    # dimension of generated random power + phase (below)
    dims_to_add = len(size) - 1
    s_scale = s_scale[(newaxis,) * dims_to_add + (Ellipsis,)]

    # prepare random number generator
    normal_dist = _get_normal_distribution(random_state)

    # Generate scaled random power + phase
    sr = normal_dist(scale=s_scale, size=size)
    si = normal_dist(scale=s_scale, size=size)

    # If the signal length is even, frequencies +/- 0.5 are equal
    # so the coefficient must be real.
    if not (samples % 2):
        si[..., -1] = 0
        sr[..., -1] *= sqrt(2)  # Fix magnitude

    # Regardless of signal length, the DC component must be real
    si[..., 0] = 0
    sr[..., 0] *= sqrt(2)  # Fix magnitude

    # Combine power + corrected phase to Fourier components
    s = sr + 1j * si

    # Transform to real time series & scale to unit variance
    y = irfft(s, n=samples, axis=-1) / sigma

    return y


def _get_normal_distribution(
    random_state: Optional[Union[int, Generator, RandomState]],
):
    normal_dist = None
    if isinstance(random_state, (integer, int)) or random_state is None:
        random_state = default_rng(random_state)
        normal_dist = random_state.normal
    elif isinstance(random_state, (Generator, RandomState)):
        normal_dist = random_state.normal
    else:
        raise ValueError(
            "random_state must be one of integer, numpy.random.Generator, "
            "numpy.random.Randomstate"
        )
    return normal_dist


class NoisyBaseNode(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        num_node,
        is_training: bool = True,
        T: int = 5,
        sigma_init: float = 0.5,
        beta: float = 0.0,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.num_node = num_node
        self.is_training = is_training
        self.T = T
        self.beta = beta

        self.sigma_v = sigma_init / math.sqrt(num_node)
        self.cn_v = None

        self.sigma_s = sigma_init / math.sqrt(num_node)
        self.cn_s = None

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1.0 - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        y = []

        if self.is_training:
            if self.cn_v is None or self.cn_s is None:
                self.noise_step += 1

            for t in range(self.T):
                if self.cn_v is None:
                    self.neuronal_charge(
                        x_seq[t]
                        + self.sigma_v
                        * self.eps_v_seq[self.noise_step][t].to(x_seq.device)
                    )
                else:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.cn_v[:, t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                if self.cn_s is None:
                    spike = spike + self.sigma_s * self.eps_s_seq[self.noise_step][
                        t
                    ].to(x_seq.device)
                else:
                    spike = spike + self.sigma_s * self.cn_s[:, t]
                y.append(spike)

        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                y.append(spike)

        return torch.stack(y)

    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        # (nodes, steps * T) -> (nodes, steps, T) -> (steps, T, nodes)
        self.eps_v_seq = torch.FloatTensor(
            powerlaw_psd_gaussian(self.beta, eps_shape).reshape(
                self.num_node, num_rl_step, self.T
            )
        ).permute(per_order)
        self.eps_s_seq = torch.FloatTensor(
            powerlaw_psd_gaussian(self.beta, eps_shape).reshape(
                self.num_node, num_rl_step, self.T
            )
        ).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        cn = [self.eps_v_seq[self.noise_step], self.eps_s_seq[self.noise_step]]
        return torch.cat(cn, dim=1)

    def load_colored_noise(self, cn):
        self.cn_v = cn[:, :, : self.num_node]
        self.cn_s = cn[:, :, self.num_node :]

    def cancel_load(self):
        self.cn_v = None
        self.cn_s = None


class NoisyCUBALIFNode(NoisyBaseNode):
    def __init__(
        self,
        num_node,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        is_training: bool = True,
        T: int = 5,
        sigma_init: float = 0.5,
        beta: float = 0.0,
        v_threshold: float = 0.5,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        super().__init__(
            num_node,
            is_training,
            T,
            sigma_init,
            beta,
            v_threshold,
            v_reset,
            surrogate_function,
        )

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)


class NoisyILCBaseNode(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        is_training: bool = True,
        T: int = 5,
        sigma_init: float = 0.5,
        beta: float = 0.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        assert isinstance(v_reset, float) or v_reset is None
        assert isinstance(v_threshold, float)
        super().__init__()

        self.act_dim = act_dim
        self.num_node = act_dim * dec_pop_dim
        self.dec_pop_dim = dec_pop_dim

        self.conn = nn.Conv1d(act_dim, self.num_node, dec_pop_dim, groups=act_dim)

        self.is_training = is_training
        self.T = T
        self.beta = beta

        self.sigma_v = sigma_init / math.sqrt(self.num_node)
        self.cn_v = None

        self.sigma_s = sigma_init / math.sqrt(self.num_node)
        self.cn_s = None

        self.v_threshold = v_threshold
        self.v_reset = v_reset

        self.surrogate_function = surrogate_function

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def neuronal_fire(self):
        return self.surrogate_function(self.v - self.v_threshold)

    def neuronal_reset(self, spike):
        if self.v_reset is None:
            self.v = self.v - spike * self.v_threshold
        else:
            self.v = (1.0 - spike) * self.v + spike * self.v_reset

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=self.v_reset)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        y = []

        if self.is_training:
            if self.cn_v is None or self.cn_s is None:
                self.noise_step += 1

            for t in range(self.T):
                if self.cn_v is None:
                    self.neuronal_charge(
                        x_seq[t]
                        + self.sigma_v
                        * self.eps_v_seq[self.noise_step][t].to(x_seq.device)
                    )
                else:
                    self.neuronal_charge(x_seq[t] + self.sigma_v * self.cn_v[:, t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                if self.cn_s is None:
                    spike = spike + self.sigma_s * self.eps_s_seq[self.noise_step][
                        t
                    ].to(x_seq.device)
                else:
                    spike = spike + self.sigma_s * self.cn_s[:, t]
                y.append(spike)

                if t < self.T - 1:
                    x_seq[t + 1] = x_seq[t + 1] + self.conn(
                        spike.view(-1, self.act_dim, self.dec_pop_dim)
                    ).view(-1, self.num_node)

        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                spike = self.neuronal_fire()
                self.neuronal_reset(spike)
                y.append(spike)

                if t < self.T - 1:
                    x_seq[t + 1] = x_seq[t + 1] + self.conn(
                        spike.view(-1, self.act_dim, self.dec_pop_dim)
                    ).view(-1, self.num_node)

        return torch.stack(y)

    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        # (nodes, steps * T) -> (nodes, steps, T) -> (steps, T, nodes)
        self.eps_v_seq = torch.FloatTensor(
            powerlaw_psd_gaussian(self.beta, eps_shape).reshape(
                self.num_node, num_rl_step, self.T
            )
        ).permute(per_order)
        self.eps_s_seq = torch.FloatTensor(
            powerlaw_psd_gaussian(self.beta, eps_shape).reshape(
                self.num_node, num_rl_step, self.T
            )
        ).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        cn = [self.eps_v_seq[self.noise_step], self.eps_s_seq[self.noise_step]]
        return torch.cat(cn, dim=1)

    def load_colored_noise(self, cn):
        self.cn_v = cn[:, :, : self.num_node]
        self.cn_s = cn[:, :, self.num_node :]

    def cancel_load(self):
        self.cn_v = None
        self.cn_s = None


class NoisyILCCUBALIFNode(NoisyILCBaseNode):
    def __init__(
        self,
        act_dim,
        dec_pop_dim,
        c_decay: float = 0.5,
        v_decay: float = 0.75,
        is_training: bool = True,
        T: int = 5,
        sigma_init: float = 0.5,
        beta: float = 0.0,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: Callable = surrogate.Rect(),
    ):
        super().__init__(
            act_dim,
            dec_pop_dim,
            is_training,
            T,
            sigma_init,
            beta,
            v_threshold,
            v_reset,
            surrogate_function,
        )

        self.c_decay = c_decay
        self.v_decay = v_decay

    def neuronal_charge(self, x: torch.Tensor):
        self.c = self.c * self.c_decay + x
        self.v = self.v * self.v_decay + self.c

    def init_tensor(self, data: torch.Tensor):
        self.c = torch.full_like(data, fill_value=0.0)
        self.v = torch.full_like(data, fill_value=self.v_reset)


class NoisyNonSpikingBaseNode(nn.Module, base.MultiStepModule):
    def __init__(
        self,
        num_node,
        is_training: bool = True,
        T: int = 5,
        sigma_init: float = 0.5,
        beta: float = 0.0,
        decode: Optional[str] = None,
    ):
        super().__init__()

        self.num_node = num_node
        self.is_training = is_training
        self.T = T
        self.beta = beta
        self.decode = decode

        self.sigma = nn.Parameter(torch.FloatTensor(num_node))
        self.sigma.data.fill_(sigma_init / math.sqrt(num_node))
        self.cn = None

    @abstractmethod
    def neuronal_charge(self, x: torch.Tensor):
        raise NotImplementedError

    def init_tensor(self, data: torch.Tensor):
        self.v = torch.full_like(data, fill_value=0.0)

    def forward(self, x_seq: torch.Tensor):
        self.init_tensor(x_seq[0].data)

        v_seq = []

        if self.is_training:
            if self.cn is None:
                self.noise_step += 1

            for t in range(self.T):
                if self.cn is None:
                    self.neuronal_charge(
                        x_seq[t]
                        + self.sigma.mul(
                            self.eps_seq[self.noise_step][t].to(x_seq.device)
                        )
                    )
                else:
                    self.neuronal_charge(
                        x_seq[t] + self.sigma.mul(self.cn[:, t].to(x_seq.device))
                    )
                v_seq.append(self.v)

        else:
            for t in range(self.T):
                self.neuronal_charge(x_seq[t])
                v_seq.append(self.v)

        if self.decode == "max-mem":
            mem = torch.max(torch.stack(v_seq, 0), 0).values
        elif self.decode == "max-abs-mem":
            v_stack = torch.stack(v_seq, 0)
            max_mem = torch.max(v_stack, 0).values
            min_mem = torch.min(v_stack, 0).values
            mem = max_mem * (max_mem.abs() > min_mem.abs()) + min_mem * (
                max_mem.abs() <= min_mem.abs()
            )
        elif self.decode == "mean-mem":
            mem = torch.mean(torch.stack(v_seq, 0), 0)
        elif self.decode == "last-me":
            mem = v_seq[-1]
        else:
            mem = v_seq
        return mem

    def reset_noise(self, num_rl_step):
        eps_shape = [self.num_node, num_rl_step * self.T]
        per_order = [1, 2, 0]
        self.eps_seq = torch.FloatTensor(
            powerlaw_psd_gaussian(self.beta, eps_shape).reshape(
                self.num_node, num_rl_step, self.T
            )
        ).permute(per_order)
        self.noise_step = -1

    def get_colored_noise(self):
        return self.eps_seq[self.noise_step]

    def load_colored_noise(self, cn):
        self.cn = cn

    def cancel_load(self):
        self.cn = None


class NoisyNonSpikingIFNode(NoisyNonSpikingBaseNode):
    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.v + x
