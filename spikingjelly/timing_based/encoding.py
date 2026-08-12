import torch

__all__ = ["GaussianTuning"]


class GaussianTuning:
    def __init__(
        self,
        n: int,
        m: int,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <GaussianTuning.__init__-cn>` | :ref:`English <GaussianTuning.__init__-en>`

        ----

        .. _GaussianTuning.__init__-cn:

        * **中文**

        使用高斯感受野将连续值编码为脉冲时间。每个输入维度由 ``m`` 个神经元
        编码，公开属性 ``mu`` 和 ``sigma2`` 分别保存感受野中心和方差。

        :param n: 输入特征维度数量
        :type n: int
        :param m: 每个特征使用的神经元数量，必须大于 2
        :type m: int
        :param x_min: 各输入特征的最小值，形状为 ``[n]``
        :type x_min: torch.Tensor
        :param x_max: 各输入特征的最大值，形状为 ``[n]``
        :type x_max: torch.Tensor
        :raises ValueError: ``n`` 或 ``m`` 非法、输入范围形状不匹配，或任一
            ``x_min`` 不小于对应的 ``x_max``

        ----

        .. _GaussianTuning.__init__-en:

        * **English**

        Encode continuous values as spike times with Gaussian receptive fields.
        Each input dimension uses ``m`` neurons. Public attributes ``mu`` and
        ``sigma2`` contain the receptive-field centers and variances.

        :param n: Number of input feature dimensions
        :type n: int
        :param m: Number of neurons per feature; must be greater than 2
        :type m: int
        :param x_min: Per-feature minimum values with shape ``[n]``
        :type x_min: torch.Tensor
        :param x_max: Per-feature maximum values with shape ``[n]``
        :type x_max: torch.Tensor
        :raises ValueError: If ``n`` or ``m`` is invalid, the range tensors have
            incompatible shapes, or any minimum is not smaller than its maximum
        """
        if n <= 0:
            raise ValueError("n must be a positive integer.")
        if m <= 2:
            raise ValueError("m must be greater than 2.")
        if x_min.shape != (n,) or x_max.shape != (n,):
            raise ValueError(
                f"x_min and x_max must both have shape ({n},), "
                f"but got {x_min.shape} and {x_max.shape}."
            )
        if not torch.all(x_min < x_max):
            raise ValueError("All elements of x_min must be less than x_max.")

        self.n = n
        self.m = m
        indices = torch.arange(
            1, m + 1, device=x_min.device, dtype=torch.float32
        ).unsqueeze(0)
        input_range = (x_max - x_min).unsqueeze(-1)
        self.mu = x_min.unsqueeze(-1) + (2 * indices - 3) * input_range / (2 * (m - 2))
        self.sigma2 = (input_range / (1.5 * (m - 2))).square().repeat(1, m)

    def encode(
        self,
        x: torch.Tensor,
        max_spike_time: int = 50,
    ) -> torch.Tensor:
        r"""
        **API Language** - :ref:`中文 <GaussianTuning.encode-cn>` | :ref:`English <GaussianTuning.encode-en>`

        ----

        .. _GaussianTuning.encode-cn:

        * **中文**

        :param x: 输入张量，形状为 ``[batch_size, n, samples_count]``
        :type x: torch.Tensor
        :param max_spike_time: 非负编码时间窗长度；达到该值的神经元以 ``-1`` 表示不发放。
            当其为 ``0`` 时，所有神经元均不发放
        :type max_spike_time: int
        :return: 形状为 ``[batch_size, n, samples_count, m]`` 的脉冲时间
        :rtype: torch.Tensor
        :raises AssertionError: 输入不是三维张量或特征维不等于 ``n``
        :raises ValueError: ``max_spike_time`` 为负数

        ----

        .. _GaussianTuning.encode-en:

        * **English**

        :param x: Input tensor with shape ``[batch_size, n, samples_count]``
        :type x: torch.Tensor
        :param max_spike_time: Non-negative encoding-window length; neurons reaching
            it are marked inactive with ``-1``. When it is ``0``, all neurons are inactive
        :type max_spike_time: int
        :return: Spike times with shape ``[batch_size, n, samples_count, m]``
        :rtype: torch.Tensor
        :raises AssertionError: If the input is not three-dimensional or its
            feature dimension differs from ``n``
        :raises ValueError: If ``max_spike_time`` is negative
        """
        if max_spike_time < 0:
            raise ValueError("max_spike_time must be non-negative.")
        if x.dim() != 3 or x.shape[1] != self.n:
            raise AssertionError(
                f"x must have shape [batch_size, {self.n}, samples_count], "
                f"but got {tuple(x.shape)}."
            )
        batch_size, _, samples_count = x.shape
        values = x.permute(0, 2, 1).reshape(-1, self.n, 1).expand(-1, -1, self.m)
        responses = torch.exp(-(values - self.mu).square() / (2 * self.sigma2))
        spike_times = (max_spike_time * (1 - responses)).round()
        spike_times[spike_times >= max_spike_time] = -1
        return spike_times.view(batch_size, samples_count, self.n, self.m).permute(
            0, 2, 1, 3
        )
