import math

import torch
from torch import nn

__all__ = ["GaussianTuning"]


class GaussianTuning(nn.Module):
    def __init__(
        self,
        n: int,
        m: int,
        x_min: torch.Tensor,
        x_max: torch.Tensor,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <timing-GaussianTuning-init-cn>` | :ref:`English <timing-GaussianTuning-init-en>`

        ----

        .. _timing-GaussianTuning-init-cn:

        * **中文**

        用每特征 m 个高斯感受野将连续输入编码为事件时刻。中心 mu 和方差
        sigma2 是固定 buffers，支持 nn.Module.to 和 state_dict，不参与学习。
        公式与高斯调谐编码一致；forward 不取整，也不输出无事件占位值。
        无跨调用时间状态。

        :param n: 正整数特征数量。
        :type n: int
        :param m: 每个特征的感受野数量，整数且大于 2。
        :type m: int
        :param x_min: 有限 FP32/FP64 特征下界，形状 ``[n]``；决定 buffers 的
            初始 dtype/device，不保留输入的计算图。
        :type x_min: torch.Tensor
        :param x_max: 有限特征上界 ``[n]``，与 x_min 同 dtype/device，逐项大于下界。
        :type x_max: torch.Tensor
        :raises TypeError: 下界与上界的 dtype 不受支持或不相同。
        :raises ValueError: 数量、形状、设备、有限性、范围或感受野方差非法。

        ----

        .. _timing-GaussianTuning-init-en:

        * **English**

        Encode continuous values as event times with m Gaussian receptive fields
        per feature. Centers mu and variances sigma2 are fixed buffers supporting
        nn.Module.to and state_dict, not learned parameters. The Gaussian tuning
        formula is unchanged; forward neither rounds times nor emits inactive
        placeholders. No temporal state persists between calls.

        :param n: Positive integer number of features.
        :type n: int
        :param m: Integer number of receptive fields per feature, greater than 2.
        :type m: int
        :param x_min: Finite FP32/FP64 feature minima ``[n]``, determining initial
            buffer dtype/device. Its computation graph is not retained.
        :type x_min: torch.Tensor
        :param x_max: Finite maxima ``[n]`` matching x_min's dtype/device, strictly
            greater than each minimum.
        :type x_max: torch.Tensor
        :raises TypeError: For unsupported or mismatched range dtypes.
        :raises ValueError: For invalid sizes, shapes, devices, finite values,
            ranges or receptive-field variances.
        """
        super().__init__()
        if not isinstance(n, int) or not isinstance(m, int) or n <= 0 or m <= 2:
            raise ValueError(
                "n must be positive and m must exceed 2; both must be integers."
            )
        if (
            x_min.dtype not in (torch.float32, torch.float64)
            or x_max.dtype != x_min.dtype
        ):
            raise TypeError("Range tensors must share FP32 or FP64 dtype.")
        if x_min.shape != (n,) or x_max.shape != (n,) or x_min.device != x_max.device:
            raise ValueError("Range tensors must share device and shape [n].")
        if not bool(
            torch.isfinite(x_min).all()
            & torch.isfinite(x_max).all()
            & (x_min < x_max).all()
        ):
            raise ValueError("Feature bounds must be finite with x_min < x_max.")
        self.n = n
        self.m = m
        with torch.no_grad():
            indices = torch.arange(1, m + 1, device=x_min.device, dtype=x_min.dtype)[
                None, :
            ]
            width = (x_max - x_min)[:, None]
            mu = x_min[:, None] + (2 * indices - 3) * width / (2 * (m - 2))
            sigma2 = (width / (1.5 * (m - 2))).square().expand(n, m).clone()
            if not bool(
                torch.isfinite(mu).all()
                & torch.isfinite(sigma2).all()
                & (sigma2 > 0).all()
            ):
                raise ValueError(
                    "Receptive fields are not representable in their dtype."
                )
        self.register_buffer("mu", mu)
        self.register_buffer("sigma2", sigma2)

    def forward(
        self, x: torch.Tensor, duration: float = 50.0
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <timing-GaussianTuning-forward-cn>` | :ref:`English <timing-GaussianTuning-forward-en>`

        ----

        .. _timing-GaussianTuning-forward-cn:

        * **中文**

        计算 ``duration * (1 - response)`` 的连续时刻并输出紧凑事件。
        输出顺序按 batch、特征、样本、感受野排列；输入连接索引为
        ``(feature * samples_count + sample) * m + field``。
        仅保留 ``0 <= time < duration``；响应下溢等导致窗口末端时刻时不输出事件。
        输入值可超出初始化范围，但必须有限。不进行时间取整。
        有效事件的时刻可对 x 求导，事件选择边界不保证唯一导数。

        :param x: 有限 FP32/FP64 张量 ``[B,n,samples_count]``，dtype/device 与
            buffers 相同。samples_count 可以为零。
        :type x: torch.Tensor
        :param duration: 正且有限、可用 x.dtype 表示的编码窗终点，默认 50；
            与目标神经元的时间单位一致。
        :type duration: float
        :return: ``(spike_times,input_indices,batch_offsets)``，形状分别为
            ``[E]``、``[E]``、``[B+1]``，同 x.device。时刻同 x.dtype，索引及
            边界为 int64。空样本通过重复边界表示，无事件占位值。调用间无状态变化。
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: duration 类型或输入 dtype 不受支持/不匹配。
        :raises ValueError: 时间窗、输入形状、设备或有限性非法，或转换后的感受野非法。

        ----

        .. _timing-GaussianTuning-forward-en:

        * **English**

        Compute continuous times ``duration * (1 - response)`` and emit compact
        events ordered by batch, feature, sample and receptive field. Input indices
        are ``(feature * samples_count + sample) * m + field``. Keep only
        ``0 <= time < duration``; window-end events from underflow are omitted.
        Input values may be outside the initialization range but must be finite.
        Times are not rounded. Selected event times are differentiable with respect
        to x; event-selection boundaries need not have unique derivatives.

        :param x: Finite FP32/FP64 tensor ``[B,n,samples_count]`` matching buffer
            dtype/device. samples_count may be zero.
        :type x: torch.Tensor
        :param duration: Finite positive encoding endpoint representable in x.dtype,
            default 50, in the target neuron's time units.
        :type duration: float
        :return: ``(spike_times,input_indices,batch_offsets)`` of shapes ``[E]``,
            ``[E]`` and ``[B+1]`` on x.device. Times share x.dtype; indices/offsets
            are int64. Repeated offsets represent empty trials, without inactive
            placeholders. No state changes between calls.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: For unsupported duration types or unsupported/mismatched dtypes.
        :raises ValueError: For invalid windows, input shapes, devices, finite values
            or receptive fields after conversion.
        """
        if not isinstance(duration, (int, float)):
            raise TypeError("duration must be a fixed real scalar.")
        if not math.isfinite(duration) or duration <= 0:
            raise ValueError("duration must be finite and positive.")
        if x.dtype not in (torch.float32, torch.float64) or x.dtype != self.mu.dtype:
            raise TypeError("Input and buffers must share FP32 or FP64 dtype.")
        if x.ndim != 3 or x.shape[1] != self.n or x.device != self.mu.device:
            raise ValueError(
                "Input must have shape [B,n,samples_count] on the buffers' device."
            )
        if not bool(
            torch.isfinite(self.mu).all()
            & torch.isfinite(self.sigma2).all()
            & (self.sigma2 > 0).all()
        ):
            raise ValueError(
                "Receptive fields must remain finite with positive variances."
            )
        endpoint = x.new_tensor(duration)
        if not bool(
            torch.isfinite(x).all() & torch.isfinite(endpoint) & (endpoint > 0)
        ):
            raise ValueError("Input and representable duration must be finite.")
        values = x.unsqueeze(-1)
        distance = (values - self.mu[None, :, None, :]) / self.sigma2[
            None, :, None, :
        ].sqrt()
        finfo = torch.finfo(x.dtype)
        cutoff = math.sqrt(-2 * math.log(finfo.tiny * finfo.eps))
        # Bound the square before exponentiation so absent events have finite backward.
        responses = torch.exp(-0.5 * distance.clamp(-cutoff, cutoff).square())
        responses = torch.where(distance.abs() > cutoff, 0, responses)
        times = (endpoint * (1 - responses)).reshape(
            x.shape[0], self.n * x.shape[2] * self.m
        )
        valid = (times >= 0) & (times < endpoint)
        columns = torch.arange(times.shape[1], device=x.device).expand_as(times)
        counts = valid.sum(1)
        offsets = torch.cat((counts.new_zeros(1), counts.cumsum(0)))
        return times[valid], columns[valid], offsets
