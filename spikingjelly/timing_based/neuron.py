import torch
from torch import nn

from . import functional

__all__ = ["Tempotron"]


class Tempotron(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        duration: float,
        tau: float = 15.0,
        tau_s: float = 3.75,
        v_threshold: float = 1.0,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <timing-Tempotron-init-cn>` | :ref:`English <timing-Tempotron-init-en>`

        ----

        .. _timing-Tempotron-init-cn:

        * **中文**

        拥有可训练突触权重的连续事件 Tempotron。每条连接允许多个输入事件。
        首次上穿阈值后屏蔽同刻及更晚输入，已有 PSP 继续演化，不硬重置。
        无跨调用时间状态；loss 和沉默恢复由调用方定义。

        权重形状为 ``[out_features,in_features]``，以均值 0、标准差 0.001
        的正态分布初始化，checkpoint 键为 ``weight``。使用普通 nn.Module
        的 ``to`` 管理设备与 dtype。支持 CPU/CUDA FP32/FP64 及一阶梯度。
        时间单位由调用方选择，各时刻、duration 和时间常数必须一致。

        :param in_features: 正整数输入连接数量。
        :type in_features: int
        :param out_features: 正整数输出神经元数量。
        :type out_features: int
        :param duration: 正且有限的观察窗终点，窗口为 ``[0,duration]``。
        :type duration: float
        :param tau: 正且有限的膜时间常数，默认 15。
        :type tau: float
        :param tau_s: 正且有限的突触时间常数，默认 3.75；可大于 tau，但不能相等。
        :type tau_s: float
        :param v_threshold: 正且有限的阈值，默认 1；不改变单位峰值 PSP。
        :type v_threshold: float
        :raises ValueError: 数量、时间窗、时间常数或阈值非法，或时间常数不可表示。
        :raises TypeError: 时间超参数或阈值不是固定实数标量。

        ----

        .. _timing-Tempotron-init-en:

        * **English**

        Continuous-event Tempotron owning trainable synaptic weights. Connections
        accept multiple events. The first upward threshold crossing shunts
        simultaneous and later inputs; existing PSPs continue without a hard reset.
        No temporal state persists between calls. Callers choose losses and recovery.

        Weights have shape ``[out_features,in_features]``, normal initialization
        with mean 0 and standard deviation 0.001, and checkpoint key ``weight``.
        Standard nn.Module ``to`` controls device/dtype. CPU/CUDA FP32/FP64 and
        first-order gradients are supported. Callers choose time units, consistently
        for all event times, duration and time constants.

        :param in_features: Positive number of input connections.
        :type in_features: int
        :param out_features: Positive number of output neurons.
        :type out_features: int
        :param duration: Finite positive endpoint of ``[0,duration]``.
        :type duration: float
        :param tau: Finite positive membrane time constant, default 15.
        :type tau: float
        :param tau_s: Finite positive synaptic time constant, default 3.75;
            may exceed tau but must differ from it.
        :type tau_s: float
        :param v_threshold: Finite positive threshold, default 1;
            independent of unit-peak PSP normalization.
        :type v_threshold: float
        :raises ValueError: For invalid sizes, window, time constants or threshold,
            or unrepresentable time constants.
        :raises TypeError: If time hyperparameters or threshold are not fixed real scalars.
        """
        super().__init__()
        if (
            not isinstance(in_features, int)
            or not isinstance(out_features, int)
            or min(in_features, out_features) <= 0
        ):
            raise ValueError("in_features and out_features must be positive integers.")
        functional._parameters(duration, tau, tau_s, v_threshold)
        self.in_features = in_features
        self.out_features = out_features
        self.duration = duration
        self.tau = tau
        self.tau_s = tau_s
        self.v_threshold = v_threshold
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.001)

    def forward(
        self,
        spike_times: torch.Tensor,
        input_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <timing-Tempotron-forward-cn>` | :ref:`English <timing-Tempotron-forward-en>`

        ----

        .. _timing-Tempotron-forward-cn:

        * **中文**

        使用本模块权重执行 :func:`spikingjelly.timing_based.functional.tempotron`。
        支持重复连接、无序及空样本，窗外输入梯度为零。内部 FP64 事件求解仅
        支持一阶反向；峰值并列选择最早分支，奇异的被使用时间导数报错。
        无 loss、替代梯度或跨调用状态。

        :param spike_times: 有限 FP32/FP64 时刻 ``[E]``，单位与 duration 一致。
        :type spike_times: torch.Tensor
        :param input_indices: int64 ``[E]``，范围 ``[0,in_features)``。
        :type input_indices: torch.Tensor
        :param batch_offsets: 同设备 int64 ``[B+1]``，从 0 到 E 的非递减分段。
        :type batch_offsets: torch.Tensor
        :return: 同输入 dtype/device 的 ``(v_max,t_peak,t_spike)``，各为
            ``[B,out_features]``；峰值属于 shunting 响应，未发放为 inf，空样本
            为 ``(0,0,inf)``。时刻与权重必须同 dtype/device。
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: 张量 dtype 不受支持。
        :raises ValueError: 张量形状、设备、有限性、索引、分段或超参数非法。
        :raises RuntimeError: 数值状态、求根或被请求的导数不可表示。

        ----

        .. _timing-Tempotron-forward-en:

        * **English**

        Evaluate :func:`spikingjelly.timing_based.functional.tempotron` with this
        module's weights. Repeated connections, unordered events and empty trials
        are supported; out-of-window inputs have zero gradient. FP64 event solving
        supports first-order backward only. Equal peaks select the earliest branch;
        singular requested time derivatives raise. No loss, surrogate or temporal
        state is built in.

        :param spike_times: Finite FP32/FP64 times ``[E]`` in duration's units.
        :type spike_times: torch.Tensor
        :param input_indices: int64 ``[E]`` in ``[0,in_features)``.
        :type input_indices: torch.Tensor
        :param batch_offsets: Same-device int64 ``[B+1]``, nondecreasing from 0 to E.
        :type batch_offsets: torch.Tensor
        :return: ``(v_max,t_peak,t_spike)``, each ``[B,out_features]`` with input
            dtype/device. Peaks follow shunting; non-firing times are inf; empty
            trials yield ``(0,0,inf)``. Times and weights must share dtype/device.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: For unsupported tensor dtypes.
        :raises ValueError: For invalid shapes, devices, finite values, indices,
            offsets or hyperparameters.
        :raises RuntimeError: For unrepresentable states, roots or requested derivatives.
        """
        return functional.tempotron(
            spike_times,
            input_indices,
            batch_offsets,
            self.weight,
            self.duration,
            self.tau,
            self.tau_s,
            self.v_threshold,
        )
