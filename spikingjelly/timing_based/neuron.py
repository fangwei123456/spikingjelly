import torch
from torch import nn

from . import functional

__all__ = ["Tempotron", "SRM", "CubicSRM", "CubicConv2d"]


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


class SRM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        duration: float,
        v_threshold: float = 1.0,
        max_spikes: int = 64,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <timing-SRM-init-cn>` | :ref:`English <timing-SRM-init-en>`

        ----

        .. _timing-SRM-init-cn:

        * **中文**

        拥有权重的连续事件 SRM；输入核 ``max(u,0)``，输出历史核
        ``-v_threshold*H(u)``。发放后电压归零、电流保留，可多次发放。
        无泄漏、不应期或跨调用状态；loss 和恢复策略由调用方定义。
        ``weight[O,I]`` 默认 N(0,0.001²)，checkpoint 键为 ``weight``。
        CPU/CUDA FP32/FP64，同设备同浮点 dtype，内部 FP64。

        :param in_features: 正整数输入连接数。
        :type in_features: int
        :param out_features: 正整数输出神经元数。
        :type out_features: int
        :param duration: 有限正数窗口终点，单位与输入时刻一致。
        :type duration: float
        :param v_threshold: 有限正数电压阈值与复位幅度，默认 1。
        :type v_threshold: float
        :param max_spikes: 每样本每输出事件容量，正整数，默认 64；超出报错。
        :type max_spikes: int
        :raises TypeError: 时间窗、阈值或事件容量类型不受支持。
        :raises ValueError: 参数非法或事件容量无法表示。

        ----

        .. _timing-SRM-init-en:

        * **English**

        Trainable continuous-event SRM with input kernel ``max(u,0)`` and output
        history kernel ``-v_threshold*H(u)``. Spiking resets voltage to zero and
        retains current, allowing multiple spikes. No leak, refractory period or
        cross-call state. Callers define losses and recovery. ``weight[O,I]``
        defaults to N(0,0.001²); its checkpoint key is ``weight``. CPU/CUDA FP32/FP64
        inputs share device and floating dtype; internals use FP64.

        :param in_features: Positive number of input connections.
        :type in_features: int
        :param out_features: Positive number of output neurons.
        :type out_features: int
        :param duration: Finite positive window endpoint, in input-time units.
        :type duration: float
        :param v_threshold: Finite positive voltage threshold/reset amplitude, default 1.
        :type v_threshold: float
        :param max_spikes: Positive per-trial, per-output capacity, default 64;
            overflow raises.
        :type max_spikes: int
        :raises TypeError: For unsupported window, threshold or event-capacity types.
        :raises ValueError: For invalid parameters or unrepresentable capacity.
        """
        super().__init__()
        if any(
            not isinstance(size, int) or isinstance(size, bool) or size < 1
            for size in (in_features, out_features)
        ):
            raise ValueError("in_features and out_features must be positive integers.")
        functional._srm_parameters(duration, v_threshold, max_spikes)
        self.in_features = in_features
        self.out_features = out_features
        self.duration = duration
        self.v_threshold = v_threshold
        self.max_spikes = max_spikes
        self.weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.normal_(self.weight, mean=0.0, std=0.001)

    def forward(
        self,
        spike_times: torch.Tensor,
        input_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <timing-SRM-forward-cn>` | :ref:`English <timing-SRM-forward-en>`

        ----

        .. _timing-SRM-forward-cn:

        * **中文**

        用模块权重执行 :func:`spikingjelly.timing_based.functional.srm`。
        每次独立求解 ``[0,duration]``，同刻输出优先；窗外输入梯度为零。
        返回紧凑多发事件，所选分支可微，事件数/顺序切换处不保证唯一导数。

        :param spike_times: 有限 FP32/FP64 ``[E]``，与权重同 dtype/device。
        :type spike_times: torch.Tensor
        :param input_indices: 同设备 int64 ``[E]``，范围 ``[0,in_features)``。
        :type input_indices: torch.Tensor
        :param batch_offsets: 同设备 int64 ``[B+1]``，从 0 到 E 非递减。
        :type batch_offsets: torch.Tensor
        :return: 输出 ``(times,indices,offsets)``，形状 ``[S]``、``[S]``、``[B+1]``；
            同设备，时刻同输入 dtype，索引和边界为 int64。按样本/输出/发放次序
            排列，无事件样本有重复边界，可直接串联下一 SRM 层。
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: 输入 dtype 不受支持。
        :raises ValueError: 输入形状、设备、有限性、索引或分段非法。
        :raises RuntimeError: 状态/根不可表示、超出事件容量或输入时刻/权重梯度非有限。

        ----

        .. _timing-SRM-forward-en:

        * **English**

        Evaluate :func:`spikingjelly.timing_based.functional.srm` with module
        weights. Independently solve ``[0,duration]``; outputs win ties and
        out-of-window inputs have zero gradient. Compact multi-spike outputs are
        differentiable within the selected branch; count/order boundaries need
        not have unique derivatives.

        :param spike_times: Finite FP32/FP64 ``[E]`` matching weight dtype/device.
        :type spike_times: torch.Tensor
        :param input_indices: Same-device int64 ``[E]`` in ``[0,in_features)``.
        :type input_indices: torch.Tensor
        :param batch_offsets: Same-device nondecreasing int64 ``[B+1]`` from 0 to E.
        :type batch_offsets: torch.Tensor
        :return: ``(times,indices,offsets)`` with shapes ``[S]``, ``[S]``, ``[B+1]``
            on the input device, matching floating time dtype and int64 metadata.
            Ordered by trial/output/spike number. Repeated offsets mark empty trials.
            Directly usable by another SRM layer.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: For unsupported dtypes.
        :raises ValueError: For invalid shapes, devices, finite values, indices or offsets.
        :raises RuntimeError: For unrepresentable states/roots, capacity overflow
            or non-finite input-time/weight gradients.
        """
        return functional.srm(
            spike_times,
            input_indices,
            batch_offsets,
            self.weight,
            self.duration,
            self.v_threshold,
            self.max_spikes,
        )


class CubicSRM(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        duration: float,
        support: float = 2.0,
        v_threshold: float = 1.0,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <timing-CubicSRM-cn>` | :ref:`English <timing-CubicSRM-en>`

        ----

        .. _timing-CubicSRM-cn:

        * **中文**

        实验性有限支撑三次 PSP 的单发 SRM。模型定义与边界见
        :func:`spikingjelly.timing_based.functional.cubic_srm`。模块仅持有权重
        与固定超参数；每次调用独立、无持久膜状态。权重默认 N(0,0.001²)，
        state_dict 使用 weight 键。标准 .to 可转换 CPU/CUDA，时刻与权重仅支持 FP32。

        :param in_features: 正整数输入连接数，每连接允许多个事件。
        :type in_features: int
        :param out_features: 正整数输出数，每输出每试次至多一次发放。
        :type out_features: int
        :param duration: 有限正数观察窗终点，与输入时刻同单位。
        :type duration: float
        :param support: 有限正数 PSP 支撑长度，默认 2，与输入时刻同单位。
        :type support: float
        :param v_threshold: 有限正数电压阈值，默认 1。
        :type v_threshold: float
        :raises TypeError: 连接数量或超参数类型非法。
        :raises ValueError: 非正连接数量或非法超参数。

        ----

        .. _timing-CubicSRM-en:

        * **English**

        Experimental single-spike SRM with a finite-support cubic PSP. See
        :func:`spikingjelly.timing_based.functional.cubic_srm` for dynamics and
        boundaries. Owns only weights and fixed hyperparameters; calls are
        independent with no persistent membrane state. Weights default to
        N(0,0.001²); state_dict uses the weight key. Standard .to supports
        CPU/CUDA; times and weights must remain FP32.

        :param in_features: Positive input count; each connection allows multiple events.
        :type in_features: int
        :param out_features: Positive output count; at most one spike per output per trial.
        :type out_features: int
        :param duration: Finite positive observation endpoint in input-time units.
        :type duration: float
        :param support: Finite positive PSP support length, default 2, in input-time units.
        :type support: float
        :param v_threshold: Finite positive voltage threshold, default 1.
        :type v_threshold: float
        :raises TypeError: For invalid feature or hyperparameter types.
        :raises ValueError: For nonpositive feature counts or invalid hyperparameters.
        """
        super().__init__()
        if any(
            not isinstance(n, int) or isinstance(n, bool)
            for n in (in_features, out_features)
        ):
            raise TypeError("Feature counts must be integers.")
        if min(in_features, out_features) < 1:
            raise ValueError("Feature counts must be positive.")
        functional._cubic_parameters(duration, support, v_threshold)
        self.in_features, self.out_features = in_features, out_features
        self.duration, self.support, self.v_threshold = duration, support, v_threshold
        self.weight = nn.Parameter(
            torch.empty(out_features, in_features, dtype=torch.float32)
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.001)

    def forward(
        self,
        spike_times: torch.Tensor,
        input_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <timing-CubicSRM-forward-cn>` | :ref:`English <timing-CubicSRM-forward-en>`

        ----

        .. _timing-CubicSRM-forward-cn:

        * **中文**

        使用模块权重调用 :func:`spikingjelly.timing_based.functional.cubic_srm`。
        每试次独立；最早上穿输出紧凑事件，沉默无事件，可直接串联下一层。

        :param spike_times: 有限 FP32 ``[E]``，与权重同 dtype/device。
        :type spike_times: torch.Tensor
        :param input_indices: 同设备 int64 ``[E]``，索引输入列，可重复/无序。
        :type input_indices: torch.Tensor
        :param batch_offsets: 同设备 int64 ``[B+1]``，从 0 到 E 非递减。
        :type batch_offsets: torch.Tensor
        :return: CSR ``(times,indices,offsets)``，时刻同输入 dtype/device，
            元数据 int64；按样本/输出排列，每输出至多一个事件。
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: 不支持的 dtype。
        :raises ValueError: 非法事件、设备或 CSR。
        :raises RuntimeError: 状态、根或梯度不可表示。

        ----

        .. _timing-CubicSRM-forward-en:

        * **English**

        Call :func:`spikingjelly.timing_based.functional.cubic_srm` with module
        weights. Independent trials emit compact earliest-upcrossing events;
        silence emits nothing. Directly composable with the next layer.

        :param spike_times: Finite FP32 ``[E]`` matching weight dtype/device.
        :type spike_times: torch.Tensor
        :param input_indices: Same-device int64 ``[E]`` input columns; repeated/unordered allowed.
        :type input_indices: torch.Tensor
        :param batch_offsets: Same-device nondecreasing int64 ``[B+1]`` from 0 to E.
        :type batch_offsets: torch.Tensor
        :return: CSR ``(times,indices,offsets)`` matching input time dtype/device
            and int64 metadata, ordered by trial/output, at most one event per output.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: For unsupported dtypes.
        :raises ValueError: For invalid events, devices or CSR.
        :raises RuntimeError: For unrepresentable states, roots or gradients.
        """
        return functional.cubic_srm(
            spike_times,
            input_indices,
            batch_offsets,
            self.weight,
            self.duration,
            self.support,
            self.v_threshold,
        )


class CubicConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        duration: float,
        support: float = 2.0,
        v_threshold: float = 1.0,
        stride: int = 1,
        padding: int = 0,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <timing-CubicConv2d-cn>` | :ref:`English <timing-CubicConv2d-en>`

        ----

        .. _timing-CubicConv2d-cn:

        * **中文**

        拥有 ``weight[Cout,Cin,k,k]`` 的单发三次PSP卷积神经元。使用
        :func:`~spikingjelly.timing_based.functional.cubic_conv2d` 的局部事件试次，
        不维护跨调用状态；无bias、groups或dilation。权重固定FP32，默认N(0,0.001²)，
        checkpoint键为weight；可用.to在CPU/CUDA之间转换。loss与权重约束由调用方定义。

        :param in_channels: 正整数输入通道数。
        :type in_channels: int
        :param out_channels: 正整数输出通道数。
        :type out_channels: int
        :param kernel_size: 正整数方形核边长。
        :type kernel_size: int
        :param duration: 有限正数窗口终点，与事件同时间单位。
        :type duration: float
        :param support: 有限正数PSP支撑长度，默认2，同时间单位。
        :type support: float
        :param v_threshold: 有限正数阈值，默认1，不改变单位峰值核。
        :type v_threshold: float
        :param stride: 两个空间轴共用的正整数步长，默认1。
        :type stride: int
        :param padding: 两个空间轴共用的非负整数边缘宽度，默认0；padding没有事件。
        :type padding: int
        :raises TypeError: 参数类型非法。
        :raises ValueError: 通道数、空间参数或FP32超参数非法。

        ----

        .. _timing-CubicConv2d-en:

        * **English**

        Single-spike cubic-PSP convolution owning ``weight[Cout,Cin,k,k]``.
        Use local event trials from :func:`~spikingjelly.timing_based.functional.cubic_conv2d`;
        no persistent temporal state, bias, groups or dilation. Weights are FP32,
        initialized to N(0,0.001²), with checkpoint key weight; .to supports CPU/CUDA
        device moves. Callers define losses and weight constraints.

        :param in_channels: Positive integer input channel count.
        :type in_channels: int
        :param out_channels: Positive integer output channel count.
        :type out_channels: int
        :param kernel_size: Positive integer square-kernel side length.
        :type kernel_size: int
        :param duration: Finite positive window endpoint in input-time units.
        :type duration: float
        :param support: Finite positive PSP support in the same time units, default 2.
        :type support: float
        :param v_threshold: Finite positive threshold, default 1, independent of the unit-peak kernel.
        :type v_threshold: float
        :param stride: Positive integer stride shared by both spatial axes, default 1.
        :type stride: int
        :param padding: Nonnegative integer border width shared by both axes, default 0; padded positions have no events.
        :type padding: int
        :raises TypeError: For invalid parameter types.
        :raises ValueError: For invalid channels, spatial parameters or FP32 hyperparameters.
        """
        super().__init__()
        if any(
            not isinstance(x, int) or isinstance(x, bool)
            for x in (in_channels, out_channels)
        ):
            raise TypeError("Channel counts must be integers.")
        if min(in_channels, out_channels) < 1:
            raise ValueError("Channel counts must be positive.")
        functional._conv_parameters(kernel_size, stride, padding)
        functional._cubic_parameters(duration, support, v_threshold)
        self.in_channels, self.out_channels = in_channels, out_channels
        self.kernel_size, self.stride, self.padding = kernel_size, stride, padding
        self.duration, self.support, self.v_threshold = duration, support, v_threshold
        self.weight = nn.Parameter(
            torch.empty(
                out_channels, in_channels, kernel_size, kernel_size, dtype=torch.float32
            )
        )
        nn.init.normal_(self.weight, mean=0.0, std=0.001)

    def forward(
        self,
        spike_times: torch.Tensor,
        input_indices: torch.Tensor,
        batch_offsets: torch.Tensor,
        input_size: tuple[int, int],
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""
        **API Language** - :ref:`中文 <timing-CubicConv2d-forward-cn>` | :ref:`English <timing-CubicConv2d-forward-en>`

        ----

        .. _timing-CubicConv2d-forward-cn:

        * **中文**

        使用共享卷积权重执行 :func:`~spikingjelly.timing_based.functional.cubic_conv2d`。
        独立试次，支持重复输入事件、无序与空样本；padding无事件，窗外梯度为零。
        FP32 CPU/CUDA，保留原CubicSRM完整分段、最早上穿和仅一阶梯度契约。

        :param spike_times: 同权重设备的有限FP32 ``[E]`` 时刻。
        :type spike_times: torch.Tensor
        :param input_indices: 同设备int64 ``[E]``，按 ``(Cin,H,W)`` 展平。
        :type input_indices: torch.Tensor
        :param batch_offsets: 同设备int64 ``[B+1]``，从0到E非递减。
        :type batch_offsets: torch.Tensor
        :param input_size: 正整数输入空间大小 ``(H,W)``。
        :type input_size: tuple[int, int]
        :return: 原B段紧凑CSR ``(times,indices,offsets)``，索引按 ``(Cout,Hout,Wout)``
            展平并按样本/输出排序。每输出至多一发，沉默无事件；时刻同dtype/device。
            Hout/Wout公式见functional接口；结果可传入下一事件层。
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: dtype或输入空间参数类型不支持。
        :raises ValueError: 事件、设备、CSR或空间形状非法。
        :raises RuntimeError: 状态、根或被请求梯度不可表示。

        ----

        .. _timing-CubicConv2d-forward-en:

        * **English**

        Evaluate :func:`~spikingjelly.timing_based.functional.cubic_conv2d` with shared
        convolution weights. Independent trials accept repeated/unordered input events
        and empty samples; padding has no events and outside-window gradients are zero.
        FP32 CPU/CUDA retains CubicSRM's complete partition, earliest upcrossing and
        first-order-only gradient contract.

        :param spike_times: Finite FP32 ``[E]`` times on the weights' device.
        :type spike_times: torch.Tensor
        :param input_indices: Same-device int64 ``[E]`` flattened over ``(Cin,H,W)``.
        :type input_indices: torch.Tensor
        :param batch_offsets: Same-device nondecreasing int64 ``[B+1]`` from 0 to E.
        :type batch_offsets: torch.Tensor
        :param input_size: Positive integer input spatial size ``(H,W)``.
        :type input_size: tuple[int, int]
        :return: Compact CSR ``(times,indices,offsets)`` with original B segments;
            indices flatten ``(Cout,Hout,Wout)`` in trial/output order. At most one
            spike per output; silence emits nothing. Times match input dtype/device.
            See the functional interface for Hout/Wout; composable with another event layer.
        :rtype: tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        :raises TypeError: For unsupported dtypes or spatial parameter types.
        :raises ValueError: For invalid events/devices/CSR/spatial shapes.
        :raises RuntimeError: For unrepresentable states, roots or requested gradients.
        """
        return functional.cubic_conv2d(
            spike_times,
            input_indices,
            batch_offsets,
            self.weight,
            input_size,
            self.duration,
            self.support,
            self.v_threshold,
            self.stride,
            self.padding,
        )
