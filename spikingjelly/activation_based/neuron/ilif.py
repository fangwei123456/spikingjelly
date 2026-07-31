import logging
import numbers
from typing import Optional, Tuple

import torch

from .. import surrogate
from .base_node import BaseNode

try:
    from ..triton_kernel.neuron_kernel import ilif as triton_ilif_kernel
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.neuron.ilif: {e}")
    triton_ilif_kernel = None

__all__ = ["ILIFNode"]


class ILIFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        max_spike_count: int = 4,
        decay: float = 0.25,
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
        grad_window: Optional[Tuple[float, float]] = None,
    ):
        r"""
        **API Language** - :ref:`中文 <ILIFNode.__init__-cn>` | :ref:`English <ILIFNode.__init__-en>`

        ----

        .. _ILIFNode.__init__-cn:

        * **中文**

        I-LIF 由 Xinhao Luo、Man Yao、Yuhong Chou、Bo Xu 和 Guoqi Li 在
        ECCV 2024 论文 *Integer-Valued Training and Spike-Driven Inference
        Spiking Neural Network for High-performance and Energy-efficient Object
        Detection* 中提出
        （`论文 <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04704.pdf>`__，
        `代码 <https://github.com/BICLab/SpikeYOLO>`__）。本实现保留论文中的
        整数发放和 soft reset，将单位阈值推广为 ``v_threshold``，并允许配置
        矩形替代梯度区间。

        令 :math:`U[t]` 为 reset 前膜电位，:math:`V[t]` 为本模块保存的
        reset 后膜电位，:math:`C[t]` 为整数发放计数。单步动力学为

        .. math::

            U[t] = \beta V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(U[t] / V_{th}, 0, D))

        .. math::

            V[t] = U[t] - C[t] V_{th}

        其中 :math:`D` 为 ``max_spike_count``，:math:`\beta` 为 ``decay``。
        ``decay`` 位于充电过程，其形式与 :class:`LIFNode` 在
        ``decay_input=False`` 且 ``v_reset=None`` 时的 charge 方程一致。
        ``train()`` 和 ``eval()`` 的前向语义相同，均返回
        :math:`C[t] \in \{0, 1, \ldots, D\}`，输入和输出的逻辑时间步数相同。
        需要梯度时使用矩形窗口直通估计；默认仅当
        :math:`U[t] / V_{th} \in [0, D]` 时传递梯度，可通过 ``grad_window``
        修改区间。调用 :meth:`reset` 会将保存的膜电位恢复为初始值。
        ``step_mode="s"`` 接收形状 ``[N, *]`` 的单步浮点张量；
        ``step_mode="m"`` 接收形状 ``[T, N, *]`` 的多步浮点张量。输出与输入的
        shape、dtype 和 device 相同。Torch 后端可按输入所在设备执行；Triton
        后端支持 CUDA 单步和多步 FP32、FP16 或 BF16 张量。

        论文所称的 spike-driven inference 需要由部署端将每个整数计数无损地
        展开为 :math:`D` 个二值槽，其中恰有 :math:`C[t]` 个单位事件：

        .. math::

            C[t] = \sum_{d=1}^{D} S[t, d], \qquad S[t, d] \in \{0, 1\},

        线性权重运算满足

        .. math::

            W C[t] = \sum_{d=1}^{D} W S[t, d].

        因此每个单位事件可触发一次权重累加（AC），无需执行整数激活与权重的
        乘加（MAC）。其中 :math:`t` 是模型的逻辑时间步，:math:`d` 是部署端的
        unary/thermometer 事件槽，不参与神经元状态更新。:class:`ILIFNode`
        只输出整数计数，不负责事件展开和跨事件累加。

        .. warning::

            I-LIF 不是传统的二值脉冲神经元。传统 IF/LIF 在每个逻辑时间步、
            每个神经元位置最多输出一个 :math:`0/1` 脉冲；I-LIF 输出
            :math:`0` 到 :math:`D` 的整数。从数值上看，其发放函数是带 LIF
            状态递推和 soft reset 的 :math:`D+1` 级激活量化器。直接计算
            :math:`W C[t]` 属于整数 MAC 推理；只有部署端完成上述事件展开后，
            下游权重运算才是 spike-driven。

            若下游卷积或线性层带有 bias :math:`b`，并且部署端固定执行全部
            :math:`D` 个事件槽（包括零值槽），则未经处理的 bias 会被累加
            :math:`D` 次：

            .. math::

                \sum_{d=1}^{D} (W S[t, d] + b)
                = W C[t] + D b \ne W C[t] + b.

            此时每个槽应使用 :math:`b_{\mathrm{slot}} = b / D`：

            .. math::

                \sum_{d=1}^{D} (W S[t, d] + b / D)
                = W C[t] + b.

            该缩放依赖固定的槽数。若稀疏硬件只在非零事件到达时执行突触累加，
            就不存在固定的 :math:`D` 次 bias 加法；把 :math:`b / D` 绑定到
            每个事件会使 bias 项随事件数变化。对上面的标量计数，结果是
            :math:`W C[t] + C[t] b / D`。这时应先做无 bias 的事件累加，再在
            逻辑时间步结束时加入一次 :math:`b`；归一化和非线性也应在累加完成后
            执行一次。

            事件串行展开最多需要每个逻辑时间步 :math:`D` 个物理槽，增加延迟；
            并行展开则增加事件通道、累加器或带宽。只支持每个神经元每步一个
            二值事件的硬件不能直接执行 I-LIF。与传统脉冲神经元比较时，除保持
            数据、网络规模、训练/评测协议和逻辑时间步 :math:`T` 一致外，还应
            报告 :math:`D`、总事件数 :math:`\sum_{t,n} C_n[t]`、突触 AC/SOP
            数、端到端延迟、能耗和额外硬件资源。串行实现的延迟口径应计入最多
            :math:`T D` 个物理槽，并行实现则应计入相应资源。缺少这些数据时，
            仅比较准确率或逻辑时间步不能说明 I-LIF 更高效。

            同一逻辑时间步下，I-LIF 每个位置携带 :math:`D+1` 个幅值等级，并可
            触发最多 :math:`D` 个事件；它与每步最多一个事件的二值 IF/LIF
            不是同等信息或执行预算。公平比较至少应同时给出两类基线：相同
            :math:`D+1` 级量化和整数 MAC 的 QANN 基线，用于判断事件展开是否
            真正降低部署代价；以及在相同物理槽、总事件、延迟或能耗预算下的
            二值 IF/LIF 基线，而不只是令逻辑 :math:`T` 相同。

            当 :math:`T=1` 且输入为静态激活时，跨时间的膜状态递推几乎没有发挥
            作用，结果主要反映多级量化，不应据此声称获得了时序 SNN 的优势。
            硬件效率结论还应计入或实测计数到事件的展开、事件路由、存储访问、
            缓冲与累加器、膜状态更新，以及 bias、归一化和非线性的调度开销；
            单独报告 AC/SOP 数不足以证明端到端能效。当前实现只定义数值计数
            语义，不实现这些部署步骤，也不估算硬件能耗。

        :param v_threshold: 神经元阈值电压，必须为有限正数
        :type v_threshold: float
        :param max_spike_count: 单个逻辑步最多释放的整数计数 ``D``
        :type max_spike_count: int
        :param decay: 充电过程中历史膜电位的衰减 ``beta``，必须位于 ``[0, 1]``。
            当 ``decay=1`` 时退化为无泄漏的整数 IF 形式
        :type decay: float
        :param detach_reset: 是否在反向传播时分离 reset 中的发放值
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步，``"m"`` 为多步
        :type step_mode: str
        :param backend: 后端名称。单步和多步模式均支持 ``"torch"`` 和
            ``"triton"``
        :type backend: str
        :param store_v_seq: 多步模式下是否保存每个输入步后的膜电位
        :type store_v_seq: bool
        :param grad_window: 矩形直通梯度的闭区间。默认值 ``None`` 表示
            ``[0, max_spike_count]``
        :type grad_window: Optional[Tuple[float, float]]
        :raises TypeError: 当 ``v_threshold`` 或 ``decay`` 不是实数，或
            ``max_spike_count`` 不是整数时
        :raises ValueError: 当 ``v_threshold``、``max_spike_count``、``decay``、
            ``grad_window``、``step_mode`` 或 ``backend`` 的取值无效时

        ----

        .. _ILIFNode.__init__-en:

        * **English**

        I-LIF was introduced by Xinhao Luo, Man Yao, Yuhong Chou, Bo Xu, and
        Guoqi Li in the ECCV 2024 paper *Integer-Valued Training and Spike-Driven
        Inference Spiking Neural Network for High-performance and Energy-efficient
        Object Detection*
        (`paper <https://www.ecva.net/papers/eccv_2024/papers_ECCV/papers/04704.pdf>`__,
        `code <https://github.com/BICLab/SpikeYOLO>`__). This implementation keeps
        the integer firing and soft reset used in the paper, generalizes its unit
        threshold to ``v_threshold``, and allows the rectangular surrogate-gradient
        interval to be configured.

        Let :math:`U[t]` be the membrane potential before reset, :math:`V[t]` the
        value stored after reset, and :math:`C[t]` the integer firing count. The
        single-step dynamics are

        .. math::

            U[t] = \beta V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(U[t] / V_{th}, 0, D))

        .. math::

            V[t] = U[t] - C[t] V_{th}

        Here :math:`D` is ``max_spike_count`` and :math:`\beta` is ``decay``.
        ``decay`` is applied during charging, matching the :class:`LIFNode` charge
        equation when ``decay_input=False`` and ``v_reset=None``. ``train()`` and
        ``eval()`` have the same forward semantics: both return
        :math:`C[t] \in \{0, 1, \ldots, D\}` and preserve the number of logical
        timesteps. When gradients are required, a rectangular straight-through
        estimator is used. By default, gradients pass where
        :math:`U[t] / V_{th} \in [0, D]`; ``grad_window`` changes this interval.
        Calling :meth:`reset` restores the stored membrane potential.
        ``step_mode="s"`` accepts a single-step floating-point tensor with shape
        ``[N, *]``; ``step_mode="m"`` accepts a multi-step floating-point tensor
        with shape ``[T, N, *]``. Output shape, dtype, and device match the input.
        The Torch backend follows the input device; the Triton backend supports
        single-step and multi-step FP32, FP16, or BF16 CUDA tensors.

        The spike-driven inference described in the paper requires the deployment
        layer to losslessly expand each integer count into :math:`D` binary slots,
        exactly :math:`C[t]` of which contain unit events:

        .. math::

            C[t] = \sum_{d=1}^{D} S[t, d], \qquad S[t, d] \in \{0, 1\},

        A linear weight operation satisfies

        .. math::

            W C[t] = \sum_{d=1}^{D} W S[t, d].

        Each unit event can therefore trigger one weight accumulation (AC), without
        a multiply-accumulate (MAC) between an integer activation and its weight.
        Here :math:`t` is a model logical timestep, while :math:`d` is a deployment
        unary/thermometer event slot and does not update the neuronal state.
        :class:`ILIFNode` returns integer counts; it does not expand events or
        accumulate across them.

        .. warning::

            I-LIF is not a conventional binary spiking neuron. A conventional
            IF/LIF neuron emits at most one :math:`0/1` spike per neuron position
            and logical timestep, whereas I-LIF emits an integer from :math:`0` to
            :math:`D`. Numerically, its firing function is a :math:`D+1`-level
            activation quantizer with LIF state recurrence and soft reset. Directly
            evaluating :math:`W C[t]` is integer-MAC inference; the downstream
            weight operation becomes spike-driven only after the event expansion
            described above.

            If the following convolution or linear layer has bias :math:`b`, and
            the deployment executes all :math:`D` event slots, including zero
            slots, an unmodified bias is accumulated :math:`D` times:

            .. math::

                \sum_{d=1}^{D} (W S[t, d] + b)
                = W C[t] + D b \ne W C[t] + b.

            The bias used in each slot should instead be
            :math:`b_{\mathrm{slot}} = b / D`:

            .. math::

                \sum_{d=1}^{D} (W S[t, d] + b / D)
                = W C[t] + b.

            This scaling assumes a fixed number of slots. If sparse hardware
            performs synaptic accumulation only when a nonzero event arrives, there
            are no longer exactly :math:`D` bias additions. Attaching
            :math:`b / D` to each event makes the bias contribution depend on the
            event count; for the scalar count above, the result is
            :math:`W C[t] + C[t] b / D`. In that case, events should first be
            accumulated with a bias-free weight operation, and :math:`b` should be
            added once at the end of the logical timestep. Normalization and
            nonlinear operations should likewise run once after accumulation.

            Serial event expansion requires up to :math:`D` physical slots per
            logical timestep and increases latency. Parallel expansion requires
            additional event lanes, accumulators, or bandwidth. Hardware that
            accepts only one binary event per neuron and timestep cannot execute
            I-LIF directly. A comparison with conventional spiking neurons should
            keep the data, model size, training/evaluation protocol, and logical
            timestep count :math:`T` fixed, while also reporting :math:`D`, total
            event count :math:`\sum_{t,n} C_n[t]`, synaptic AC/SOP count,
            end-to-end latency, energy, and additional hardware resources. Serial
            implementations should account for up to :math:`T D` physical slots;
            parallel implementations should account for their additional
            resources. Accuracy or logical timestep count alone is insufficient
            evidence of greater efficiency.

            At the same logical timestep count, I-LIF carries :math:`D+1`
            amplitude levels per position and may trigger up to :math:`D` events.
            It therefore does not have the same information or execution budget
            as a binary IF/LIF neuron that emits at most one event per step. A
            fair evaluation should include both a QANN baseline with the same
            :math:`D+1` quantization levels and integer MACs, to isolate whether
            event expansion reduces deployment cost, and a binary IF/LIF baseline
            under the same physical-slot, event, latency, or energy budget rather
            than only the same logical :math:`T`.

            With :math:`T=1` and static activations, temporal membrane recurrence
            has little opportunity to contribute; such a result mainly evaluates
            multilevel quantization and should not be presented as an advantage
            of temporal SNN computation. Hardware-efficiency claims should also
            measure or account for count-to-event expansion, event routing,
            memory traffic, buffering and accumulation, membrane updates, and
            scheduling of bias, normalization, and nonlinear operations. AC/SOP
            count alone does not establish end-to-end energy efficiency. This
            implementation defines only the numerical count semantics; it does
            not perform these deployment steps or estimate hardware energy.

        :param v_threshold: Threshold voltage, which must be finite and positive
        :type v_threshold: float
        :param max_spike_count: Maximum integer count ``D`` emitted in one
            logical step
        :type max_spike_count: int
        :param decay: Historical membrane decay ``beta`` in charge, in ``[0, 1]``.
            ``decay=1`` degenerates to a non-leaky integer IF form
        :type decay: float
        :param detach_reset: Whether to detach the emitted value used by reset in
            backward
        :type detach_reset: bool
        :param step_mode: Step mode, ``"s"`` or ``"m"``
        :type step_mode: str
        :param backend: Backend name. Both modes support ``"torch"`` and
            ``"triton"``
        :type backend: str
        :param store_v_seq: Whether to store membrane voltage after each input
            step in multi-step mode
        :type store_v_seq: bool
        :param grad_window: Closed interval for the rectangular straight-through
            gradient. ``None`` uses ``[0, max_spike_count]``
        :type grad_window: Optional[Tuple[float, float]]
        :raises TypeError: If ``v_threshold`` or ``decay`` is not real, or
            ``max_spike_count`` is not an integer
        :raises ValueError: If ``v_threshold``, ``max_spike_count``, ``decay``,
            ``grad_window``, ``step_mode``, or ``backend`` has an invalid value
        """
        if not isinstance(v_threshold, numbers.Real):
            raise TypeError("v_threshold must be a real number.")
        v_threshold = float(v_threshold)
        if not torch.isfinite(torch.tensor(v_threshold)) or v_threshold <= 0.0:
            raise ValueError("v_threshold must be finite positive.")
        if not isinstance(max_spike_count, numbers.Integral) or isinstance(
            max_spike_count, bool
        ):
            raise TypeError("max_spike_count must be an integer.")
        max_spike_count = int(max_spike_count)
        if max_spike_count < 1:
            raise ValueError("max_spike_count must be >= 1.")
        if not isinstance(decay, numbers.Real):
            raise TypeError("decay must be a real number.")
        decay = float(decay)
        if not torch.isfinite(torch.tensor(decay)) or decay < 0.0 or decay > 1.0:
            raise ValueError("decay must be finite and in [0, 1].")

        super().__init__(
            v_threshold=v_threshold,
            v_reset=None,
            surrogate_function=surrogate.MultiLevelSpikeCount(
                max_spike_count,
                grad_window=grad_window,
            ),
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )
        self.decay = decay

    @property
    def supported_backends(self):
        if self.step_mode in ("s", "m"):
            return ("torch", "triton")
        raise ValueError(self.step_mode)

    def neuronal_charge(self, x: torch.Tensor):
        self.v = self.decay * self.v + x

    def neuronal_fire(self):
        return self.surrogate_function(self.v / self.v_threshold)

    def neuronal_reset(self, spike):
        if self.detach_reset:
            spike = spike.detach()
        self.v = self.v - spike * self.v_threshold

    def single_step_forward(self, x: torch.Tensor):
        self.v_float_to_tensor(x)
        if self.backend == "triton":
            spike, self.v = triton_ilif_kernel.single_step_ilif(
                x,
                self.v,
                self.decay,
                self.v_threshold,
                self.surrogate_function.max_spike_count,
                self.surrogate_function.grad_min,
                self.surrogate_function.grad_max,
                self.detach_reset,
            )
            return spike
        self.neuronal_charge(x)
        spike = self.neuronal_fire()
        self.neuronal_reset(spike)
        return spike

    def multi_step_forward(self, x_seq: torch.Tensor):
        if self.backend == "triton":
            self.v_float_to_tensor(x_seq[0])
            spike_seq, v_out = triton_ilif_kernel.multistep_ilif(
                x_seq,
                self.v,
                self.decay,
                self.v_threshold,
                self.surrogate_function.max_spike_count,
                self.surrogate_function.grad_min,
                self.surrogate_function.grad_max,
                self.detach_reset,
                self.store_v_seq,
            )
            if self.store_v_seq:
                self.v_seq = v_out
                self.v = v_out[-1].clone()
            else:
                self.v = v_out
            return spike_seq

        spike_seq = []
        if self.store_v_seq:
            v_seq = []
        for t in range(x_seq.shape[0]):
            spike_seq.append(self.single_step_forward(x_seq[t]))
            if self.store_v_seq:
                v_seq.append(self.v)
        if self.store_v_seq:
            self.v_seq = torch.stack(v_seq)
        return torch.stack(spike_seq)

    def extra_repr(self):
        return (
            f"v_threshold={self.v_threshold}, "
            f"max_spike_count={self.surrogate_function.max_spike_count}, "
            f"grad_window=({self.surrogate_function.grad_min}, "
            f"{self.surrogate_function.grad_max}), "
            f"decay={self.decay}, detach_reset={self.detach_reset}, "
            f"step_mode={self.step_mode}, backend={self.backend}"
        )
