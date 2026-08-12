from spikingjelly.logger import logger

import torch

from .. import functional, surrogate
from .lif import LIFNode

try:
    from ..triton_kernel.neuron_kernel import ilif as triton_ilif_kernel
except (ImportError, OSError) as e:
    logger.debug("Optional Triton kernel unavailable: {}", e)
    triton_ilif_kernel = None

__all__ = ["ILIFNode"]


class ILIFNode(LIFNode):
    def __init__(
        self,
        tau: float = 4.0 / 3.0,
        v_threshold: float = 1.0,
        surrogate_function: surrogate.MultiLevelSpikeCount = surrogate.MultiLevelSpikeCount(
            4
        ),
        detach_reset: bool = False,
        step_mode: str = "s",
        backend: str = "torch",
        store_v_seq: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <ILIFNode.__init__-cn>` | :ref:`English <ILIFNode.__init__-en>`

        ----

        .. _ILIFNode.__init__-cn:

        * **中文**

        I-LIF 由 ECCV 2024 论文 *Integer-Valued Training and Spike-Driven Inference
        Spiking Neural Network for High-performance and Energy-efficient Object
        Detection* 中提出。本实现保留论文中的整数发放和 soft reset，
        将单位阈值推广为 ``v_threshold``。整数发放上限和替代梯度由
        ``surrogate_function`` 配置。

        令 :math:`H[t]` 为 reset 前膜电位，:math:`V[t]` 为本模块保存的
        reset 后膜电位，:math:`C[t]` 为整数发放计数。单步动力学为

        .. math::

            H[t] = \left(1 - \frac{1}{\tau}\right) V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(H[t] / V_{th}, 0, D))

        .. math::

            V[t] = H[t] - C[t] V_{th}

        其中 :math:`D` 为 ``surrogate_function.max_spike_count``。该充电过程
        直接复用 :class:`LIFNode` 在 ``decay_input=False``、``v_reset=None``
        时的实现；论文及作者代码中的衰减系数 :math:`\beta = 1 - 1 / \tau`。
        作者代码默认 :math:`\beta=0.25`，对应本类默认的 :math:`\tau=4/3`。
        ``train()`` 和 ``eval()`` 的前向语义相同，均返回
        :math:`C[t] \in \{0, 1, \ldots, D\}`，输入和输出的逻辑时间步数相同。

        需要梯度时使用矩形窗口直通估计；默认仅当
        :math:`H[t] / V_{th} \in [0, D]` 时传递梯度。可在构造
        ``surrogate_function`` 时修改该区间。

        论文所称的 spike-driven inference 需要由部署端将每个整数计数
        展开为 :math:`D` 个二值槽，其中恰有 :math:`C[t]` 个单位事件：

        .. math::

            C[t] = \sum_{d=1}^{D} S[t, d], \qquad S[t, d] \in \{0, 1\},

        线性权重运算满足

        .. math::

            W C[t] = \sum_{d=1}^{D} W S[t, d].

        因此每个单位事件可触发一次权重累加（AC），无需执行整数激活与权重的
        乘加（MAC）。其中 :math:`t` 是模型的逻辑时间步；而 :math:`d` 是部署端的
        unary/thermometer 事件槽，不参与神经元状态更新。:class:`ILIFNode`
        只输出整数计数，不负责二值脉冲展开和累加。

        :class:`~spikingjelly.activation_based.layer.misc.SpikeCountToBinary`
        可以将整数计数展开到时间维，随后由无 bias 权重层处理二值事件；
        :class:`~spikingjelly.activation_based.layer.misc.TemporalBinSum`
        将权重层输出还原到逻辑时间维。两者默认在训练模式保持恒等映射、在
        eval 模式执行转换，从而支持整数训练和二值脉冲推理。

        .. warning::

            I-LIF 不是传统的二值脉冲神经元。从数值上看，其发放函数是带 LIF
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
            执行一次。使用 SpikingJelly 提供的转换层时，中间权重层必须设置
            ``bias=False``；若需要 bias，应在
            :class:`~spikingjelly.activation_based.layer.misc.TemporalBinSum`
            之后单独加入。标准流程是
            ``I-LIF - SpikeCountToBinary - bias-free weight - TemporalBinSum``。

            同一逻辑时间步下，I-LIF 每个位置携带 :math:`D+1` 个幅值等级，并可
            触发最多 :math:`D` 个事件；它与每步最多一个事件的二值 IF/LIF
            不是同等信息或执行预算。公平比较至少应同时给出两类基线：相同
            :math:`D+1` 级量化和整数 MAC 的 QANN 基线，用于判断事件展开是否
            真正降低部署代价；以及在相同物理槽、总事件、延迟或能耗预算下的
            二值 IF/LIF 基线，而不只是令逻辑 :math:`T` 相同。

        :param tau: 膜电位时间常数。论文作者代码默认的衰减系数 0.25 对应
            ``tau=4/3``
        :type tau: float
        :param v_threshold: 神经元阈值电压，默认为 1.0
        :type v_threshold: float
        :param surrogate_function: 多级整数发放函数，必须是 ``spiking=True`` 的
            :class:`~spikingjelly.activation_based.surrogate.MultiLevelSpikeCount`。
            ``max_spike_count`` 和矩形替代梯度区间均由该对象配置。默认为
            ``MultiLevelSpikeCount(4)``，对应 :math:`D=4` 和梯度区间 ``[0, 4]``
        :type surrogate_function: surrogate.MultiLevelSpikeCount
        :param detach_reset: 是否在反向传播时分离 reset 中的发放值，默认为
            ``False``
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步，``"m"`` 为多步，默认为
            ``"s"``
        :type step_mode: str
        :param backend: 后端名称。单步模式仅支持 ``"torch"``；多步模式支持
            ``"torch"`` 和 ``"triton"``，默认为 ``"torch"``
        :type backend: str
        :param store_v_seq: 多步模式下是否保存每个输入步后的膜电位，默认为
            ``False``
        :type store_v_seq: bool
        :raises ValueError: 当 ``surrogate_function`` 不是
            ``MultiLevelSpikeCount`` 或 ``surrogate_function.spiking=False`` 时

        ----

        .. _ILIFNode.__init__-en:

        * **English**

        I-LIF was introduced in the ECCV 2024 paper *Integer-Valued Training and
        Spike-Driven Inference Spiking Neural Network for High-performance and
        Energy-efficient Object Detection*. This implementation keeps the integer
        firing and soft reset used in the paper and generalizes its unit threshold
        to ``v_threshold``. The integer firing limit and surrogate gradient are
        configured by ``surrogate_function``.

        Let :math:`H[t]` be the membrane potential before reset, :math:`V[t]` the
        value stored after reset, and :math:`C[t]` the integer firing count. The
        single-step dynamics are

        .. math::

            H[t] = \left(1 - \frac{1}{\tau}\right) V[t - 1] + X[t]

        .. math::

            C[t] = \operatorname{round}(\operatorname{clip}(H[t] / V_{th}, 0, D))

        .. math::

            V[t] = H[t] - C[t] V_{th}

        Here :math:`D` is ``surrogate_function.max_spike_count``. Charging directly
        reuses the :class:`LIFNode` implementation with ``decay_input=False`` and
        ``v_reset=None``. The decay factor used by the paper and its reference code
        is :math:`\beta = 1 - 1 / \tau`; its default :math:`\beta=0.25` maps to this
        class's default :math:`\tau=4/3`. ``train()`` and ``eval()`` have the same
        forward semantics:
        both return :math:`C[t] \in \{0, 1, \ldots, D\}` and preserve the number of
        logical timesteps.

        When gradients are required, a rectangular straight-through estimator is
        used. By default, gradients pass where
        :math:`H[t] / V_{th} \in [0, D]`; this interval can be changed when
        constructing ``surrogate_function``.

        The spike-driven inference described in the paper requires the deployment
        layer to expand each integer count into :math:`D` binary slots, exactly
        :math:`C[t]` of which contain unit events:

        .. math::

            C[t] = \sum_{d=1}^{D} S[t, d], \qquad S[t, d] \in \{0, 1\},

        A linear weight operation satisfies

        .. math::

            W C[t] = \sum_{d=1}^{D} W S[t, d].

        Each unit event can therefore trigger one weight accumulation (AC), without
        a multiply-accumulate (MAC) between an integer activation and its weight.
        Here :math:`t` is a model logical timestep, while :math:`d` is a deployment
        unary/thermometer event slot and does not update the neuronal state.
        :class:`ILIFNode` returns integer counts; it does not expand binary spikes
        or accumulate across them.

        :class:`~spikingjelly.activation_based.layer.misc.SpikeCountToBinary`
        expands the integer counts along the time dimension so that a bias-free
        weight layer processes binary events. Then
        :class:`~spikingjelly.activation_based.layer.misc.TemporalBinSum`
        restores the logical time dimension. By default, both modules are identity
        mappings in training mode and perform their conversions in evaluation mode,
        enabling integer-valued training and binary-spike inference.

        .. warning::

            I-LIF is not a conventional binary spiking neuron. Numerically, its
            firing function is a :math:`D+1`-level activation quantizer with LIF
            state recurrence and soft reset. Directly evaluating :math:`W C[t]` is
            integer-MAC inference; the downstream weight operation becomes
            spike-driven only after the event expansion described above.

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
            nonlinear operations should likewise run once after accumulation. When
            using the SpikingJelly conversion layers, the intervening weight layer
            must set ``bias=False``; add any required bias once after
            :class:`~spikingjelly.activation_based.layer.misc.TemporalBinSum`. The
            standard flow is
            ``I-LIF - SpikeCountToBinary - bias-free weight - TemporalBinSum``.

            At the same logical timestep count, I-LIF carries :math:`D+1`
            amplitude levels per position and may trigger up to :math:`D` events.
            It therefore does not have the same information or execution budget
            as a binary IF/LIF neuron that emits at most one event per step. A
            fair evaluation should include both a QANN baseline with the same
            :math:`D+1` quantization levels and integer MACs, to isolate whether
            event expansion reduces deployment cost, and a binary IF/LIF baseline
            under the same physical-slot, event, latency, or energy budget rather
            than only the same logical :math:`T`.

        :param tau: Membrane time constant. The reference code's default decay
            factor 0.25 maps to ``tau=4/3``
        :type tau: float
        :param v_threshold: Threshold voltage; defaults to 1.0
        :type v_threshold: float
        :param surrogate_function: Multi-level integer firing function. It must be a
            :class:`~spikingjelly.activation_based.surrogate.MultiLevelSpikeCount`
            with ``spiking=True``. Its ``max_spike_count`` and rectangular
            surrogate-gradient window define the corresponding neuron settings.
            Defaults to ``MultiLevelSpikeCount(4)``, giving :math:`D=4` and the
            gradient window ``[0, 4]``
        :type surrogate_function: surrogate.MultiLevelSpikeCount
        :param detach_reset: Whether to detach the emitted value used by reset in
            backward; defaults to ``False``
        :type detach_reset: bool
        :param step_mode: Step mode, ``"s"`` or ``"m"``; defaults to ``"s"``
        :type step_mode: str
        :param backend: Backend name. Single-step mode supports ``"torch"`` only;
            multi-step mode supports ``"torch"`` and ``"triton"``; defaults to
            ``"torch"``
        :type backend: str
        :param store_v_seq: Whether to store membrane voltage after each input
            step in multi-step mode; defaults to ``False``
        :type store_v_seq: bool
        :raises ValueError: If ``surrogate_function`` is not
            ``MultiLevelSpikeCount`` or if ``surrogate_function.spiking=False``
        """
        if (
            not isinstance(surrogate_function, surrogate.MultiLevelSpikeCount)
            or not surrogate_function.spiking
        ):
            raise ValueError(
                "surrogate_function must be a spiking MultiLevelSpikeCount."
            )

        super().__init__(
            tau=tau,
            decay_input=False,
            v_threshold=v_threshold,
            v_reset=None,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            backend=backend,
            store_v_seq=store_v_seq,
        )

    @property
    def supported_backends(self) -> tuple[str, ...]:
        return ("torch",) if self.step_mode == "s" else ("torch", "triton")

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x = inputs[0]
        v = states[0]

        spike, v = functional.ilif_step(
            x,
            v,
            self.tau,
            self.v_threshold,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])

    def multi_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        x_seq = inputs[0]
        v = states[0]

        if self.backend == "triton":
            if triton_ilif_kernel is None:
                raise ImportError(
                    "ILIFNode backend='triton' requires the optional Triton backend."
                )
            spike_seq, v_out = triton_ilif_kernel._multistep_ilif(
                x_seq,
                v,
                1.0 - 1.0 / self.tau,
                self.v_threshold,
                self.surrogate_function.max_spike_count,
                self.surrogate_function.grad_min,
                self.surrogate_function.grad_max,
                self.detach_reset,
                False,
            )
            v = v_out
        elif self.backend == "torch":
            return super().multi_step_functional_forward(inputs, states, **kwargs)
        else:
            raise ValueError(self.backend)

        return (spike_seq,), (v,)

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        if not self.store_v_seq or self.backend == "torch":
            return super().multi_step_forward(x_seq, *args, **kwargs)

        states = self.materialize_states(
            (x_seq, *args), tuple(self._memories.values()), "m"
        )
        if triton_ilif_kernel is None:
            raise ImportError(
                "ILIFNode backend='triton' requires the optional Triton backend."
            )
        spike_seq, v_seq = triton_ilif_kernel._multistep_ilif(
            x_seq,
            states[0],
            1.0 - 1.0 / self.tau,
            self.v_threshold,
            self.surrogate_function.max_spike_count,
            self.surrogate_function.grad_min,
            self.surrogate_function.grad_max,
            self.detach_reset,
            True,
        )
        self.v = v_seq[-1].clone()
        self.v_seq = v_seq
        return spike_seq
