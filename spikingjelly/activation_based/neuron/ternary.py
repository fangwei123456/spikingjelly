import math
from typing import Optional

import torch

from .. import functional, surrogate
from .base_node import BaseNode

__all__ = ["TernaryIFNode", "TernaryLIFNode"]


def _validate_ternary_parameters(
    v_threshold: float, v_reset: Optional[float], detach_reset: bool
) -> None:
    if not isinstance(v_threshold, float):
        raise TypeError("v_threshold must be a float.")
    if not math.isfinite(v_threshold) or v_threshold <= 0.0:
        raise ValueError("v_threshold must be finite and positive.")
    if v_reset is not None:
        if not isinstance(v_reset, float):
            raise TypeError("v_reset must be a float or None.")
        if not math.isfinite(v_reset) or not -v_threshold < v_reset < v_threshold:
            raise ValueError("v_reset must lie strictly between the two thresholds.")
    if not isinstance(detach_reset, bool):
        raise TypeError("detach_reset must be a bool.")


def _ternary_fire_and_reset(
    h: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    positive = surrogate_function(h - v_threshold)
    negative = surrogate_function(-h - v_threshold)
    spike = positive - negative
    if v_reset is None:
        spike_reset = spike.detach() if detach_reset else spike
        v = BaseNode.apply_soft_reset(h, spike_reset, v_threshold)
    else:
        # The two branch decisions retain their surrogate gradients when silent.
        # abs(spike) has the same discrete value but a different gradient at zero.
        reset_mask = positive + negative
        if detach_reset:
            reset_mask = reset_mask.detach()
        v = BaseNode.apply_hard_reset(h, reset_mask, v_reset)
    return spike, v


class TernaryIFNode(BaseNode):
    def __init__(
        self,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode: str = "s",
        store_v_seq: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <TernaryIFNode.__init__-cn>` | :ref:`English <TernaryIFNode.__init__-en>`

        ----

        .. _TernaryIFNode.__init__-cn:

        * **中文**

        基础的有符号三值 Integrate-and-Fire 神经元。输入为浮点张量，单步
        形状为 ``[N, *]``，多步形状为 ``[T, N, *]``。使用原生 PyTorch
        状态转移，训练与评估执行相同的方程；没有专用的加速 kernel。
        在替代函数的 ``spiking=True`` 模式下，每一步只输出 ``-1``、
        ``0`` 或 ``1``，形状、dtype 和 device 与输入相同：

        .. math::
            H[t] &= V[t-1] + X[t], \\
            P[t] &= \Theta(H[t] - V_{threshold}), \\
            N[t] &= \Theta(-H[t] - V_{threshold}), \\
            S[t] &= P[t] - N[t].

        这里 :math:`\Theta(0)=1` ，所以两个阈值均包含等号。令
        :math:`R[t]=P[t]+N[t]` ，硬重置为
        :math:`V[t]=(1-R[t])H[t]+R[t]V_{reset}` ；``v_reset=None`` 时，
        软重置为 :math:`V[t]=H[t]-V_{threshold}S[t]` 。任一方向的硬重置
        都到同一个 ``v_reset``。这些规则是算法模型，不是离子通道推导。

        反向传播分别对两个阈值使用 ``surrogate_function``。硬重置保留
        ``P + N`` 的替代梯度，而不对 ``abs(S)`` 求导；二者在离散前向中
        等价，但后者在 ``S=0`` 处会产生不同的重置梯度。``detach_reset``
        只分离重置中的 ``R`` 或 ``S``，不分离输出脉冲的梯度。如果替代
        函数使用 ``spiking=False``，则使用其连续原函数，输出和重置掩码
        也连续，不再保证三值输出。

        本模型采用 `Ternary Spike: Learning Ternary Spikes for Spiking Neural
        Networks (Guo et al., AAAI 2024)
        <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_
        的对称三值事件定义，并使用无泄漏的 IF 充电方程。软重置和非零
        重置是库扩展；不实现论文的可学习脉冲幅值或训练用 STE。
        :meth:`reset` 将电位恢复为 ``v_reset``（软重置时为 ``0.0``），
        并清空 ``v_seq``。``v`` 保存重置后的电位。

        有符号输出适用于普通稠密 PyTorch 层。不要用 ``memopt`` 中的
        ``BooleanSpikeCompressor``、``Uint8SpikeCompressor``、
        ``BitSpikeCompressor`` 或 ``SparseSpikeCompressor`` 压缩这些输出；
        使用 :func:`~spikingjelly.activation_based.memopt.optimize_memory` 时
        设置 ``compress=False``，使用
        :func:`~spikingjelly.activation_based.memopt.checkpoint_module` 时设置
        ``compressor=None``。检测任一极性的首次事件时，应先取绝对值，
        再按 :func:`~spikingjelly.activation_based.functional.first_spike_index`
        的时间维约定传入。默认二值 SynOps 计数和能耗模型不适用于有符号事件。
        本类不支持 NIR 或 Lava 导出。

        :param v_threshold: 对称阈值的正幅值，默认 ``1.0``，必须为有限正浮点数。
        :type v_threshold: float
        :param v_reset: 两个方向共用的硬重置电位，默认 ``0.0``，必须是严格
            位于两个阈值之间的有限浮点数。``None`` 表示有符号软重置。
        :type v_reset: Optional[float]
        :param surrogate_function: 两个阈值共用的替代梯度函数，默认
            :class:`~spikingjelly.activation_based.surrogate.Sigmoid` 。
            ``spiking=True`` 时前向输出离散事件，``False`` 时输出连续值。
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否分离重置掩码或有符号脉冲的梯度，默认 ``False``。
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步（默认），``"m"`` 为多步。
        :type step_mode: str
        :param store_v_seq: 是否在 ``v_seq`` 中记录每一步重置后的电位，默认
            ``False``。单步模式逐步追加，直到 ``reset()``；多步模式记录当前序列。
        :type store_v_seq: bool
        :raises TypeError: 阈值或非空重置电位不是浮点数，或 ``detach_reset`` 不是布尔值。
        :raises ValueError: 阈值不是有限正数、重置电位不在开放阈值区间内，
            或 ``step_mode`` 不是 ``"s"`` 或 ``"m"``。

        .. code-block:: python

            import torch
            from spikingjelly.activation_based import neuron

            node = neuron.TernaryIFNode()
            node(torch.tensor([-1.0, 0.0, 1.0]))
            # tensor([-1., 0., 1.])
            node.reset()

        ----

        .. _TernaryIFNode.__init__-en:

        * **English**

        A basic signed ternary Integrate-and-Fire neuron. Floating-point inputs
        have shape ``[N, *]`` in single-step mode and ``[T, N, *]`` in multi-step
        mode. The native PyTorch transition uses the same equations in training
        and evaluation and has no dedicated accelerated kernel. When the
        surrogate uses ``spiking=True``, each step emits only ``-1``, ``0`` or
        ``1`` with the input shape, dtype and device:

        .. math::
            H[t] &= V[t-1] + X[t], \\
            P[t] &= \Theta(H[t] - V_{threshold}), \\
            N[t] &= \Theta(-H[t] - V_{threshold}), \\
            S[t] &= P[t] - N[t].

        Here :math:`\Theta(0)=1`, so both thresholds are inclusive. With
        :math:`R[t]=P[t]+N[t]`, hard reset is
        :math:`V[t]=(1-R[t])H[t]+R[t]V_{reset}`. With ``v_reset=None``, signed
        soft reset is :math:`V[t]=H[t]-V_{threshold}S[t]`. Both event polarities
        hard-reset to the same ``v_reset``. These are algorithmic rules, not a
        derivation from ion-channel dynamics.

        Backward uses ``surrogate_function`` at each threshold. Hard reset keeps
        the surrogate gradients of ``P + N`` instead of differentiating
        ``abs(S)``: they agree in discrete forward execution but the latter
        gives different reset gradients at ``S=0``. ``detach_reset`` detaches
        only ``R`` or ``S`` in reset, not the output spike gradients. With
        ``spiking=False``, the surrogate's continuous primitive is used; outputs
        and reset masks are continuous, so the ternary guarantee does not apply.

        The symmetric event definition follows `Ternary Spike: Learning Ternary
        Spikes for Spiking Neural Networks (Guo et al., AAAI 2024)
        <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_,
        with non-leaky IF charging. Soft reset and nonzero reset are library
        extensions. The paper's learnable spike amplitude and training STE are
        not implemented. :meth:`reset` restores voltage to ``v_reset`` (``0.0``
        for soft reset) and clears ``v_seq``. ``v`` holds the post-reset voltage.

        Signed outputs work with ordinary dense PyTorch layers. Do not compress
        them with ``memopt.BooleanSpikeCompressor``, ``Uint8SpikeCompressor``,
        ``BitSpikeCompressor`` or ``SparseSpikeCompressor``. Set ``compress=False``
        in :func:`~spikingjelly.activation_based.memopt.optimize_memory` and
        ``compressor=None`` in
        :func:`~spikingjelly.activation_based.memopt.checkpoint_module`. To detect
        the first event of either polarity, take absolute values first and follow
        the time-axis convention of
        :func:`~spikingjelly.activation_based.functional.first_spike_index`.
        Default binary SynOps counters and energy models do not apply to signed events.
        NIR and Lava export are not supported for this class.

        :param v_threshold: Positive magnitude of the symmetric thresholds;
            defaults to ``1.0`` and must be a finite positive float.
        :type v_threshold: float
        :param v_reset: Common hard-reset voltage for both polarities; defaults
            to ``0.0`` and must be a finite float strictly between the thresholds.
            ``None`` selects signed soft reset.
        :type v_reset: Optional[float]
        :param surrogate_function: Shared surrogate gradient function for both
            thresholds; defaults to
            :class:`~spikingjelly.activation_based.surrogate.Sigmoid`.
            ``spiking=True`` gives discrete events; ``False`` gives continuous values.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach the reset mask or signed spike
            during reset; defaults to ``False``.
        :type detach_reset: bool
        :param step_mode: ``"s"`` for single-step mode (default) or ``"m"`` for
            multi-step mode.
        :type step_mode: str
        :param store_v_seq: Whether to record post-reset voltage at each step in
            ``v_seq``; defaults to ``False``. Single-step mode appends until
            ``reset()``; multi-step mode records the current sequence.
        :type store_v_seq: bool
        :raises TypeError: If the threshold or non-``None`` reset is not a float,
            or ``detach_reset`` is not a bool.
        :raises ValueError: If the threshold is not finite positive, the reset
            is outside the open threshold interval, or ``step_mode`` is not
            ``"s"`` or ``"m"``.

        .. code-block:: python

            import torch
            from spikingjelly.activation_based import neuron

            node = neuron.TernaryIFNode()
            node(torch.tensor([-1.0, 0.0, 1.0]))
            # tensor([-1., 0., 1.])
            node.reset()
        """
        _validate_ternary_parameters(v_threshold, v_reset, detach_reset)
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
        )

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <TernaryIFNode.single_step_functional_forward-cn>` | :ref:`English <TernaryIFNode.single_step_functional_forward-en>`

        ----

        .. _TernaryIFNode.single_step_functional_forward-cn:

        * **中文**

        用显式状态执行一次充电、双阈值放电和重置。不修改模块 memory 或
        输入状态；阈值、重置和替代梯度约定见构造函数。

        :param inputs: ``(x,)``；``x`` 是形状为 ``[N, *]`` 的浮点张量。
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(v,)``；物化后的 ``v`` 与输入形状、dtype 和 device 相同。
        :type states: tuple[object, ...]
        :param kwargs: 预留接口参数，当前不使用。
        :type kwargs: object
        :return: ``((spike,), (v_next,))``；输出和新电位与输入形状、dtype、device 相同。
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]

        ----

        .. _TernaryIFNode.single_step_functional_forward-en:

        * **English**

        Charge, fire at both thresholds, and reset using explicit state without
        mutating module memories or input states. See the constructor for
        threshold, reset, and surrogate-gradient conventions.

        :param inputs: ``(x,)`` with a floating-point tensor of shape ``[N, *]``.
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(v,)`` with materialized voltage matching the input
            shape, dtype and device.
        :type states: tuple[object, ...]
        :param kwargs: Reserved interface arguments; currently unused.
        :type kwargs: object
        :return: ``((spike,), (v_next,))`` with outputs and voltage matching the
            input shape, dtype and device.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        """
        h = states[0] + inputs[0]
        spike, v = _ternary_fire_and_reset(
            h,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])


class TernaryLIFNode(BaseNode):
    def __init__(
        self,
        tau: float = 2.0,
        decay_input: bool = True,
        v_threshold: float = 1.0,
        v_reset: Optional[float] = 0.0,
        surrogate_function: surrogate.SurrogateFunctionBase = surrogate.Sigmoid(),
        detach_reset: bool = False,
        step_mode: str = "s",
        store_v_seq: bool = False,
    ) -> None:
        r"""
        **API Language** - :ref:`中文 <TernaryLIFNode.__init__-cn>` | :ref:`English <TernaryLIFNode.__init__-en>`

        ----

        .. _TernaryLIFNode.__init__-cn:

        * **中文**

        基础的有符号三值 Leaky Integrate-and-Fire 神经元。输入为浮点张量，
        单步形状为 ``[N, *]``，多步形状为 ``[T, N, *]``。使用原生
        PyTorch 状态转移，无专用加速 kernel；训练和评估使用相同方程。
        令 :math:`V_{rest}=V_{reset}` （软重置时为零），充电方程为：

        .. math::
            H[t] = V[t-1] + \frac{X[t]-(V[t-1]-V_{rest})}{\tau}
            \quad (\mathrm{decay\_input=True}),

        .. math::
            H[t] = V[t-1] - \frac{V[t-1]-V_{rest}}{\tau} + X[t]
            \quad (\mathrm{decay\_input=False}).

        两个阈值的放电与重置方程为：

        .. math::
            P[t] &= \Theta(H[t]-V_{threshold}), \\
            N[t] &= \Theta(-H[t]-V_{threshold}), \\
            S[t] &= P[t]-N[t], \qquad R[t]=P[t]+N[t], \\
            V[t] &= (1-R[t])H[t]+R[t]V_{reset}
            \quad (V_{reset}\ne\mathrm{None}), \\
            V[t] &= H[t]-V_{threshold}S[t]
            \quad (V_{reset}=\mathrm{None}).

        :math:`\Theta(0)=1` ；在替代函数 ``spiking=True`` 模式下输出
        ``-1``、``0`` 或 ``1``，形状、dtype 和 device 与输入相同。
        任一方向的硬重置都到同一个 ``v_reset``。``v`` 保存重置后的电位；
        :meth:`reset` 恢复 ``v_reset``（软重置时为 ``0.0``）并清空 ``v_seq``。

        ``surrogate_function`` 分别提供两个阈值的替代梯度。硬重置对
        ``P + N`` 求导，而不是 ``abs(S)``；虽然离散前向相同，但在静默
        输出处梯度约定不同。``detach_reset`` 只分离重置中的 ``R`` 或
        ``S``，保留输出脉冲的梯度。``spiking=False`` 使用连续原函数，
        因而输出和重置掩码连续，不保证三值。

        `Ternary Spike: Learning Ternary Spikes for Spiking Neural Networks
        (Guo et al., AAAI 2024)
        <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_
        的公式 (7)-(8) 使用衰减系数 :math:`\lambda=0.25` 。
        ``tau=4.0/3.0, decay_input=False, v_reset=0.0`` 在零初态下匹配
        其单位幅值三值前向动力学：本类 ``H[t]`` 对应论文 ``u[t]``，
        ``V[t]`` 是放电后重置的状态。默认配置沿用库中 LIF 的惯例。
        本类不实现论文可学习的脉冲幅值或训练 STE；非零重置、软重置和
        ``decay_input=True`` 是库扩展。这是算法模型，不是离子通道推导。

        有符号输出适用于普通稠密 PyTorch 层。不要用 ``memopt`` 中的
        ``BooleanSpikeCompressor``、``Uint8SpikeCompressor``、
        ``BitSpikeCompressor`` 或 ``SparseSpikeCompressor`` 压缩这些输出；
        使用 :func:`~spikingjelly.activation_based.memopt.optimize_memory` 时
        设置 ``compress=False``，使用
        :func:`~spikingjelly.activation_based.memopt.checkpoint_module` 时设置
        ``compressor=None``。检测任一极性的首次事件时，应先取绝对值，
        再按 :func:`~spikingjelly.activation_based.functional.first_spike_index`
        的时间维约定传入。默认二值 SynOps 计数和能耗模型不适用于有符号事件。
        本类不支持 NIR 或 Lava 导出。

        :param tau: 以离散时间步为单位的膜电位时间常数，默认 ``2.0``，
            必须为大于 ``1.0`` 的有限浮点数；衰减系数为 ``1-1/tau``。
        :type tau: float
        :param decay_input: 输入是否也乘以 ``1/tau``，默认 ``True``。
        :type decay_input: bool
        :param v_threshold: 对称阈值的正幅值，默认 ``1.0``，必须为有限正浮点数。
        :type v_threshold: float
        :param v_reset: 两个方向共用的硬重置和静息电位，默认 ``0.0``，
            必须是严格位于两个阈值之间的有限浮点数。``None`` 使用有符号
            软重置和零静息电位。
        :type v_reset: Optional[float]
        :param surrogate_function: 两个阈值共用的替代梯度函数，默认
            :class:`~spikingjelly.activation_based.surrogate.Sigmoid` 。
            ``spiking=True`` 输出离散事件；``False`` 输出连续值。
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: 是否分离重置掩码或有符号脉冲的梯度，默认 ``False``。
        :type detach_reset: bool
        :param step_mode: 步进模式，``"s"`` 为单步（默认），``"m"`` 为多步。
        :type step_mode: str
        :param store_v_seq: 是否在 ``v_seq`` 记录每一步重置后的电位，默认
            ``False``。单步模式逐步追加，直到 ``reset()``；多步模式记录当前序列。
        :type store_v_seq: bool
        :raises TypeError: 阈值、时间常数或非空重置电位不是浮点数，或
            ``decay_input``、``detach_reset`` 不是布尔值。
        :raises ValueError: 阈值不是有限正数、时间常数不是有限且大于一、
            重置电位不在开放阈值区间内，或 ``step_mode`` 不是 ``"s"`` 或 ``"m"``。

        .. code-block:: python

            import torch
            from spikingjelly.activation_based import neuron

            node = neuron.TernaryLIFNode(
                tau=4.0 / 3.0, decay_input=False, v_reset=0.0, step_mode="m"
            )
            node(torch.tensor([[1.0, -1.0], [0.0, 0.0]]))
            # tensor([[ 1., -1.], [ 0.,  0.]])
            node.reset()

        ----

        .. _TernaryLIFNode.__init__-en:

        * **English**

        A basic signed ternary Leaky Integrate-and-Fire neuron. Floating-point
        inputs have shape ``[N, *]`` in single-step mode and ``[T, N, *]`` in
        multi-step mode. Native PyTorch transitions have no dedicated accelerated
        kernel and use the same equations in training and evaluation. With
        :math:`V_{rest}=V_{reset}` (zero for soft reset), charging is:

        .. math::
            H[t] = V[t-1] + \frac{X[t]-(V[t-1]-V_{rest})}{\tau}
            \quad (\mathrm{decay\_input=True}),

        .. math::
            H[t] = V[t-1] - \frac{V[t-1]-V_{rest}}{\tau} + X[t]
            \quad (\mathrm{decay\_input=False}).

        Firing at both thresholds and resetting are defined by:

        .. math::
            P[t] &= \Theta(H[t]-V_{threshold}), \\
            N[t] &= \Theta(-H[t]-V_{threshold}), \\
            S[t] &= P[t]-N[t], \qquad R[t]=P[t]+N[t], \\
            V[t] &= (1-R[t])H[t]+R[t]V_{reset}
            \quad (V_{reset}\ne\mathrm{None}), \\
            V[t] &= H[t]-V_{threshold}S[t]
            \quad (V_{reset}=\mathrm{None}).

        With :math:`\Theta(0)=1` and ``spiking=True``, the output is ``-1``,
        ``0`` or ``1`` with the input shape, dtype and device. Both event
        polarities hard-reset to the same ``v_reset``. ``v`` holds the post-reset
        voltage. :meth:`reset` restores ``v_reset`` (``0.0`` for soft reset) and
        clears ``v_seq``.

        ``surrogate_function`` supplies gradients independently at both
        thresholds. Hard reset differentiates ``P + N`` instead of ``abs(S)``;
        they agree in discrete forward execution but differ in the gradient
        convention for silent outputs. ``detach_reset`` detaches only ``R`` or
        ``S`` in reset, preserving output spike gradients. ``spiking=False``
        uses the continuous primitive, giving continuous outputs and reset masks
        without the ternary guarantee.

        Equations (7)-(8) of `Ternary Spike: Learning Ternary Spikes for Spiking
        Neural Networks (Guo et al., AAAI 2024)
        <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_
        use decay :math:`\lambda=0.25`. With zero initial state,
        ``tau=4.0/3.0, decay_input=False, v_reset=0.0`` matches their unit-amplitude
        ternary forward dynamics: ``H[t]`` corresponds to the paper's ``u[t]``
        and ``V[t]`` is the post-spike reset state. Defaults follow the library's
        LIF conventions. The paper's learnable spike amplitude and training STE
        are not implemented. Nonzero reset, soft reset and ``decay_input=True``
        are library extensions. This is an algorithmic model, not a derivation
        from ion-channel dynamics.

        Signed outputs work with ordinary dense PyTorch layers. Do not compress
        them with ``memopt.BooleanSpikeCompressor``, ``Uint8SpikeCompressor``,
        ``BitSpikeCompressor`` or ``SparseSpikeCompressor``. Set ``compress=False``
        in :func:`~spikingjelly.activation_based.memopt.optimize_memory` and
        ``compressor=None`` in
        :func:`~spikingjelly.activation_based.memopt.checkpoint_module`. To detect
        the first event of either polarity, take absolute values first and follow
        the time-axis convention of
        :func:`~spikingjelly.activation_based.functional.first_spike_index`.
        Default binary SynOps counters and energy models do not apply to signed events.
        NIR and Lava export are not supported for this class.

        :param tau: Membrane time constant in discrete time steps; defaults to
            ``2.0`` and must be a finite float greater than ``1.0``. The decay
            coefficient is ``1-1/tau``.
        :type tau: float
        :param decay_input: Whether input is also scaled by ``1/tau``; defaults
            to ``True``.
        :type decay_input: bool
        :param v_threshold: Positive magnitude of the symmetric thresholds;
            defaults to ``1.0`` and must be a finite positive float.
        :type v_threshold: float
        :param v_reset: Common hard-reset and resting voltage; defaults to
            ``0.0`` and must be a finite float strictly between the thresholds.
            ``None`` selects signed soft reset and zero resting voltage.
        :type v_reset: Optional[float]
        :param surrogate_function: Shared surrogate gradient function for both
            thresholds; defaults to
            :class:`~spikingjelly.activation_based.surrogate.Sigmoid`.
            ``spiking=True`` gives discrete events; ``False`` gives continuous values.
        :type surrogate_function: surrogate.SurrogateFunctionBase
        :param detach_reset: Whether to detach the reset mask or signed spike
            during reset; defaults to ``False``.
        :type detach_reset: bool
        :param step_mode: ``"s"`` for single-step mode (default) or ``"m"`` for
            multi-step mode.
        :type step_mode: str
        :param store_v_seq: Whether to record post-reset voltage at each step in
            ``v_seq``; defaults to ``False``. Single-step mode appends until
            ``reset()``; multi-step mode records the current sequence.
        :type store_v_seq: bool
        :raises TypeError: If the threshold, time constant or non-``None`` reset
            is not a float, or ``decay_input`` or ``detach_reset`` is not a bool.
        :raises ValueError: If the threshold is not finite positive, the time
            constant is not finite and greater than one, the reset is outside
            the open threshold interval, or ``step_mode`` is not ``"s"`` or ``"m"``.

        .. code-block:: python

            import torch
            from spikingjelly.activation_based import neuron

            node = neuron.TernaryLIFNode(
                tau=4.0 / 3.0, decay_input=False, v_reset=0.0, step_mode="m"
            )
            node(torch.tensor([[1.0, -1.0], [0.0, 0.0]]))
            # tensor([[ 1., -1.], [ 0.,  0.]])
            node.reset()
        """
        _validate_ternary_parameters(v_threshold, v_reset, detach_reset)
        if not isinstance(tau, float):
            raise TypeError("tau must be a float.")
        if not math.isfinite(tau) or tau <= 1.0:
            raise ValueError("tau must be finite and greater than 1.0.")
        if not isinstance(decay_input, bool):
            raise TypeError("decay_input must be a bool.")
        super().__init__(
            v_threshold=v_threshold,
            v_reset=v_reset,
            surrogate_function=surrogate_function,
            detach_reset=detach_reset,
            step_mode=step_mode,
            store_v_seq=store_v_seq,
        )
        self.tau = tau
        self.decay_input = decay_input

    def extra_repr(self) -> str:
        r"""
        **API Language** - :ref:`中文 <TernaryLIFNode.extra_repr-cn>` | :ref:`English <TernaryLIFNode.extra_repr-en>`

        ----

        .. _TernaryLIFNode.extra_repr-cn:

        * **中文**

        :return: 阈值、重置、步进模式、时间常数和输入衰减设置的字符串。
        :rtype: str

        ----

        .. _TernaryLIFNode.extra_repr-en:

        * **English**

        :return: String describing threshold, reset, step mode, time constant,
            and input decay settings.
        :rtype: str
        """
        return (
            super().extra_repr() + f", tau={self.tau}, decay_input={self.decay_input}"
        )

    def single_step_functional_forward(
        self,
        inputs: tuple[torch.Tensor, ...],
        states: tuple[object, ...],
        **kwargs: object,
    ) -> tuple[tuple[torch.Tensor, ...], tuple[object, ...]]:
        r"""
        **API Language** - :ref:`中文 <TernaryLIFNode.single_step_functional_forward-cn>` | :ref:`English <TernaryLIFNode.single_step_functional_forward-en>`

        ----

        .. _TernaryLIFNode.single_step_functional_forward-cn:

        * **中文**

        用显式状态执行一次泄漏充电、双阈值放电和重置。不修改模块 memory
        或输入状态；动力学及替代梯度约定见构造函数。

        :param inputs: ``(x,)``；``x`` 是形状为 ``[N, *]`` 的浮点张量。
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(v,)``；物化后的 ``v`` 与输入形状、dtype 和 device 相同。
        :type states: tuple[object, ...]
        :param kwargs: 预留接口参数，当前不使用。
        :type kwargs: object
        :return: ``((spike,), (v_next,))``；输出和新电位与输入形状、dtype、device 相同。
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]

        ----

        .. _TernaryLIFNode.single_step_functional_forward-en:

        * **English**

        Apply leaky charging, firing at both thresholds, and reset with explicit
        state without mutating module memories or input states. See the
        constructor for dynamics and surrogate-gradient conventions.

        :param inputs: ``(x,)`` with a floating-point tensor of shape ``[N, *]``.
        :type inputs: tuple[torch.Tensor, ...]
        :param states: ``(v,)`` with materialized voltage matching the input
            shape, dtype and device.
        :type states: tuple[object, ...]
        :param kwargs: Reserved interface arguments; currently unused.
        :type kwargs: object
        :return: ``((spike,), (v_next,))`` with outputs and voltage matching the
            input shape, dtype and device.
        :rtype: tuple[tuple[torch.Tensor, ...], tuple[object, ...]]
        """
        h = functional.lif_charge(
            inputs[0], states[0], self.tau, self.decay_input, self.v_reset
        )
        spike, v = _ternary_fire_and_reset(
            h,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])
