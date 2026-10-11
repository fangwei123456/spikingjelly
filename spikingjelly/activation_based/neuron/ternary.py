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

        有符号三值 IF 神经元，以正负阈值区分 ``-1``、``0``、``1`` 事件。
        浮点输入的单步形状为 ``[N, *]``，多步为 ``[T, N, *]``；输出保持
        输入的形状、dtype 和 device。训练和评估均使用以下 PyTorch 方程：

        .. math::
            H[t] &= V[t-1] + X[t], \\
            P[t] &= \Theta(H[t] - V_{threshold}), \\
            N[t] &= \Theta(-H[t] - V_{threshold}), \\
            S[t] &= P[t] - N[t].

        :math:`\Theta` 为二值阶跃函数，:math:`\Theta(0)=1`，两个阈值均
        包含等号。令 :math:`R[t]=P[t]+N[t]`，硬重置为
        :math:`V[t]=(1-R[t])H[t]+R[t]V_{reset}`；``v_reset=None`` 时，
        有符号软重置为 :math:`V[t]=H[t]-V_{threshold}S[t]`。

        两个阈值分别使用替代梯度，硬重置对 ``P + N`` 求导。
        ``detach_reset`` 仅分离重置分支，保留输出脉冲的梯度。
        ``spiking=False`` 使用替代函数的连续原函数，输出和重置掩码也连续。
        ``v`` 保存重置后的电位；:meth:`reset` 恢复 ``v_reset``
        （软重置时为 ``0.0``）并清空 ``v_seq``。

        有符号输出的 :ref:`工具兼容性 <signed-ternary-compatibility>` 见模块文档。

        :param v_threshold: 对称阈值的正幅值，默认 ``1.0``，必须为有限正浮点数。
        :type v_threshold: float
        :param v_reset: 两个方向共用的硬重置电位，默认 ``0.0``，必须是严格
            位于两个阈值之间的有限浮点数。``None`` 表示有符号软重置。
        :type v_reset: Optional[float]
        :param surrogate_function: 两个阈值共用的替代梯度函数，默认
            :class:`~spikingjelly.activation_based.surrogate.Sigmoid` 。
            离散模式要求前向为二值 Heaviside 函数，且 :math:`\Theta(0)=1`；
            ``spiking=False`` 时输出连续值。
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

        A signed ternary IF neuron with positive and negative thresholds for
        ``-1``, ``0`` and ``1`` events. Floating-point inputs have shape
        ``[N, *]`` in single-step mode or ``[T, N, *]`` in multi-step mode.
        Outputs retain the input shape, dtype and device. Training and evaluation
        use the same PyTorch equations:

        .. math::
            H[t] &= V[t-1] + X[t], \\
            P[t] &= \Theta(H[t] - V_{threshold}), \\
            N[t] &= \Theta(-H[t] - V_{threshold}), \\
            S[t] &= P[t] - N[t].

        :math:`\Theta` is a binary step function with :math:`\Theta(0)=1`, so
        both thresholds are inclusive. With :math:`R[t]=P[t]+N[t]`, hard reset is
        :math:`V[t]=(1-R[t])H[t]+R[t]V_{reset}`. With ``v_reset=None``, signed
        soft reset is :math:`V[t]=H[t]-V_{threshold}S[t]`.

        Each threshold uses the surrogate gradient; hard reset differentiates
        ``P + N``. ``detach_reset`` detaches only the reset branch, preserving
        output spike gradients. With ``spiking=False``, the surrogate's continuous
        primitive gives continuous outputs and reset masks. ``v`` holds the
        post-reset voltage; :meth:`reset` restores ``v_reset`` (``0.0`` for soft
        reset) and clears ``v_seq``.

        See the module documentation for
        :ref:`tool compatibility <signed-ternary-compatibility>`.

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
            Discrete mode requires binary Heaviside forward with :math:`\Theta(0)=1`;
            ``spiking=False`` gives continuous values.
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
        spike, v = functional.ternary_if_step(
            inputs[0],
            states[0],
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

        有符号三值 LIF 神经元，在双阈值放电前对电位施加泄漏。
        浮点输入的单步形状为 ``[N, *]``，多步为 ``[T, N, *]``；输出保持
        输入的形状、dtype 和 device。训练和评估均使用以下 PyTorch 方程。
        令 :math:`V_{rest}=V_{reset}` （软重置时为零）：

        .. math::
            H[t] = V[t-1] + \frac{X[t]-(V[t-1]-V_{rest})}{\tau}
            \quad (\mathrm{decay\_input=True}),

        .. math::
            H[t] = V[t-1] - \frac{V[t-1]-V_{rest}}{\tau} + X[t]
            \quad (\mathrm{decay\_input=False}).

        放电、重置及替代梯度规则与 :class:`TernaryIFNode` 相同：达到
        正负阈值（包含等号）时输出 ``1`` 或 ``-1``，其余输出 ``0``。
        硬重置均回到 ``v_reset``；``None`` 使用有符号软重置。
        硬重置对两个阈值事件之和求导，``detach_reset`` 仅分离重置分支。
        ``spiking=False`` 使用连续输出和重置掩码。``v`` 保存重置后的电位；
        :meth:`reset` 恢复 ``v_reset``（软重置时为 ``0.0``）并清空 ``v_seq``。

        .. note::

            ``tau=4.0/3.0, decay_input=False, v_reset=0.0`` 在零初态下匹配
            `Guo et al. (AAAI 2024)
            <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_
            公式 (7)-(8) 的单位幅值前向动力学（保留系数为 ``0.25``）。
            ``H[t]`` 对应论文的 ``u[t]``，``V[t]`` 为重置后的状态。
            可学习幅值和论文的训练 STE 不在本类范围内；非零重置、软重置
            和输入衰减是库扩展。

        有符号输出的 :ref:`工具兼容性 <signed-ternary-compatibility>` 见模块文档。

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
            离散模式要求前向为二值 Heaviside 函数，且 :math:`\Theta(0)=1`；
            ``spiking=False`` 时输出连续值。
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

        A signed ternary LIF neuron that applies voltage leakage before firing
        at two thresholds. Floating-point inputs have shape ``[N, *]`` in
        single-step mode or ``[T, N, *]`` in multi-step mode. Outputs retain the
        input shape, dtype and device. Training and evaluation use the same
        PyTorch equations. With :math:`V_{rest}=V_{reset}` (zero for soft reset):

        .. math::
            H[t] = V[t-1] + \frac{X[t]-(V[t-1]-V_{rest})}{\tau}
            \quad (\mathrm{decay\_input=True}),

        .. math::
            H[t] = V[t-1] - \frac{V[t-1]-V_{rest}}{\tau} + X[t]
            \quad (\mathrm{decay\_input=False}).

        Firing, reset and surrogate gradients follow :class:`TernaryIFNode`:
        reaching either inclusive threshold emits ``1`` or ``-1``; otherwise the
        output is ``0``. Both polarities hard-reset to ``v_reset``; ``None`` selects
        signed soft reset. Hard reset differentiates the sum of both threshold
        events, and ``detach_reset`` detaches only the reset branch.
        ``spiking=False`` gives continuous outputs and reset masks. ``v`` holds
        the post-reset voltage; :meth:`reset` restores ``v_reset`` (``0.0`` for soft
        reset) and clears ``v_seq``.

        .. note::

            With zero initial state, ``tau=4.0/3.0, decay_input=False, v_reset=0.0``
            matches the unit-amplitude forward dynamics in equations (7)-(8) of
            `Guo et al. (AAAI 2024)
            <https://ojs.aaai.org/index.php/AAAI/article/download/29114/30106>`_
            (retained fraction ``0.25``). ``H[t]`` corresponds to the paper's
            ``u[t]``; ``V[t]`` is the post-reset state. Learnable amplitudes and
            the paper's training STE are outside this class's scope. Nonzero
            reset, soft reset and input decay are library extensions.

        See the module documentation for
        :ref:`tool compatibility <signed-ternary-compatibility>`.

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
            Discrete mode requires binary Heaviside forward with :math:`\Theta(0)=1`;
            ``spiking=False`` gives continuous values.
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
        spike, v = functional.ternary_lif_step(
            inputs[0],
            states[0],
            self.tau,
            self.decay_input,
            self.v_threshold,
            self.v_reset,
            self.surrogate_function,
            self.detach_reset,
        )
        return (spike,), (v, *states[1:])
