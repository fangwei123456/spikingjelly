import math
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _reverse_repeat_tuple

from .. import base


__all__ = [
    "TDModule",
    "TDSoftmax",
    "TDLayerNorm",
    "TDRMSNorm",
    "TDGELU",
    "TDSiLU",
    "TDLinear",
    "TDConv2d",
    "SNNMatrixOperator",
    "SNNElementWiseProduct",
    "TDScaledDotProductAttention",
    "TDMultiheadAttention",
]


def _temporal_difference(y_cum: torch.Tensor) -> torch.Tensor:
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


def _resolve_dim(dim: int, ndim: int) -> int:
    resolved = dim
    if resolved < 0:
        resolved += ndim
    if resolved < 0 or resolved >= ndim:
        raise ValueError(
            f"dim must be in the range [{-ndim}, {ndim - 1}], "
            f"but got {dim} for an input with {ndim} dimensions."
        )
    return resolved


class TDModule(base.MemoryModule):
    def __init__(self, step_mode: str = "m") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDModule.__init__-cn>` |
        :ref:`English <TDModule.__init__-en>`

        ----

        .. _TDModule.__init__-cn:

        * **中文**

        Temporal-difference / sequence-preserving 算子的基类。该类继承
        :class:`spikingjelly.activation_based.base.MemoryModule`，使用
        ``step_mode``、memory 和 ``reset`` 语义实现 TD 状态传播。

        ``step_mode="s"`` 时，输入被解释为当前差分时间步，模块更新内部累积
        memory 并返回当前差分输出；``step_mode="m"`` 时，输入第 0 维被解释为
        时间维，模块返回完整差分序列并保留最终 memory。普通 ANN/PyTorch 数值
        路径由 :meth:`ann_forward` 提供，且不读写 memory。子类必须实现
        :meth:`ann_forward` 和 :meth:`multi_step_forward`。子类 ``__init__``
        应调用 ``super().__init__(step_mode)`` 初始化步进模式。

        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"`` 保持既有
            TD operator 行为。
        :type step_mode: str
        :raises ValueError: 当 ``step_mode`` 非法时，由
            :class:`~spikingjelly.activation_based.base.StepModule` 的 setter
            抛出；若子类绕过 setter 写入非法模式，``forward`` 也会抛出。

        ----

        .. _TDModule.__init__-en:

        * **English**

        Base class for temporal-difference / sequence-preserving operators. It
        inherits :class:`spikingjelly.activation_based.base.MemoryModule` and
        uses ``step_mode``, memory, and ``reset`` semantics for TD state
        propagation.

        With ``step_mode="s"``, inputs are interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. With ``step_mode="m"``,
        dimension 0 is interpreted as the time dimension; the module returns a
        full differential sequence and keeps the final memory. The ordinary
        ANN/PyTorch numeric path is exposed by :meth:`ann_forward` and does not
        read or write memory. Subclasses must implement :meth:`ann_forward` and
        :meth:`multi_step_forward`. Subclass ``__init__`` methods should call
        ``super().__init__(step_mode)`` to initialize the step mode.

        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default ``"m"``
            preserves existing TD operator behavior.
        :type step_mode: str
        :raises ValueError: Raised by
            :class:`~spikingjelly.activation_based.base.StepModule`'s setter
            when ``step_mode`` is invalid; ``forward`` also raises if a
            subclass bypasses the setter and writes an invalid mode.
        """
        super().__init__()
        self.register_memory("x_cum", None)
        self.register_memory("y_cum", None)
        self.step_mode = step_mode

    def ann_forward(self, *args, **kwargs):
        raise NotImplementedError

    def multi_step_forward(self, *args, **kwargs):
        raise NotImplementedError(
            f"{type(self).__name__} must implement multi_step_forward."
        )

    def single_step_forward(self, *args, **kwargs):
        y_cum = self.ann_forward(*self._accumulate_inputs(*args), **kwargs)
        return self._diff_output(y_cum)

    @staticmethod
    def _same_tensor_meta(a: torch.Tensor, b: torch.Tensor) -> bool:
        return a.shape == b.shape and a.device == b.device and a.dtype == b.dtype

    def _accumulate_one_input(self, x: torch.Tensor, index: int = 0) -> torch.Tensor:
        if self.x_cum is None:
            self.y_cum = None
            x_cum = x
        elif isinstance(self.x_cum, tuple):
            prev = self.x_cum[index]
            if prev is None or not self._same_tensor_meta(prev, x):
                self.y_cum = None
                x_cum = x
            else:
                x_cum = prev + x
        else:
            if not self._same_tensor_meta(self.x_cum, x):
                self.y_cum = None
                x_cum = x
            else:
                x_cum = self.x_cum + x

        if isinstance(self.x_cum, tuple):
            x_cum_values = list(self.x_cum)
            x_cum_values[index] = x_cum
            self.x_cum = tuple(x_cum_values)
        else:
            self.x_cum = x_cum
        return x_cum

    def _accumulate_inputs(self, *xs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        if len(xs) == 1:
            if isinstance(self.x_cum, tuple):
                self.x_cum = None
                self.y_cum = None
            return (self._accumulate_one_input(xs[0]),)
        if self.x_cum is None:
            self.x_cum = tuple(None for _ in xs)
            self.y_cum = None
        elif not isinstance(self.x_cum, tuple) or len(self.x_cum) != len(xs):
            self.x_cum = tuple(None for _ in xs)
            self.y_cum = None
        elif not all(
            prev is not None
            and isinstance(prev, torch.Tensor)
            and self._same_tensor_meta(prev, x)
            for prev, x in zip(self.x_cum, xs)
        ):
            self.x_cum = tuple(None for _ in xs)
            self.y_cum = None
        return tuple(self._accumulate_one_input(x, i) for i, x in enumerate(xs))

    def _diff_output(self, y_cum: torch.Tensor) -> torch.Tensor:
        if (
            self.y_cum is None
            or not isinstance(self.y_cum, torch.Tensor)
            or not self._same_tensor_meta(self.y_cum, y_cum)
        ):
            y = y_cum
        else:
            y = y_cum - self.y_cum
        self.y_cum = y_cum
        return y

    def _diff_sequence_output(self, y_cum_seq: torch.Tensor) -> torch.Tensor:
        if y_cum_seq.shape[0] == 0:
            self.y_cum = None
            return y_cum_seq
        if (
            self.y_cum is None
            or not isinstance(self.y_cum, torch.Tensor)
            or not self._same_tensor_meta(self.y_cum, y_cum_seq[0])
        ):
            y_seq = _temporal_difference(y_cum_seq)
        else:
            y_seq = torch.empty_like(y_cum_seq)
            y_seq[0] = y_cum_seq[0] - self.y_cum
            y_seq[1:] = y_cum_seq[1:] - y_cum_seq[:-1]
        self.y_cum = y_cum_seq[-1].clone()
        return y_seq

    def _td_sequence_forward(self, input_seqs: Tuple[torch.Tensor, ...], ann_forward):
        for x_seq in input_seqs:
            if x_seq.shape[0] == 0:
                raise ValueError(
                    f"{self.__class__.__name__} expects a non-empty time "
                    f"dimension, but got shape {tuple(x_seq.shape)}."
                )
        cum_seqs = tuple(x_seq.cumsum(dim=0) for x_seq in input_seqs)
        if len(cum_seqs) == 1:
            prev_inputs = self.x_cum if isinstance(self.x_cum, tuple) else (self.x_cum,)
        else:
            prev_inputs = self.x_cum

        should_continue = (
            isinstance(prev_inputs, tuple)
            and len(prev_inputs) == len(cum_seqs)
            and all(
                prev is not None
                and isinstance(prev, torch.Tensor)
                and self._same_tensor_meta(prev, x_cum_seq[0])
                for prev, x_cum_seq in zip(prev_inputs, cum_seqs)
            )
        )
        if should_continue:
            cum_seqs = tuple(
                prev + x_cum_seq
                for prev, x_cum_seq in zip(prev_inputs, cum_seqs, strict=True)
            )
        else:
            self.y_cum = None

        y_cum_seq = ann_forward(*cum_seqs)
        y_seq = self._diff_sequence_output(y_cum_seq)
        final_inputs = tuple(x_cum_seq[-1].clone() for x_cum_seq in cum_seqs)
        self.x_cum = final_inputs[0] if len(final_inputs) == 1 else final_inputs
        return y_seq


def _check_time_sequence(x_seq: torch.Tensor, module_name: str) -> None:
    if x_seq.dim() < 2:
        raise ValueError(
            f"{module_name} expects an input sequence with shape [T, ...] "
            f"and at least 2 dimensions, but got shape {tuple(x_seq.shape)}."
        )
    if x_seq.shape[0] == 0:
        raise ValueError(
            f"{module_name} expects a non-empty time dimension, but got "
            f"shape {tuple(x_seq.shape)}."
        )


def _check_pair_time_sequence(
    a_seq: torch.Tensor,
    b_seq: torch.Tensor,
    a_name: str,
    b_name: str,
    module_name: str,
) -> None:
    _check_time_sequence(a_seq, module_name)
    _check_time_sequence(b_seq, module_name)
    if a_seq.shape[0] != b_seq.shape[0]:
        raise ValueError(
            f"{module_name} expects {a_name} and {b_name} to have the same "
            f"time length, but got {a_seq.shape[0]} and {b_seq.shape[0]}."
        )


def _align_sequence_ranks(
    a_seq: torch.Tensor, b_seq: torch.Tensor
) -> Tuple[torch.Tensor, torch.Tensor]:
    while a_seq.dim() < b_seq.dim():
        a_seq = a_seq.unsqueeze(1)
    while b_seq.dim() < a_seq.dim():
        b_seq = b_seq.unsqueeze(1)
    return a_seq, b_seq


def _check_attention_sequence(
    x_seq: torch.Tensor, tensor_name: str, module_name: str
) -> None:
    _check_time_sequence(x_seq, module_name)
    if x_seq.dim() < 3:
        raise ValueError(
            f"{module_name} expects {tensor_name} with shape [T, ..., L, E] "
            f"and at least 3 dimensions, but got shape {tuple(x_seq.shape)}."
        )


class TDSoftmax(TDModule):
    def __init__(self, dim: int = -1, step_mode: str = "m") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDSoftmax.__init__-cn>` |
        :ref:`English <TDSoftmax.__init__-en>`

        ----

        .. _TDSoftmax.__init__-cn:

        * **中文**

        Temporal-difference (TD) Softmax 算子。``step_mode="m"`` 时输入
        必须是完整时间序列，时间维固定为第 0 维，形状为 ``[T, ...]``；
        模块先对输入在时间维做累积，再沿 ``dim`` 计算
        ``torch.softmax``，最后返回累积输出在时间维上的差分。
        ``step_mode="s"`` 时输入被解释为当前差分时间步，模块更新内部累积
        memory 并返回当前差分输出；普通 ``torch.softmax`` 路径由
        :meth:`ann_forward` 提供。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Softmax。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD
        Softmax 只实现张量级算子：在多步模式下它仍调用
        ``torch.softmax``，需要完整时间序列输入，不是逐时间步在线算子，
        也不是面向神经形态硬件的 fully spike-driven Softmax。

        .. code-block:: python

            op = TDSoftmax(dim=-1)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param dim: Softmax 归一化维度。``step_mode="m"`` 时不能为第 0 维，
            因为第 0 维保留为时间维；``step_mode="s"`` 时作用在当前差分
            时间步的对应维度。
        :type dim: int
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _TDSoftmax.__init__-en:

        * **English**

        Temporal-difference (TD) Softmax operator. With ``step_mode="m"``, the
        input must be a complete time sequence whose time dimension is fixed at
        dimension 0, with shape ``[T, ...]``. This module first accumulates the
        input over time, applies ``torch.softmax`` along ``dim`` to each
        cumulative input, and returns the temporal difference of the cumulative
        outputs. With ``step_mode="s"``, the input is interpreted as the
        current differential time step; the module updates its cumulative
        memory and returns the current differential output. The ordinary
        ``torch.softmax`` path is exposed by :meth:`ann_forward`.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven Softmax. The output dtype matches the input dtype;
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: in
        multi-step mode it still calls ``torch.softmax``, requires a complete
        time sequence, is not a step-wise online operator, and is not a fully
        spike-driven Softmax for neuromorphic hardware.

        .. code-block:: python

            op = TDSoftmax(dim=-1)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param dim: Softmax normalization dimension. With ``step_mode="m"``, it
            must not be dimension 0, which is reserved as the time dimension.
            With ``step_mode="s"``, it applies to the corresponding dimension
            of the current differential time step.
        :type dim: int
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)
        self.dim = dim

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.softmax(x, dim=self.dim)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDSoftmax.forward-cn>` |
        :ref:`English <TDSoftmax.forward-en>`

        ----

        .. _TDSoftmax.forward-cn:

        * **中文**

        对完整时间序列执行 TD Softmax。计算过程为：

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{Softmax}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        因此 ``Y.cumsum(dim=0)`` 与对 ``X.cumsum(dim=0)`` 逐时间步执行 ANN
        Softmax 的结果一致。输出是浮点差分值，可能为负，不是二值脉冲。
        当 ``T = 1`` 时，``Y[0]`` 直接等于 ``torch.softmax(X[0], dim=dim)``。
        输出 dtype 与输入 dtype 相同，且该算子对 autograd 透明。

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``，且 ``T > 0``。
        :type x_seq: torch.Tensor
        :return: TD Softmax 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维、时间维为空，或 ``dim``
            指向时间维。

        ----

        .. _TDSoftmax.forward-en:

        * **English**

        Apply TD Softmax to a complete time sequence:

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{Softmax}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        Thus, ``Y.cumsum(dim=0)`` matches ANN Softmax applied to
        ``X.cumsum(dim=0)`` at each time step. The output contains
        floating-point differential values, may be negative, and is not a binary
        spike tensor. When ``T = 1``, ``Y[0]`` is exactly
        ``torch.softmax(X[0], dim=dim)``. The output dtype matches the input
        dtype, and the operator is transparent to autograd.

        :param x_seq: Input time sequence with shape ``[T, ...]`` and ``T > 0``.
        :type x_seq: torch.Tensor
        :return: TD Softmax differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions, the time
            dimension is empty, or ``dim`` refers to the time dimension.
        """
        _check_time_sequence(x_seq, "TDSoftmax")

        dim = _resolve_dim(self.dim, x_seq.dim())
        if dim == 0:
            raise ValueError(
                "TDSoftmax reserves dimension 0 as the time dimension; "
                "softmax dim must not resolve to 0."
            )

        return self._td_sequence_forward(
            (x_seq,), lambda x_cum: torch.softmax(x_cum, dim=dim)
        )

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class TDLayerNorm(TDModule):
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDLayerNorm.__init__-cn>` |
        :ref:`English <TDLayerNorm.__init__-en>`

        ----

        .. _TDLayerNorm.__init__-cn:

        * **中文**

        Temporal-difference (TD) LayerNorm 算子。``step_mode="m"`` 时输入
        必须是完整时间序列，时间维固定为第 0 维，形状为 ``[T, ...]``；
        模块先对输入在时间维做累积，再对每个累积输入执行
        :func:`torch.nn.functional.layer_norm`，最后返回累积输出在时间维上的
        差分。``step_mode="s"`` 时输入被解释为当前差分时间步，模块更新内部
        累积 memory 并返回当前差分输出；普通 LayerNorm 路径由
        :meth:`ann_forward` 提供。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven LayerNorm。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。该算子是 stateful TD MemoryModule；
        重复处理独立序列前应调用 ``reset``。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD
        LayerNorm 只实现张量级算子：在多步模式下它仍调用
        :func:`torch.nn.functional.layer_norm`，需要完整时间序列输入，不是逐
        时间步在线算子，也不是面向神经形态硬件的 fully spike-driven
        LayerNorm。

        .. code-block:: python

            op = TDLayerNorm(normalized_shape=3)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param normalized_shape: 输入尾部需要归一化的形状，与
            :class:`torch.nn.LayerNorm` 的 ``normalized_shape`` 语义一致。
        :type normalized_shape: int or list[int] or torch.Size
        :param eps: 加到方差上的数值稳定项。
        :type eps: float
        :param elementwise_affine: 若为 ``True``，使用可学习的逐元素仿射
            参数。
        :type elementwise_affine: bool
        :param bias: 若 ``elementwise_affine`` 和 ``bias`` 均为 ``True``，
            使用可学习 bias 参数。若 ``elementwise_affine`` 为 ``False``，
            则忽略 ``bias``。
        :type bias: bool
        :param device: 参数初始化设备。
        :type device: torch.device or str or None
        :param dtype: 参数初始化 dtype。
        :type dtype: torch.dtype or None
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _TDLayerNorm.__init__-en:

        * **English**

        Temporal-difference (TD) LayerNorm operator. With ``step_mode="m"``,
        the input must be a complete time sequence whose time dimension is
        fixed at dimension 0, with shape ``[T, ...]``. This module first
        accumulates the input over time, applies
        :func:`torch.nn.functional.layer_norm` to each cumulative input, and
        returns the temporal difference of the cumulative outputs. With
        ``step_mode="s"``, the input is interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. The ordinary LayerNorm path is
        exposed by :meth:`ann_forward`.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven LayerNorm. The output dtype matches the input dtype;
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd. The operator is a stateful TD
        MemoryModule; call ``reset`` before processing an independent sequence.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: in
        multi-step mode it still calls
        :func:`torch.nn.functional.layer_norm`, requires a complete time
        sequence, is not a step-wise online operator, and is not a fully
        spike-driven LayerNorm for neuromorphic hardware.

        .. code-block:: python

            op = TDLayerNorm(normalized_shape=3)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param normalized_shape: Input trailing shape to normalize, with the
            same semantics as ``normalized_shape`` in :class:`torch.nn.LayerNorm`.
        :type normalized_shape: int or list[int] or torch.Size
        :param eps: Value added to the variance for numerical stability.
        :type eps: float
        :param elementwise_affine: If ``True``, use learnable per-element affine
            parameters.
        :type elementwise_affine: bool
        :param bias: If both ``elementwise_affine`` and ``bias`` are ``True``,
            use a learnable bias parameter. If ``elementwise_affine`` is
            ``False``, ``bias`` is ignored.
        :type bias: bool
        :param device: Device used to initialize parameters.
        :type device: torch.device or str or None
        :param dtype: Dtype used to initialize parameters.
        :type dtype: torch.dtype or None
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine

        factory_kwargs = {"device": device, "dtype": dtype}
        if self.elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, **factory_kwargs)
            )
            if bias:
                self.bias = nn.Parameter(
                    torch.empty(self.normalized_shape, **factory_kwargs)
                )
            else:
                # bias=False mirrors nn.LayerNorm by making bias an explicit
                # None parameter while preserving a learnable weight.
                self.register_parameter("bias", None)
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        if self.elementwise_affine:
            nn.init.ones_(self.weight)
            if self.bias is not None:
                nn.init.zeros_(self.bias)

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.layer_norm(
            x,
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDLayerNorm.forward-cn>` |
        :ref:`English <TDLayerNorm.forward-en>`

        ----

        .. _TDLayerNorm.forward-cn:

        * **中文**

        对完整时间序列执行 TD LayerNorm。计算过程为：

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{LayerNorm}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        因此 ``Y.cumsum(dim=0)`` 与对 ``X.cumsum(dim=0)`` 逐时间步执行 ANN
        LayerNorm 的结果一致。输出是浮点差分值，可能为负，不是二值
        脉冲。
        当 ``T = 1`` 时，``Y[0]`` 直接等于对 ``X[0]`` 执行 LayerNorm 的
        结果。
        输出 dtype 与输入 dtype 相同，且该算子对 autograd 透明。

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``，且 ``T > 0``，尾部形状必须
            匹配 ``normalized_shape``。
        :type x_seq: torch.Tensor
        :return: TD LayerNorm 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维、时间维为空或尾部形状不匹配。

        ----

        .. _TDLayerNorm.forward-en:

        * **English**

        Apply TD LayerNorm to a complete time sequence:

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{LayerNorm}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        Thus, ``Y.cumsum(dim=0)`` matches ANN LayerNorm applied to
        ``X.cumsum(dim=0)`` at each time step. The output contains
        floating-point differential values, may be negative, and is not a binary
        spike tensor. When ``T = 1``, ``Y[0]`` is exactly LayerNorm applied to
        ``X[0]``. The output dtype matches the input dtype, and the operator is
        transparent to autograd.

        :param x_seq: Input time sequence with shape ``[T, ...]`` and
            ``T > 0``. The trailing shape must match ``normalized_shape``.
        :type x_seq: torch.Tensor
        :return: TD LayerNorm differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions, the time
            dimension is empty, or the trailing shape does not match.
        """
        _check_time_sequence(x_seq, "TDLayerNorm")
        if len(self.normalized_shape) > x_seq.dim() - 1:
            trailing_shape = tuple(x_seq.shape[1:])
        else:
            trailing_shape = tuple(x_seq.shape[-len(self.normalized_shape) :])
        if trailing_shape != self.normalized_shape:
            raise ValueError(
                "TDLayerNorm expects the trailing shape of x_seq to match "
                f"normalized_shape={self.normalized_shape}, but got "
                f"{trailing_shape}."
            )

        return self._td_sequence_forward(
            (x_seq,),
            lambda x_cum: F.layer_norm(
                x_cum,
                self.normalized_shape,
                self.weight,
                self.bias,
                self.eps,
            ),
        )

    def extra_repr(self) -> str:
        has_bias = self.bias is not None
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, bias={has_bias}"
        )


class TDRMSNorm(TDModule):
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int], torch.Size],
        eps: Optional[float] = None,
        elementwise_affine: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDRMSNorm.__init__-cn>` |
        :ref:`English <TDRMSNorm.__init__-en>`

        ----

        .. _TDRMSNorm.__init__-cn:

        * **中文**

        Temporal-difference (TD) RMSNorm 算子。``step_mode="m"`` 接收形状
        为 ``[T, ...]`` 的完整时间差分序列，先沿时间累积，对每个累积输入执行
        :func:`torch.nn.functional.rms_norm`，再返回输出的时间差分。
        ``step_mode="s"`` 接收单个时间步并维护累积状态。输出是可正可负的浮点
        差分，不是二值脉冲或 fully spike-driven RMSNorm。独立序列之间必须调用
        ``reset``。算子由 PyTorch 可微操作组成，支持 autograd，device 和 dtype
        约束与 :class:`torch.nn.RMSNorm` 一致。

        :param normalized_shape: 需要归一化的输入尾部形状。
        :type normalized_shape: int or Sequence[int] or torch.Size
        :param eps: 数值稳定项；``None`` 使用 PyTorch 基于输入 dtype 的默认值。
        :type eps: Optional[float]
        :param elementwise_affine: 是否使用可学习的逐元素 weight。
        :type elementwise_affine: bool
        :param device: 参数初始化设备。
        :type device: torch.device or str or None
        :param dtype: 参数初始化 dtype。
        :type dtype: torch.dtype or None
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。
        :type step_mode: str
        :raises ValueError: 若 ``step_mode`` 不是 ``"s"`` 或 ``"m"``。

        ----

        .. _TDRMSNorm.__init__-en:

        * **English**

        Temporal-difference (TD) RMSNorm operator. With ``step_mode="m"``, it
        accepts a complete differential sequence with shape ``[T, ...]``,
        accumulates the sequence over time, applies
        :func:`torch.nn.functional.rms_norm` to every cumulative input, and
        returns temporal differences of those outputs. With ``step_mode="s"``,
        it accepts one differential step and maintains cumulative state. Outputs
        are floating-point differences that may be negative; they are not binary
        spikes or a fully spike-driven RMSNorm. Call ``reset`` between
        independent sequences. The operator consists of differentiable PyTorch
        operations, supports autograd, and follows :class:`torch.nn.RMSNorm`
        device and dtype constraints.

        :param normalized_shape: Trailing input shape to normalize.
        :type normalized_shape: int or Sequence[int] or torch.Size
        :param eps: Numerical stability term. ``None`` uses PyTorch's input-dtype
            default.
        :type eps: Optional[float]
        :param elementwise_affine: Whether to use a learnable elementwise weight.
        :type elementwise_affine: bool
        :param device: Parameter initialization device.
        :type device: torch.device or str or None
        :param dtype: Parameter initialization dtype.
        :type dtype: torch.dtype or None
        :param step_mode: Step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        :raises ValueError: If ``step_mode`` is neither ``"s"`` nor ``"m"``.
        """
        super().__init__(step_mode)
        if isinstance(normalized_shape, int):
            normalized_shape = (normalized_shape,)
        else:
            normalized_shape = tuple(normalized_shape)
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.elementwise_affine = elementwise_affine
        if elementwise_affine:
            self.weight = nn.Parameter(
                torch.empty(self.normalized_shape, device=device, dtype=dtype)
            )
        else:
            self.register_parameter("weight", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        r"""
        **API Language** - 中文 | English

        将可学习的 RMSNorm weight 重置为 1。

        Reset the learnable RMSNorm weight to one.
        """
        if self.weight is not None:
            nn.init.ones_(self.weight)

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - 中文 | English

        对普通非时间序列输入执行 RMSNorm，不读取或修改 TD memory。

        Apply RMSNorm to an ordinary non-temporal input without reading or
        modifying TD memory.

        :param x: 尾部形状匹配 ``normalized_shape`` 的输入张量。 / Input tensor
            whose trailing shape matches ``normalized_shape``.
        :type x: torch.Tensor
        :return: RMSNorm 输出，形状、device 和 dtype 与输入一致。 / RMSNorm
            output with the same shape, device, and dtype as the input.
        :rtype: torch.Tensor
        :raises RuntimeError: 若输入尾部形状与 ``normalized_shape`` 不匹配。 /
            If the input trailing shape does not match ``normalized_shape``.
        """
        return F.rms_norm(x, self.normalized_shape, self.weight, self.eps)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        对完整时间序列执行 TD RMSNorm。
        Apply TD RMSNorm to a complete time sequence.

        :param x_seq: 输入时间差分序列，形状为 ``[T, ...]``，``T > 0``，尾部
            形状必须匹配 ``normalized_shape``。 / Input differential sequence
            with shape ``[T, ...]`` and ``T > 0`` whose trailing shape matches
            ``normalized_shape``.
        :type x_seq: torch.Tensor
        :return: 与输入形状相同的 TD RMSNorm 差分序列。 / TD RMSNorm
            differential sequence with the same shape as the input.
        :rtype: torch.Tensor
        :raises ValueError: 若时间维为空、输入维数不足或尾部形状不匹配。 / If
            the time dimension is empty, rank is insufficient, or the trailing
            shape does not match.
        """
        _check_time_sequence(x_seq, "TDRMSNorm")
        if len(self.normalized_shape) > x_seq.dim() - 1:
            trailing_shape = tuple(x_seq.shape[1:])
        else:
            trailing_shape = tuple(x_seq.shape[-len(self.normalized_shape) :])
        if trailing_shape != self.normalized_shape:
            raise ValueError(
                "TDRMSNorm expects the trailing shape of x_seq to match "
                f"normalized_shape={self.normalized_shape}, but got "
                f"{trailing_shape}."
            )
        return self._td_sequence_forward(
            (x_seq,),
            lambda x_cum: F.rms_norm(
                x_cum,
                self.normalized_shape,
                self.weight,
                self.eps,
            ),
        )

    def extra_repr(self) -> str:
        r"""
        **API Language** - 中文 | English

        返回用于 ``repr`` 的归一化形状、epsilon 和 affine 配置。

        Return the normalized shape, epsilon, and affine configuration used by
        ``repr``.

        :return: 模块配置字符串。 / Module configuration string.
        :rtype: str
        """
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}"
        )


class TDSiLU(TDModule):
    def __init__(self, step_mode: str = "m") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDSiLU.__init__-cn>` |
        :ref:`English <TDSiLU.__init__-en>`

        ----

        .. _TDSiLU.__init__-cn:

        * **中文**

        Temporal-difference (TD) SiLU 算子。多步模式先累积输入，对每个累积
        输入执行 :func:`torch.nn.functional.silu`，再返回输出差分；单步模式
        维护相同的跨时间状态。输出是可正可负的浮点差分，不是二值脉冲或 fully
        spike-driven SiLU。独立序列之间必须调用 ``reset``。该算子支持 PyTorch
        autograd，device 和 dtype 约束与 :func:`torch.nn.functional.silu` 一致。

        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。
        :type step_mode: str
        :raises ValueError: 若 ``step_mode`` 不是 ``"s"`` 或 ``"m"``。

        ----

        .. _TDSiLU.__init__-en:

        * **English**

        Temporal-difference (TD) SiLU operator. Multi-step mode accumulates the
        input, applies :func:`torch.nn.functional.silu` to every cumulative
        input, and returns output differences; single-step mode maintains the
        same temporal state. Outputs are floating-point differences that may be
        negative, not binary spikes or a fully spike-driven SiLU. Call ``reset``
        between independent sequences. The operator supports PyTorch autograd
        and follows :func:`torch.nn.functional.silu` device and dtype constraints.

        :param step_mode: Step mode, ``"s"`` or ``"m"``.
        :type step_mode: str
        :raises ValueError: If ``step_mode`` is neither ``"s"`` nor ``"m"``.
        """
        super().__init__(step_mode)

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language** - 中文 | English

        对普通非时间序列输入执行 SiLU，不读取或修改 TD memory。

        Apply SiLU to an ordinary non-temporal input without reading or
        modifying TD memory.

        :param x: 任意形状的浮点输入张量。 / Floating-point input tensor of any
            shape.
        :type x: torch.Tensor
        :return: SiLU 输出，形状、device 和 dtype 与输入一致。 / SiLU output
            with the same shape, device, and dtype as the input.
        :rtype: torch.Tensor
        """
        return F.silu(x)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        对完整时间序列执行 TD SiLU。 / Apply TD SiLU to a complete sequence.

        :param x_seq: 形状为 ``[T, ...]`` 且 ``T > 0`` 的输入差分序列。 /
            Input differential sequence with shape ``[T, ...]`` and ``T > 0``.
        :type x_seq: torch.Tensor
        :return: 与输入形状相同的 TD SiLU 差分序列。 / TD SiLU differential
            sequence with the same shape as the input.
        :rtype: torch.Tensor
        :raises ValueError: 若输入维数不足或时间维为空。 / If input rank is
            insufficient or the time dimension is empty.
        """
        _check_time_sequence(x_seq, "TDSiLU")
        return self._td_sequence_forward((x_seq,), F.silu)


class TDGELU(TDModule):
    def __init__(
        self,
        approximate: Literal["none", "tanh"] = "none",
        step_mode: str = "m",
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDGELU.__init__-cn>` |
        :ref:`English <TDGELU.__init__-en>`

        ----

        .. _TDGELU.__init__-cn:

        * **中文**

        Temporal-difference (TD) GELU（Gaussian Error Linear Unit）算子。
        ``step_mode="m"`` 时输入必须是完整时间序列，时间维固定为第 0 维，
        形状为 ``[T, ...]``；模块先对输入在时间维做累积，再对每个累积输入
        执行 :func:`torch.nn.functional.gelu`，最后返回累积输出在时间维上的
        差分。``step_mode="s"`` 时输入被解释为当前差分时间步，模块更新内部
        累积 memory 并返回当前差分输出；普通 GELU 路径由
        :meth:`ann_forward` 提供。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven GELU。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子是 stateful
        TD MemoryModule；重复处理独立序列前应调用 ``reset``。该算子仅依赖
        :func:`torch.nn.functional.gelu`，支持 CPU 与 CUDA，后端与
        :mod:`torch` 一致，无 CuPy / Triton 专用路径。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD GELU
        只实现张量级算子：在多步模式下它仍调用
        :func:`torch.nn.functional.gelu`，需要完整时间序列输入，不是逐时间步
        在线算子，也不是面向神经形态硬件的 fully spike-driven GELU。

        .. code-block:: python

            op = TDGELU(approximate="none")
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param approximate: GELU 近似模式，与 :class:`torch.nn.GELU` 的
            ``approximate`` 语义一致。
        :type approximate: Literal["none", "tanh"]
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str
        :raises ValueError: 若 ``approximate`` 不是 ``"none"`` 或 ``"tanh"``。

        ----

        .. _TDGELU.__init__-en:

        * **English**

        Temporal-difference (TD) GELU (Gaussian Error Linear Unit) operator.
        With ``step_mode="m"``, the input must be a complete time sequence
        whose time dimension is fixed at dimension 0, with shape ``[T, ...]``.
        This module first accumulates the input over time, applies
        :func:`torch.nn.functional.gelu` to each cumulative input, and returns
        the temporal difference of the cumulative outputs. With
        ``step_mode="s"``, the input is interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. The ordinary GELU path is
        exposed by :meth:`ann_forward`.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven GELU. The output dtype matches the input dtype;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is a
        stateful TD MemoryModule; call ``reset`` before processing an
        independent sequence. It
        only depends on :func:`torch.nn.functional.gelu`, supports CPU and CUDA,
        follows the :mod:`torch` backend behavior, and has no CuPy / Triton
        specific path.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: in
        multi-step mode it still calls :func:`torch.nn.functional.gelu`,
        requires a complete time sequence, is not a step-wise online operator,
        and is not a fully spike-driven GELU for neuromorphic hardware.

        .. code-block:: python

            op = TDGELU(approximate="none")
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param approximate: GELU approximation mode, with the same semantics as
            ``approximate`` in :class:`torch.nn.GELU`.
        :type approximate: Literal["none", "tanh"]
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        :raises ValueError: If ``approximate`` is not ``"none"`` or ``"tanh"``.
        """
        super().__init__(step_mode)
        if approximate not in ("none", "tanh"):
            raise ValueError(
                "TDGELU: approximate must be 'none' or 'tanh', "
                f"but got {approximate!r}."
            )
        self.approximate = approximate

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(x, approximate=self.approximate)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDGELU.forward-cn>` |
        :ref:`English <TDGELU.forward-en>`

        ----

        .. _TDGELU.forward-cn:

        * **中文**

        对完整时间序列执行 TD GELU。计算过程为：

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{GELU}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        因此 ``Y.cumsum(dim=0)`` 与对 ``X.cumsum(dim=0)`` 逐时间步执行 ANN
        GELU 的结果一致。输出是浮点差分值，可能为负，不是二值脉冲。
        当 ``T = 1`` 时，``Y[0]`` 直接等于对 ``X[0]`` 执行 GELU 的结果。
        输出 dtype 与输入 dtype 相同，且该算子对 autograd 透明。

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``，且 ``T > 0``。
        :type x_seq: torch.Tensor
        :return: TD GELU 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维或时间维为空。

        ----

        .. _TDGELU.forward-en:

        * **English**

        Apply TD GELU to a complete time sequence:

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = \operatorname{GELU}(X_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        Thus, ``Y.cumsum(dim=0)`` matches ANN GELU applied to
        ``X.cumsum(dim=0)`` at each time step. The output contains
        floating-point differential values, may be negative, and is not a binary
        spike tensor. When ``T = 1``, ``Y[0]`` is exactly GELU applied to
        ``X[0]``. The output dtype matches the input dtype, and the operator is
        transparent to autograd.

        :param x_seq: Input time sequence with shape ``[T, ...]`` and ``T > 0``.
        :type x_seq: torch.Tensor
        :return: TD GELU differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions or the time
            dimension is empty.
        """
        _check_time_sequence(x_seq, "TDGELU")

        return self._td_sequence_forward(
            (x_seq,), lambda x_cum: F.gelu(x_cum, approximate=self.approximate)
        )

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class TDLinear(TDModule):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <TDLinear.__init__-cn>` |
        :ref:`English <TDLinear.__init__-en>`

        ----

        .. _TDLinear.__init__-cn:

        * **中文**

        Temporal-difference (TD) Linear 算子。``step_mode="m"`` 时输入必须
        是完整时间序列，时间维固定为第 0 维，形状为
        ``[T, ..., in_features]``；模块返回 sequence-preserving affine
        差分序列，使 ``Y.cumsum(dim=0)`` 等于对 ``X.cumsum(dim=0)`` 逐时间
        步执行 :func:`torch.nn.functional.linear`。``step_mode="s"`` 时输入
        被解释为当前差分时间步，模块更新内部累积 memory 并返回当前差分输出；
        普通 Linear 路径由 :meth:`ann_forward` 提供。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Linear。输出 dtype 与 PyTorch Linear 一致；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子是 stateful
        TD MemoryModule；重复处理独立序列前应调用 ``reset``。该算子仅依赖 PyTorch
        Linear，支持 CPU 与 CUDA，后端与 :mod:`torch` 一致，无 CuPy / Triton
        专用路径。

        该算子用于处理带 bias 的 affine projection。普通
        :class:`torch.nn.Linear` 直接作用在 TD 差分序列上会在时间累积后得到
        ``T * bias``；TD Linear 使累计输出保持 ``W @ x_cum + bias``。当
        ``bias=False`` 时，该算子退化为普通逐时间步 Linear；当 ``bias=True``
        时，bias 只在第 0 个时间步进入差分序列。

        .. code-block:: python

            op = TDLinear(3, 5)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param in_features: 输入特征数。
        :type in_features: int
        :param out_features: 输出特征数。
        :type out_features: int
        :param bias: 若为 ``True``，使用可学习 bias 参数。
        :type bias: bool
        :param device: 参数初始化设备。
        :type device: torch.device or str or None
        :param dtype: 参数初始化 dtype。
        :type dtype: torch.dtype or None
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _TDLinear.__init__-en:

        * **English**

        Temporal-difference (TD) Linear operator. With ``step_mode="m"``, the
        input must be a complete time sequence whose time dimension is fixed at
        dimension 0, with shape ``[T, ..., in_features]``. This module returns a
        sequence-preserving affine differential sequence such that
        ``Y.cumsum(dim=0)`` matches :func:`torch.nn.functional.linear` applied
        to ``X.cumsum(dim=0)`` at every time step. With ``step_mode="s"``, the
        input is interpreted as the current differential time step; the module
        updates its cumulative memory and returns the current differential
        output. The ordinary Linear path is exposed by :meth:`ann_forward`.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        a fully spike-driven Linear. The output dtype follows PyTorch Linear;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is a
        stateful TD MemoryModule; call ``reset`` before processing an
        independent sequence. It
        only depends on PyTorch Linear, supports CPU and CUDA, follows the
        :mod:`torch` backend behavior, and has no CuPy / Triton specific path.

        This operator handles affine projections with bias. Applying ordinary
        :class:`torch.nn.Linear` directly to a TD differential sequence would
        accumulate the bias as ``T * bias``. TD Linear applies Linear to the
        cumulative input and then differences the cumulative output, preserving
        ``W @ x_cum + bias``. When ``bias=False``, this operator degenerates to
        ordinary per-time-step Linear. When ``bias=True``, the bias appears only
        at the first time step of the differential sequence.

        .. code-block:: python

            op = TDLinear(3, 5)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param in_features: Number of input features.
        :type in_features: int
        :param out_features: Number of output features.
        :type out_features: int
        :param bias: If ``True``, use a learnable bias parameter.
        :type bias: bool
        :param device: Device used to initialize parameters.
        :type device: torch.device or str or None
        :param dtype: Dtype used to initialize parameters.
        :type dtype: torch.dtype or None
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(
            torch.empty((out_features, in_features), **factory_kwargs)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight, self.bias)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language:**
        :ref:`中文 <TDLinear.forward-cn>` |
        :ref:`English <TDLinear.forward-en>`

        ----

        .. _TDLinear.forward-cn:

        * **中文**

        对完整时间序列执行 TD Linear。计算过程为：

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = X_{cum}[t] W^T + b

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        因此 ``Y.cumsum(dim=0)`` 与对 ``X.cumsum(dim=0)`` 逐时间步执行 ANN
        Linear 的结果一致。若 ``bias`` 为 ``None``，该计算等价于直接对
        ``X`` 逐时间步执行 Linear；若存在 bias，bias 只出现在 ``Y[0]`` 中，
        避免累计后得到 ``T * bias``。输出是浮点差分值，可能为负，不是二值
        脉冲。当 ``T = 1`` 时，``Y[0]`` 直接等于对 ``X[0]`` 执行 Linear 的
        结果。输出 dtype 与 PyTorch Linear 一致，且该算子对 autograd 透明。

        :param x_seq: 输入时间序列，形状为 ``[T, ..., in_features]``，且
            ``T > 0``。
        :type x_seq: torch.Tensor
        :return: TD Linear 差分序列，形状为 ``[T, ..., out_features]``。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维或时间维为空。

        ----

        .. _TDLinear.forward-en:

        * **English**

        Apply TD Linear to a complete time sequence:

        .. math::

            X_{cum}[t] = \sum_{i=0}^{t} X[i]

        .. math::

            Y_{cum}[t] = X_{cum}[t] W^T + b

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        Thus, ``Y.cumsum(dim=0)`` matches ANN Linear applied to
        ``X.cumsum(dim=0)`` at each time step. If ``bias`` is ``None``, this is
        equivalent to applying Linear to ``X`` at each time step directly. If a
        bias exists, the bias appears only in ``Y[0]``, avoiding ``T * bias``
        after accumulation. The output contains floating-point differential
        values, may be negative, and is not a binary spike tensor. When
        ``T = 1``, ``Y[0]`` is exactly Linear applied to ``X[0]``. The output
        dtype follows PyTorch Linear, and the operator is transparent to
        autograd.

        :param x_seq: Input time sequence with shape
            ``[T, ..., in_features]`` and ``T > 0``.
        :type x_seq: torch.Tensor
        :return: TD Linear differential sequence with shape
            ``[T, ..., out_features]``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions or the
            time dimension is empty.
        """
        _check_time_sequence(x_seq, "TDLinear")

        return self._td_sequence_forward((x_seq,), self.ann_forward)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class TDConv2d(TDModule):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: Union[int, Tuple[int, int]],
        stride: Union[int, Tuple[int, int]] = 1,
        padding: Union[str, int, Tuple[int, int]] = 0,
        dilation: Union[int, Tuple[int, int]] = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = "zeros",
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <TDConv2d.__init__-cn>` |
        :ref:`English <TDConv2d.__init__-en>`

        ----

        .. _TDConv2d.__init__-cn:

        * **中文**

        Temporal-difference (TD) Conv2d 算子。``step_mode="m"`` 时输入必须是
        完整时间序列，形状为 ``[T, N, C, H, W]``；返回的浮点差分序列满足
        ``Y.cumsum(dim=0)`` 等于对 ``X.cumsum(dim=0)`` 逐时间步执行
        :func:`torch.nn.functional.conv2d`。当存在 bias 时，bias 只出现在第
        0 个差分时间步，避免累计后得到 ``T * bias``。``step_mode="s"`` 时
        输入被解释为当前差分时间步，模块更新内部累积 memory 并返回当前差分
        输出；普通 Conv2d 路径由 :meth:`ann_forward` 提供。

        输出是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Conv2d。构造参数对齐 :class:`torch.nn.Conv2d` 的 2D
        convolution 参数，支持 ``padding="same"`` 和 ``padding="valid"``。

        :param in_channels: 输入通道数。
        :type in_channels: int
        :param out_channels: 输出通道数。
        :type out_channels: int
        :param kernel_size: 卷积核大小。
        :type kernel_size: int or Tuple[int, int]
        :param stride: 卷积步幅。
        :type stride: int or Tuple[int, int]
        :param padding: padding 参数，可为整数、tuple、``"same"`` 或
            ``"valid"``。
        :param dilation: dilation 参数。
        :param groups: 分组卷积组数。
        :param bias: 是否使用 learnable bias。
        :param padding_mode: padding 模式。
        :param device: 参数初始化设备。
        :param dtype: 参数初始化 dtype。
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。

        ----

        .. _TDConv2d.__init__-en:

        * **English**

        Temporal-difference (TD) Conv2d operator. With ``step_mode="m"``, input
        must be a complete time sequence with shape ``[T, N, C, H, W]``. The
        returned floating differential sequence satisfies ``Y.cumsum(dim=0)``
        matching :func:`torch.nn.functional.conv2d` applied to
        ``X.cumsum(dim=0)`` at each timestep. When bias is present, it appears
        only in ``Y[0]`` to avoid accumulating ``T * bias``. With
        ``step_mode="s"``, the input is interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. The ordinary Conv2d path is
        exposed by :meth:`ann_forward`.

        The output may contain negative floating-point differential values. It
        is not a binary spike tensor and does not represent a fully
        spike-driven Conv2d. Constructor arguments mirror the supported 2D
        convolution arguments of :class:`torch.nn.Conv2d`, including
        ``padding="same"`` and ``padding="valid"``.

        :param in_channels: Number of input channels.
        :type in_channels: int
        :param out_channels: Number of output channels.
        :type out_channels: int
        :param kernel_size: Convolution kernel size.
        :type kernel_size: int or Tuple[int, int]
        :param stride: Convolution stride.
        :type stride: int or Tuple[int, int]
        :param padding: Padding argument, which can be an int, tuple,
            ``"same"`` or ``"valid"``.
        :param dilation: Convolution dilation.
        :param groups: Number of convolution groups.
        :param bias: If ``True``, use a learnable bias parameter.
        :param padding_mode: Padding mode.
        :param device: Device used to initialize parameters.
        :param dtype: Dtype used to initialize parameters.
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        """
        super().__init__(step_mode)
        if groups <= 0:
            raise ValueError("groups must be a positive integer")
        if in_channels % groups != 0:
            raise ValueError("in_channels must be divisible by groups")
        if out_channels % groups != 0:
            raise ValueError("out_channels must be divisible by groups")
        if isinstance(padding, str):
            if padding not in {"same", "valid"}:
                raise ValueError("padding must be an int, a tuple, 'same', or 'valid'.")
            if padding == "same" and any(s != 1 for s in _pair(stride)):
                raise ValueError(
                    "padding='same' is not supported for strided convolutions"
                )
        if padding_mode not in {"zeros", "reflect", "replicate", "circular"}:
            raise ValueError(
                "padding_mode must be 'zeros', 'reflect', 'replicate' or 'circular'."
            )
        factory_kwargs = {"device": device, "dtype": dtype}
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = _pair(kernel_size)
        self.stride = _pair(stride)
        self.padding = padding if isinstance(padding, str) else _pair(padding)
        self.dilation = _pair(dilation)
        self.groups = groups
        self.padding_mode = padding_mode
        if isinstance(self.padding, str):
            self._reversed_padding_repeated_twice = [0, 0] * len(self.kernel_size)
            if self.padding == "same":
                for d, k, i in zip(
                    self.dilation,
                    self.kernel_size,
                    range(len(self.kernel_size) - 1, -1, -1),
                ):
                    total_padding = d * (k - 1)
                    left_pad = total_padding // 2
                    self._reversed_padding_repeated_twice[2 * i] = left_pad
                    self._reversed_padding_repeated_twice[2 * i + 1] = (
                        total_padding - left_pad
                    )
        else:
            self._reversed_padding_repeated_twice = _reverse_repeat_tuple(
                self.padding, 2
            )
        self.weight = nn.Parameter(
            torch.empty(
                (out_channels, in_channels // groups, *self.kernel_size),
                **factory_kwargs,
            )
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(out_channels, **factory_kwargs))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def _conv2d(self, x: torch.Tensor, bias: Optional[torch.Tensor]) -> torch.Tensor:
        padding = _pair(0) if self.padding == "valid" else self.padding
        if self.padding_mode != "zeros":
            x = F.pad(
                x,
                self._reversed_padding_repeated_twice,
                mode=self.padding_mode,
            )
            return F.conv2d(
                x,
                self.weight,
                bias,
                self.stride,
                _pair(0),
                self.dilation,
                self.groups,
            )
        return F.conv2d(
            x,
            self.weight,
            bias,
            self.stride,
            padding,
            self.dilation,
            self.groups,
        )

    def ann_forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._conv2d(x, self.bias)

    def multi_step_forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        _check_time_sequence(x_seq, "TDConv2d")
        if x_seq.dim() != 5:
            raise ValueError(
                "TDConv2d expects an input sequence with shape [T, N, C, H, W], "
                f"but got shape {tuple(x_seq.shape)}."
            )

        t, n = x_seq.shape[:2]

        def ann_forward(x_cum_seq: torch.Tensor) -> torch.Tensor:
            y = self._conv2d(x_cum_seq.flatten(0, 1), self.bias)
            return y.reshape(t, n, *y.shape[1:])

        return self._td_sequence_forward((x_seq,), ann_forward)

    def extra_repr(self) -> str:
        s = (
            f"{self.in_channels}, {self.out_channels}, "
            f"kernel_size={self.kernel_size}, stride={self.stride}"
        )
        if self.padding != (0, 0):
            padding_repr = (
                repr(self.padding) if isinstance(self.padding, str) else self.padding
            )
            s += f", padding={padding_repr}"
        if self.dilation != (1, 1):
            s += f", dilation={self.dilation}"
        if self.groups != 1:
            s += f", groups={self.groups}"
        if self.bias is None:
            s += ", bias=False"
        if self.padding_mode != "zeros":
            s += f", padding_mode={self.padding_mode!r}"
        return s


class SNNMatrixOperator(TDModule):
    def __init__(self, step_mode: str = "m") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNMatrixOperator.__init__-cn>` |
        :ref:`English <SNNMatrixOperator.__init__-en>`

        ----

        .. _SNNMatrixOperator.__init__-cn:

        * **中文**

        Sequence-preserving SNN 矩阵乘法算子。``step_mode="m"`` 时输入必须
        是两条完整时间序列，时间维固定为第 0 维，形状分别为
        ``[T, ..., M, N]`` 和 ``[T, ..., N, P]``；模块先分别对两条输入在时间
        维做累积，再执行 :func:`torch.matmul`，最后返回累积输出在时间维上的
        差分。``step_mode="s"`` 时输入被解释为当前差分时间步，模块更新内部
        累积 memory 并返回当前差分输出；普通 matmul 路径由
        :meth:`ann_forward` 提供。

        该算子满足
        ``Y.cumsum(dim=0) == torch.matmul(A.cumsum(dim=0), B.cumsum(dim=0))``。
        因而它保留 cross-time terms，例如 ``A[0] @ B[1]`` 与
        ``A[1] @ B[0]``；这不同于逐时间步执行 ``A[t] @ B[t]``。该算子是
        LAS ``SNNMatrixOperator`` prefix recurrence 的 sequence-preserving
        张量级形式，但不会在内部自动 ``sum(0)``。

        输出是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven matrix multiplication。dtype、device 与 broadcast 语义遵循
        :func:`torch.matmul`。该算子是 stateful TD MemoryModule；重复处理独立
        序列前应调用 ``reset``。

        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _SNNMatrixOperator.__init__-en:

        * **English**

        Sequence-preserving SNN matrix multiplication operator. With
        ``step_mode="m"``, the inputs must be two complete time sequences whose
        time dimension is fixed at dimension 0, with shapes
        ``[T, ..., M, N]`` and ``[T, ..., N, P]``. This module accumulates both
        inputs over time, applies :func:`torch.matmul`, and returns the
        temporal difference of the cumulative outputs. With ``step_mode="s"``,
        the inputs are interpreted as the current differential time step; the
        module updates its cumulative memory and returns the current
        differential output. The ordinary matmul path is exposed by
        :meth:`ann_forward`.

        The operator satisfies
        ``Y.cumsum(dim=0) == torch.matmul(A.cumsum(dim=0), B.cumsum(dim=0))``.
        Therefore it preserves cross-time terms such as ``A[0] @ B[1]`` and
        ``A[1] @ B[0]``; it is not equivalent to applying ``A[t] @ B[t]`` at
        each time step independently. It is the sequence-preserving tensor
        form of the LAS ``SNNMatrixOperator`` prefix recurrence, but it does
        not implicitly call ``sum(0)``.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven matrix multiplication. Dtype, device and broadcasting
        semantics follow :func:`torch.matmul`. The operator is a stateful TD
        MemoryModule; call ``reset`` before processing an independent sequence.

        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)

    def ann_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return torch.matmul(a, b)

    def multi_step_forward(
        self, a_seq: torch.Tensor, b_seq: torch.Tensor
    ) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNMatrixOperator.forward-cn>` |
        :ref:`English <SNNMatrixOperator.forward-en>`

        ----

        .. _SNNMatrixOperator.forward-cn:

        * **中文**

        对两条完整时间序列执行 sequence-preserving SNN 矩阵乘法：

        .. math::

            A_{cum}[t] = \sum_{i=0}^{t} A[i]

        .. math::

            B_{cum}[t] = \sum_{i=0}^{t} B[i]

        .. math::

            Y_{cum}[t] = A_{cum}[t] B_{cum}[t]

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        :param a_seq: 左输入序列，形状为 ``[T, ..., M, N]``，且 ``T > 0``。
        :type a_seq: torch.Tensor
        :param b_seq: 右输入序列，形状为 ``[T, ..., N, P]``，且 ``T > 0``。
        :type b_seq: torch.Tensor
        :return: 差分输出序列，形状为 ``[T, ..., M, P]``。
        :rtype: torch.Tensor
        :raises ValueError: 若输入少于 3 维、时间维为空或时间长度不一致。

        ----

        .. _SNNMatrixOperator.forward-en:

        * **English**

        Apply sequence-preserving SNN matrix multiplication to two complete time
        sequences:

        .. math::

            A_{cum}[t] = \sum_{i=0}^{t} A[i]

        .. math::

            B_{cum}[t] = \sum_{i=0}^{t} B[i]

        .. math::

            Y_{cum}[t] = A_{cum}[t] B_{cum}[t]

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        :param a_seq: Left input sequence with shape ``[T, ..., M, N]`` and
            ``T > 0``.
        :type a_seq: torch.Tensor
        :param b_seq: Right input sequence with shape ``[T, ..., N, P]`` and
            ``T > 0``.
        :type b_seq: torch.Tensor
        :return: Differential output sequence with shape ``[T, ..., M, P]``.
        :rtype: torch.Tensor
        :raises ValueError: If an input has fewer than 3 dimensions, the time
            dimension is empty, or the time lengths differ.
        """
        if a_seq.dim() < 3:
            raise ValueError(
                "SNNMatrixOperator expects a_seq with shape [T, ..., M, N] "
                f"and at least 3 dimensions, but got shape {tuple(a_seq.shape)}."
            )
        if b_seq.dim() < 3:
            raise ValueError(
                "SNNMatrixOperator expects b_seq with shape [T, ..., N, P] "
                f"and at least 3 dimensions, but got shape {tuple(b_seq.shape)}."
            )
        _check_pair_time_sequence(a_seq, b_seq, "a_seq", "b_seq", "SNNMatrixOperator")

        a_seq, b_seq = _align_sequence_ranks(a_seq, b_seq)
        return self._td_sequence_forward((a_seq, b_seq), torch.matmul)


class SNNElementWiseProduct(TDModule):
    def __init__(self, step_mode: str = "m") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNElementWiseProduct.__init__-cn>` |
        :ref:`English <SNNElementWiseProduct.__init__-en>`

        ----

        .. _SNNElementWiseProduct.__init__-cn:

        * **中文**

        Sequence-preserving SNN 逐元素乘法算子。``step_mode="m"`` 时输入
        必须是两条完整时间序列，时间维固定为第 0 维，形状为 ``[T, ...]``，
        非时间维遵循 PyTorch broadcast 规则；模块先分别对两条输入在时间维做
        累积，再执行逐元素乘法，最后返回累积输出在时间维上的差分。
        ``step_mode="s"`` 时输入被解释为当前差分时间步，模块更新内部累积
        memory 并返回当前差分输出；普通逐元素乘法路径由
        :meth:`ann_forward` 提供。

        该算子满足 ``Y.cumsum(dim=0) == A.cumsum(dim=0) * B.cumsum(dim=0)``。
        它是 LAS ``SNNMACOperator`` 核心乘法-累积语义的 sequence-preserving
        张量级形式，但不会在内部自动 ``sum(0)``；需要单步聚合时由调用方显式
        完成。

        输出是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven multiply-accumulate。dtype、device 与 broadcast 语义遵循
        PyTorch 逐元素乘法。该算子是 stateful TD MemoryModule；重复处理独立
        序列前应调用 ``reset``。

        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _SNNElementWiseProduct.__init__-en:

        * **English**

        Sequence-preserving SNN element-wise product operator. With
        ``step_mode="m"``, the inputs must be two complete time sequences whose
        time dimension is fixed at dimension 0, with shape ``[T, ...]``.
        Non-time dimensions follow PyTorch broadcasting rules. This module
        accumulates both inputs over time, applies element-wise multiplication,
        and returns the temporal difference of the cumulative outputs. With
        ``step_mode="s"``, the inputs are interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. The ordinary element-wise
        multiplication path is exposed by :meth:`ann_forward`.

        The operator satisfies
        ``Y.cumsum(dim=0) == A.cumsum(dim=0) * B.cumsum(dim=0)``. It is the
        sequence-preserving tensor form of the core multiply-accumulate
        semantics in LAS ``SNNMACOperator``, but it does not implicitly call
        ``sum(0)``; callers should aggregate explicitly when a single-step
        output is required.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven multiply-accumulate. Dtype, device and broadcasting
        semantics follow PyTorch element-wise multiplication. The operator is a
        stateful TD MemoryModule; call ``reset`` before processing an
        independent sequence.

        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)

    def ann_forward(self, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
        return a * b

    def multi_step_forward(
        self, a_seq: torch.Tensor, b_seq: torch.Tensor
    ) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNElementWiseProduct.forward-cn>` |
        :ref:`English <SNNElementWiseProduct.forward-en>`

        ----

        .. _SNNElementWiseProduct.forward-cn:

        * **中文**

        对两条完整时间序列执行 sequence-preserving SNN 逐元素乘法：

        .. math::

            A_{cum}[t] = \sum_{i=0}^{t} A[i]

        .. math::

            B_{cum}[t] = \sum_{i=0}^{t} B[i]

        .. math::

            Y_{cum}[t] = A_{cum}[t] \odot B_{cum}[t]

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        :param a_seq: 左输入序列，形状为 ``[T, ...]``，且 ``T > 0``。
        :type a_seq: torch.Tensor
        :param b_seq: 右输入序列，形状为 ``[T, ...]``，且 ``T > 0``。
        :type b_seq: torch.Tensor
        :return: 差分输出序列，形状由 PyTorch broadcast 规则决定。
        :rtype: torch.Tensor
        :raises ValueError: 若输入少于 2 维、时间维为空或时间长度不一致。

        ----

        .. _SNNElementWiseProduct.forward-en:

        * **English**

        Apply sequence-preserving SNN element-wise product to two complete time
        sequences:

        .. math::

            A_{cum}[t] = \sum_{i=0}^{t} A[i]

        .. math::

            B_{cum}[t] = \sum_{i=0}^{t} B[i]

        .. math::

            Y_{cum}[t] = A_{cum}[t] \odot B_{cum}[t]

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        :param a_seq: Left input sequence with shape ``[T, ...]`` and
            ``T > 0``.
        :type a_seq: torch.Tensor
        :param b_seq: Right input sequence with shape ``[T, ...]`` and
            ``T > 0``.
        :type b_seq: torch.Tensor
        :return: Differential output sequence whose shape follows PyTorch
            broadcasting rules.
        :rtype: torch.Tensor
        :raises ValueError: If an input has fewer than 2 dimensions, the time
            dimension is empty, or the time lengths differ.
        """
        _check_pair_time_sequence(
            a_seq, b_seq, "a_seq", "b_seq", "SNNElementWiseProduct"
        )

        a_seq, b_seq = _align_sequence_ranks(a_seq, b_seq)
        return self._td_sequence_forward((a_seq, b_seq), torch.mul)


class TDScaledDotProductAttention(TDModule):
    def __init__(
        self,
        is_causal: bool = False,
        scale: Optional[float] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDScaledDotProductAttention.__init__-cn>` |
        :ref:`English <TDScaledDotProductAttention.__init__-en>`

        ----

        .. _TDScaledDotProductAttention.__init__-cn:

        * **中文**

        Temporal-difference (TD) scaled dot-product attention 算子。
        ``step_mode="m"`` 时输入必须是完整时间序列，时间维固定为第 0 维。
        ``query_seq`` 的形状为 ``[T, ..., L, E]``，``key_seq`` 的形状为
        ``[T, ..., S, E]``，``value_seq`` 的形状为 ``[T, ..., S, Ev]``；
        模块先分别对 query、key、value 在时间维做累积，再调用
        :func:`torch.nn.functional.scaled_dot_product_attention`，最后返回
        累积输出在时间维上的差分。``step_mode="s"`` 时输入被解释为当前差分
        时间步，模块更新内部累积 memory 并返回当前差分输出；普通 SDPA 路径
        由 :meth:`ann_forward` 提供。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven attention。dtype、device 与 mask broadcast 语义遵循
        :func:`torch.nn.functional.scaled_dot_product_attention`；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子是 stateful
        TD MemoryModule；重复处理独立序列前应调用 ``reset``。该算子仅依赖 PyTorch
        SDPA，支持 CPU 与 CUDA，后端与 :mod:`torch` 一致，无 CuPy / Triton
        专用路径。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 算子的累积-差分等价转换思路。本文档中的 TD scaled
        dot-product attention 只实现张量级最小 primitive：在多步模式下它仍
        调用 PyTorch SDPA，需要完整时间序列输入，不是逐时间步在线算子，也
        不是面向神经形态硬件的 fully spike-driven attention。本实现固定
        ``dropout_p=0.0``，且不暴露 ``enable_gqa``。组合 TD Transformer
        block 时，普通带 bias 的 :class:`torch.nn.Linear` 不能直接作用在差分
        序列上，因为累计后 bias 会被重复累加；应使用 ``bias=False`` 或专门的
        TD Linear。

        .. code-block:: python

            op = TDScaledDotProductAttention()
            q_seq = torch.randn(4, 2, 3, 8)
            k_seq = torch.randn(4, 2, 5, 8)
            v_seq = torch.randn(4, 2, 5, 6)
            y_seq = op(q_seq, k_seq, v_seq)

        :param is_causal: 是否应用 causal attention mask。若为 ``True``，
            ``forward`` 中不能同时传入 ``attn_mask``。
        :type is_causal: bool
        :param scale: attention scale。若为 ``None``，使用 PyTorch SDPA 默认值。
        :type scale: Optional[float]
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str

        ----

        .. _TDScaledDotProductAttention.__init__-en:

        * **English**

        Temporal-difference (TD) scaled dot-product attention operator. With
        ``step_mode="m"``, the inputs must be complete time sequences whose
        time dimension is fixed at dimension 0. ``query_seq`` has shape
        ``[T, ..., L, E]``, ``key_seq`` has shape ``[T, ..., S, E]``, and
        ``value_seq`` has shape ``[T, ..., S, Ev]``. This module first
        accumulates query, key, and value over time, calls
        :func:`torch.nn.functional.scaled_dot_product_attention`, and returns
        the temporal difference of the cumulative outputs. With
        ``step_mode="s"``, the inputs are interpreted as the current
        differential time step; the module updates its cumulative memory and
        returns the current differential output. The ordinary SDPA path is
        exposed by :meth:`ann_forward`.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven attention. Dtype, device, and mask broadcasting
        follow :func:`torch.nn.functional.scaled_dot_product_attention`;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is a
        stateful TD MemoryModule; call ``reset`` before processing an
        independent sequence. It
        only depends on PyTorch SDPA, supports CPU and CUDA, follows the
        :mod:`torch` backend behavior, and has no CuPy / Triton specific path.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer operators in `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_. This
        implementation provides only a tensor-level minimal primitive: in
        multi-step mode it still calls PyTorch SDPA, requires a complete time
        sequence, is not a step-wise online operator, and is not fully
        spike-driven attention for neuromorphic hardware. This implementation
        fixes ``dropout_p=0.0`` and
        does not expose ``enable_gqa``. When composing TD Transformer blocks,
        ordinary :class:`torch.nn.Linear` layers with bias must not be applied
        directly to differential sequences, because the bias would be
        accumulated repeatedly; use ``bias=False`` or a dedicated TD Linear.

        .. code-block:: python

            op = TDScaledDotProductAttention()
            q_seq = torch.randn(4, 2, 3, 8)
            k_seq = torch.randn(4, 2, 5, 8)
            v_seq = torch.randn(4, 2, 5, 6)
            y_seq = op(q_seq, k_seq, v_seq)

        :param is_causal: Whether to apply causal attention masking. If
            ``True``, ``attn_mask`` must not be passed to ``forward``.
        :type is_causal: bool
        :param scale: Attention scale. If ``None``, use the PyTorch SDPA
            default.
        :type scale: Optional[float]
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        """
        super().__init__(step_mode)
        self.is_causal = is_causal
        self.scale = scale

    def ann_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_causal and attn_mask is not None:
            raise ValueError(
                "TDScaledDotProductAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )
        return F.scaled_dot_product_attention(
            query,
            key,
            value,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
            scale=self.scale,
        )

    def single_step_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        if self.is_causal and attn_mask is not None:
            raise ValueError(
                "TDScaledDotProductAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )
        query_cum, key_cum, value_cum = self._accumulate_inputs(query, key, value)
        y_cum = F.scaled_dot_product_attention(
            query_cum,
            key_cum,
            value_cum,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        return self._diff_output(y_cum)

    def multi_step_forward(
        self,
        query_seq: torch.Tensor,
        key_seq: torch.Tensor,
        value_seq: torch.Tensor,
        attn_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDScaledDotProductAttention.forward-cn>` |
        :ref:`English <TDScaledDotProductAttention.forward-en>`

        ----

        .. _TDScaledDotProductAttention.forward-cn:

        * **中文**

        对完整 query、key、value 时间序列执行 TD scaled dot-product
        attention。计算过程为：

        .. math::

            Q_{cum}[t] = \sum_{i=0}^{t} Q[i], \quad
            K_{cum}[t] = \sum_{i=0}^{t} K[i], \quad
            V_{cum}[t] = \sum_{i=0}^{t} V[i]

        .. math::

            Y_{cum}[t] =
            \operatorname{SDPA}(Q_{cum}[t], K_{cum}[t], V_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        因此 ``Y.cumsum(dim=0)`` 与对累积 query、key、value 逐时间步执行 ANN
        SDPA 的结果一致。输出是浮点差分值，可能为负，不是二值脉冲。当
        ``T = 1`` 时，``Y[0]`` 直接等于对第一步 query、key、value 执行
        SDPA 的结果。输出 dtype 与 PyTorch SDPA 一致，且该算子对 autograd
        透明。

        :param query_seq: query 时间序列，形状为 ``[T, ..., L, E]``，且
            ``T > 0``。
        :type query_seq: torch.Tensor
        :param key_seq: key 时间序列，形状为 ``[T, ..., S, E]``，且时间维长度
            必须与 ``query_seq`` 相同。
        :type key_seq: torch.Tensor
        :param value_seq: value 时间序列，形状为 ``[T, ..., S, Ev]``，且时间维
            长度必须与 ``query_seq`` 相同。
        :type value_seq: torch.Tensor
        :param attn_mask: attention mask，broadcast 语义与 PyTorch SDPA 一致。
        :type attn_mask: torch.Tensor or None
        :return: TD scaled dot-product attention 差分序列，形状为
            ``[T, ..., L, Ev]``。
        :rtype: torch.Tensor
        :raises ValueError: 若任一输入少于 3 维、时间维为空、三者时间维长度不一致，
            或 ``is_causal=True`` 时同时传入 ``attn_mask``。

        ----

        .. _TDScaledDotProductAttention.forward-en:

        * **English**

        Apply TD scaled dot-product attention to complete query, key, and value
        time sequences:

        .. math::

            Q_{cum}[t] = \sum_{i=0}^{t} Q[i], \quad
            K_{cum}[t] = \sum_{i=0}^{t} K[i], \quad
            V_{cum}[t] = \sum_{i=0}^{t} V[i]

        .. math::

            Y_{cum}[t] =
            \operatorname{SDPA}(Q_{cum}[t], K_{cum}[t], V_{cum}[t])

        .. math::

            Y[0] = Y_{cum}[0], \quad
            Y[t] = Y_{cum}[t] - Y_{cum}[t-1]

        Thus, ``Y.cumsum(dim=0)`` matches ANN SDPA applied to cumulative query,
        key, and value at each time step. The output contains floating-point
        differential values, may be negative, and is not a binary spike tensor.
        When ``T = 1``, ``Y[0]`` is exactly SDPA applied to the first query,
        key, and value step. The output dtype follows PyTorch SDPA, and the
        operator is transparent to autograd.

        :param query_seq: Query time sequence with shape ``[T, ..., L, E]`` and
            ``T > 0``.
        :type query_seq: torch.Tensor
        :param key_seq: Key time sequence with shape ``[T, ..., S, E]``. Its
            time dimension length must match ``query_seq``.
        :type key_seq: torch.Tensor
        :param value_seq: Value time sequence with shape ``[T, ..., S, Ev]``.
            Its time dimension length must match ``query_seq``.
        :type value_seq: torch.Tensor
        :param attn_mask: Attention mask with the same broadcast semantics as
            PyTorch SDPA.
        :type attn_mask: torch.Tensor or None
        :return: TD scaled dot-product attention differential sequence with
            shape ``[T, ..., L, Ev]``.
        :rtype: torch.Tensor
        :raises ValueError: If any input has fewer than 3 dimensions, any time
            dimension is empty, the time lengths differ, or ``attn_mask`` is
            passed when ``is_causal=True``.
        """
        _check_attention_sequence(query_seq, "query_seq", "TDScaledDotProductAttention")
        _check_attention_sequence(key_seq, "key_seq", "TDScaledDotProductAttention")
        _check_attention_sequence(value_seq, "value_seq", "TDScaledDotProductAttention")
        if (
            query_seq.shape[0] != key_seq.shape[0]
            or query_seq.shape[0] != value_seq.shape[0]
        ):
            raise ValueError(
                "TDScaledDotProductAttention expects query_seq, key_seq, and "
                "value_seq to have the same time length, but got "
                f"{query_seq.shape[0]}, {key_seq.shape[0]}, and "
                f"{value_seq.shape[0]}."
            )
        if self.is_causal and attn_mask is not None:
            raise ValueError(
                "TDScaledDotProductAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )
        return self._td_sequence_forward(
            (query_seq, key_seq, value_seq),
            lambda query_cum, key_cum, value_cum: F.scaled_dot_product_attention(
                query_cum,
                key_cum,
                value_cum,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=self.is_causal,
                scale=self.scale,
            ),
        )

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, scale={self.scale}"


class TDMultiheadAttention(TDModule):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
        step_mode: str = "m",
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <TDMultiheadAttention.__init__-cn>` |
        :ref:`English <TDMultiheadAttention.__init__-en>`

        ----

        .. _TDMultiheadAttention.__init__-cn:

        * **中文**

        Temporal-difference (TD) MultiheadAttention 的窄子集实现。
        ``step_mode="m"`` 时输入必须是完整时间序列，时间维固定为第 0 维，
        形状为 ``[T, batch, seq, embed_dim]``；该模块使用 ``TDLinear`` 生成
        q/k/v projection，执行 TD scaled dot-product attention，再用
        ``TDLinear`` 执行输出 projection。``step_mode="s"`` 时输入被解释为
        当前差分时间步，形状为 ``[batch, seq, embed_dim]``，模块更新内部累积
        memory 并返回当前差分输出；普通 MultiheadAttention 数值路径由
        :meth:`ann_forward` 提供。

        返回值是 ``(attn_output_seq, None)``，用于匹配
        :class:`torch.nn.MultiheadAttention` 在 ``need_weights=False`` 时的
        tuple 返回结构。输出是浮点差分值，不是二值脉冲，也不是 fully
        spike-driven attention。输出 dtype 跟随 PyTorch Linear / SDPA；
        推荐使用 ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。
        该算子完全由 PyTorch 可微算子组成，对 autograd 透明。该算子是
        stateful TD MemoryModule；重复处理独立序列前应调用 ``reset``；支持 CPU 与 CUDA，
        后端与 :mod:`torch` 一致，无 CuPy / Triton 专用路径。当前只支持
        ``dropout=0.0``、``batch_first=True`` 和 ``need_weights=False``。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中的
        累积-差分等价转换思路。本实现是窄子集 TD wrapper，仍使用浮点
        ``TDLinear`` 和 PyTorch SDPA，不是逐时间步在线 attention，也不是面向
        神经形态硬件的 fully spike-driven MultiheadAttention。``bias=True``
        时 projection bias 由 ``TDLinear`` 在累积输入上处理，避免普通
        ``nn.Linear`` 直接作用在差分序列时产生重复累计 bias。
        父模块的 ``step_mode`` 会同步到内部 q/k/v/out projection。常规
        ``forward`` 调用由父模块的 ``step_mode`` 分发；直接调用
        ``single_step_forward`` 或 ``multi_step_forward`` 时，父模块会显式调用内部
        projection 的对应 step 方法，而不依赖子模块当前 ``step_mode``。

        .. code-block:: python

            op = TDMultiheadAttention(embed_dim=8, num_heads=2)
            x_seq = torch.randn(4, 2, 5, 8)
            y_seq, weights = op(x_seq, x_seq, x_seq, need_weights=False)

        :param embed_dim: 输入和输出 embedding 维度。
        :type embed_dim: int
        :param num_heads: attention head 数量，必须整除 ``embed_dim``。
        :type num_heads: int
        :param dropout: attention dropout。当前必须为 ``0.0``。
        :type dropout: float
        :param bias: 若为 ``True``，q/k/v 和 out projection 使用 bias。
        :type bias: bool
        :param batch_first: 当前必须为 ``True``，即每个时间步的输入形状为
            ``[batch, seq, embed_dim]``。
        :type batch_first: bool
        :param device: 参数初始化设备。
        :type device: torch.device or str or None
        :param dtype: 参数初始化 dtype。
        :type dtype: torch.dtype or None
        :param step_mode: 步进模式，``"s"`` 或 ``"m"``。默认 ``"m"``。
        :type step_mode: str
        :raises ValueError: 若 ``embed_dim`` 不能被 ``num_heads`` 整除、或传入
            当前不支持的 ``dropout`` / ``batch_first``。

        ----

        .. _TDMultiheadAttention.__init__-en:

        * **English**

        Narrow temporal-difference (TD) MultiheadAttention implementation. With
        ``step_mode="m"``, the input must be a complete time sequence whose
        time dimension is fixed at dimension 0, with shape
        ``[T, batch, seq, embed_dim]``. This module uses ``TDLinear`` for q/k/v
        projections, applies TD scaled dot-product attention, and then applies
        a ``TDLinear`` output projection. With ``step_mode="s"``, the input is
        interpreted as the current differential time step with shape
        ``[batch, seq, embed_dim]``; the module updates its cumulative memory
        and returns the current differential output. The ordinary
        MultiheadAttention numeric path is exposed by :meth:`ann_forward`.

        The return value is ``(attn_output_seq, None)`` to match the tuple
        structure of :class:`torch.nn.MultiheadAttention` when
        ``need_weights=False``. The output contains floating-point differential
        values, is not a binary spike tensor, and is not fully spike-driven
        attention. The output dtype follows PyTorch Linear / SDPA;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is a
        stateful TD MemoryModule; call ``reset`` before processing an
        independent sequence. It
        supports CPU and CUDA, follows the :mod:`torch` backend behavior, and
        has no CuPy / Triton specific path. Currently only ``dropout=0.0``,
        ``batch_first=True`` and ``need_weights=False`` are supported.

        The mechanism follows the cumulative-difference equivalence idea in
        `SpikeZIP-TF: Conversion is All You Need for Transformer-based SNN
        <https://arxiv.org/abs/2406.03470>`_. This implementation is a narrow
        TD wrapper: it still uses floating-point ``TDLinear`` and PyTorch SDPA,
        is not step-wise online attention, and is not fully spike-driven
        MultiheadAttention for neuromorphic hardware. When ``bias=True``,
        projection biases are handled by ``TDLinear`` on cumulative inputs,
        avoiding the repeated bias accumulation that would occur if ordinary
        ``nn.Linear`` were applied directly to differential sequences.
        The parent module's ``step_mode`` is synchronized to the internal
        q/k/v/out projections. Regular ``forward`` calls are dispatched by the
        parent ``step_mode``; when ``single_step_forward`` or
        ``multi_step_forward`` is called directly, the parent explicitly invokes
        the matching child projection step method instead of depending on the
        child modules' current ``step_mode``.

        .. code-block:: python

            op = TDMultiheadAttention(embed_dim=8, num_heads=2)
            x_seq = torch.randn(4, 2, 5, 8)
            y_seq, weights = op(x_seq, x_seq, x_seq, need_weights=False)

        :param embed_dim: Input and output embedding dimension.
        :type embed_dim: int
        :param num_heads: Number of attention heads. Must divide ``embed_dim``.
        :type num_heads: int
        :param dropout: Attention dropout. It must be ``0.0`` currently.
        :type dropout: float
        :param bias: If ``True``, use bias in q/k/v and output projections.
        :type bias: bool
        :param batch_first: Must be ``True`` currently. Each time step has shape
            ``[batch, seq, embed_dim]``.
        :type batch_first: bool
        :param device: Device used to initialize parameters.
        :type device: torch.device or str or None
        :param dtype: Dtype used to initialize parameters.
        :type dtype: torch.dtype or None
        :param step_mode: Step mode, ``"s"`` or ``"m"``. The default is ``"m"``.
        :type step_mode: str
        :raises ValueError: If ``embed_dim`` is not divisible by ``num_heads``,
            or unsupported ``dropout`` / ``batch_first`` is passed.
        """
        super().__init__(step_mode)
        if embed_dim <= 0:
            raise ValueError("embed_dim must be positive.")
        if num_heads <= 0:
            raise ValueError("num_heads must be positive.")
        if embed_dim % num_heads != 0:
            raise ValueError("embed_dim must be divisible by num_heads.")
        if dropout != 0.0:
            raise ValueError("TDMultiheadAttention only supports dropout=0.0.")
        if not batch_first:
            raise ValueError("TDMultiheadAttention only supports batch_first=True.")

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.dropout = dropout
        self.batch_first = batch_first
        self.head_dim = embed_dim // num_heads

        factory_kwargs = {"device": device, "dtype": dtype}
        self.q_proj = TDLinear(
            embed_dim, embed_dim, bias=bias, step_mode=step_mode, **factory_kwargs
        )
        self.k_proj = TDLinear(
            embed_dim, embed_dim, bias=bias, step_mode=step_mode, **factory_kwargs
        )
        self.v_proj = TDLinear(
            embed_dim, embed_dim, bias=bias, step_mode=step_mode, **factory_kwargs
        )
        self.out_proj = TDLinear(
            embed_dim, embed_dim, bias=bias, step_mode=step_mode, **factory_kwargs
        )
        self.step_mode = step_mode

    def reset(self):
        super().reset()
        self.q_proj.reset()
        self.k_proj.reset()
        self.v_proj.reset()
        self.out_proj.reset()

    @TDModule.step_mode.setter
    def step_mode(self, value: str):
        base.StepModule.step_mode.fset(self, value)
        if hasattr(self, "q_proj"):
            self.q_proj.step_mode = value
            self.k_proj.step_mode = value
            self.v_proj.step_mode = value
            self.out_proj.step_mode = value

    def _split_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        if x_seq.dim() != 4:
            raise ValueError(
                "TDMultiheadAttention expects input with shape "
                f"[T, batch, seq, embed_dim], but got {tuple(x_seq.shape)}."
            )
        if x_seq.shape[-1] != self.embed_dim:
            raise ValueError(
                "TDMultiheadAttention expects the last dimension to match "
                f"embed_dim={self.embed_dim}, but got {x_seq.shape[-1]}."
            )
        t, batch_size, seq_len, _ = x_seq.shape
        x_seq = x_seq.reshape(t, batch_size, seq_len, self.num_heads, self.head_dim)
        return x_seq.transpose(2, 3)

    def _merge_heads(self, x_seq: torch.Tensor) -> torch.Tensor:
        t, batch_size, _, seq_len, _ = x_seq.shape
        x_seq = x_seq.transpose(2, 3).contiguous()
        return x_seq.reshape(t, batch_size, seq_len, self.embed_dim)

    def _split_heads_single(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                "TDMultiheadAttention expects single-step input with shape "
                f"[batch, seq, embed_dim], but got {tuple(x.shape)}."
            )
        if x.shape[-1] != self.embed_dim:
            raise ValueError(
                "TDMultiheadAttention expects the last dimension to match "
                f"embed_dim={self.embed_dim}, but got {x.shape[-1]}."
            )
        batch_size, seq_len, _ = x.shape
        x = x.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
        return x.transpose(1, 2)

    def _merge_heads_single(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, _, seq_len, _ = x.shape
        x = x.transpose(1, 2).contiguous()
        return x.reshape(batch_size, seq_len, self.embed_dim)

    def _canonical_mha_attn_mask(
        self,
        attn_mask: Optional[torch.Tensor],
        batch_size: int,
    ) -> Optional[torch.Tensor]:
        if attn_mask is None:
            return None
        if attn_mask.dtype == torch.bool:
            attn_mask = torch.logical_not(attn_mask)
        if attn_mask.dim() == 3 and attn_mask.shape[0] == batch_size * self.num_heads:
            return attn_mask.reshape(batch_size, self.num_heads, *attn_mask.shape[1:])
        return attn_mask

    def _check_forward_options(
        self,
        key_padding_mask: Optional[torch.Tensor],
        need_weights: bool,
        average_attn_weights: bool,
    ) -> None:
        if need_weights:
            raise ValueError("TDMultiheadAttention only supports need_weights=False.")
        if key_padding_mask is not None:
            raise ValueError("TDMultiheadAttention does not support key_padding_mask.")
        if not average_attn_weights:
            raise ValueError(
                "TDMultiheadAttention does not support average_attn_weights=False."
            )

    def _check_attention_leading_dims(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        name: str,
    ) -> None:
        if q.shape[:-2] != k.shape[:-2] or q.shape[:-2] != v.shape[:-2]:
            raise ValueError(
                f"{name} requires query, key and value leading dimensions to "
                f"match exactly before SDPA, but got {tuple(q.shape[:-2])}, "
                f"{tuple(k.shape[:-2])} and {tuple(v.shape[:-2])}."
            )

    def ann_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        self._check_forward_options(
            key_padding_mask, need_weights, average_attn_weights
        )

        q = self._split_heads_single(self.q_proj.ann_forward(query))
        k = self._split_heads_single(self.k_proj.ann_forward(key))
        v = self._split_heads_single(self.v_proj.ann_forward(value))
        self._check_attention_leading_dims(q, k, v, "TDMultiheadAttention")
        if is_causal and attn_mask is not None:
            raise ValueError(
                "TDMultiheadAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )
        attn_mask = self._canonical_mha_attn_mask(attn_mask, q.shape[0])
        attn = F.scaled_dot_product_attention(
            q,
            k,
            v,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        out = self.out_proj.ann_forward(self._merge_heads_single(attn))
        return out, None

    def single_step_forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        self._check_forward_options(
            key_padding_mask, need_weights, average_attn_weights
        )

        q = self._split_heads_single(self.q_proj.single_step_forward(query))
        k = self._split_heads_single(self.k_proj.single_step_forward(key))
        v = self._split_heads_single(self.v_proj.single_step_forward(value))
        self._check_attention_leading_dims(q, k, v, "TDMultiheadAttention")
        if is_causal and attn_mask is not None:
            raise ValueError(
                "TDMultiheadAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )
        attn_mask = self._canonical_mha_attn_mask(attn_mask, q.shape[0])
        q_cum, k_cum, v_cum = self._accumulate_inputs(q, k, v)
        attn_cum = F.scaled_dot_product_attention(
            q_cum,
            k_cum,
            v_cum,
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=is_causal,
        )
        attn = self._diff_output(attn_cum)
        out = self.out_proj.single_step_forward(self._merge_heads_single(attn))
        return out, None

    def multi_step_forward(
        self,
        query_seq: torch.Tensor,
        key_seq: torch.Tensor,
        value_seq: torch.Tensor,
        key_padding_mask: Optional[torch.Tensor] = None,
        need_weights: bool = False,
        attn_mask: Optional[torch.Tensor] = None,
        average_attn_weights: bool = True,
        is_causal: bool = False,
    ) -> Tuple[torch.Tensor, None]:
        r"""
        **API Language:**
        :ref:`中文 <TDMultiheadAttention.forward-cn>` |
        :ref:`English <TDMultiheadAttention.forward-en>`

        ----

        .. _TDMultiheadAttention.forward-cn:

        * **中文**

        对完整 query/key/value 时间序列执行 TD multi-head attention。输入形状
        为 ``[T, batch, seq, embed_dim]``，且 ``T > 0``。当
        ``need_weights=False`` 时返回 ``(attn_output_seq, None)``。输出是浮点
        差分值，且 ``attn_output_seq.cumsum(dim=0)`` 与对累积输入逐时间步执行
        支持子集内的 ANN MultiheadAttention 输出一致。当 ``T = 1`` 时，
        ``attn_output_seq[0]`` 等于支持子集内 ANN MultiheadAttention 对第一步
        输入的输出。输出 dtype 与 PyTorch Linear / SDPA 一致，且该算子对
        autograd 透明。

        :param query_seq: query 时间序列，形状为
            ``[T, batch, target_len, embed_dim]``。
        :type query_seq: torch.Tensor
        :param key_seq: key 时间序列，形状为
            ``[T, batch, source_len, embed_dim]``。
        :type key_seq: torch.Tensor
        :param value_seq: value 时间序列，形状为
            ``[T, batch, source_len, embed_dim]``。
        :type value_seq: torch.Tensor
        :param key_padding_mask: 当前不支持，必须为 ``None``。
        :type key_padding_mask: torch.Tensor or None
        :param need_weights: 当前必须为 ``False``。
        :type need_weights: bool
        :param attn_mask: attention mask，语义与
            :class:`torch.nn.MultiheadAttention` 一致；bool mask 中 ``True``
            表示禁止 attention。
        :type attn_mask: torch.Tensor or None
        :param average_attn_weights: 为兼容
            :class:`torch.nn.MultiheadAttention` 调用签名保留；由于当前不返回
            attention weights，必须为 ``True``。
        :type average_attn_weights: bool
        :param is_causal: 是否应用 causal attention mask。
        :type is_causal: bool
        :return: ``(attn_output_seq, None)``，其中 ``attn_output_seq`` 形状为
            ``[T, batch, target_len, embed_dim]``。
        :rtype: Tuple[torch.Tensor, None]
        :raises ValueError: 若传入不支持的 mask/options 或非法输入形状。

        ----

        .. _TDMultiheadAttention.forward-en:

        * **English**

        Apply TD multi-head attention to complete query/key/value time
        sequences. Inputs have shape ``[T, batch, seq, embed_dim]`` with
        ``T > 0``. When ``need_weights=False``, this method returns
        ``(attn_output_seq, None)``. The output contains floating-point
        differential values, and ``attn_output_seq.cumsum(dim=0)`` matches ANN
        MultiheadAttention in the supported subset applied to cumulative inputs
        at each time step. When ``T = 1``, ``attn_output_seq[0]`` equals the
        output of ANN MultiheadAttention in the supported subset applied to the
        first input step. The output dtype follows PyTorch Linear / SDPA, and
        the operator is transparent to autograd.

        :param query_seq: Query sequence with shape
            ``[T, batch, target_len, embed_dim]`` and ``T > 0``.
        :type query_seq: torch.Tensor
        :param key_seq: Key sequence with shape
            ``[T, batch, source_len, embed_dim]``.
        :type key_seq: torch.Tensor
        :param value_seq: Value sequence with shape
            ``[T, batch, source_len, embed_dim]``.
        :type value_seq: torch.Tensor
        :param key_padding_mask: Unsupported in this narrow implementation.
        :type key_padding_mask: torch.Tensor or None
        :param need_weights: Must be ``False``. Attention weights are not
            implemented.
        :type need_weights: bool
        :param attn_mask: Optional attention mask with the same semantics as
            :class:`torch.nn.MultiheadAttention`; ``True`` values in a bool mask
            disallow attention.
        :type attn_mask: torch.Tensor or None
        :param average_attn_weights: Kept for
            :class:`torch.nn.MultiheadAttention` signature compatibility. It
            must be ``True`` because attention weights are not returned.
        :type average_attn_weights: bool
        :param is_causal: Whether to apply causal masking.
        :type is_causal: bool
        :return: ``(attn_output_seq, None)`` where ``attn_output_seq`` has shape
            ``[T, batch, target_len, embed_dim]``.
        :rtype: Tuple[torch.Tensor, None]
        :raises ValueError: If unsupported masks/options or invalid shapes are
            passed.
        """
        self._check_forward_options(
            key_padding_mask, need_weights, average_attn_weights
        )
        if is_causal and attn_mask is not None:
            raise ValueError(
                "TDMultiheadAttention does not allow attn_mask when "
                "is_causal=True; use one masking mode at a time."
            )

        q_seq = self._split_heads(self.q_proj.multi_step_forward(query_seq))
        k_seq = self._split_heads(self.k_proj.multi_step_forward(key_seq))
        v_seq = self._split_heads(self.v_proj.multi_step_forward(value_seq))
        self._check_attention_leading_dims(q_seq, k_seq, v_seq, "TDMultiheadAttention")
        attn_mask = self._canonical_mha_attn_mask(attn_mask, q_seq.shape[1])
        attn_seq = self._td_sequence_forward(
            (q_seq, k_seq, v_seq),
            lambda q_cum, k_cum, v_cum: F.scaled_dot_product_attention(
                q_cum,
                k_cum,
                v_cum,
                attn_mask=attn_mask,
                dropout_p=0.0,
                is_causal=is_causal,
            ),
        )
        out_seq = self.out_proj.multi_step_forward(self._merge_heads(attn_seq))
        return out_seq, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
