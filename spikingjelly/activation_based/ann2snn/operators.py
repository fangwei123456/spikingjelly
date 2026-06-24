import math
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = [
    "TDSoftmax",
    "TDLayerNorm",
    "TDGELU",
    "TDLinear",
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


def _td_scaled_dot_product_attention(
    query_seq: torch.Tensor,
    key_seq: torch.Tensor,
    value_seq: torch.Tensor,
    attn_mask: Optional[torch.Tensor],
    is_causal: bool,
    scale: Optional[float],
    module_name: str,
) -> torch.Tensor:
    _check_attention_sequence(query_seq, "query_seq", module_name)
    _check_attention_sequence(key_seq, "key_seq", module_name)
    _check_attention_sequence(value_seq, "value_seq", module_name)

    if (
        query_seq.shape[0] != key_seq.shape[0]
        or query_seq.shape[0] != value_seq.shape[0]
    ):
        raise ValueError(
            f"{module_name} expects query_seq, key_seq, and value_seq to have "
            "the same time length, but got "
            f"{query_seq.shape[0]}, {key_seq.shape[0]}, and "
            f"{value_seq.shape[0]}."
        )
    if is_causal and attn_mask is not None:
        raise ValueError(
            f"{module_name} does not allow attn_mask when is_causal=True; "
            "use one masking mode at a time."
        )

    y_cum = F.scaled_dot_product_attention(
        query_seq.cumsum(dim=0),
        key_seq.cumsum(dim=0),
        value_seq.cumsum(dim=0),
        attn_mask=attn_mask,
        dropout_p=0.0,
        is_causal=is_causal,
        scale=scale,
    )
    return _temporal_difference(y_cum)


class TDSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDSoftmax.__init__-cn>` |
        :ref:`English <TDSoftmax.__init__-en>`

        ----

        .. _TDSoftmax.__init__-cn:

        * **中文**

        Temporal-difference (TD) Softmax 算子。输入必须是完整时间序列，
        时间维固定为第 0 维，形状为 ``[T, ...]``。该模块先对输入在时间维
        做累积，再沿 ``dim`` 计算 ``torch.softmax``，最后返回累积输出在
        时间维上的差分。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Softmax。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD
        Softmax 只实现张量级算子：它仍调用 ``torch.softmax``，需要完整时间
        序列输入，不是逐时间步在线算子，也不是面向神经形态硬件的 fully
        spike-driven Softmax。

        .. code-block:: python

            op = TDSoftmax(dim=-1)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param dim: Softmax 归一化维度。不能为第 0 维，因为第 0 维保留为时间维。
        :type dim: int

        ----

        .. _TDSoftmax.__init__-en:

        * **English**

        Temporal-difference (TD) Softmax operator. The input must be a complete
        time sequence whose time dimension is fixed at dimension 0, with shape
        ``[T, ...]``. This module first accumulates the input over time, applies
        ``torch.softmax`` along ``dim`` to each cumulative input, and returns
        the temporal difference of the cumulative outputs.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven Softmax. The output dtype matches the input dtype;
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: it still
        calls ``torch.softmax``, requires a complete time sequence, is not a
        step-wise online operator, and is not a fully spike-driven Softmax for
        neuromorphic hardware.

        .. code-block:: python

            op = TDSoftmax(dim=-1)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param dim: Softmax normalization dimension. It must not be dimension 0,
            which is reserved as the time dimension.
        :type dim: int
        """
        super().__init__()
        self.dim = dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
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

        dim = self.dim
        if dim < 0:
            dim += x_seq.dim()
        if dim < 0 or dim >= x_seq.dim():
            raise ValueError(
                f"dim must be in the range [{-x_seq.dim()}, {x_seq.dim() - 1}], "
                f"but got {self.dim} for an input with {x_seq.dim()} dimensions."
            )
        if dim == 0:
            raise ValueError(
                "TDSoftmax reserves dimension 0 as the time dimension; "
                "softmax dim must not resolve to 0."
            )

        y_cum = torch.softmax(x_seq.cumsum(dim=0), dim=dim)
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        return f"dim={self.dim}"


class TDLayerNorm(nn.Module):
    def __init__(
        self,
        normalized_shape: Union[int, Sequence[int], torch.Size],
        eps: float = 1e-5,
        elementwise_affine: bool = True,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDLayerNorm.__init__-cn>` |
        :ref:`English <TDLayerNorm.__init__-en>`

        ----

        .. _TDLayerNorm.__init__-cn:

        * **中文**

        Temporal-difference (TD) LayerNorm 算子。输入必须是完整时间序列，
        时间维固定为第 0 维，形状为 ``[T, ...]``。该模块先对输入在时间维
        做累积，再对每个累积输入执行
        :func:`torch.nn.functional.layer_norm`，最后返回累积输出在时间维上的
        差分。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven LayerNorm。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。该算子无内部状态，多次 ``forward``
        之间不需要调用 ``reset``。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD
        LayerNorm 只实现张量级算子：它仍调用
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

        ----

        .. _TDLayerNorm.__init__-en:

        * **English**

        Temporal-difference (TD) LayerNorm operator. The input must be a
        complete time sequence whose time dimension is fixed at dimension 0,
        with shape ``[T, ...]``. This module first accumulates the input over
        time, applies :func:`torch.nn.functional.layer_norm` to each cumulative
        input, and returns the temporal difference of the cumulative outputs.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven LayerNorm. The output dtype matches the input dtype;
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd. The operator is stateless, and repeated
        ``forward`` calls do not require ``reset``.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: it still
        calls :func:`torch.nn.functional.layer_norm`, requires a complete time
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
        """
        super().__init__()
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

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
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

        y_cum = F.layer_norm(
            x_seq.cumsum(dim=0),
            self.normalized_shape,
            self.weight,
            self.bias,
            self.eps,
        )
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        has_bias = self.bias is not None
        return (
            f"{self.normalized_shape}, eps={self.eps}, "
            f"elementwise_affine={self.elementwise_affine}, bias={has_bias}"
        )


class TDGELU(nn.Module):
    def __init__(self, approximate: Literal["none", "tanh"] = "none") -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDGELU.__init__-cn>` |
        :ref:`English <TDGELU.__init__-en>`

        ----

        .. _TDGELU.__init__-cn:

        * **中文**

        Temporal-difference (TD) GELU（Gaussian Error Linear Unit）算子。
        输入必须是完整时间序列，时间维固定为第 0 维，形状为 ``[T, ...]``。
        该模块先对输入在时间维做累积，再对每个累积输入执行
        :func:`torch.nn.functional.gelu`，最后返回累积输出在时间维上的差分。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven GELU。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子无内部状态，
        多次 ``forward`` 之间不需要调用 ``reset``。该算子仅依赖
        :func:`torch.nn.functional.gelu`，支持 CPU 与 CUDA，后端与
        :mod:`torch` 一致，无 CuPy / Triton 专用路径。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 非线性算子的累积-差分等价转换思路。本文档中的 TD GELU
        只实现张量级算子：它仍调用 :func:`torch.nn.functional.gelu`，需要
        完整时间序列输入，不是逐时间步在线算子，也不是面向神经形态硬件的
        fully spike-driven GELU。

        .. code-block:: python

            op = TDGELU(approximate="none")
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param approximate: GELU 近似模式，与 :class:`torch.nn.GELU` 的
            ``approximate`` 语义一致。
        :type approximate: Literal["none", "tanh"]
        :raises ValueError: 若 ``approximate`` 不是 ``"none"`` 或 ``"tanh"``。

        ----

        .. _TDGELU.__init__-en:

        * **English**

        Temporal-difference (TD) GELU (Gaussian Error Linear Unit) operator.
        The input must be a complete time sequence whose time dimension is
        fixed at dimension 0, with shape ``[T, ...]``. This module first
        accumulates the input over time, applies
        :func:`torch.nn.functional.gelu` to each cumulative input, and returns
        the temporal difference of the cumulative outputs.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven GELU. The output dtype matches the input dtype;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is
        stateless, and repeated ``forward`` calls do not require ``reset``. It
        only depends on :func:`torch.nn.functional.gelu`, supports CPU and CUDA,
        follows the :mod:`torch` backend behavior, and has no CuPy / Triton
        specific path.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer nonlinear operators in `SpikeZIP-TF: Conversion is All You
        Need for Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_.
        This implementation provides only a tensor-level operator: it still
        calls :func:`torch.nn.functional.gelu`, requires a complete time
        sequence, is not a step-wise online operator, and is not a fully
        spike-driven GELU for neuromorphic hardware.

        .. code-block:: python

            op = TDGELU(approximate="none")
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param approximate: GELU approximation mode, with the same semantics as
            ``approximate`` in :class:`torch.nn.GELU`.
        :type approximate: Literal["none", "tanh"]
        :raises ValueError: If ``approximate`` is not ``"none"`` or ``"tanh"``.
        """
        super().__init__()
        if approximate not in ("none", "tanh"):
            raise ValueError(
                "TDGELU: approximate must be 'none' or 'tanh', "
                f"but got {approximate!r}."
            )
        self.approximate = approximate

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
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

        y_cum = F.gelu(x_seq.cumsum(dim=0), approximate=self.approximate)
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"


class TDLinear(nn.Module):
    def __init__(
        self,
        in_features: int,
        out_features: int,
        bias: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <TDLinear.__init__-cn>` |
        :ref:`English <TDLinear.__init__-en>`

        ----

        .. _TDLinear.__init__-cn:

        * **中文**

        Temporal-difference (TD) Linear 算子。输入必须是完整时间序列，
        时间维固定为第 0 维，形状为 ``[T, ..., in_features]``。该模块返回
        sequence-preserving affine 差分序列，使 ``Y.cumsum(dim=0)`` 等于对
        ``X.cumsum(dim=0)`` 逐时间步执行 :func:`torch.nn.functional.linear`。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Linear。输出 dtype 与 PyTorch Linear 一致；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子无内部状态，
        多次 ``forward`` 之间不需要调用 ``reset``。该算子仅依赖 PyTorch
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

        ----

        .. _TDLinear.__init__-en:

        * **English**

        Temporal-difference (TD) Linear operator. The input must be a complete
        time sequence whose time dimension is fixed at dimension 0, with shape
        ``[T, ..., in_features]``. This module returns a sequence-preserving
        affine differential sequence such that ``Y.cumsum(dim=0)`` matches
        :func:`torch.nn.functional.linear` applied to ``X.cumsum(dim=0)`` at
        every time step.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        a fully spike-driven Linear. The output dtype follows PyTorch Linear;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is
        stateless, and repeated ``forward`` calls do not require ``reset``. It
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
        """
        super().__init__()
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

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
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

        if self.bias is None:
            return F.linear(x_seq, self.weight, None)

        y_cum = F.linear(x_seq.cumsum(dim=0), self.weight, self.bias)
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        return (
            f"in_features={self.in_features}, out_features={self.out_features}, "
            f"bias={self.bias is not None}"
        )


class SNNMatrixOperator(nn.Module):
    def __init__(self) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNMatrixOperator.__init__-cn>` |
        :ref:`English <SNNMatrixOperator.__init__-en>`

        ----

        .. _SNNMatrixOperator.__init__-cn:

        * **中文**

        Sequence-preserving SNN 矩阵乘法算子。输入必须是两条完整时间序列，
        时间维固定为第 0 维，形状分别为 ``[T, ..., M, N]`` 和
        ``[T, ..., N, P]``。该模块先分别对两条输入在时间维做累积，再执行
        :func:`torch.matmul`，最后返回累积输出在时间维上的差分。

        该算子满足
        ``Y.cumsum(dim=0) == torch.matmul(A.cumsum(dim=0), B.cumsum(dim=0))``。
        因而它保留 cross-time terms，例如 ``A[0] @ B[1]`` 与
        ``A[1] @ B[0]``；这不同于逐时间步执行 ``A[t] @ B[t]``。该算子是
        LAS ``SNNMatrixOperater`` prefix recurrence 的 sequence-preserving
        张量级形式，但不会在内部自动 ``sum(0)``。

        输出是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven matrix multiplication。dtype、device 与 broadcast 语义遵循
        :func:`torch.matmul`。该算子无内部状态，多次 ``forward`` 之间不需要
        调用 ``reset``。

        ----

        .. _SNNMatrixOperator.__init__-en:

        * **English**

        Sequence-preserving SNN matrix multiplication operator. The inputs must
        be two complete time sequences whose time dimension is fixed at
        dimension 0, with shapes ``[T, ..., M, N]`` and ``[T, ..., N, P]``.
        This module accumulates both inputs over time, applies
        :func:`torch.matmul`, and returns the temporal difference of the
        cumulative outputs.

        The operator satisfies
        ``Y.cumsum(dim=0) == torch.matmul(A.cumsum(dim=0), B.cumsum(dim=0))``.
        Therefore it preserves cross-time terms such as ``A[0] @ B[1]`` and
        ``A[1] @ B[0]``; it is not equivalent to applying ``A[t] @ B[t]`` at
        each time step independently. It is the sequence-preserving tensor
        form of the LAS ``SNNMatrixOperater`` prefix recurrence, but it does
        not implicitly call ``sum(0)``.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven matrix multiplication. Dtype, device and broadcasting
        semantics follow :func:`torch.matmul`. The operator is stateless, and
        repeated ``forward`` calls do not require ``reset``.
        """
        super().__init__()

    def forward(self, a_seq: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
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
        y_cum = torch.matmul(a_seq.cumsum(dim=0), b_seq.cumsum(dim=0))
        return _temporal_difference(y_cum)


class SNNElementWiseProduct(nn.Module):
    def __init__(self) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <SNNElementWiseProduct.__init__-cn>` |
        :ref:`English <SNNElementWiseProduct.__init__-en>`

        ----

        .. _SNNElementWiseProduct.__init__-cn:

        * **中文**

        Sequence-preserving SNN 逐元素乘法算子。输入必须是两条完整时间序列，
        时间维固定为第 0 维，形状为 ``[T, ...]``，非时间维遵循 PyTorch
        broadcast 规则。该模块先分别对两条输入在时间维做累积，再执行逐元素
        乘法，最后返回累积输出在时间维上的差分。

        该算子满足 ``Y.cumsum(dim=0) == A.cumsum(dim=0) * B.cumsum(dim=0)``。
        它是 LAS ``SNNMACOperater`` 核心乘法-累积语义的 sequence-preserving
        张量级形式，但不会在内部自动 ``sum(0)``；需要单步聚合时由调用方显式
        完成。

        输出是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven multiply-accumulate。dtype、device 与 broadcast 语义遵循
        PyTorch 逐元素乘法。该算子无内部状态，多次 ``forward`` 之间不需要调用
        ``reset``。

        ----

        .. _SNNElementWiseProduct.__init__-en:

        * **English**

        Sequence-preserving SNN element-wise product operator. The inputs must
        be two complete time sequences whose time dimension is fixed at
        dimension 0, with shape ``[T, ...]``. Non-time dimensions follow
        PyTorch broadcasting rules. This module accumulates both inputs over
        time, applies element-wise multiplication, and returns the temporal
        difference of the cumulative outputs.

        The operator satisfies
        ``Y.cumsum(dim=0) == A.cumsum(dim=0) * B.cumsum(dim=0)``. It is the
        sequence-preserving tensor form of the core multiply-accumulate
        semantics in LAS ``SNNMACOperater``, but it does not implicitly call
        ``sum(0)``; callers should aggregate explicitly when a single-step
        output is required.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven multiply-accumulate. Dtype, device and broadcasting
        semantics follow PyTorch element-wise multiplication. The operator is
        stateless, and repeated ``forward`` calls do not require ``reset``.
        """
        super().__init__()

    def forward(self, a_seq: torch.Tensor, b_seq: torch.Tensor) -> torch.Tensor:
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
        y_cum = a_seq.cumsum(dim=0) * b_seq.cumsum(dim=0)
        return _temporal_difference(y_cum)


class TDScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        r"""
        .. rubric:: API Language

        :ref:`中文 <TDScaledDotProductAttention.__init__-cn>` |
        :ref:`English <TDScaledDotProductAttention.__init__-en>`

        ----

        .. _TDScaledDotProductAttention.__init__-cn:

        * **中文**

        Temporal-difference (TD) scaled dot-product attention 算子。输入必须
        是完整时间序列，时间维固定为第 0 维。``query_seq`` 的形状为
        ``[T, ..., L, E]``，``key_seq`` 的形状为 ``[T, ..., S, E]``，
        ``value_seq`` 的形状为 ``[T, ..., S, Ev]``。该模块先分别对
        query、key、value 在时间维做累积，再调用
        :func:`torch.nn.functional.scaled_dot_product_attention`，最后返回
        累积输出在时间维上的差分。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven attention。dtype、device 与 mask broadcast 语义遵循
        :func:`torch.nn.functional.scaled_dot_product_attention`；推荐使用
        ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。该算子
        完全由 PyTorch 可微算子组成，对 autograd 透明。该算子无内部状态，
        多次 ``forward`` 之间不需要调用 ``reset``。该算子仅依赖 PyTorch
        SDPA，支持 CPU 与 CUDA，后端与 :mod:`torch` 一致，无 CuPy / Triton
        专用路径。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中对
        Transformer 算子的累积-差分等价转换思路。本文档中的 TD scaled
        dot-product attention 只实现张量级最小 primitive：它仍调用 PyTorch
        SDPA，需要完整时间序列输入，不是逐时间步在线算子，也不是面向神经
        形态硬件的 fully spike-driven attention。本实现固定
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

        ----

        .. _TDScaledDotProductAttention.__init__-en:

        * **English**

        Temporal-difference (TD) scaled dot-product attention operator. The
        inputs must be complete time sequences whose time dimension is fixed at
        dimension 0. ``query_seq`` has shape ``[T, ..., L, E]``, ``key_seq``
        has shape ``[T, ..., S, E]``, and ``value_seq`` has shape
        ``[T, ..., S, Ev]``. This module first accumulates query, key, and
        value over time, calls
        :func:`torch.nn.functional.scaled_dot_product_attention`, and returns
        the temporal difference of the cumulative outputs.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent
        fully spike-driven attention. Dtype, device, and mask broadcasting
        follow :func:`torch.nn.functional.scaled_dot_product_attention`;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is
        stateless, and repeated ``forward`` calls do not require ``reset``. It
        only depends on PyTorch SDPA, supports CPU and CUDA, follows the
        :mod:`torch` backend behavior, and has no CuPy / Triton specific path.

        The mechanism follows the cumulative-difference equivalence idea for
        Transformer operators in `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_. This
        implementation provides only a tensor-level minimal primitive: it still
        calls PyTorch SDPA, requires a complete time sequence, is not a
        step-wise online operator, and is not fully spike-driven attention for
        neuromorphic hardware. This implementation fixes ``dropout_p=0.0`` and
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
        """
        super().__init__()
        self.is_causal = is_causal
        self.scale = scale

    def forward(
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
        return _td_scaled_dot_product_attention(
            query_seq,
            key_seq,
            value_seq,
            attn_mask,
            self.is_causal,
            self.scale,
            "TDScaledDotProductAttention",
        )

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, scale={self.scale}"


class TDMultiheadAttention(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        dropout: float = 0.0,
        bias: bool = True,
        batch_first: bool = True,
        device: Optional[Union[torch.device, str]] = None,
        dtype: Optional[torch.dtype] = None,
    ) -> None:
        r"""
        **API Language:**
        :ref:`中文 <TDMultiheadAttention.__init__-cn>` |
        :ref:`English <TDMultiheadAttention.__init__-en>`

        ----

        .. _TDMultiheadAttention.__init__-cn:

        * **中文**

        Temporal-difference (TD) MultiheadAttention 的窄子集实现。输入必须是
        完整时间序列，时间维固定为第 0 维，形状为
        ``[T, batch, seq, embed_dim]``。该模块使用 ``TDLinear`` 生成 q/k/v
        projection，执行 TD scaled dot-product attention，再用 ``TDLinear``
        执行输出 projection。

        返回值是 ``(attn_output_seq, None)``，用于匹配
        :class:`torch.nn.MultiheadAttention` 在 ``need_weights=False`` 时的
        tuple 返回结构。输出是浮点差分值，不是二值脉冲，也不是 fully
        spike-driven attention。输出 dtype 跟随 PyTorch Linear / SDPA；
        推荐使用 ``float32``、``float16``、``bfloat16`` 或 ``float64`` 输入。
        该算子完全由 PyTorch 可微算子组成，对 autograd 透明。该算子无内部
        状态，多次 ``forward`` 之间不需要调用 ``reset``；支持 CPU 与 CUDA，
        后端与 :mod:`torch` 一致，无 CuPy / Triton 专用路径。当前只支持
        ``dropout=0.0``、``batch_first=True`` 和 ``need_weights=False``。

        该算子的机制来源于 `SpikeZIP-TF: Conversion is All You Need for
        Transformer-based SNN <https://arxiv.org/abs/2406.03470>`_ 中的
        累积-差分等价转换思路。本实现是窄子集 TD wrapper，仍使用浮点
        ``TDLinear`` 和 PyTorch SDPA，不是逐时间步在线 attention，也不是面向
        神经形态硬件的 fully spike-driven MultiheadAttention。``bias=True``
        时 projection bias 由 ``TDLinear`` 在累积输入上处理，避免普通
        ``nn.Linear`` 直接作用在差分序列时产生重复累计 bias。

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
        :raises ValueError: 若 ``embed_dim`` 不能被 ``num_heads`` 整除、或传入
            当前不支持的 ``dropout`` / ``batch_first``。

        ----

        .. _TDMultiheadAttention.__init__-en:

        * **English**

        Narrow temporal-difference (TD) MultiheadAttention implementation. The
        input must be a complete time sequence whose time dimension is fixed at
        dimension 0, with shape ``[T, batch, seq, embed_dim]``. This module uses
        ``TDLinear`` for q/k/v projections, applies TD scaled dot-product
        attention, and then applies a ``TDLinear`` output projection.

        The return value is ``(attn_output_seq, None)`` to match the tuple
        structure of :class:`torch.nn.MultiheadAttention` when
        ``need_weights=False``. The output contains floating-point differential
        values, is not a binary spike tensor, and is not fully spike-driven
        attention. The output dtype follows PyTorch Linear / SDPA;
        ``float32``, ``float16``, ``bfloat16`` and ``float64`` inputs are
        recommended. The operator is composed entirely of differentiable
        PyTorch operations and is transparent to autograd. The operator is
        stateless, and repeated ``forward`` calls do not require ``reset``. It
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
        :raises ValueError: If ``embed_dim`` is not divisible by ``num_heads``,
            or unsupported ``dropout`` / ``batch_first`` is passed.
        """
        super().__init__()
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
        self.q_proj = TDLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.k_proj = TDLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.v_proj = TDLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)
        self.out_proj = TDLinear(embed_dim, embed_dim, bias=bias, **factory_kwargs)

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

    def forward(
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
        if need_weights:
            raise ValueError("TDMultiheadAttention only supports need_weights=False.")
        if key_padding_mask is not None:
            raise ValueError("TDMultiheadAttention does not support key_padding_mask.")
        if not average_attn_weights:
            raise ValueError(
                "TDMultiheadAttention does not support average_attn_weights=False."
            )

        q_seq = self._split_heads(self.q_proj(query_seq))
        k_seq = self._split_heads(self.k_proj(key_seq))
        v_seq = self._split_heads(self.v_proj(value_seq))
        attn_mask = self._canonical_mha_attn_mask(attn_mask, q_seq.shape[1])
        attn_seq = _td_scaled_dot_product_attention(
            q_seq,
            k_seq,
            v_seq,
            attn_mask,
            is_causal,
            None,
            "TDMultiheadAttention",
        )
        out_seq = self.out_proj(self._merge_heads(attn_seq))
        return out_seq, None

    def extra_repr(self) -> str:
        return (
            f"embed_dim={self.embed_dim}, num_heads={self.num_heads}, "
            f"dropout={self.dropout}, batch_first={self.batch_first}"
        )
