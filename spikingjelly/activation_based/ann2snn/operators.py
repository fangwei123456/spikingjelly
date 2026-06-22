from typing import Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["TDSoftmax", "TDLayerNorm", "TDGELU"]


def _temporal_difference(y_cum: torch.Tensor) -> torch.Tensor:
    y_seq = torch.empty_like(y_cum)
    y_seq[0] = y_cum[0]
    y_seq[1:] = y_cum[1:] - y_cum[:-1]
    return y_seq


class TDSoftmax(nn.Module):
    def __init__(self, dim: int = -1) -> None:
        r"""
        **API Language:**
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
        :return: None
        :rtype: None

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
        :return: None
        :rtype: None
        """
        super().__init__()
        self.dim = dim

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language:**
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

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``。
        :type x_seq: torch.Tensor
        :return: TD Softmax 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维，或 ``dim`` 指向时间维。

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

        :param x_seq: Input time sequence with shape ``[T, ...]``.
        :type x_seq: torch.Tensor
        :return: TD Softmax differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions, or ``dim``
            refers to the time dimension.
        """
        if x_seq.dim() < 2:
            raise ValueError(
                "TDSoftmax expects an input sequence with shape [T, ...] "
                f"and at least 2 dimensions, but got shape {tuple(x_seq.shape)}."
            )

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
        **API Language:**
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
        :return: None
        :rtype: None

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
        :return: None
        :rtype: None
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
        **API Language:**
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

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``，尾部形状必须
            匹配 ``normalized_shape``。
        :type x_seq: torch.Tensor
        :return: TD LayerNorm 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维。

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

        :param x_seq: Input time sequence with shape ``[T, ...]``. The trailing
            shape must match ``normalized_shape``.
        :type x_seq: torch.Tensor
        :return: TD LayerNorm differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions.
        """
        if x_seq.dim() < 2:
            raise ValueError(
                "TDLayerNorm expects an input sequence with shape [T, ...] "
                f"and at least 2 dimensions, but got shape {tuple(x_seq.shape)}."
            )
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
    def __init__(self, approximate: str = "none") -> None:
        r"""
        **API Language:**
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
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。该算子无内部状态，多次 ``forward``
        之间不需要调用 ``reset``。

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
        :type approximate: str
        :return: None
        :rtype: None

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
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd. The operator is stateless, and repeated
        ``forward`` calls do not require ``reset``.

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
        :type approximate: str
        :return: None
        :rtype: None
        """
        super().__init__()
        self.approximate = approximate

    def forward(self, x_seq: torch.Tensor) -> torch.Tensor:
        r"""
        **API Language:**
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

        :param x_seq: 输入时间序列，形状为 ``[T, ...]``。
        :type x_seq: torch.Tensor
        :return: TD GELU 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维。

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

        :param x_seq: Input time sequence with shape ``[T, ...]``.
        :type x_seq: torch.Tensor
        :return: TD GELU differential sequence with the same shape as
            ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions.
        """
        if x_seq.dim() < 2:
            raise ValueError(
                "TDGELU expects an input sequence with shape [T, ...] "
                f"and at least 2 dimensions, but got shape {tuple(x_seq.shape)}."
            )

        y_cum = F.gelu(x_seq.cumsum(dim=0), approximate=self.approximate)
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        return f"approximate={self.approximate!r}"
