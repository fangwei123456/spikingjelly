from typing import Literal, Optional, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F


__all__ = ["TDSoftmax", "TDLayerNorm", "TDGELU", "TDScaledDotProductAttention"]


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


def _check_attention_sequence(x_seq: torch.Tensor, tensor_name: str) -> None:
    module_name = "TDScaledDotProductAttention"
    _check_time_sequence(x_seq, module_name)
    if x_seq.dim() < 3:
        raise ValueError(
            f"{module_name} expects {tensor_name} with shape [T, ..., L, E] "
            f"and at least 3 dimensions, but got shape {tuple(x_seq.shape)}."
        )


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
        :return: None
        :rtype: None
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
        :return: None
        :rtype: None
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


class TDScaledDotProductAttention(nn.Module):
    def __init__(
        self,
        is_causal: bool = False,
        scale: Optional[float] = None,
    ) -> None:
        r"""
        **API Language:**
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
        :return: None
        :rtype: None

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
        :return: None
        :rtype: None
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
        **API Language:**
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
        _check_attention_sequence(query_seq, "query_seq")
        _check_attention_sequence(key_seq, "key_seq")
        _check_attention_sequence(value_seq, "value_seq")

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

        y_cum = F.scaled_dot_product_attention(
            query_seq.cumsum(dim=0),
            key_seq.cumsum(dim=0),
            value_seq.cumsum(dim=0),
            attn_mask=attn_mask,
            dropout_p=0.0,
            is_causal=self.is_causal,
            scale=self.scale,
        )
        return _temporal_difference(y_cum)

    def extra_repr(self) -> str:
        return f"is_causal={self.is_causal}, scale={self.scale}"
