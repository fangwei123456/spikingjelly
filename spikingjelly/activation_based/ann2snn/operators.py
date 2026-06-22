import torch
import torch.nn as nn


__all__ = ["SpikeSoftmax"]


class SpikeSoftmax(nn.Module):
    def __init__(self, dim: int = -1):
        r"""
        **API Language:**
        :ref:`中文 <SpikeSoftmax.__init__-cn>` |
        :ref:`English <SpikeSoftmax.__init__-en>`

        ----

        .. _SpikeSoftmax.__init__-cn:

        * **中文**

        Spike-equivalent Softmax 算子。输入必须是完整时间序列，时间维固定为
        第 0 维，形状为 ``[T, ...]``。该模块先对输入在时间维做累积，再沿
        ``dim`` 计算 ``torch.softmax``，最后返回累积输出在时间维上的差分。

        返回值是浮点差分值，可能包含负值；它不是二值脉冲，也不表示 fully
        spike-driven Softmax。输出 dtype 与输入 dtype 相同；推荐使用
        ``float32``、``float16`` 或 ``float64`` 输入。该算子完全由 PyTorch
        可微算子组成，对 autograd 透明。

        .. code-block:: python

            op = SpikeSoftmax(dim=-1)
            x_seq = torch.randn(4, 2, 3)
            y_seq = op(x_seq)

        :param dim: Softmax 归一化维度。不能为第 0 维，因为第 0 维保留为时间维。
        :type dim: int
        :return: None
        :rtype: None

        ----

        .. _SpikeSoftmax.__init__-en:

        * **English**

        Spike-equivalent Softmax operator. The input must be a complete time
        sequence whose time dimension is fixed at dimension 0, with shape
        ``[T, ...]``. This module first accumulates the input over time, applies
        ``torch.softmax`` along ``dim`` to each cumulative input, and returns the
        temporal difference of the cumulative outputs.

        The output contains floating-point differential values and may contain
        negative values. It is not a binary spike tensor and does not represent a
        fully spike-driven Softmax. The output dtype matches the input dtype;
        ``float32``, ``float16`` and ``float64`` inputs are recommended. The
        operator is composed entirely of differentiable PyTorch operations and
        is transparent to autograd.

        .. code-block:: python

            op = SpikeSoftmax(dim=-1)
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
        :ref:`中文 <SpikeSoftmax.forward-cn>` |
        :ref:`English <SpikeSoftmax.forward-en>`

        ----

        .. _SpikeSoftmax.forward-cn:

        * **中文**

        对完整时间序列执行 spike-equivalent Softmax。计算过程为：

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
        :return: Spike-equivalent Softmax 差分序列，形状与 ``x_seq`` 相同。
        :rtype: torch.Tensor
        :raises ValueError: 若 ``x_seq`` 少于 2 维，或 ``dim`` 指向时间维。

        ----

        .. _SpikeSoftmax.forward-en:

        * **English**

        Apply spike-equivalent Softmax to a complete time sequence:

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
        :return: Spike-equivalent Softmax differential sequence with the same
            shape as ``x_seq``.
        :rtype: torch.Tensor
        :raises ValueError: If ``x_seq`` has fewer than 2 dimensions, or ``dim``
            refers to the time dimension.
        """
        if x_seq.dim() < 2:
            raise ValueError(
                "SpikeSoftmax expects an input sequence with shape [T, ...] "
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
                "SpikeSoftmax reserves dimension 0 as the time dimension; "
                "softmax dim must not resolve to 0."
            )

        y_cum = torch.softmax(x_seq.cumsum(dim=0), dim=dim)
        y_seq = torch.empty_like(y_cum)
        y_seq[0] = y_cum[0]
        y_seq[1:] = y_cum[1:] - y_cum[:-1]
        return y_seq

    def extra_repr(self) -> str:
        return f"dim={self.dim}"
