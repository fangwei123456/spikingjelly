import torch


class round_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <round_atgf-cn>` | :ref:`English <round_atgf-en>`

    ----

    .. _round_atgf-cn:

    * **中文**

    :class:`round` 的自动梯度函数。前向传播执行 ``y = torch.round(x)``，反向传播将梯度原样传递（即梯度直通估计器 Straight-Through Estimator）。

    :param x: 输入张量
    :type x: torch.Tensor
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _round_atgf-en:

    * **English**

    The autograd function for :class:`round`. The forward executes ``y = torch.round(x)``, and the backward
    passes the gradient through unchanged (i.e., a Straight-Through Estimator).

    :param x: the input tensor
    :type x: torch.Tensor
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def round(x: torch.Tensor):
    r"""
    **API Language** - :ref:`中文 <round-cn>` | :ref:`English <round-en>`

    ----

    .. _round-cn:

    * **中文**

    对输入张量应用 ``y = torch.round(x)`` 操作，并重新定义梯度为 :math:`\frac{\partial y}{\partial x} = 1`。

    :param x: 输入张量
    :type x: torch.Tensor

    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _round-en:

    * **English**

    Apply ``y = torch.round(x)`` with re-defining gradient as :math:`\frac{\partial y}{\partial x} = 1`.

    :param x: the input tensor
    :type x: torch.Tensor

    :return: the output tensor
    :rtype: torch.Tensor
    """
    return round_atgf.apply(x)


class ceil_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <ceil_atgf-cn>` | :ref:`English <ceil_atgf-en>`

    ----

    .. _ceil_atgf-cn:

    * **中文**

    :class:`ceil` 的自动梯度函数。前向传播执行 ``y = torch.ceil(x)``，反向传播将梯度原样传递（即梯度直通估计器 Straight-Through Estimator）。

    :param x: 输入张量
    :type x: torch.Tensor
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _ceil_atgf-en:

    * **English**

    The autograd function for :class:`ceil`. The forward executes ``y = torch.ceil(x)``, and the backward
    passes the gradient through unchanged (i.e., a Straight-Through Estimator).

    :param x: the input tensor
    :type x: torch.Tensor
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def ceil(x: torch.Tensor):
    r"""
    **API Language** - :ref:`中文 <ceil-cn>` | :ref:`English <ceil-en>`

    ----

    .. _ceil-cn:

    * **中文**

    对输入张量应用 ``y = torch.ceil(x)`` 操作，并重新定义梯度为 :math:`\frac{\partial y}{\partial x} = 1`。

    :param x: 输入张量
    :type x: torch.Tensor

    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _ceil-en:

    * **English**

    Apply ``y = torch.ceil(x)`` with re-defining gradient as :math:`\frac{\partial y}{\partial x} = 1`.

    :param x: the input tensor
    :type x: torch.Tensor

    :return: the output tensor
    :rtype: torch.Tensor
    """
    return ceil_atgf.apply(x)


class floor_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <floor_atgf-cn>` | :ref:`English <floor_atgf-en>`

    ----

    .. _floor_atgf-cn:

    * **中文**

    :class:`floor` 的自动梯度函数。前向传播执行 ``y = torch.floor(x)``，反向传播将梯度原样传递（即梯度直通估计器 Straight-Through Estimator）。

    :param x: 输入张量
    :type x: torch.Tensor
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _floor_atgf-en:

    * **English**

    The autograd function for :class:`floor`. The forward executes ``y = torch.floor(x)``, and the backward
    passes the gradient through unchanged (i.e., a Straight-Through Estimator).

    :param x: the input tensor
    :type x: torch.Tensor
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


def floor(x: torch.Tensor):
    r"""
    **API Language** - :ref:`中文 <floor-cn>` | :ref:`English <floor-en>`

    ----

    .. _floor-cn:

    * **中文**

    对输入张量应用 ``y = torch.floor(x)`` 操作，并重新定义梯度为 :math:`\frac{\partial y}{\partial x} = 1`。

    :param x: 输入张量
    :type x: torch.Tensor

    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _floor-en:

    * **English**

    Apply ``y = torch.floor(x)`` with re-defining gradient as :math:`\frac{\partial y}{\partial x} = 1`.

    :param x: the input tensor
    :type x: torch.Tensor

    :return: the output tensor
    :rtype: torch.Tensor
    """
    return floor_atgf.apply(x)


def clamp_backward(
    grad_output: torch.Tensor, x: torch.Tensor, min_value: float, max_value: float
):
    r"""
    **API Language** - :ref:`中文 <clamp_backward-cn>` | :ref:`English <clamp_backward-en>`

    ----

    .. _clamp_backward-cn:

    * **中文**

    :class:`clamp_atgf` 的反向传播辅助函数。计算梯度掩码：对于 ``min_value <= x <= max_value`` 范围内的元素梯度保持不变，否则梯度为 0。

    .. math::

        \frac{\partial y}{\partial x} = \begin{cases}
            1, \mathrm{min\_value} \leq x \leq \mathrm{max\_value} \\
            0, \mathrm{otherwise}
        \end{cases}

    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :param x: 前向传播时的输入张量
    :type x: torch.Tensor
    :param min_value: 夹紧范围的下界
    :type min_value: float
    :param max_value: 夹紧范围的上界
    :type max_value: float
    :return: 施加掩码后的梯度
    :rtype: torch.Tensor

    ----

    .. _clamp_backward-en:

    * **English**

    The backward helper function for :class:`clamp_atgf`. Computes a gradient mask:
    for elements within ``min_value <= x <= max_value`` the gradient passes through,
    otherwise it is zero.

    .. math::

        \frac{\partial y}{\partial x} = \begin{cases}
            1, \mathrm{min\_value} \leq x \leq \mathrm{max\_value} \\
            0, \mathrm{otherwise}
        \end{cases}

    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :param x: the input tensor from the forward pass
    :type x: torch.Tensor
    :param min_value: the lower-bound of the clamping range
    :type min_value: float
    :param max_value: the upper-bound of the clamping range
    :type max_value: float
    :return: the masked gradient
    :rtype: torch.Tensor
    """
    mask = (x >= min_value).to(x) * (x <= max_value).to(x)
    return grad_output * mask


class clamp_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <clamp_atgf-cn>` | :ref:`English <clamp_atgf-en>`

    ----

    .. _clamp_atgf-cn:

    * **中文**

    :class:`clamp` 的自动梯度函数。前向传播执行 ``y = torch.clamp(x, min_value, max_value)``，
    反向传播使用 :func:`clamp_backward` 对在 ``[min_value, max_value]`` 范围内的元素传递梯度，范围外的元素梯度为 0。

    :param x: 输入张量
    :type x: torch.Tensor
    :param min_value: 夹紧范围的下界
    :type min_value: float
    :param max_value: 夹紧范围的上界
    :type max_value: float
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 夹紧后的张量及 ``None`` 占位
    :rtype: torch.Tensor / tuple

    ----

    .. _clamp_atgf-en:

    * **English**

    The autograd function for :class:`clamp`. The forward executes ``y = torch.clamp(x, min_value, max_value)``,
    and the backward uses :func:`clamp_backward` to pass gradients for elements within ``[min_value, max_value]``
    and zero out gradients outside.

    :param x: the input tensor
    :type x: torch.Tensor
    :param min_value: the lower-bound of the clamping range
    :type min_value: float
    :param max_value: the upper-bound of the clamping range
    :type max_value: float
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the clamped tensor and ``None`` placeholders
    :rtype: torch.Tensor / tuple
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, min_value: float, max_value: float):
        if x.requires_grad:
            ctx.save_for_backward(x)
            ctx.min_value = min_value
            ctx.max_value = max_value
        return torch.clamp(x, min_value, max_value)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return (
            clamp_backward(
                grad_output, ctx.saved_tensors[0], ctx.min_value, ctx.max_value
            ),
            None,
            None,
        )


def clamp(x: torch.Tensor, min_value: float, max_value: float):
    r"""
    **API Language** - :ref:`中文 <clamp-cn>` | :ref:`English <clamp-en>`

    ----

    .. _clamp-cn:

    * **中文**

    应用 ``y = torch.clamp(x, min_value, max_value)`` 操作，并重新定义梯度为：

    .. math::

        \frac{\partial y}{\partial x} = \begin{cases}
            1, \mathrm{min\_value} \leq x \leq \mathrm{max\_value} \\
            0, \mathrm{otherwise}
        \end{cases}

    :param x: 输入张量
    :type x: torch.Tensor

    :param min_value: 要夹紧到的范围的下界
    :type min_value: float

    :param max_value: 要夹紧到的范围的上界
    :type max_value: float

    :return: 输出张量
    :rtype: torch.Tensor

    ----

    .. _clamp-en:

    * **English**

    Apply ``y = torch.clamp(x, min_value, max_value)`` with re-defining gradient as:

    .. math::

        \frac{\partial y}{\partial x} = \begin{cases}
            1, \mathrm{min\_value} \leq x \leq \mathrm{max\_value} \\
            0, \mathrm{otherwise}
        \end{cases}

    :param x: the input tensor
    :type x: torch.Tensor

    :param min_value: lower-bound of the range to be clamped to
    :type min_value: float

    :param max_value: upper-bound of the range to be clamped to
    :type max_value: float

    :return: the output tensor
    :rtype: torch.Tensor
    """
    return clamp_atgf.apply(x, min_value, max_value)


def step_quantize_forward(x: torch.Tensor, step: float):
    r"""
    **API Language** - :ref:`中文 <step_quantize_forward-cn>` | :ref:`English <step_quantize_forward-en>`

    ----

    .. _step_quantize_forward-cn:

    * **中文**

    :class:`step_quantize_atgf` 的前向传播辅助函数。将 ``x`` 量化到最近的 ``i * step``，其中 ``i`` 是整数。

    .. math::

        y = \\mathrm{round}(x / \\mathrm{step}) \\times \\mathrm{step}

    :param x: 输入张量
    :type x: torch.Tensor
    :param step: 量化步长
    :type step: float
    :return: 量化后的张量
    :rtype: torch.Tensor

    ----

    .. _step_quantize_forward-en:

    * **English**

    The forward helper function for :class:`step_quantize_atgf`. Quantizes ``x`` to the nearest ``i * step``,
    where ``i`` is an integer.

    .. math::

        y = \\mathrm{round}(x / \\mathrm{step}) \\times \\mathrm{step}

    :param x: the input tensor
    :type x: torch.Tensor
    :param step: the quantization step size
    :type step: float
    :return: the quantized tensor
    :rtype: torch.Tensor
    """
    return torch.round_(x / step) * step


class step_quantize_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <step_quantize_atgf-cn>` | :ref:`English <step_quantize_atgf-en>`

    ----

    .. _step_quantize_atgf-cn:

    * **中文**

    :class:`step_quantize` 的自动梯度函数。前向传播调用 :func:`step_quantize_forward` 执行步长量化，
    反向传播将梯度原样传递（即梯度直通估计器 Straight-Through Estimator）。

    :param x: 输入张量
    :type x: torch.Tensor
    :param step: 量化步长
    :type step: float
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 量化结果及 ``None`` 占位
    :rtype: torch.Tensor / tuple

    ----

    .. _step_quantize_atgf-en:

    * **English**

    The autograd function for :class:`step_quantize`. The forward calls :func:`step_quantize_forward` to perform
    step quantization, and the backward passes the gradient through unchanged (i.e., a Straight-Through Estimator).

    :param x: the input tensor
    :type x: torch.Tensor
    :param step: the quantization step size
    :type step: float
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the quantized result and ``None`` placeholders
    :rtype: torch.Tensor / tuple
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, step: float):
        return step_quantize_forward(x, step)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


def step_quantize(x: torch.Tensor, step: float):
    r"""
    **API Language** - :ref:`中文 <step_quantize-cn>` | :ref:`English <step_quantize-en>`

    ----

    .. _step_quantize-cn:

    * **中文**

    将 ``x`` 量化到最近的 ``i * step``，其中 ``i`` 是整数。

    注意梯度定义为 :math:`\frac{\partial y}{\partial x} = 1`。

    .. image:: ../_static/API/activation_based//quantize/step_quantize.*
        :width: 100%

    :param x: 输入张量
    :type x: torch.Tensor

    :param step: 量化步长
    :type step: float

    :return: 量化后的张量
    :rtype: torch.Tensor

    ----

    .. _step_quantize-en:

    * **English**

    Quantize ``x`` to the nearest ``i * step``, where ``i`` is an integer.

    Note that the gradient is defined by :math:`\frac{\partial y}{\partial x} = 1`.

    .. image:: ../_static/API/activation_based//quantize/step_quantize.*
        :width: 100%

    :param x: the input tensor
    :type x: torch.Tensor

    :param step: the quantize step
    :type step: float

    :return: the quantized tensor
    :rtype: torch.Tensor
    """
    return step_quantize_atgf.apply(x, step)


def k_bit_quantize_forward(x: torch.Tensor, k: int):
    r"""
    **API Language** - :ref:`中文 <k_bit_quantize_forward-cn>` | :ref:`English <k_bit_quantize_forward-en>`

    ----

    .. _k_bit_quantize_forward-cn:

    * **中文**

    :class:`k_bit_quantize_atgf` 的前向传播辅助函数。将范围为 ``[0, 1]`` 的输入量化到最近的 ``i / (2 ** k - 1)``，
    其中 ``i = 0, 1, ..., (2 ** k - 1)``。

    .. math::

        y = \\frac{\\mathrm{round}((2^k - 1) \\cdot x)}{2^k - 1}

    :param x: 范围为 ``[0, 1]`` 的浮点张量
    :type x: torch.Tensor
    :param k: 输出位数
    :type k: int
    :return: 量化后的张量
    :rtype: torch.Tensor

    ----

    .. _k_bit_quantize_forward-en:

    * **English**

    The forward helper function for :class:`k_bit_quantize_atgf`. Quantizes the input in range ``[0, 1]``
    to the nearest ``i / (2 ** k - 1)``, where ``i = 0, 1, ..., (2 ** k - 1)``.

    .. math::

        y = \\frac{\\mathrm{round}((2^k - 1) \\cdot x)}{2^k - 1}

    :param x: a float tensor whose range is ``[0, 1]``
    :type x: torch.Tensor
    :param k: the bit number of output
    :type k: int
    :return: the quantized tensor
    :rtype: torch.Tensor
    """
    c = float(1 << k) - 1.0
    x = x * c
    torch.round_(x)
    return x / c


class k_bit_quantize_atgf(torch.autograd.Function):
    r"""
    **API Language** - :ref:`中文 <k_bit_quantize_atgf-cn>` | :ref:`English <k_bit_quantize_atgf-en>`

    ----

    .. _k_bit_quantize_atgf-cn:

    * **中文**

    :class:`k_bit_quantize` 的自动梯度函数。前向传播调用 :func:`k_bit_quantize_forward` 执行 DoReFa-Net 风格的 k 位量化，
    反向传播将梯度原样传递（即梯度直通估计器 Straight-Through Estimator）。

    :param x: 范围为 ``[0, 1]`` 的浮点张量
    :type x: torch.Tensor
    :param k: 输出位数
    :type k: int
    :param grad_output: 上游梯度
    :type grad_output: torch.Tensor
    :return: 量化结果及 ``None`` 占位
    :rtype: torch.Tensor / tuple

    ----

    .. _k_bit_quantize_atgf-en:

    * **English**

    The autograd function for :class:`k_bit_quantize`. The forward calls :func:`k_bit_quantize_forward` to perform
    DoReFa-Net style k-bit quantization, and the backward passes the gradient through unchanged (i.e., a Straight-Through Estimator).

    :param x: a float tensor whose range is ``[0, 1]``
    :type x: torch.Tensor
    :param k: the bit number of output
    :type k: int
    :param grad_output: the upstream gradient
    :type grad_output: torch.Tensor
    :return: the quantized result and ``None`` placeholders
    :rtype: torch.Tensor / tuple
    """

    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int):
        return k_bit_quantize_forward(x, k)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def k_bit_quantize(x: torch.Tensor, k: int):
    r"""
    **API Language** - :ref:`中文 <k_bit_quantize-cn>` | :ref:`English <k_bit_quantize-en>`

    ----

    .. _k_bit_quantize-cn:

    * **中文**

    在 `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_ 中定义的k位量化器。

    范围为 ``[0, 1]`` 的输入将被量化到最近的 ``i / (2 ** k - 1)``，其中 ``i = 0, 1, ..., (2 ** k - 1)``。

    注意梯度定义为 :math:`\frac{\partial y}{\partial x} = 1`。

    要将范围为 ``(-inf, inf)`` 的输入夹紧到范围 ``(0, 1)``，可以使用 :class:`torch.sigmoid`、:class:`torch.nn.Hardtanh` 或
    ``spikingjelly.activation_based.quantize`` 中的 ``clamp_*`` 函数（例如 :class:`spikingjelly.activation_based.quantize.clamp_by_linear`）。

    .. image:: ../_static/API/activation_based//quantize/k_bit_quantize.*
        :width: 100%

    :param x: 范围为 ``[0, 1]`` 的浮点张量
    :type x: torch.Tensor

    :param k: 输出的位数
    :type k: int

    :return: ``y = round((2 ** k - 1) * x) / (2 ** k - 1)``
    :rtype: torch.Tensor

    ----

    .. _k_bit_quantize-en:

    * **English**

    The k-bit quantizer defined in `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_.

    The input whose range is ``[0, 1]`` will be quantized to the nearest ``i / (2 ** k - 1)``, where ``i = 0, 1, ..., (2 ** k - 1)``.

    Note that the gradient is defined by :math:`\frac{\partial y}{\partial x} = 1`.

    To clamp the input whose range is ``(-inf, inf)`` to range ``(0, 1)``, using :class:`torch.sigmoid`, :class:`torch.nn.Hardtanh` or
    ``clamp_*`` functions (e.g., :class:`spikingjelly.activation_based.quantize.clamp_by_linear`) in ``spikingjelly.activation_based.quantize``.

    .. image:: ../_static/API/activation_based//quantize/k_bit_quantize.*
        :width: 100%

    :param x: a float tensor whose range is ``[0, 1]``.
    :type x: torch.Tensor

    :param k: the bit number of output
    :type k: int

    :return: ``y = round((2 ** k - 1) * x) / (2 ** k - 1)``
    :rtype: torch.Tensor

    :raises ValueError: if ``k < 1``
    """
    if k < 1:
        raise ValueError(f"k must be a positive integer, but got {k}")
    return k_bit_quantize_atgf.apply(x, k)


def affine_k_bit_quantize(x: torch.Tensor, k: int, w: torch.Tensor, b: torch.Tensor):
    r"""
    **API Language** - :ref:`中文 <affine_k_bit_quantize-cn>` | :ref:`English <affine_k_bit_quantize-en>`

    ----

    .. _affine_k_bit_quantize-cn:

    * **中文**

    应用仿射量化 ``y = w * round((2 ** k - 1) * x) / (2 ** k - 1) + b``。

    :param x: 范围为 ``[0, 1]`` 的浮点张量
    :type x: torch.Tensor

    :param k: 输出的位数
    :type k: int

    :param w: 仿射变换的权重
    :type w: torch.Tensor

    :param b: 仿射变换的偏置
    :type b: torch.Tensor

    :return: ``y = w * round((2 ** k - 1) * x) / (2 ** k - 1) + b``
    :rtype: torch.Tensor

    ----

    .. _affine_k_bit_quantize-en:

    * **English**

    Apply an affine quantization with ``y = w * round((2 ** k - 1) * x) / (2 ** k - 1) + b``.

    :param x: a float tensor whose range is ``[0, 1]``.
    :type x: torch.Tensor

    :param k: the bit number of output
    :type k: int

    :param w: the weight of the affine transform
    :type w: torch.Tensor

    :param b: the bias of the affine transform
    :type b: torch.Tensor

    :return: ``y = w * round((2 ** k - 1) * x) / (2 ** k - 1) + b``
    :rtype: torch.Tensor
    """
    return w * k_bit_quantize(x, k) + b


def clamp_by_linear(x: torch.Tensor, eps: float = 1e-5):
    r"""
    **API Language** - :ref:`中文 <clamp_by_linear-cn>` | :ref:`English <clamp_by_linear-en>`

    ----

    .. _clamp_by_linear-cn:

    * **中文**

    使用线性变换将输入范围从 ``(-inf, inf)`` 夹紧到 ``[0., 1.]``：

    .. math::

        y = \frac{x - \mathrm{min}(x)}{\mathrm{max}(x) - \mathrm{min}(x) + eps}

    :param x: 要归一化的输入张量，其范围为 ``(-inf, inf)``
    :type x: torch.Tensor

    :param eps: 添加到分母的小值以保证数值稳定性，默认值为 ``1e-5``
    :type eps: float

    :return: 归一化后的张量，其范围为 ``[0., 1.]``
    :rtype: torch.Tensor

    ----

    .. _clamp_by_linear-en:

    * **English**

    Using the linear transform to clamp the input range from ``(-inf, inf)`` to ``[0., 1.]``:

    .. math::

        y = \frac{x - \mathrm{min}(x)}{\mathrm{max}(x) - \mathrm{min}(x) + eps}

    :param x: the input tensor to be normed, whose range is ``(-inf, inf)``
    :type x: torch.Tensor

    :param eps: a value added to the denominator for numerical stability. The default value is ``1e-5``
    :type eps: float

    :return: the normed tensor, whose range is ``[0., 1.]``
    :rtype: torch.Tensor
    """
    x_max = torch.max(x) + eps
    x_min = torch.min(x)
    return (x - x_min) / (x_max - x_min)
