import torch


class round_atgf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.round(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


@torch.jit.ignore
def round(x: torch.Tensor):
    r"""
    **API Language:**
    :ref:`中文 <round-cn>` | :ref:`English <round-en>`

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
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.ceil(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


@torch.jit.ignore
def ceil(x: torch.Tensor):
    r"""
    **API Language:**
    :ref:`中文 <ceil-cn>` | :ref:`English <ceil-en>`

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
    @staticmethod
    def forward(ctx, x: torch.Tensor):
        return torch.floor(x)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output


@torch.jit.ignore
def floor(x: torch.Tensor):
    r"""
    **API Language:**
    :ref:`中文 <floor-cn>` | :ref:`English <floor-en>`

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


@torch.jit.script
def clamp_backward(
    grad_output: torch.Tensor, x: torch.Tensor, min_value: float, max_value: float
):
    mask = (x >= min_value).to(x) * (x <= max_value).to(x)
    return grad_output * mask


class clamp_atgf(torch.autograd.Function):
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


@torch.jit.ignore
def clamp(x: torch.Tensor, min_value: float, max_value: float):
    r"""
    **API Language:**
    :ref:`中文 <clamp-cn>` | :ref:`English <clamp-en>`

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


@torch.jit.script
def step_quantize_forward(x: torch.Tensor, step: float):
    return torch.round_(x / step) * step


class step_quantize_atgf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, step: float):
        return step_quantize_forward(x, step)

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor):
        return grad_output, None


@torch.jit.ignore
def step_quantize(x: torch.Tensor, step: float):
    r"""
    **API Language:**
    :ref:`中文 <step_quantize-cn>` | :ref:`English <step_quantize-en>`

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


@torch.jit.script
def k_bit_quantize_forward(x: torch.Tensor, k: int):
    c = float(1 << k) - 1.0
    x = x * c
    torch.round_(x)
    return x / c


class k_bit_quantize_atgf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, k: int):
        return k_bit_quantize_forward(x, k)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


@torch.jit.ignore
def k_bit_quantize(x: torch.Tensor, k: int):
    r"""
    **API Language:**
    :ref:`中文 <k_bit_quantize-cn>` | :ref:`English <k_bit_quantize-en>`

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
    """
    return k_bit_quantize_atgf.apply(x, k)


def affine_k_bit_quantize(x: torch.Tensor, k: int, w: torch.Tensor, b: torch.Tensor):
    r"""
    **API Language:**
    :ref:`中文 <affine_k_bit_quantize-cn>` | :ref:`English <affine_k_bit_quantize-en>`

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


@torch.jit.script
def clamp_by_linear(x: torch.Tensor, eps: float = 1e-5):
    r"""
    **API Language:**
    :ref:`中文 <clamp_by_linear-cn>` | :ref:`English <clamp_by_linear-en>`

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
