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
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor

    Apply ``y = torch.round(x)`` with re-defining gradient as :math:`\\frac{\\partial y}{\\partial x} = 1`.

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
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor

    Apply ``y = torch.ceil(x)`` with re-defining gradient as :math:`\\frac{\\partial y}{\\partial x} = 1`.

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
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor

    Apply ``y = torch.floor(x)`` with re-defining gradient as :math:`\\frac{\\partial y}{\\partial x} = 1`.

    """
    return floor_atgf.apply(x)

@torch.jit.script
def clamp_backward(grad_output: torch.Tensor, x: torch.Tensor, min_value: float, max_value: float):
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
        return clamp_backward(grad_output, ctx.saved_tensors[0], ctx.min_value, ctx.max_value), None, None
@torch.jit.ignore
def clamp(x: torch.Tensor, min_value: float, max_value: float):
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :param min_value:  lower-bound of the range to be clamped to
    :type min_value: float
    :param max_value: upper-bound of the range to be clamped to
    :type max_value: torch.Tensor
    :return: the output tensor
    :rtype: torch.Tensor

    Apply ``y = torch.clamp(x, min_value, max_value)`` with re-defining gradient as:

    .. math::

        \\frac{\\partial y}{\\partial x} = \\begin{cases}
            1, \\rm{min\\_value} \\leq x \\leq \\rm{max\\_value} \\\\
            0, \\rm{otherwise}
        \\end{cases}

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
    """
    :param x: the input tensor
    :type x: torch.Tensor
    :param step: the quantize step
    :type step: float
    :return: the quantized tensor
    :rtype: torch.Tensor

    Quantize ``x`` to the nearest ``i * step``, where ``i`` is an integer.

    Note that the gradient is defined by :math:`\\frac{\\partial y}{\\partial x} = 1`.

    .. image:: ../_static/API/activation_based//quantize/step_quantize.*
        :width: 100%

    """
    return step_quantize_atgf.apply(x, step)
"""
import torch
from spikingjelly.activation_based import quantize
from matplotlib import pyplot as plt
plt.style.use(['science', 'grid'])
fig = plt.figure(dpi=200, figsize=(8, 4))
x = torch.arange(-4, 4, 0.01)
colormap = plt.get_cmap('tab10')
for i, step in zip(range(2), [1, 2]):

    plt.subplot(1, 2, i + 1)
    y = quantize.step_quantize(x, step)
    plt.plot(x, y, label=f'y = step_quantize(x, {step})', c=colormap(i))

    plt.xlabel('Input')
    plt.ylabel('Output')
    plt.xticks(step / 2 * torch.as_tensor([-3, -1, 1, 3]))

    plt.grid(ls='--')
    plt.legend()
# plt.show()
plt.savefig('./docs/source/_static/API/activation_based/quantize/step_quantize.pdf')
plt.savefig('./docs/source/_static/API/activation_based/quantize/step_quantize.svg')
plt.savefig('./docs/source/_static/API/activation_based/quantize/step_quantize.png')
"""



@torch.jit.script
def k_bit_quantize_forward(x: torch.Tensor, k: int):
    c = float(1 << k) - 1.
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
    """
    :param x: a float tensor whose range is ``[0, 1]``.
    :type x: torch.Tensor
    :param k: the bit number of output
    :type k: int
    :return: ``y = round((2 ** k - 1) * x) / (2 ** k - 1)``
    :rtype: torch.Tensor

    The k-bit quantizer defined in `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_.

    The input whose range is ``[0, 1]`` will be quantized to the nearest ``i / (2 ** k - 1)``, where ``i = 0, 1, ..., (2 ** k - 1)``.

    Note that the gradient is defined by :math:`\\frac{\\partial y}{\\partial x} = 1`.

    To clamp the input whose range is ``(-inf, inf)`` to range ``(0, 1)``, using :class:`torch.sigmoid`, :class:`torch.nn.Hardtanh` or
    ``clamp_*`` functions (e.g., :class:`spikingjelly.activation_based.quantize.clamp_by_linear`) in ``spikingjelly.activation_based.quantize``.

    .. image:: ../_static/API/activation_based//quantize/k_bit_quantize.*
        :width: 100%

    Codes example:

    .. code-block:: python

        x = torch.rand(8)
        y = k_bit_quantize(x, 2)
        print(f'x={x}')
        print(f'y={y}')
        # x=tensor([0.6965, 0.5697, 0.9883, 0.0438, 0.1332, 0.7613, 0.9704, 0.2384])
        # y=tensor([0.6667, 0.6667, 1.0000, 0.0000, 0.0000, 0.6667, 1.0000, 0.3333])
    """
    return k_bit_quantize_atgf.apply(x, k)

def affine_k_bit_quantize(x: torch.Tensor, k: int, w: torch.Tensor, b: torch.Tensor):
    """
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

    Apply an affine quantization with ``y = w * round((2 ** k - 1) * x) / (2 ** k - 1) + b``.
    """
    return w * k_bit_quantize(x, k) + b

"""
import torch
from spikingjelly.activation_based import quantize
from matplotlib import pyplot as plt
plt.style.use(['science', 'grid'])
fig = plt.figure(dpi=200, figsize=(8, 4))
x = torch.arange(0, 1, 0.001)
colormap = plt.get_cmap('tab10')
for i, k in zip(range(2), [2, 3]):

    plt.subplot(1, 2, i + 1)
    y = quantize.k_bit_quantize(x, k=k)
    plt.plot(x, y, label=f'y = k_bit_quantize(x, {k})', c=colormap(i))

    plt.xlabel('Input')
    plt.ylabel('Output')

    plt.grid(ls='--')
    plt.legend()
# plt.show()
plt.savefig('./docs/source/_static/API/activation_based/quantize/k_bit_quantize.pdf')
plt.savefig('./docs/source/_static/API/activation_based/quantize/k_bit_quantize.svg')
plt.savefig('./docs/source/_static/API/activation_based/quantize/k_bit_quantize.png')
"""

@torch.jit.script
def clamp_by_linear(x: torch.Tensor, eps: float = 1e-5):
    """
    :param x: the input tensor to be normed, whose range is ``(-inf, inf)``
    :type x: torch.Tensor
    :param eps: a value added to the denominator for numerical stability. The default value is ``1e-5``
    :type eps: float
    :type max_value: float
    :return: the normed tensor, whose range is ``[min_value, max_value]``
    :rtype: torch.Tensor

    Using the linear transform to clamp the input range from ``(-inf, inf)`` to ``[0., 1.]``:

    .. math::

        y = \\frac{x - \\rm{min}(x)}{\\rm{max}(x) - \\rm{min}(x) + eps}
    """
    x_max = torch.max(x) + eps
    x_min = torch.min(x)
    return (x - x_min) / (x_max - x_min)
