import torch

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


def k_bit_quantize(x: torch.Tensor, k: int):
    """
    :param x: a float tensor whose range is ``0 <= x <= 1``.
    :type x: torch.Tensor
    :param k: the bit number of output
    :type k: int
    :return: ``y = round((2 ** k - 1) * x) / (2 ** k - 1)``
    :rtype: torch.Tensor

    The k-bit quantizer defined in `DoReFa-Net: Training Low Bitwidth Convolutional Neural Networks with Low Bitwidth Gradients <https://arxiv.org/abs/1606.06160>`_.

    Note that the gradient is defined by :math:`\frac{\partial y}{\partial x} = 1`.

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


@torch.jit.script
def step_quantize_forward(x: torch.Tensor, step: float):
    x = x / step
    torch.round_(x)
    return x * step


class step_quantize_atgf(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x: torch.Tensor, step: float = 1.):
        return step_quantize_forward(x, step)

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None


def step_quantize(x: torch.Tensor, step: float = 1.):
    """
    :param x: a float tensor whose range is ``0 <= x <= 1``.
    :type x: torch.Tensor

    :param step: the quantization step
    :type step: float

    :return: ``y = round(x / step) * step``
    :rtype: torch.Tensor

    The step quantizer defined in `Lava`.
    """
    return step_quantize_atgf.apply(x, step)


def quantize_8b(x, scale, descale = False):
    if not descale:
        return step_quantize(x, step = 2 / scale).clamp(-256/scale, 255/scale)
    else:
        return step_quantize(x, step = 2 / scale).clamp(-256/scale, 255/scale) * scale


@torch.jit.script
def norm_to_01_by_sigmoid(x: torch.Tensor):
    sgx = torch.sigmoid(x)
    return sgx / torch.max(sgx)

@torch.jit.script
def norm_to_01_by_linear(x: torch.Tensor, eps: float = 1e-5):
    x_max = torch.max(x) + eps
    x_min = torch.min(x)
    return (x - x_min) / (x_max - x_min)


def right_shift_to_zero(x, bits):
    if x.dtype not in (torch.int32, torch.int64):
        raise AssertionError(
            f"Expected x.dtype to be torch.int32 or torch.int64,"
            f"but get {x.dtype} instead."
        )

    x_sign = 2 * (x > 0) - 1
    return (x_sign * ((x_sign * x) >> bits)).to(x.dtype)