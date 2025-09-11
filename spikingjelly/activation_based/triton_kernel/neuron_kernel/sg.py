import torch.nn as nn
import triton
import triton.language as tl

from ..triton_utils import compile_triton_code_str
from ... import surrogate as sjsg


def get_triton_surrogate_kernel(sjsg_module: nn.Module):
    if isinstance(sjsg_module, sjsg.ATan):
        return atan_surrogate(sjsg_module.alpha)
    elif isinstance(sjsg_module, sjsg.Sigmoid):
        return sigmoid_surrogate(sjsg_module.alpha)
    else:
        raise NotImplementedError(
            f"{sjsg_module}'s triton kernel has not been implemented."
        )


atan_surrogate_template = """
@triton.jit
def atan_surrogate_{alpha_str}(h):
    sg = 3.141592653589793 * h * {alpha} / 2.
    sg = {alpha} / 2. / tl.fma(sg, sg, 1.)
    return sg.to(h.dtype)
"""


def atan_surrogate(alpha: float = 2.):
    alpha_str = str(alpha).replace(".", "_")
    code_str = atan_surrogate_template.format(
        alpha=alpha,
        alpha_str=alpha_str,
    ).strip()
    kernel_name = f"atan_surrogate_{alpha_str}"
    return compile_triton_code_str(code_str, kernel_name)


sigmoid_surrogate_template = """
@triton.jit
def sigmoid_surrogate_{alpha_str}(h):
    # triton's exp() supports only fp32 and fp64, so we must convert it to fp32!
    sg = tl.sigmoid(h.to(tl.float32) * {alpha})
    sg = {alpha} * sg * (1. - sg)
    return sg.to(h.dtype)
"""


def sigmoid_surrogate(alpha: float = 4.):
    alpha_str = str(alpha).replace(".", "_")
    code_str = sigmoid_surrogate_template.format(
        alpha=alpha, alpha_str=alpha_str
    ).strip()
    kernel_name = f"sigmoid_surrogate_{alpha_str}"
    return compile_triton_code_str(code_str, kernel_name)
