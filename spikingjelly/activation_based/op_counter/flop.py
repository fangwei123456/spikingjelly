from collections import defaultdict
from typing import Any, Callable

import torch

aten = torch.ops.aten
import torch.nn as nn

from .base import BaseCounter


__all__ = ["FlopCounter"]


def _prod(dims):
    p = 1
    for v in dims:
        p *= v
    return p

def _flop_null(args, kwargs, out):
    return 0

def _flop_mm(args, kwargs, out):
    """out = x @ y"""
    x, y = args[:2]
    m, k = x.shape
    kk, n = y.shape
    if k != kk:
        raise AssertionError(f"mm: inner dimensions mismatch [{x.shape} and {y.shape}]")
    return m * n * (2 * k - 1)


def _flop_addmm(args, kwargs, out):
    """out = beta * bias + alpha * (x @ y)"""
    bias, x, y = args[:3]
    m, k = x.shape
    kk, n = y.shape
    if k != kk:
        raise AssertionError(
            f"addmm: inner dimensions mismatch [{x.shape} and {y.shape}]"
        )

    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)

    flops = m * n * (2 * k - 1)  # matmul; 2k-1 flops for each output element
    if alpha != 1:
        flops += m * n  # scale by alpha
    if beta == 1:
        flops += m * n  # add b to the m*n matrix
    elif beta != 0:
        flops += bias.numel() + m * n  # scale bias, and add it to the m*n matrix
    return flops


def _flop_bmm(args, kwargs, out):
    """Batch matrix multiply: out[b] = x[b] @ y[b]"""
    x, y = args[:2]
    b, m, k = x.shape
    bb, kk, n = y.shape
    if b != bb or k != kk:
        raise AssertionError(
            f"bmm: batch or inner dimensions mismatch [{x.shape} and {y.shape}]"
        )
    return b * m * n * (2 * k - 1)


def _flop_baddbmm(args, kwargs, out):
    """out[b] = beta * bias[b] + alpha * (x[b] @ y[b])"""
    bias, x, y = args[:3]
    b, m, k = x.shape
    bb, kk, n = y.shape
    if b != bb or k != kk:
        raise AssertionError(
            f"baddmm: batch or inner dimensions mismatch [{x.shape}, {y.shape}]"
        )

    alpha = kwargs.get("alpha", 1)
    beta = kwargs.get("beta", 1)

    flops = b * m * n * (2 * k - 1)  # batched matmul
    if alpha != 1:
        flops += b * m * n  # scale by alpha
    if beta == 1:
        flops += b * m * n  # add b to the b*m*n matrix
    elif beta != 0:
        flops += bias.numel() + b * m * n  # scale bias, then add it to the b*m*n matrix
    return flops


def _flop_convolution(args, kwargs, out):
    """
    args[0]: x, shape [B, C_in, ...]
    args[1]: weight, shape [C_out, C_in, *kernel_shape]
    args[2]: bias or None
    """
    x, w, bias = args[:3]
    transposed = kwargs.get("transposed", False)

    b = x.shape[0]
    c_out, c_in, *kernel_shape = w.shape

    spatial_shape = x.shape[2:] if transposed else out.shape[2:]
    flops_per_position = 2 * c_in * _prod(kernel_shape)
    flops = flops_per_position * _prod(spatial_shape) * c_out * b
    flops -= out.numel()  # for each output element, the first add can be avoided
    if bias is not None:
        flops += out.numel()
    return flops


def _flop_convolution_backward(args, kwargs, out):
    """
    Outputs (by output_mask):
        0: grad_x
        1: grad_weight
        2: grad_bias
    """
    (
        grad_out,
        x,
        w,
        bias,
        _stride,
        _padding,
        _dilation,
        transposed,
        _output_padding,
        _groups,
        output_mask,
    ) = args
    flops = 0

    if output_mask[0]:
        grad_x = out[0]
        flops += _flop_convolution(
            [grad_out, w, None], {"transposed": not transposed}, grad_x
        )

    if output_mask[1]:
        grad_weight = out[1]
        if transposed:
            pseudo_x = grad_out
            pseudo_w = x
        else:
            pseudo_x = x
            pseudo_w = grad_out
        pseudo_x = pseudo_x.transpose(0, 1)
        pseudo_w = pseudo_w.transpose(0, 1)

        flops += _flop_convolution(
            [pseudo_x, pseudo_w, None], {"transposed": False}, grad_weight
        )

    if output_mask[2] and bias is not None:
        B = grad_out.shape[0]
        C_out = grad_out.shape[1]
        spatial_shape = grad_out.shape[2:]
        flops += C_out * (B * _prod(spatial_shape) - 1)

    return flops


def _flop_max_pool2d_with_indices(args, kwargs, out):
    kernel_size = args[1]
    y = out[0]
    return y.numel() * (_prod(kernel_size) - 1)  # K-1 * max


def _flop_avg_pool2d(args, kwargs, out):
    kernel_size = args[1]
    return out.numel() * _prod(kernel_size)  # K-1 * add, 1 * div


def _flop_sum(args, kwargs, out):
    x = args[0]
    y = out
    return x.numel() - y.numel()


def _flop_mean(args, kwargs, out):
    x = args[0]
    return x.numel()


def _flop_add(args, kwargs, out):
    alpha = kwargs.get("alpha", 1.0)
    if alpha == 1.0:
        return out.numel()
    else:
        nb = args[1].numel() if torch.is_tensor(args[1]) else 1
        return nb + out.numel()


def _flop_element_wise(args, kwargs, out):
    return out.numel()


def _flop_sigmoid(args, kwargs, out):
    return 4 * out.numel()

def _flop_native_batch_norm(args, kwargs, out):
    x, train = args[0], args[5]
    n, c = x.numel(), x.shape[1]
    flops = 0
    if train:
        flops += n # batch mean
        flops += 3*n - c # batch var
        flops += 2*c # sqrt(var + eps)
        flops += 2*n # x - mean / std
        flops += 2*n # * gamma, + beta
        flops += 6*c # (1-momentum)*stat + momentum*stat, stat in [mean, var]
    else:
        flops += 2*c # sqrt(var + eps)
        flops += 2*n # x - mean / std
        flops += 2*n # * gamma, + beta
    return flops


def _flop_native_batch_norm_backward(args, kwargs, out):
    grad_output, gamma, train, output_mask = args[0], args[2], args[-3], args[-1]
    n = grad_output.numel()
    c = gamma.numel()

    flops = 0
    if train:
        if output_mask[0]:  # grad_input
            flops += 2*n # x_hat = (x - mean) * invstd
            flops += n - c # term1: sum(grad_output) per channel (grad_beta)
            flops += 2*n - c # term2: sum(grad_output * x_hat) per channel (grad_gamma)
            flops += 5*n + 2*c # invstd*gamma/n * (grad_output*n - term1 - term2*x_hat)
        if output_mask[1] and not output_mask[0]:  # grad_gamma
            flops += 2*n # x_hat
            flops += 2*n - c
        if output_mask[2] and not output_mask[0]: # grad_beta
            flops = flops + n - c
    else:
        if output_mask[0]:  # grad_input
            flops += 2*n # grad_output * saved_invstd * gamma
        if output_mask[1]: # grad_gamma
            flops += 2*n # x_hat = (x - mean) / std
            flops += 2*n - c
        if output_mask[2]: # grad_beta
            flops += n - c
    return flops


class FlopCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        r"""
        **API Language:**
        :ref:`中文 <FlopCounter.__init__-cn>` | :ref:`English <FlopCounter.__init__-en>`

        ----

        .. _FlopCounter.__init__-cn:

        * **中文**

        浮点运算计数器，用于计算深度神经网络中的浮点运算次数。

        **FLOP（Floating Point Operations）** 是一个衡量计算复杂度的常用指标：

        - 1 次乘法 = 1 FLOP；1 次加法 = 1 FLOP；......
        - 逐元素操作的FLOP也会纳入考量。

        ``FlopCounter`` 应与 :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` 搭配使用。

        .. warning::

            目前，``FlopCounter`` 支持的 aten 操作类型有限。查看源代码以获取操作列表。如需添加新操作，
            可以使用 ``extra_rules`` 参数；也欢迎提交 pull request 来完善默认的 :attr:`rules` ！

        :param extra_rules: 额外的操作规则，格式为 ``{aten_op: func}`` ，
            其中 ``func`` 是一个函数，接受 ``(args, kwargs, out)`` 并返回计数值
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: 额外需要忽略的模块列表，这些模块中的操作不会被计数
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        .. _FlopCounter.__init__-en:

        * **English**

        FLOP counter for calculating the number of floating-point operations in deep networks.

        FLOP (Floating Point Operations) is a common metric for measuring computational complexity:

        - 1 multiplication = 1 FLOP; 1 addition = 1 FLOP; ......
        - Element-wise operations are also considered.

        ``FlopCounter`` should be used with :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` .

        .. warning::

            Currently, ``FlopCounter`` supports a limited number of aten operations.
            See the source code for the operation list.
            If you want to add new operations, use the ``extra_rules`` parameter.
            Welcome to submit a pull request to improve the default :attr:`rules` !

        :param extra_rules: additional operation rules, format as ``{aten_op: func}``,
            where ``func`` is a function that takes ``(args, kwargs, out)`` and returns the count value
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: additional list of modules to ignore.
            Operations within these modules will not be counted
        :type extra_ignore_modules: list[torch.nn.Module]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                FlopCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
            x = torch.randn(32, 100)

            flop_counter = FlopCounter()

            with DispatchCounterMode([flop_counter]):
                output = model(x)

            # Get FLOP counts
            total_flops = flop_counter.get_total()
            print(f"Total FLOPs: {total_flops}")
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _flop_mm,
            aten.addmm.default: _flop_addmm,
            aten.bmm.default: _flop_bmm,
            aten.baddbmm.default: _flop_baddbmm,
            aten.convolution.default: _flop_convolution,
            aten.convolution_backward.default: _flop_convolution_backward,
            aten.native_batch_norm.default: _flop_native_batch_norm,
            aten.native_batch_norm_backward.default: _flop_native_batch_norm_backward,
            aten.max_pool2d_with_indices.default: _flop_max_pool2d_with_indices,
            aten.max_pool2d_with_indices_backward.default: _flop_null,
            aten.avg_pool2d.default: _flop_avg_pool2d,
            aten.sum.default: _flop_sum,
            aten.sum.dim_IntList: _flop_sum,
            aten.mean.dim: _flop_mean,
            aten.add.Tensor: _flop_add,
            aten.add_.Tensor: _flop_add,
            aten.add.Scalar: _flop_add,
            aten.add_.Scalar: _flop_add,
            aten.sub.Tensor: _flop_add,
            aten.sub_.Tensor: _flop_add,
            aten.sub.Scalar: _flop_add,
            aten.sub_.Scalar: _flop_add,
            aten.rsub.Tensor: _flop_add,
            aten.rsub.Scalar: _flop_add,
            aten.neg.default: _flop_element_wise,
            aten.neg_.default: _flop_element_wise,
            aten.mul.Tensor: _flop_element_wise,
            aten.mul_.Tensor: _flop_element_wise,
            aten.mul.Scalar: _flop_element_wise,
            aten.mul_.Scalar: _flop_element_wise,
            aten.div.Tensor: _flop_element_wise,
            aten.div_.Tensor: _flop_element_wise,
            aten.div.Scalar: _flop_element_wise,
            aten.div_.Scalar: _flop_element_wise,
            aten.eq.Tensor: _flop_element_wise,
            aten.eq.Scalar: _flop_element_wise,
            aten.ne.Tensor: _flop_element_wise,
            aten.ne.Scalar: _flop_element_wise,
            aten.lt.Tensor: _flop_element_wise,
            aten.lt.Scalar: _flop_element_wise,
            aten.le.Tensor: _flop_element_wise,
            aten.le.Scalar: _flop_element_wise,
            aten.gt.Tensor: _flop_element_wise,
            aten.gt.Scalar: _flop_element_wise,
            aten.ge.Tensor: _flop_element_wise,
            aten.ge.Scalar: _flop_element_wise,
            aten.logical_and.default: _flop_element_wise,
            aten.logical_or.default: _flop_element_wise,
            aten.logical_xor.default: _flop_element_wise,
            aten.logical_not.default: _flop_element_wise,
            aten.sigmoid_.default: _flop_sigmoid,
            aten.stack.default: _flop_null,
            aten.clone.default: _flop_null,
            aten._to_copy.default: _flop_null,
            aten.full_like.default: _flop_null,
            aten.ones_like.default: _flop_null,
            aten.view.default: _flop_null,
            aten.empty.memory_format: _flop_null,
            aten.select.int: _flop_null,
            aten.select_backward.default: _flop_null,
            aten.detach.default: _flop_null,
            aten.t.default: _flop_null,
            aten.expand.default: _flop_null,
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
