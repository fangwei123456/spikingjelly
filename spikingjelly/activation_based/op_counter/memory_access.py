from collections import defaultdict
from typing import Any, Callable

import torch

aten = torch.ops.aten
import torch.nn as nn

from .base import BaseCounter

__all__ = ["MemoryAccessCounter"]


def _bytes(x: torch.Tensor):
    return x.element_size() * x.numel() if torch.is_tensor(x) else 0


def _memory_null(args, kwargs, out):
    return 0

def _memory_mm(args, kwargs, out):
    """out = x @ y"""
    x, y = args[:2]
    _, k = x.shape
    kk, _ = y.shape
    if k != kk:
        raise AssertionError(f"mm: inner dimensions mismatch [{x.shape} and {y.shape}]")
    return _bytes(x) + _bytes(y) + _bytes(out)


def _memory_addmm(args, kwargs, out):
    """out = beta * bias + alpha * (x @ y)"""
    bias, x, y = args[:3]
    _, k = x.shape
    kk, _ = y.shape
    if k != kk:
        raise AssertionError(
            f"addmm: inner dimensions mismatch [{x.shape} and {y.shape}]"
        )

    m = _bytes(x) + _bytes(y) + _bytes(out)
    beta = kwargs.get("beta", 1.0)
    if beta != 0:
        m += _bytes(bias)
    return m


def _memory_bmm(args, kwargs, out):
    """Batch matrix multiply: out[b] = x[b] @ y[b]"""
    x, y = args[:2]
    b, _, k = x.shape
    bb, kk, _ = y.shape
    if b != bb or k != kk:
        raise AssertionError(
            f"bmm: batch or inner dimensions mismatch [{x.shape} and {y.shape}]"
        )
    return _bytes(x) + _bytes(y) + _bytes(out)


def _memory_baddbmm(args, kwargs, out):
    """out[b] = beta * b[b] + alpha * (x[b] @ y[b])"""
    bias, x, y = args[:3]
    b, m, k = x.shape
    bb, kk, n = y.shape
    if b != bb or k != kk:
        raise AssertionError(
            f"baddmm: batch or inner dimensions mismatch [{x.shape}, {y.shape}]"
        )

    m = _bytes(x) + _bytes(y) + _bytes(out)
    beta = kwargs.get("beta", 1.0)
    if beta != 0:
        m += _bytes(bias)
    return m


def _memory_convolution(args, kwargs, out):
    x, w, bias = args[:3]
    m = _bytes(x) + _bytes(w) + _bytes(out)
    if bias is not None:
        m += _bytes(bias)
    return m


def _memory_convolution_backward(args, kwargs, out):
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
        _transposed,
        _output_padding,
        _groups,
        output_mask,
    ) = args
    m = _bytes(grad_out)

    if output_mask[0]:  # grad_x
        grad_x = out[0]
        m += _bytes(w)
        m += _bytes(grad_x)

    if output_mask[1]:  # grad_weight
        grad_weight = out[1]
        m += _bytes(x)
        m += _bytes(grad_weight)

    if output_mask[2]:  # grad_bias
        grad_bias = out[2]
        m += _bytes(grad_bias)

    return m


def _memory_max_pool2d_with_indices(args, kwargs, out):
    x = args[0]
    y, indices = out
    return _bytes(x) + _bytes(y) + _bytes(indices)


def _memory_max_pool2d_with_indices_backward(args, kwargs, out):
    grad_output, indices = args[0], args[1]
    grad_x = out
    return _bytes(grad_output) + _bytes(indices) + _bytes(grad_x)

def _memory_avg_pool2d(args, kwargs, out):
    x = args[0]
    return _bytes(x) + _bytes(out)


def _memory_mean(args, kwargs, out):
    x = args[0]
    return _bytes(x) + _bytes(out)


def _memory_element_wise_binary(args, kwargs, out):
    x, y = args[:2]
    return _bytes(x) + _bytes(y) + _bytes(out)


def _memory_element_wise_unary(args, kwargs, out):
    x = args[0]
    return _bytes(x) + _bytes(out)


def _memory_stack(args, kwargs, out):
    tensor_list = args[0]
    return sum(_bytes(x) for x in tensor_list) + _bytes(out)


def _memory_clone(args, kwargs, out):
    x = args[0]
    return _bytes(x) + _bytes(out)


def _memory_full_like(args, kwargs, out):
    return _bytes(out)


def _memory_select_backward(args, kwargs, out):
    return _bytes(args[0]) + _bytes(out)


def _memory_native_batch_norm(args, kwargs, out):
    x, mean, var, gamma, beta, train = args[:6]
    m = _bytes(x) + _bytes(mean) + _bytes(var) + _bytes(gamma) + _bytes(beta)
    if train:
        m += _bytes(out[0]) + _bytes(out[1]) + _bytes(out[2]) # write x, mean, var
    else:
        m += _bytes(out[0]) # write only x
    return m


def _memory_native_batch_norm_backward(args, kwargs, out):
    grad_output, x, gamma, running_mean, running_var, saved_mean, saved_invstd = args[:7]
    train, output_mask = args[-3], args[-1]
    n = grad_output.numel()
    c = gamma.numel()

    m = 0
    if train:
        if output_mask[0]:  # grad_input
            m += _bytes(grad_output)
            m += _bytes(x) + _bytes(saved_mean) +  _bytes(saved_invstd)
            m += _bytes(gamma)
            # grad_gamma and grad_beta has been computed!
        elif output_mask[1]:  # grad_gamma
            m += _bytes(grad_output)
            m += _bytes(x) + _bytes(saved_mean) +  _bytes(saved_invstd)
        elif output_mask[2]:  # grad_beta
            m += _bytes(grad_output)
    else:
        if output_mask[0]:  # grad_input
            m += _bytes(grad_output) + _bytes(saved_invstd) + _bytes(gamma)
        if output_mask[1]: # grad_gamma
            m += _bytes(x) + _bytes(saved_mean)
            if not output_mask[0]:
                m += _bytes(saved_invstd) + _bytes(grad_output)
        if output_mask[2] and not output_mask[0] and not output_mask[1]:
            m += _bytes(grad_output)
    return m


class MemoryAccessCounter(BaseCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] = {},
        extra_ignore_modules: list[nn.Module] = [],
    ):
        r"""
        **API Language:**
        :ref:`中文 <MemoryAccessCounter.__init__-cn>` | :ref:`English <MemoryAccessCounter.__init__-en>`

        ----

        .. _MemoryAccessCounter.__init__-cn:

        * **中文**

        内存访问计数器，用于粗略估计深度神经网络的内存访问量。

        该计数器统计操作所需的输入、输出张量的 **字节** 数，作为对内存访问量的 **下界估计** 。真实的内存访问量由算子的load store模式决定，取决于具体实现，在此不做考虑。

        ``MemoryAccessCounter`` 应与 :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>` 搭配使用。

        .. warning::

            目前，``MemoryAccessCounter`` 支持的 aten 操作类型有限。查看源代码以获取操作列表。如需添加新操作，
            可以使用 ``extra_rules`` 参数；也欢迎提交 pull request 来完善默认的 :attr:`rules` ！

        :param extra_rules: 额外的操作规则，格式为 ``{aten_op, func}``，
            其中 ``func`` 是一个函数，接受 ``(args, kwargs, out)`` 并返回字节数
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: 额外需要忽略的模块列表，这些模块中的操作不会被计数
        :type extra_ignore_modules: list[nn.Module]

        ----

        .. _MemoryAccessCounter.__init__-en:

        * **English**

        Memory access counter for estimating memory access in deep networks.

        This counter tracks the **byte** count of input and output tensors for operations as a **lower bound estimate** of memory access. Actual amount of memory access depends on the load store patterns of specific implementations, so it is not considered here.

        ``MemoryAccessCounter`` should be used with :class:`DispatchCounterMode <spikingjelly.activation_based.op_counter.base.DispatchCounterMode>`.

        .. warning::

            Currently, ``MemoryAccessCounter`` supports a limited number of aten operations. See the source code for the list of operations. If you want to add a new operation, you can use the ``extra_rules`` parameter. Welcome to submit a pull request to improve the default :attr:`rules`!

        :param extra_rules: additional operation rules, format as ``{aten_op: func}``,
            where ``func`` is a function that takes ``(args, kwargs, out)`` and returns byte count
        :type extra_rules: dict[Any, Callable]

        :param extra_ignore_modules: additional list of modules to ignore.
            Operations within these modules will not be counted
        :type extra_ignore_modules: list[nn.Module]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            from spikingjelly.activation_based.op_counter import (
                MemoryAccessCounter,
                DispatchCounterMode,
            )
            import torch
            import torch.nn as nn

            model = nn.Sequential(nn.Linear(100, 50), nn.ReLU(), nn.Linear(50, 10))
            x = torch.randn(32, 100)

            memory_counter = MemoryAccessCounter()

            with DispatchCounterMode([memory_counter]):
                output = model(x)

            total_bytes = memory_counter.get_total()
            print(f"Total memory access: {total_bytes / 1024:.2f} KB")
        """
        self.records: dict[str, dict[Any, int]] = defaultdict(lambda: defaultdict(int))
        self.rules: dict[Any, Callable] = {
            aten.mm.default: _memory_mm,
            aten.addmm.default: _memory_addmm,
            aten.bmm.default: _memory_bmm,
            aten.baddbmm.default: _memory_baddbmm,
            aten.convolution.default: _memory_convolution,
            aten.convolution_backward.default: _memory_convolution_backward,
            aten.native_batch_norm.default: _memory_native_batch_norm,
            aten.native_batch_norm_backward.default: _memory_native_batch_norm_backward,
            aten.max_pool2d_with_indices.default: _memory_max_pool2d_with_indices,
            aten.max_pool2d_with_indices_backward.default: _memory_max_pool2d_with_indices_backward,
            aten.avg_pool2d.default: _memory_avg_pool2d,
            aten.sum.default: _memory_mean,
            aten.sum.dim_IntList: _memory_mean,
            aten.mean.dim: _memory_mean,
            aten.add.Tensor: _memory_element_wise_binary,
            aten.add_.Tensor: _memory_element_wise_binary,
            aten.add.Scalar: _memory_element_wise_binary,
            aten.add_.Scalar: _memory_element_wise_binary,
            aten.sub.Tensor: _memory_element_wise_binary,
            aten.sub_.Tensor: _memory_element_wise_binary,
            aten.sub.Scalar: _memory_element_wise_binary,
            aten.sub_.Scalar: _memory_element_wise_binary,
            aten.rsub.Tensor: _memory_element_wise_binary,
            aten.rsub.Scalar: _memory_element_wise_binary,
            aten.neg.default: _memory_element_wise_unary,
            aten.neg_.default: _memory_element_wise_unary,
            aten.mul.Tensor: _memory_element_wise_binary,
            aten.mul_.Tensor: _memory_element_wise_binary,
            aten.mul.Scalar: _memory_element_wise_binary,
            aten.mul_.Scalar: _memory_element_wise_binary,
            aten.div.Tensor: _memory_element_wise_binary,
            aten.div_.Tensor: _memory_element_wise_binary,
            aten.div.Scalar: _memory_element_wise_binary,
            aten.div_.Scalar: _memory_element_wise_binary,
            aten.eq.Tensor: _memory_element_wise_binary,
            aten.eq.Scalar: _memory_element_wise_binary,
            aten.ne.Tensor: _memory_element_wise_binary,
            aten.ne.Scalar: _memory_element_wise_binary,
            aten.lt.Tensor: _memory_element_wise_binary,
            aten.lt.Scalar: _memory_element_wise_binary,
            aten.le.Tensor: _memory_element_wise_binary,
            aten.le.Scalar: _memory_element_wise_binary,
            aten.gt.Tensor: _memory_element_wise_binary,
            aten.gt.Scalar: _memory_element_wise_binary,
            aten.ge.Tensor: _memory_element_wise_binary,
            aten.ge.Scalar: _memory_element_wise_binary,
            aten.logical_and.default: _memory_element_wise_binary,
            aten.logical_or.default: _memory_element_wise_binary,
            aten.logical_xor.default: _memory_element_wise_binary,
            aten.logical_not.default: _memory_element_wise_binary,
            aten.sigmoid_.default: _memory_element_wise_unary,
            aten.stack.default: _memory_stack,
            aten.clone.default: _memory_clone,
            aten._to_copy.default: _memory_clone,
            aten.full_like.default: _memory_full_like,
            aten.ones_like.default: _memory_full_like,
            aten.view.default: _memory_null,
            aten.empty.memory_format: _memory_null,
            aten.select.int: _memory_null, # return a view
            aten.select_backward.default: _memory_select_backward, # involve load store
            aten.detach.default: _memory_null,
            aten.t.default: _memory_null,
            aten.expand.default: _memory_null,
        }
        self.ignore_modules = []
        self.rules.update(extra_rules)
        self.ignore_modules.extend(extra_ignore_modules)
