from typing import Optional, Callable, Tuple
import threading
import contextlib
import functools

import torch
import torch.nn as nn
import torch.autograd as autograd

from .compress import *
from .. import base

__all__ = [
    "in_gc_1st_forward",
    "query_autocast",
    "input_compressed_gc",
    "to_gc_function",
    "GCContainer",
    "TCGCContainer",
]


_thread_local = threading.local()


@contextlib.contextmanager
def _gc_1st_forward():
    """
    Context manager that marks execution as being inside the first forward pass
    of gradient checkpointing.

    This implementation:
    - Is thread-safe (uses threading.local)
    - Supports nested usage
    - Guarantees proper restoration even if exceptions occur
    """
    depth = getattr(_thread_local, "gc_1st_forward_depth", 0)
    _thread_local.gc_1st_forward_depth = depth + 1
    try:
        yield
    finally:
        _thread_local.gc_1st_forward_depth = depth


def in_gc_1st_forward() -> bool:
    r"""
    **API Language:**
    :ref:`中文 <in_gc_1st_forward-cn>` | :ref:`English <in_gc_1st_forward-en>`

    ----

    .. _in_gc_1st_forward-cn:

    * **中文**

    判断当前是否处于梯度检查点的第一次前向传播过程中。

    :rtype: bool

    ----

    .. _in_gc_1st_forward-en:

    * **English**

    Determine whether the current execution is inside the first forward pass of gradient checkpointing.

    :rtype: bool
    """
    return getattr(_thread_local, "gc_1st_forward_depth", 0) > 0


def query_autocast() -> Tuple[str, torch.dtype, bool]:
    r"""
    **API Language:**
    :ref:`中文 <query_autocast-cn>` | :ref:`English <query_autocast-en>`

    ----

    .. _query_autocast-cn:

    * **中文**

    查询当前自动混合精度设置。

    :return: 一个包含 ``(设备类型, 数据类型, 是否启用)`` 的元组。如果 ``is_enabled == False`` ，
        ``device_type`` 和 ``dtype`` 将分别设置为 ``"cpu"`` 和 ``torch.get_autocast_dtype("cpu")`` 
    :rtype: Tuple[str, torch.dtype, bool]

    ----

    .. _query_autocast-en:

    * **English**

    Query the current autocast settings.

    :return: a tuple of ``(device_type, dtype, is_enabled)`` . If ``is_enabled == False``,
        ``device_type`` and ``dtype`` will be set as ``"cpu"`` and
        ``torch.get_autocast_dtype("cpu")``, respectively.
    :rtype: Tuple[str, torch.dtype, bool]
    """
    for device_type in ("cuda", "cpu"):
        if torch.is_autocast_enabled(device_type):
            return device_type, torch.get_autocast_dtype(device_type), True
    return "cpu", torch.get_autocast_dtype("cpu"), False


def _separate_args(*args) -> Tuple[list, list, list]:
    input_args = []  # *args, but tensors -> None
    tensor_args = []  # tensors in *args
    tensor_args_indices = []  # indices of the tensors in *args
    for i, arg in enumerate(args):
        if torch.is_tensor(arg):
            tensor_args.append(arg)
            input_args.append(None)
            tensor_args_indices.append(i)
        else:
            input_args.append(arg)
    return input_args, tensor_args, tensor_args_indices


def _combine_args(input_args, tensor_args, tensor_args_indices) -> list:
    for i, idx in enumerate(tensor_args_indices):
        input_args[idx] = tensor_args[i]
    return input_args


class InputCompressedGC(autograd.Function):
    """Gradient checkpointing with input compression.

    Reference:
    https://github.com/pytorch/pytorch/blob/v2.6.0/torch/utils/checkpoint.py
    """

    @staticmethod
    def forward(
        ctx, f_forward: Callable, x_compressor: BaseSpikeCompressor, x_seq, *args
    ):
        ctx.f_forward = f_forward
        ctx.x_compressor = x_compressor
        ctx.x_seq_shape = x_seq.shape
        ctx.ac_device_type, ctx.ac_dtype, ctx.ac_enabled = query_autocast()

        input_args, tensor_args, tensor_args_indices = _separate_args(
            x_compressor.compress(x_seq), *args
        )
        ctx.input_args = input_args  # (x_seq_compressed, *args), whose tensor -> None
        ctx.save_for_backward(*tensor_args) # tensors in (x_seq_compressed, *args)
        ctx.tensor_args_indices = tensor_args_indices  # idx of tensors in input_args

        # save RNG states
        ctx.fwd_rng_state_cpu = torch.get_rng_state()
        if torch.cuda._initialized:
            ctx.fwd_rng_state_cuda = torch.cuda.get_rng_state_all()
        else:
            ctx.fwd_rng_state_cuda = []

        # depend on external autocast context
        with _gc_1st_forward(), torch.no_grad():
            outputs = f_forward(x_seq, *args)
        return outputs  # tensor or tuple

    @staticmethod
    def backward(ctx, *grad_outputs):
        cnt_input = len(ctx.input_args) + 2
        grads = [None] * cnt_input

        if any(ctx.needs_input_grad):
            x_seq_compressed, *args = _combine_args(
                ctx.input_args, ctx.saved_tensors, ctx.tensor_args_indices
            )

            with torch.set_grad_enabled(True):
                with torch.autocast(ctx.ac_device_type, ctx.ac_dtype, ctx.ac_enabled):
                    x_seq = ctx.x_compressor.decompress(
                        x_seq_compressed, ctx.x_seq_shape
                    )
                    x_seq = x_seq.detach().requires_grad_(True)
                    for i, r in enumerate(ctx.needs_input_grad[3:]):
                        rg = r and args[i].requires_grad
                        if torch.is_tensor(args[i]):
                            args[i] = args[i].detach().requires_grad_(rg)

                    devices = range(torch.cuda.device_count())
                    with torch.random.fork_rng(devices):
                        torch.set_rng_state(ctx.fwd_rng_state_cpu)
                        torch.cuda.set_rng_state_all(ctx.fwd_rng_state_cuda)
                        outputs = ctx.f_forward(x_seq, *args)

                # grad_outputs is a tuple, while outputs can be a tensor or a tuple
                if isinstance(outputs, torch.Tensor):
                    outputs = (outputs,)
                torch.autograd.backward(outputs, grad_outputs)

            if ctx.needs_input_grad[2]:
                grads[2] = x_seq.grad
            for i in range(len(args)):
                if ctx.needs_input_grad[3 + i]:
                    grads[3 + i] = args[i].grad

        return tuple(grads)


def input_compressed_gc(f_forward, x_compressor: BaseSpikeCompressor, x_seq, *args):
    r"""
    **API Language:**
    :ref:`中文 <input_compressed_gc-cn>` | :ref:`English <input_compressed_gc-en>`

    ----

    .. _input_compressed_gc-cn:

    * **中文**

    带有输入压缩的梯度检查点。

    :param f_forward: 要进行检查点的前向函数
    :type f_forward: Callable

    :param x_compressor: 施加于 ``x_seq`` 的压缩器
    :type x_compressor: BaseSpikeCompressor

    :param x_seq: 主要输入参数，通常是脉冲序列。该张量将先被压缩，后暂存
    :type x_seq: torch.Tensor

    :param args: 其他输入参数。这些张量不会被压缩，直接被暂存
    :type args: tuple

    :return: 张量或元组
    :rtype: torch.Tensor or tuple

    ----

    .. _input_compressed_gc-en:

    * **English**

    Gradient checkpointing with input compression.

    :param f_forward: the forward function whose arguments will be checkpointed
    :type f_forward: Callable

    :param x_compressor: the compressor for ``x_seq``
    :type x_compressor: BaseSpikeCompressor

    :param x_seq: the input argument to be compressed and then checkpointed. Typically,
        ``x_seq`` is a spike train
    :type x_seq: torch.Tensor

    :param args: other arguments that will be checkpointed without compression
    :type args: tuple

    :return: a Tensor or a tuple
    :rtype: torch.Tensor or tuple

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based.memopt import input_compressed_gc
        from spikingjelly.activation_based.memopt import NullSpikeCompressor

        def simple_forward(x, weight):
            return torch.matmul(x, weight.t())

        x = torch.randn(5, 3, requires_grad=True)
        weight = torch.randn(4, 3, requires_grad=True)
        result = input_compressed_gc(simple_forward, NullSpikeCompressor(), x, weight)
        loss = result.sum()
        loss.backward()
    """
    if torch.is_grad_enabled():
        x_seq.requires_grad_(True)  # make sure the retval requires grad
        return InputCompressedGC.apply(f_forward, x_compressor, x_seq, *args)
    else:
        # If gradients are not enabled, call the forward function directly
        return f_forward(x_seq, *args)


def to_gc_function(
    x_compressor: BaseSpikeCompressor, f_forward: Optional[Callable] = None
):
    r"""
    **API Language:**
    :ref:`中文 <to_gc_function-cn>` | :ref:`English <to_gc_function-en>`

    ----

    .. _to_gc_function-cn:

    * **中文**

    将函数转换为被 ``input_compressed_gc`` 包装后的函数。本接口可作为装饰器或转换函数。

    :param x_compressor: 压缩器
    :type x_compressor: BaseSpikeCompressor

    :param f_forward: 前向函数，如果为 ``None`` 则使用装饰器模式；否则使用转换函数模式。
        默认为 ``None``
    :type f_forward: Optional[Callable]

    :return: 检查点包装后的函数
    :rtype: Callable

    ----

    .. _to_gc_function-en:

    * **English**

    Convert a forward function to a GC-wrapped forward function.
    This API can be used as a decorator or a conversion function.

    :param x_compressor: compressor
    :type x_compressor: BaseSpikeCompressor

    :param f_forward: forward function. If ``None``, use the decorator mode;
        otherwise, use the conversion function mode. Defaults to ``None``.
    :type f_forward: Optional[Callable]

    :return: the GC-wrapped forward function
    :rtype: Callable

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        from spikingjelly.activation_based.memopt import to_gc_function
        from spikingjelly.activation_based.memopt import NullSpikeCompressor

        x = torch.randn(5, 3, requires_grad=True)
        weight = torch.randn(4, 3, requires_grad=True)
        compressor = NullSpikeCompressor()

        # Usage 1: as decorator
        @to_gc_function(compressor)
        def decorated_forward(x, weight):
            return torch.matmul(x, weight.t())

        result1 = decorated_forward(x, weight)

        # Usage 2: as conversion function
        def simple_forward(x, weight):
            return torch.matmul(x, weight.t())

        converted_forward = to_gc_function(compressor, simple_forward)
        result2 = converted_forward(x, weight)
    """
    def decorator_function(forward_fn):
        @functools.wraps(forward_fn)
        def wrapped_f_forward(x_seq, *args):
            return input_compressed_gc(forward_fn, x_compressor, x_seq, *args)

        return wrapped_f_forward

    if f_forward is None:  # as a decorator
        return decorator_function
    else:  # as a conversion function
        return decorator_function(f_forward)


class GCContainer(nn.Sequential):
    def __init__(self, x_compressor: Optional[BaseSpikeCompressor], *args):
        r"""
        **API Language:**
        :ref:`中文 <GCContainer.__init__-cn>` | :ref:`English <GCContainer.__init__-en>`

        ----

        .. _GCContainer.__init__-cn:

        * **中文**

        以 ``nn.Sequential`` 风格构造梯度检查点片段（GC segment）。

        :param x_compressor: 脉冲压缩器。如果为 ``None`` 则使用 ``NullSpikeCompressor``
        :type x_compressor: Optional[BaseSpikeCompressor]

        ----

        .. _GCContainer.__init__-en:

        * **English**

        Construct a GC block module in nn.Sequential style.

        :param x_compressor: spike compressor. If None, use ``NullSpikeCompressor``
        :type x_compressor: Optional[BaseSpikeCompressor]

        ----

        * **代码示例 | Example**

        .. code-block:: python

            import torch
            import torch.nn as nn
            from spikingjelly.activation_based.memopt import GCContainer
            from spikingjelly.activation_based.memopt import NullSpikeCompressor

            container = GCContainer(
                NullSpikeCompressor(),
                nn.Linear(10, 20),
                nn.ReLU(),
                nn.Linear(20, 5)
            )

            x = torch.randn(3, 10, requires_grad=True)
            result = container(x)
        """
        super().__init__(*args)
        self.x_compressor = (
            NullSpikeCompressor() if x_compressor is None else x_compressor
        )
        self.f_forward = base.to_functional_forward(self, fn=self.super_forward)
        self.num_states = len(list(base.memories(self)))
        self._forward = (
            self.stateless_forward if self.num_states == 0 else self.stateful_forward
        )

    def super_forward(self, input):
        """
        The same as ``nn.Sequential.forward`` .

        We have to explicitly specify and use this function in ``__init__`` instead of
        using ``super().forward`` in order to avoid infinite recursion in multiprocess
        scenarios!!
        """
        for module in self:
            input = module(input)
        return input

    def stateless_forward(self, x, *args):
        return input_compressed_gc(self.super_forward, self.x_compressor, x, *args)

    def stateful_forward(self, x, *args):
        states = base.extract_memories(self)
        ret = input_compressed_gc(self.f_forward, self.x_compressor, x, *args, *states)
        outputs, states = ret[: -self.num_states], ret[-self.num_states :]
        base.load_memories(self, states)
        return outputs[0] if len(outputs) == 1 else outputs

    def forward(self, x, *args):
        return self._forward(x, *args)

    def extra_repr(self) -> str:
        return f"x_compressor={self.x_compressor.__class__.__name__},"


class TCGCContainer(GCContainer):
    r"""
    **API Language:**
    :ref:`中文 <TCGCContainer-cn>` | :ref:`English <TCGCContainer-en>`

    ----

    .. _TCGCContainer-cn:

    * **中文**

    时间分块的 ``GCContainer`` 。

    :param x_compressor: 脉冲压缩器。如果为 ``None`` 则使用 ``NullSpikeCompressor``
    :type x_compressor: Optional[BaseSpikeCompressor]

    :param *args: 传递给 ``nn.Sequential`` 的若干模块。必须以位置参数形式传入

    :param n_chunk: 分块数量。默认为1。必须以关键字参数形式传入
    :type n_chunk: int

    :param n_seq_inputs: 需要分块处理的序列输入数量。默认为1。必须以关键字参数形式传入
    :type n_seq_inputs: int

    :param n_outputs: 输出数量。本模块假设输出都是 ``torch.Tensor`` 。默认为1。必须以关键字参数形式传入
    :type n_outputs: int

    ----

    .. _TCGCContainer-en:

    * **English**

    Temporally Chunked ``GCContainer`` .

    :param x_compressor: spike compressor. If None, use ``NullSpikeCompressor``
    :type x_compressor: Optional[BaseSpikeCompressor]

    :param *args: modules as arguments of ``nn.Sequential``. Must act as positional arguments

    :param n_chunk: number of chunks. Default to 1. Must act as keyword arguments
    :type n_chunk: int

    :param n_seq_inputs: number of sequence inputs. Default to 1. Must act as keyword arguments
    :type n_seq_inputs: int

    :param n_outputs: number of outputs. This container assumes that all outputs are ``torch.Tensor``.
        Default to 1. Must act as keyword arguments
    :type n_outputs: int

    ----

    * **代码示例 | Example**

    .. code-block:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based.memopt import TCGCContainer
        from spikingjelly.activation_based.memopt import NullSpikeCompressor

        # Basic usage
        tc_container = TCGCContainer(
            NullSpikeCompressor(),
            nn.Linear(10, 20),
            nn.ReLU(),
            nn.Linear(20, 5),
            n_chunk=4
        )
        x_seq = torch.randn(8, 3, 10, requires_grad=True)  # T=8
        result = tc_container(x_seq)
        print(f"Input shape: {x_seq.shape}")
        print(f"Output shape: {result.shape}")
    """

    def __init__(
        self,
        x_compressor: Optional[BaseSpikeCompressor],
        *args,
        n_chunk: int = 1,
        n_seq_inputs: int = 1,
        n_outputs: int = 1,
    ):
        super().__init__(x_compressor, *args)
        self.n_chunk = n_chunk
        self.n_seq_inputs = n_seq_inputs
        self.n_outputs = n_outputs

    def forward(self, x_seq: torch.Tensor, *args):
        n_chunk = min(self.n_chunk, x_seq.shape[0])  # n_chunk should not exceed T
        seq_inputs = args[: self.n_seq_inputs - 1]
        other_inputs = args[self.n_seq_inputs - 1 :]

        chunked = [torch.chunk(x_seq, n_chunk, dim=0)] + [
            torch.chunk(seq, n_chunk, dim=0) for seq in seq_inputs
        ]
        outputs_per_chunk = [[] for _ in range(self.n_outputs)]

        for i in range(n_chunk):
            current_inputs = [c[i] for c in chunked] + list(other_inputs)
            outs = super().forward(*current_inputs)
            if not isinstance(outs, tuple):
                outs = (outs,)
            for j, o in enumerate(outs):
                outputs_per_chunk[j].append(o)

        final_outputs = [torch.cat(chunks, dim=0) for chunks in outputs_per_chunk]

        return final_outputs[0] if len(final_outputs) == 1 else tuple(final_outputs)

    def extra_repr(self):
        return (
            f"x_compressor={self.x_compressor.__class__.__name__}, "
            f"n_chunk={self.n_chunk}, "
            f"n_seq_inputs={self.n_seq_inputs}, "
            f"n_seq_outputs={self.n_outputs}"
        )
