from typing import Optional, Callable, Tuple
import threading
import contextlib
import functools

import torch
import torch.nn as nn
import torch.autograd as autograd

from .compress import *
from ...profiler import *
from ... import base

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
    _thread_local.in_gc_1st_forward = True
    try:
        yield
    finally:
        _thread_local.in_gc_1st_forward = False


def in_gc_1st_forward():
    """
    Whether in the first FP of gradient checkpointing. Used for BN momentum correction.
    """
    return getattr(_thread_local, "in_gc_1st_forward", False)


def query_autocast() -> Tuple[Optional[str], Optional[torch.dtype], bool]:
    """
    Query the current autocast settings.

    :return: a tuple of (device_type, dtype, is_enabled)
    :rtype: Tuple[Optional[str], Optional[torch.dtype], bool]
    """
    for device_type in ("cuda", "cpu"):
        if torch.is_autocast_enabled(device_type):
            return device_type, torch.get_autocast_dtype(device_type), True
    return None, None, False


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

    Args:
        f_forward (Callable): the forward function whose arguments will be checkpointed.
        x_compressor (BaseSpikeCompressor): the compressor for x_seq
        x_seq (torch.Tensor): the input to be compressed and checkpointed.
        *args: other arguments that will be checkpointed without compression.

    Returns:
        a Tensor or a tuple

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
        ctx.save_for_backward(*tensor_args)
        ctx.input_args = input_args  # (x_seq_compressed, *args)
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
                ctx.input_args, ctx.tensor_args, ctx.tensor_args_indices
            )

            with torch.set_grad_enabled(True):
                with torch.autocast(ctx.ac_device_type, ctx.ac_dtype, ctx.ac_enabled):
                    x_seq = ctx.x_compressor.decompress(
                        x_seq_compressed, ctx.x_seq_shape
                    )
                    x_seq = x_seq.detach().requires_grad_(True)
                    for i, r in enumerate(ctx.needs_input_grad[3:]):
                        rg = r and args[i].requires_grad
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
    if torch.is_grad_enabled():
        x_seq.requires_grad_(True)  # make sure the retval requires grad
        return InputCompressedGC.apply(f_forward, x_compressor, x_seq, *args)
    else:
        # If gradients are not enabled, call the forward function directly
        return f_forward(x_seq, *args)


def to_gc_function(
    x_compressor: BaseSpikeCompressor, f_forward: Optional[Callable] = None
):
    """Convert a forward function to a GC-blocked forward function.

    Usage 1. as a decorator:
    ```
    @to_gc_block(x_compressor)
    def f_forward(x_seq, *args): ...
    ```

    Usage 2. as a conversion function:
    ```
    f_forward = to_gc_block(x_compressor, f_forward)
    ```

    Args:
        x_compressor
        f_forward (Callable, optional): if None, use the decorator mode;
            otherwise, use the conversion function mode. Defaults to None.

    Returns:
        Callable: the GC-blocked forward function
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
        """Construct a GC block module in nn.Sequential style."""
        super().__init__(*args)
        self.x_compressor = (
            NullSpikeCompressor() if x_compressor is None else x_compressor
        )
        self.f_forward = base.to_functional_forward(
            self, fn=super().forward
        )  # avoid infinite recursion
        self.num_states = len(list(base.memories(self)))
        self._forward = (
            self.stateless_forward if self.num_states == 0 else self.stateful_forward
        )

    def stateless_forward(self, x, *args):
        return input_compressed_gc(super().forward, self.x_compressor, x, *args)

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
    """Temporally Chunked GCContainer."""

    def __init__(
        self,
        x_compressor: BaseSpikeCompressor,
        n_chunk: int = 1,
        n_seq_inputs: int = 1,
        n_outputs: int = 1,
        *args,
    ):
        super().__init__(x_compressor, *args)
        self.n_chunk = n_chunk
        self.n_seq_inputs = n_seq_inputs
        self.n_outputs = n_outputs

    def forward(self, x_seq: torch.Tensor, *args) -> torch.Tensor:
        x_seqs = torch.chunk(x_seq, self.n_chunk, dim=0)  # single primary input
        out_seq = []
        for xc in x_seqs:
            yc = super().forward(xc, *args)  # single tensor output
            out_seq.append(yc)
        return torch.cat(out_seq, dim=0)

    def forward(self, x_seq: torch.Tensor, *args):
        seq_inputs = args[: self.n_seq_inputs - 1]
        other_inputs = args[self.n_seq_inputs - 1 :]

        chunked = [torch.chunk(x_seq, self.n_chunk, dim=0)] + [
            torch.chunk(seq, self.n_chunk, dim=0) for seq in seq_inputs
        ]
        outputs_per_chunk = [[] for _ in range(self.n_outputs)]

        for i in range(self.n_chunk):
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
