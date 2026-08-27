from typing import Callable, Optional

import torch
import torch.nn as nn
from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
    checkpoint_wrapper,
)
from torch.utils.checkpoint import checkpoint as torch_checkpoint
from torch.utils._pytree import tree_flatten, tree_unflatten

from .. import base
from .compress import SpikeCompressor

__all__ = ["checkpoint", "checkpoint_module"]


class _GradientToken(torch.autograd.Function):
    @staticmethod
    def forward(ctx, tensor: torch.Tensor) -> torch.Tensor:
        return tensor.new_zeros(()).expand_as(tensor)

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return grad


def checkpoint(
    function: Callable,
    *args: object,
    compressor: Optional[SpikeCompressor] = None,
    **kwargs: object,
) -> object:
    r"""
    **API Language** - :ref:`中文 <memopt-checkpoint-cn>` | :ref:`English <memopt-checkpoint-en>`

    ----

    .. _memopt-checkpoint-cn:

    * **中文**

    使用 PyTorch non-reentrant gradient checkpoint 执行 ``function``，支持关键字
    参数和 pytree 输入输出。指定 ``compressor`` 时，仅压缩第一个位置 tensor
    参数；其他参数由 PyTorch checkpoint 原样管理。输入的 device 和 dtype 支持范围
    由 ``function`` 与 ``compressor`` 决定。

    :param function: 需要 checkpoint 的可调用对象。
    :type function: Callable
    :param args: 传给 ``function`` 的位置参数。
    :type args: object
    :param compressor: 可选的首个位置 tensor 输入压缩器。``None`` 表示不压缩。
    :type compressor: Optional[SpikeCompressor]
    :param kwargs: 传给 ``function`` 的关键字参数。
    :type kwargs: object
    :return: ``function`` 的输出。
    :rtype: object

    ----

    .. _memopt-checkpoint-en:

    * **English**

    Run ``function`` with PyTorch non-reentrant gradient checkpointing, including
    keyword arguments and pytree inputs and outputs. When ``compressor`` is set,
    only the first positional tensor is compressed; PyTorch checkpointing manages
    all other inputs unchanged. Supported devices and dtypes are determined by
    ``function`` and ``compressor``.

    :param function: Callable to checkpoint.
    :type function: Callable
    :param args: Positional arguments passed to ``function``.
    :type args: object
    :param compressor: Optional compressor for the first positional tensor input;
        ``None`` disables compression.
    :type compressor: Optional[SpikeCompressor]
    :param kwargs: Keyword arguments passed to ``function``.
    :type kwargs: object
    :return: Output of ``function``.
    :rtype: object
    """
    target_index = next(
        (index for index, arg in enumerate(args) if isinstance(arg, torch.Tensor)),
        None,
    )
    if compressor is None or target_index is None:
        return torch_checkpoint(function, *args, use_reentrant=False, **kwargs)

    target = args[target_index]
    with torch.no_grad():
        packed = compressor.compress(target)
    token = _GradientToken.apply(target)
    remaining_args = args[:target_index] + args[target_index + 1 :]

    def compressed_function(packed_input, gradient_token, *inner_args, **inner_kwargs):
        restored = compressor.decompress(packed_input)
        restored = gradient_token + (restored - gradient_token).detach()
        function_args = (
            inner_args[:target_index] + (restored,) + inner_args[target_index:]
        )
        return function(*function_args, **inner_kwargs)

    return torch_checkpoint(
        compressed_function,
        packed,
        token,
        *remaining_args,
        use_reentrant=False,
        **kwargs,
    )


def _run_functional(
    module: nn.Module,
    functional_forward: Callable,
    buffer_names: tuple[str, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    states: tuple[object, ...],
    buffers: tuple[torch.Tensor, ...],
):
    if not buffer_names:
        outputs, updated_states = functional_forward(args, states, **kwargs)
        return outputs, updated_states, ()

    original_states = base.extract_memories(module)
    base.load_memories(module, list(states))
    working_buffers = {
        name: value.clone() for name, value in zip(buffer_names, buffers)
    }
    try:
        outputs = torch.func.functional_call(
            module, (dict(module.named_parameters()), working_buffers), args, kwargs
        )
        updated_states = tuple(base.extract_memories(module))
    finally:
        base.load_memories(module, original_states)
    if not isinstance(outputs, (tuple, list)):
        outputs = (outputs,)
    return tuple(outputs), updated_states, tuple(working_buffers.values())


def _run_module(
    module: nn.Module,
    functional_forward: Callable,
    buffer_names: tuple[str, ...],
    args: tuple[object, ...],
    kwargs: dict[str, object],
    states: tuple[object, ...],
    buffers: tuple[torch.Tensor, ...],
    compressor: Optional[SpikeCompressor],
):
    arg_count = len(args)
    state_count = len(states)

    def function(*flat_args, **inner_kwargs):
        module_args = flat_args[:arg_count]
        module_states = flat_args[arg_count : arg_count + state_count]
        module_buffers = flat_args[arg_count + state_count :]
        return _run_functional(
            module,
            functional_forward,
            buffer_names,
            module_args,
            inner_kwargs,
            module_states,
            module_buffers,
        )

    outputs, updated_states, updated_buffers = checkpoint(
        function, *args, *states, *buffers, compressor=compressor, **kwargs
    )
    return outputs, updated_states, updated_buffers


def _merge_chunk_outputs(outputs: list[object], time_dim: int):
    flat_chunks = []
    spec = None
    for output in outputs:
        flat, current_spec = tree_flatten(output)
        if spec is None:
            spec = current_spec
        elif current_spec != spec:
            raise ValueError(
                "All temporal chunks must return the same pytree structure."
            )
        flat_chunks.append(flat)

    merged = []
    for leaves in zip(*flat_chunks):
        first = leaves[0]
        if isinstance(first, torch.Tensor):
            if not all(isinstance(leaf, torch.Tensor) for leaf in leaves):
                raise ValueError("Output leaf types differ between temporal chunks.")
            merged.append(torch.cat(leaves, dim=time_dim))
        else:
            if any(leaf != first for leaf in leaves[1:]):
                raise ValueError("Non-tensor output leaves must be chunk-invariant.")
            merged.append(first)
    return tree_unflatten(merged, spec)


def _checkpoint_module_forward(
    module: nn.Module,
    *args: object,
    functional_forward: Callable,
    compressor: Optional[SpikeCompressor],
    chunks: int,
    chunked_args: tuple[int, ...],
    time_dim: int,
    **kwargs: object,
):
    states = tuple(base.extract_memories(module))
    live_buffers = dict(module.named_buffers())
    buffer_names = tuple(live_buffers)
    buffer_refs = tuple(live_buffers.values())
    buffers = tuple(buffer.detach().clone() for buffer in buffer_refs)
    if chunks == 1:
        outputs, states, buffers = _run_module(
            module,
            functional_forward,
            buffer_names,
            args,
            kwargs,
            states,
            buffers,
            compressor,
        )
        result = outputs[0] if len(outputs) == 1 else outputs
    else:
        sequence_length = None
        chunk_values: dict[int, tuple[torch.Tensor, ...]] = {}
        for index in chunked_args:
            if index >= len(args) or not isinstance(args[index], torch.Tensor):
                raise ValueError(
                    f"chunked_args contains non-tensor argument index {index}."
                )
            length = args[index].shape[time_dim]
            if sequence_length is None:
                sequence_length = length
            elif length != sequence_length:
                raise ValueError(
                    "All chunked arguments must have the same temporal length."
                )
            chunk_values[index] = torch.tensor_split(args[index], chunks, dim=time_dim)

        if sequence_length is None:
            raise ValueError("chunked_args must contain at least one tensor argument.")
        if sequence_length == 0 or chunks > sequence_length:
            raise ValueError(
                f"chunks={chunks} requires a non-empty temporal length T >= chunks; "
                f"got T={sequence_length}."
            )

        chunk_outputs = []
        for chunk_index in range(chunks):
            current_args = list(args)
            for index, values in chunk_values.items():
                current_args[index] = values[chunk_index]
            outputs, states, buffers = _run_module(
                module,
                functional_forward,
                buffer_names,
                tuple(current_args),
                kwargs,
                states,
                buffers,
                compressor,
            )
            chunk_outputs.append(outputs[0] if len(outputs) == 1 else outputs)
        result = _merge_chunk_outputs(chunk_outputs, time_dim)

    base.load_memories(module, list(states))
    with torch.no_grad():
        for buffer, updated in zip(buffer_refs, buffers, strict=True):
            buffer.copy_(updated)
    return result


def checkpoint_module(
    module: nn.Module,
    *,
    compressor: Optional[SpikeCompressor] = None,
    chunks: int = 1,
    chunked_args: tuple[int, ...] = (0,),
    time_dim: int = 0,
) -> nn.Module:
    r"""
    **API Language** - :ref:`中文 <checkpoint-module-cn>` | :ref:`English <checkpoint-module-en>`

    ----

    .. _checkpoint-module-cn:

    * **中文**

    返回保留参数名称、参数对象和 ``state_dict`` 键的 checkpoint wrapper。包含
    :class:`~spikingjelly.activation_based.base.MemoryModule` 的模块使用显式状态重算；
    神经元状态和 buffer 只提交一次。``chunks > 1`` 时，``chunked_args`` 指定的
    tensor 沿 ``time_dim`` 分块，tensor 输出沿同一维拼接，非 tensor 输出必须在各
    分块中相同。输入的 device、dtype 和 backend 支持范围由被包装模块和压缩器决定。

    :param module: 需要包装的模块。
    :type module: nn.Module
    :param compressor: 可选的首个位置 tensor 输入压缩器。``None`` 表示不压缩。
    :type compressor: Optional[SpikeCompressor]
    :param chunks: 时间分块数，必须大于 0，且不能超过时间维长度。
    :type chunks: int
    :param chunked_args: 需要分块的位置 tensor 参数索引。
    :type chunked_args: tuple[int, ...]
    :param time_dim: 输入分块和输出拼接使用的时间维。
    :type time_dim: int
    :return: 保持 ``state_dict`` 键不变的 checkpoint wrapper。
    :rtype: nn.Module
    :raises ValueError: 分块数、参数索引、时间长度或各分块输出不兼容时抛出。

    ----

    .. _checkpoint-module-en:

    * **English**

    Return a checkpoint wrapper that preserves parameter names, parameter objects,
    and ``state_dict`` keys. Modules containing
    :class:`~spikingjelly.activation_based.base.MemoryModule` use explicit state
    during recomputation; neuron state and buffers are committed once. When
    ``chunks > 1``, tensors selected by ``chunked_args`` are split along
    ``time_dim``. Tensor outputs are concatenated along the same dimension, while
    non-tensor outputs must be identical across chunks. Supported devices, dtypes,
    and backends are determined by the wrapped module and compressor.

    :param module: Module to wrap.
    :type module: nn.Module
    :param compressor: Optional compressor for the first positional tensor input;
        ``None`` disables compression.
    :type compressor: Optional[SpikeCompressor]
    :param chunks: Number of temporal chunks; must be positive and no greater than
        the temporal length.
    :type chunks: int
    :param chunked_args: Indices of positional tensor arguments to split.
    :type chunked_args: tuple[int, ...]
    :param time_dim: Temporal dimension used for input splitting and output joining.
    :type time_dim: int
    :return: Checkpoint wrapper with unchanged ``state_dict`` keys.
    :rtype: nn.Module
    :raises ValueError: If the chunk count, argument indices, temporal lengths, or
        chunk outputs are incompatible.
    """
    if chunks < 1:
        raise ValueError(f"chunks must be positive, got {chunks}.")
    return checkpoint_wrapper(
        module,
        checkpoint_fn=_checkpoint_module_forward,
        functional_forward=base.to_functional_forward(module),
        compressor=compressor,
        chunks=chunks,
        chunked_args=chunked_args,
        time_dim=time_dim,
    )


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    return module._checkpoint_wrapped_module


def _checkpoint_options(module: nn.Module) -> dict[str, object]:
    return module.checkpoint_fn.keywords
