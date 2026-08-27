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
    r"""Checkpoint a function and optionally compress its first tensor input.

    **中文：** 使用 PyTorch non-reentrant gradient checkpoint。支持关键字参数和
    pytree 输入输出。``compressor`` 仅压缩第一个位置 tensor 参数；其他参数由
    PyTorch checkpoint 原样管理。

    **English:** Use PyTorch non-reentrant gradient checkpointing with kwargs
    and pytree support. ``compressor`` applies only to the first positional tensor;
    PyTorch manages all remaining inputs unchanged.

    :param function: function to checkpoint / 被检查点包装的函数
    :type function: Callable
    :param compressor: optional input compressor / 可选输入压缩器
    :type compressor: Optional[SpikeCompressor]
    :return: function outputs / 函数输出
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
    buffer_names: tuple[str, ...],
    compressor: Optional[SpikeCompressor],
    chunks: int,
    chunked_args: tuple[int, ...],
    time_dim: int,
    **kwargs: object,
):
    states = tuple(base.extract_memories(module))
    buffer_refs = tuple(dict(module.named_buffers()).values())
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
        for buffer, updated in zip(buffer_refs, buffers):
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
    r"""Return a state-dict-transparent checkpoint wrapper for ``module``.

    **中文：** 包装模块并保持参数名称与 ``state_dict`` 键不变。包含
    :class:`MemoryModule` 的模块通过显式 functional state 重算。``chunks > 1``
    时，指定位置参数沿 ``time_dim`` 分块，tensor 输出沿同一维拼接。

    **English:** Wrap a module without changing parameter names or state-dict
    keys. Modules containing :class:`MemoryModule` use explicit functional state
    during recomputation. With ``chunks > 1``, selected positional inputs are split
    along ``time_dim`` and tensor outputs are concatenated along that dimension.

    :param module: module to wrap / 待包装模块
    :type module: nn.Module
    :param compressor: optional first-input compressor / 可选首输入压缩器
    :type compressor: Optional[SpikeCompressor]
    :param chunks: temporal chunk count / 时间分块数
    :type chunks: int
    :param chunked_args: positional tensor indices to split / 待切分位置参数索引
    :type chunked_args: tuple[int, ...]
    :param time_dim: temporal dimension / 时间维
    :type time_dim: int
    :return: transparent checkpoint wrapper / 透明检查点 wrapper
    :rtype: nn.Module
    :raises ValueError: if chunk configuration or outputs are incompatible / 分块配置或输出不兼容
    """
    if chunks < 1:
        raise ValueError(f"chunks must be positive, got {chunks}.")
    buffer_names = tuple(dict(module.named_buffers()))
    return checkpoint_wrapper(
        module,
        checkpoint_fn=_checkpoint_module_forward,
        functional_forward=base.to_functional_forward(module),
        buffer_names=buffer_names,
        compressor=compressor,
        chunks=chunks,
        chunked_args=chunked_args,
        time_dim=time_dim,
    )


def _unwrap_checkpoint_module(module: nn.Module) -> nn.Module:
    return module._checkpoint_wrapped_module


def _checkpoint_options(module: nn.Module) -> dict[str, object]:
    return module.checkpoint_fn.keywords
