"""
**API Language:**
:ref:`中文 <flexsn.custom_ops-cn>` | :ref:`English <flexsn.custom_ops-en>`

----

.. _flexsn.custom_ops-cn:

* **中文**

``custom_ops`` 模块为 FlexSN 的共享 CUDA 路径提供底层 opaque custom op。
它负责在 Python 侧保存 Triton kernel 与元数据，并通过轻量级整数 ``handle``
把这些 kernel 暴露给 ``torch.compile`` / AOTAutograd。

该模块的主要职责包括：

* 在 Python 注册表中维护 FlexSN kernel handle；
* 将前向 / 反向封装为 ``torch.library`` custom op，避免编译器追踪 Python
  kernel 对象与 Triton launcher 细节；
* 为 fake tensor、autograd setup/backward、final-state fast path 提供辅助实现。

通常用户不需要直接调用这些函数；它们主要由
:class:`spikingjelly.activation_based.neuron.flexsn.FlexSN` 与
:class:`spikingjelly.activation_based.neuron.flexsn.FlexSNKernel` 间接使用。

----

.. _flexsn.custom_ops-en:

* **English**

The ``custom_ops`` module provides the low-level opaque custom ops used by
FlexSN's shared CUDA execution path. It stores Triton kernels and metadata in
a Python-side registry and exposes them to ``torch.compile`` / AOTAutograd
through lightweight integer ``handle`` values.

Its main responsibilities are:

* maintaining the Python-side FlexSN kernel-handle registry;
* wrapping forward and backward as ``torch.library`` custom ops so the
  compiler does not trace Python kernel objects or Triton launcher internals;
* providing helpers for fake tensors, autograd setup/backward, and final-state
  fast paths.

Most users do not need to call these functions directly; they are primarily
used indirectly by
:class:`spikingjelly.activation_based.neuron.flexsn.FlexSN` and
:class:`spikingjelly.activation_based.neuron.flexsn.FlexSNKernel`.
"""

from __future__ import annotations

import contextlib
import weakref
from dataclasses import dataclass
from itertools import count
from threading import Lock
from typing import Dict, List, Optional

import torch

from .info import FlexSNInfo
from .wrapper import (
    flexsn_backward,
    flexsn_forward,
    flexsn_inference,
    flexsn_inference_final_state,
)

__all__ = [
    "FlexSNKernelHandle",
    "register_flexsn_kernel_handle",
    "retain_flexsn_kernel_handle",
    "retain_owner_flexsn_kernel_handle",
    "release_flexsn_kernel_handle",
    "release_active_flexsn_kernel_handle",
    "flexsn_inductor_inference",
    "flexsn_inductor_inference_final_state",
    "flexsn_inductor_training",
    "flexsn_inductor_training_final_state",
    "flexsn_inductor_backward",
    "attach_flexsn_handle_finalizer",
]


@dataclass
class FlexSNKernelHandle:
    """
    **API Language:**
    :ref:`中文 <FlexSNKernelHandle-cn>` | :ref:`English <FlexSNKernelHandle-en>`

    ----

    .. _FlexSNKernelHandle-cn:

    * **中文**

    保存 FlexSN kernel 句柄所关联的 Triton kernel、元数据和引用计数信息。
    该结构由 ``custom_ops`` 模块内部注册表维护，用于把 Python 侧 kernel 对象
    绑定到整数 ``handle`` 。

    ----

    .. _FlexSNKernelHandle-en:

    * **English**

    Store the Triton kernels, metadata, and reference-counting state associated
    with a FlexSN kernel handle. Instances are managed by the internal
    ``custom_ops`` registry and bind Python-side kernel objects to integer
    ``handle`` values.
    """
    inference_kernel: Optional[object]
    inference_info: Optional[FlexSNInfo]
    inference_final_state_kernel: Optional[object]
    inference_final_state_info: Optional[FlexSNInfo]
    forward_kernel: Optional[object]
    backward_kernel: Optional[object]
    training_info: Optional[FlexSNInfo]
    owner_refs: int = 1
    active_refs: int = 0


_KERNEL_REGISTRY: Dict[int, FlexSNKernelHandle] = {}
_KERNEL_REGISTRY_LOCK = Lock()
_KERNEL_ID_GEN = count(1)


def _normalize_kernel_handle(handle) -> int:
    if isinstance(handle, int):
        return handle
    if isinstance(handle, torch.Tensor):
        if handle.numel() != 1:
            raise TypeError(
                f"Unsupported FlexSN kernel handle tensor shape: {tuple(handle.shape)}"
            )
        return int(handle.item())
    try:
        return int(handle)
    except (TypeError, ValueError) as exc:
        raise TypeError(
            f"Unsupported FlexSN kernel handle type: {type(handle)!r}"
        ) from exc


def register_flexsn_kernel_handle(
    *,
    inference_kernel,
    inference_info,
    inference_final_state_kernel,
    inference_final_state_info,
    forward_kernel,
    backward_kernel,
    training_info,
) -> int:
    """
    **API Language:**
    :ref:`中文 <register_flexsn_kernel_handle-cn>` | :ref:`English <register_flexsn_kernel_handle-en>`

    ----

    .. _register_flexsn_kernel_handle-cn:

    * **中文**

    将一组 FlexSN Triton kernel 与对应元数据注册到 Python 侧注册表，并返回
    一个整数 ``handle`` 。该 ``handle`` 可在后续 custom op 调用中引用这组 kernel。

    :param inference_kernel: 推理 kernel
    :param inference_info: 推理路径对应的 :class:`FlexSNInfo`
    :param inference_final_state_kernel: 仅返回最终状态的推理 kernel
    :param inference_final_state_info: 最终状态推理路径对应的 :class:`FlexSNInfo`
    :param forward_kernel: 训练前向 kernel
    :param backward_kernel: 训练反向 kernel
    :param training_info: 训练路径对应的 :class:`FlexSNInfo`
    :return: 新注册的整数句柄
    :rtype: int

    ----

    .. _register_flexsn_kernel_handle-en:

    * **English**

    Register a bundle of FlexSN Triton kernels and their metadata in the
    Python-side registry and return an integer ``handle``. The returned handle
    can be referenced by later custom-op calls.

    :param inference_kernel: Inference kernel
    :param inference_info: :class:`FlexSNInfo` for the inference path
    :param inference_final_state_kernel: Inference kernel that returns final states only
    :param inference_final_state_info: :class:`FlexSNInfo` for the inference-final-state path
    :param forward_kernel: Training forward kernel
    :param backward_kernel: Training backward kernel
    :param training_info: :class:`FlexSNInfo` for the training path
    :return: Newly registered integer handle
    :rtype: int
    """
    with _KERNEL_REGISTRY_LOCK:
        handle = next(_KERNEL_ID_GEN)
        _KERNEL_REGISTRY[handle] = FlexSNKernelHandle(
            inference_kernel=inference_kernel,
            inference_info=inference_info,
            inference_final_state_kernel=inference_final_state_kernel,
            inference_final_state_info=inference_final_state_info,
            forward_kernel=forward_kernel,
            backward_kernel=backward_kernel,
            training_info=training_info,
        )
    return handle


def _cleanup_kernel_handle(bundle: FlexSNKernelHandle) -> None:
    for obj in (
        bundle.inference_kernel,
        bundle.inference_final_state_kernel,
        bundle.forward_kernel,
        bundle.backward_kernel,
    ):
        closer = getattr(obj, "close", None)
        if callable(closer):
            closer()


def _lookup_kernel_handle(handle: int) -> FlexSNKernelHandle:
    handle = _normalize_kernel_handle(handle)
    try:
        return _KERNEL_REGISTRY[handle]
    except KeyError as e:
        raise RuntimeError(f"Unknown FlexSN kernel handle: {handle}") from e


def retain_flexsn_kernel_handle(handle: int) -> None:
    """
    **API Language:**
    :ref:`中文 <retain_flexsn_kernel_handle-cn>` | :ref:`English <retain_flexsn_kernel_handle-en>`

    ----

    .. _retain_flexsn_kernel_handle-cn:

    * **中文**

    增加指定 FlexSN kernel handle 的活动引用计数。通常在 autograd context
    保存该 handle 时调用，确保相关 kernel 在 backward 完成前不会被清理。

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None

    ----

    .. _retain_flexsn_kernel_handle-en:

    * **English**

    Increase the active-reference count of the specified FlexSN kernel handle.
    This is typically used when an autograd context needs to keep the handle
    alive until backward finishes.

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None
    """
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _lookup_kernel_handle(handle)
        bundle.active_refs += 1


def retain_owner_flexsn_kernel_handle(handle: int) -> None:
    """
    **API Language:**
    :ref:`中文 <retain_owner_flexsn_kernel_handle-cn>` | :ref:`English <retain_owner_flexsn_kernel_handle-en>`

    ----

    .. _retain_owner_flexsn_kernel_handle-cn:

    * **中文**

    增加指定 FlexSN kernel handle 的所有者引用计数。通常在对象拷贝或新的拥有者
    接管该 handle 时调用。

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None

    ----

    .. _retain_owner_flexsn_kernel_handle-en:

    * **English**

    Increase the owner-reference count of the specified FlexSN kernel handle.
    This is typically used when an object copy or another owner takes over the
    handle.

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None
    """
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _lookup_kernel_handle(handle)
        bundle.owner_refs += 1


def release_flexsn_kernel_handle(handle: int) -> None:
    """
    **API Language:**
    :ref:`中文 <release_flexsn_kernel_handle-cn>` | :ref:`English <release_flexsn_kernel_handle-en>`

    ----

    .. _release_flexsn_kernel_handle-cn:

    * **中文**

    释放一个所有者引用。若所有者引用与活动引用都归零，则相关 kernel 会从注册表中
    删除并尝试执行清理。

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None

    ----

    .. _release_flexsn_kernel_handle-en:

    * **English**

    Release one owner reference. When both owner and active references reach
    zero, the associated kernels are removed from the registry and cleaned up.

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None
    """
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _KERNEL_REGISTRY.get(handle)
        if bundle is None:
            return
        bundle.owner_refs = max(0, bundle.owner_refs - 1)
        should_cleanup = bundle.owner_refs == 0 and bundle.active_refs == 0
        if should_cleanup:
            _KERNEL_REGISTRY.pop(handle, None)
    if should_cleanup:
        _cleanup_kernel_handle(bundle)


def release_active_flexsn_kernel_handle(handle: int) -> None:
    """
    **API Language:**
    :ref:`中文 <release_active_flexsn_kernel_handle-cn>` | :ref:`English <release_active_flexsn_kernel_handle-en>`

    ----

    .. _release_active_flexsn_kernel_handle-cn:

    * **中文**

    释放一个活动引用。若所有者引用与活动引用都归零，则相关 kernel 会从注册表中
    删除并尝试执行清理。

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None

    ----

    .. _release_active_flexsn_kernel_handle-en:

    * **English**

    Release one active reference. When both owner and active references reach
    zero, the associated kernels are removed from the registry and cleaned up.

    :param handle: FlexSN kernel handle
    :type handle: int
    :return: None
    :rtype: None
    """
    handle = _normalize_kernel_handle(handle)
    with _KERNEL_REGISTRY_LOCK:
        bundle = _KERNEL_REGISTRY.get(handle)
        if bundle is None:
            return
        bundle.active_refs = max(0, bundle.active_refs - 1)
        should_cleanup = bundle.owner_refs == 0 and bundle.active_refs == 0
        if should_cleanup:
            _KERNEL_REGISTRY.pop(handle, None)
    if should_cleanup:
        _cleanup_kernel_handle(bundle)


def _make_seq_outputs_like(
    info: FlexSNInfo, flat_args: List[torch.Tensor], n: int
) -> List[torch.Tensor]:
    if not flat_args:
        raise ValueError("Expected at least one FlexSN argument tensor.")
    # The underlying FlexSN Triton wrappers allocate all sequence outputs with
    # ``empty_like(flat_args[0])``. Keep fake tensors aligned with that runtime
    # contract so Dynamo/AOTAutograd sees the same shapes during tracing.
    seq_template = flat_args[0]
    return [seq_template.new_empty(seq_template.shape) for _ in range(n)]


def _template_spec(tensor: torch.Tensor):
    return tuple(tensor.shape), tensor.dtype, tensor.device


def _materialize_template(spec):
    shape, dtype, device = spec
    return torch.empty((), dtype=dtype, device=device).expand(shape)


def _grad_or_zeros(grad_out, index: int, spec):
    grad = grad_out[index] if index < len(grad_out) else None
    if grad is not None:
        return grad
    shape, dtype, device = spec
    return torch.zeros(shape, dtype=dtype, device=device)


def _make_state_templates_like(
    info: FlexSNInfo, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    if not flat_args:
        raise ValueError("Expected at least one input tensor for FlexSN fake op.")
    explicit_states = flat_args[info.num_inputs : info.num_inputs + info.num_states]
    if len(explicit_states) == info.num_states:
        return [state.new_empty(state.shape) for state in explicit_states]
    seq_template = flat_args[0]
    if seq_template.dim() == 0:
        state_shape = ()
    else:
        state_shape = tuple(seq_template.shape[1:])
    return [seq_template.new_empty(state_shape) for _ in range(info.num_states)]


def _device_guard(tensors: List[torch.Tensor]):
    for tensor in tensors:
        if isinstance(tensor, torch.Tensor) and tensor.is_cuda:
            return torch.cuda.device(tensor.device)
    return contextlib.nullcontext()


def _final_state_saved_return_indices(info: FlexSNInfo) -> List[int]:
    return [idx for idx in info.c2k_return_mapping if idx >= info.num_outputs]


def _materialize_zero_state_args(
    info: FlexSNInfo, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    if len(flat_args) == info.num_inputs:
        # In the common reset-before-forward path, FlexSN initial states are
        # zero tensors matching the per-step input shape. Materialize them
        # inside the opaque custom op so compile graphs do not need explicit
        # ``zeros_like`` nodes in front of every neuron layer.
        if info.num_inputs == 0:
            raise ValueError("FlexSN custom ops require at least one input sequence.")
        seq0 = flat_args[0]
        zero_states = [seq0.new_zeros(seq0.shape[1:]) for _ in range(info.num_states)]
        return [*flat_args, *zero_states]
    return flat_args


def _flexsn_inductor_inference_impl(
    bundle: FlexSNKernelHandle, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.inference_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_inference(
                bundle.inference_kernel,
                bundle.inference_info,
                *args,
            )
        )


def _flexsn_inductor_inference_final_state_impl(
    bundle: FlexSNKernelHandle, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.inference_final_state_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_inference_final_state(
                bundle.inference_final_state_kernel,
                bundle.inference_final_state_info,
                *args,
            )
        )


def _flexsn_inductor_training_impl(
    bundle: FlexSNKernelHandle, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    args = _materialize_zero_state_args(bundle.training_info, flat_args)
    args = [arg.contiguous() for arg in args]
    with _device_guard(args):
        return list(
            flexsn_forward(
                bundle.forward_kernel,
                bundle.training_info,
                *args,
            )
        )


def _flexsn_inductor_training_final_state_impl(
    bundle: FlexSNKernelHandle, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    info = bundle.training_info
    assert info is not None
    args = _materialize_zero_state_args(info, flat_args)
    full_returns = _flexsn_inductor_training_impl(bundle, args)
    visible_outputs = list(full_returns[: info.num_outputs])
    state_seqs = list(
        full_returns[info.num_outputs : info.num_outputs + info.num_states]
    )
    init_states = list(args[info.num_inputs : info.num_inputs + info.num_states])
    final_states = [
        (init_states[i] if state_seq.shape[0] == 0 else state_seq[-1]).clone()
        for i, state_seq in enumerate(state_seqs)
    ]
    extra_saved_tensors = [
        full_returns[i] for i in _final_state_saved_return_indices(info)
    ]
    return [*visible_outputs, *final_states, *extra_saved_tensors]


def _flexsn_inductor_backward_impl(
    bundle: FlexSNKernelHandle,
    grad_outputs: List[torch.Tensor],
    saved_tensors: List[torch.Tensor],
    input_templates: List[torch.Tensor],
) -> List[torch.Tensor]:
    grads = [grad.contiguous() for grad in grad_outputs]
    saved = [tensor.contiguous() for tensor in saved_tensors]
    templates = [tensor.contiguous() for tensor in input_templates]
    info = bundle.training_info
    assert info is not None
    seq_templates = templates[: info.num_inputs]
    state_templates = templates[info.num_inputs : info.num_inputs + info.num_states]
    with _device_guard([*grads, *saved, *templates]):
        return list(
            flexsn_backward(
                bundle.backward_kernel,
                info,
                *grads,
                *saved,
                input_templates=seq_templates,
                state_templates=state_templates,
            )
        )


@torch.library.custom_op("sj::flexsn_inductor_inference", mutates_args=())
def flexsn_inductor_inference(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    **API Language:**
    :ref:`中文 <flexsn_inductor_inference-cn>` | :ref:`English <flexsn_inductor_inference-en>`

    ----

    .. _flexsn_inductor_inference-cn:

    * **中文**

    执行 FlexSN 的推理 custom op，返回输出序列与状态序列。

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: 扁平化参数列表，顺序为 ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *state_seqs]``
    :rtype: List[torch.Tensor]

    ----

    .. _flexsn_inductor_inference-en:

    * **English**

    Execute the FlexSN inference custom op and return output sequences together
    with state sequences.

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: Flattened argument list in the order
        ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *state_seqs]``
    :rtype: List[torch.Tensor]
    """
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_kernel is None or bundle.inference_info is None:
        raise RuntimeError("FlexSN inference kernel is unavailable for this handle.")
    return _flexsn_inductor_inference_impl(bundle, flat_args)


@torch.library.custom_op("sj::flexsn_inductor_inference_final_state", mutates_args=())
def flexsn_inductor_inference_final_state(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    **API Language:**
    :ref:`中文 <flexsn_inductor_inference_final_state-cn>` | :ref:`English <flexsn_inductor_inference_final_state-en>`

    ----

    .. _flexsn_inductor_inference_final_state-cn:

    * **中文**

    执行仅返回最终状态的 FlexSN 推理 custom op，返回输出序列与最终状态张量。

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: 扁平化参数列表，顺序为 ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *final_states]``
    :rtype: List[torch.Tensor]

    ----

    .. _flexsn_inductor_inference_final_state-en:

    * **English**

    Execute the FlexSN inference custom op that returns final states only, and
    produce output sequences together with final state tensors.

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: Flattened argument list in the order
        ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *final_states]``
    :rtype: List[torch.Tensor]
    """
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.inference_final_state_kernel is None
        or bundle.inference_final_state_info is None
    ):
        raise RuntimeError(
            "FlexSN inference-final-state kernel is unavailable for this handle."
        )
    return _flexsn_inductor_inference_final_state_impl(bundle, flat_args)


@torch.library.register_fake("sj::flexsn_inductor_inference")
def _flexsn_inductor_inference_fake(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_info is None:
        raise RuntimeError("FlexSN inference metadata is unavailable for this handle.")
    return _make_seq_outputs_like(
        bundle.inference_info,
        flat_args,
        bundle.inference_info.num_outputs + bundle.inference_info.num_states,
    )


@torch.library.register_fake("sj::flexsn_inductor_inference_final_state")
def _flexsn_inductor_inference_final_state_fake(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.inference_final_state_info is None:
        raise RuntimeError(
            "FlexSN inference-final-state metadata is unavailable for this handle."
        )
    seq_outputs = _make_seq_outputs_like(
        bundle.inference_final_state_info,
        flat_args,
        bundle.inference_final_state_info.num_outputs,
    )
    final_states = _make_state_templates_like(
        bundle.inference_final_state_info, flat_args
    )
    return [*seq_outputs, *final_states]


@torch.library.custom_op("sj::flexsn_inductor_training", mutates_args=())
def flexsn_inductor_training(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    **API Language:**
    :ref:`中文 <flexsn_inductor_training-cn>` | :ref:`English <flexsn_inductor_training-en>`

    ----

    .. _flexsn_inductor_training-cn:

    * **中文**

    执行 FlexSN 的训练前向 custom op，返回可见输出与 backward 所需保存张量。

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: 扁平化参数列表，顺序为 ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: 前向可见输出与 backward 保存张量组成的列表
    :rtype: List[torch.Tensor]

    ----

    .. _flexsn_inductor_training-en:

    * **English**

    Execute the FlexSN training-forward custom op and return visible outputs
    together with the saved tensors required by backward.

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: Flattened argument list in the order
        ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: A list containing visible forward outputs and backward-saved tensors
    :rtype: List[torch.Tensor]
    """
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.forward_kernel is None
        or bundle.backward_kernel is None
        or bundle.training_info is None
    ):
        raise RuntimeError("FlexSN training kernels are unavailable for this handle.")
    return _flexsn_inductor_training_impl(bundle, flat_args)


@torch.library.custom_op("sj::flexsn_inductor_training_final_state", mutates_args=())
def flexsn_inductor_training_final_state(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    """
    **API Language:**
    :ref:`中文 <flexsn_inductor_training_final_state-cn>` | :ref:`English <flexsn_inductor_training_final_state-en>`

    ----

    .. _flexsn_inductor_training_final_state-cn:

    * **中文**

    执行 FlexSN 的训练前向 custom op，并仅物化最终状态而非完整状态序列。

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: 扁平化参数列表，顺序为 ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *final_states, *saved_tensors]``
    :rtype: List[torch.Tensor]

    ----

    .. _flexsn_inductor_training_final_state-en:

    * **English**

    Execute the FlexSN training-forward custom op while materializing final
    states only instead of full state sequences.

    :param handle: FlexSN kernel handle
    :type handle: int
    :param flat_args: Flattened argument list in the order
        ``[*input_seqs, *init_states]``
    :type flat_args: List[torch.Tensor]
    :return: ``[*output_seqs, *final_states, *saved_tensors]``
    :rtype: List[torch.Tensor]
    """
    bundle = _lookup_kernel_handle(handle)
    if (
        bundle.forward_kernel is None
        or bundle.backward_kernel is None
        or bundle.training_info is None
    ):
        raise RuntimeError("FlexSN training kernels are unavailable for this handle.")
    return _flexsn_inductor_training_final_state_impl(bundle, flat_args)


@torch.library.register_fake("sj::flexsn_inductor_training")
def _flexsn_inductor_training_fake(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    return _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        bundle.training_info.num_fwd_kernel_returns,
    )


@torch.library.register_fake("sj::flexsn_inductor_training_final_state")
def _flexsn_inductor_training_final_state_fake(
    handle: int, flat_args: List[torch.Tensor]
) -> List[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    seq_outputs = _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        bundle.training_info.num_outputs,
    )
    final_states = _make_state_templates_like(bundle.training_info, flat_args)
    extra_saved_tensors = _make_seq_outputs_like(
        bundle.training_info,
        flat_args,
        len(_final_state_saved_return_indices(bundle.training_info)),
    )
    return [*seq_outputs, *final_states, *extra_saved_tensors]


@torch.library.custom_op("sj::flexsn_inductor_backward", mutates_args=())
def flexsn_inductor_backward(
    handle: int,
    grad_outputs: List[torch.Tensor],
    saved_tensors: List[torch.Tensor],
    input_templates: List[torch.Tensor],
) -> List[torch.Tensor]:
    """
    **API Language:**
    :ref:`中文 <flexsn_inductor_backward-cn>` | :ref:`English <flexsn_inductor_backward-en>`

    ----

    .. _flexsn_inductor_backward-cn:

    * **中文**

    执行 FlexSN 的 backward custom op，返回输入序列与初始状态的梯度。

    :param handle: FlexSN kernel handle
    :type handle: int
    :param grad_outputs: 输出序列/状态序列对应的梯度列表
    :type grad_outputs: List[torch.Tensor]
    :param saved_tensors: 前向保存张量列表
    :type saved_tensors: List[torch.Tensor]
    :param input_templates: 输入与初始状态模板，用于分配梯度张量
    :type input_templates: List[torch.Tensor]
    :return: 输入序列与初始状态的梯度列表
    :rtype: List[torch.Tensor]

    ----

    .. _flexsn_inductor_backward-en:

    * **English**

    Execute the FlexSN backward custom op and return gradients for input
    sequences and initial states.

    :param handle: FlexSN kernel handle
    :type handle: int
    :param grad_outputs: Gradient list for output sequences and state sequences
    :type grad_outputs: List[torch.Tensor]
    :param saved_tensors: Saved tensors from the forward pass
    :type saved_tensors: List[torch.Tensor]
    :param input_templates: Input/state templates used to allocate gradient tensors
    :type input_templates: List[torch.Tensor]
    :return: Gradient list for input sequences and initial states
    :rtype: List[torch.Tensor]
    """
    bundle = _lookup_kernel_handle(handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")
    return _flexsn_inductor_backward_impl(
        bundle,
        grad_outputs,
        saved_tensors,
        input_templates,
    )


@torch.library.register_fake("sj::flexsn_inductor_backward")
def _flexsn_inductor_backward_fake(
    handle: int,
    grad_outputs: List[torch.Tensor],
    saved_tensors: List[torch.Tensor],
    input_templates: List[torch.Tensor],
) -> List[torch.Tensor]:
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    seq_grads = [
        input_templates[i].new_empty(input_templates[i].shape)
        for i in range(bundle.training_info.num_inputs)
    ]
    state_offset = bundle.training_info.num_inputs
    state_grads = [
        input_templates[state_offset + i].new_empty(
            input_templates[state_offset + i].shape
        )
        for i in range(bundle.training_info.num_states)
    ]
    return [*seq_grads, *state_grads]


def _flexsn_training_setup_context(ctx, inputs, output):
    handle = _normalize_kernel_handle(inputs[0])
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    retain_flexsn_kernel_handle(handle)
    ctx._active_ref_finalizer = weakref.finalize(
        ctx, release_active_flexsn_kernel_handle, handle
    )
    ctx.handle = handle
    ctx.input_template_specs = [_template_spec(t) for t in inputs[1]]
    ctx.output_template_specs = [
        _template_spec(t)
        for t in output[
            : bundle.training_info.num_outputs + bundle.training_info.num_states
        ]
    ]
    saved = [output[i] for i in bundle.training_info.c2k_return_mapping]
    ctx.save_for_backward(*saved)


def _flexsn_training_final_state_setup_context(ctx, inputs, output):
    handle = _normalize_kernel_handle(inputs[0])
    bundle = _lookup_kernel_handle(handle)
    if bundle.training_info is None:
        raise RuntimeError("FlexSN training metadata is unavailable for this handle.")
    retain_flexsn_kernel_handle(handle)
    ctx._active_ref_finalizer = weakref.finalize(
        ctx, release_active_flexsn_kernel_handle, handle
    )
    ctx.handle = handle
    ctx.input_template_specs = [_template_spec(t) for t in inputs[1]]
    ctx.output_template_specs = [
        _template_spec(t) for t in output[: bundle.training_info.num_outputs]
    ]
    if bundle.training_info.num_inputs == 0:
        raise RuntimeError("FlexSN training requires at least one input sequence.")
    seq_len = ctx.input_template_specs[0][0][0]
    state_start = bundle.training_info.num_outputs
    state_end = state_start + bundle.training_info.num_states
    ctx.state_seq_template_specs = [
        ((seq_len, *state_shape), state_dtype, state_device)
        for state_shape, state_dtype, state_device in (
            _template_spec(t) for t in output[state_start:state_end]
        )
    ]
    visible_outputs = bundle.training_info.num_outputs
    visible = bundle.training_info.num_outputs + bundle.training_info.num_states
    extra_saved = list(output[visible:])
    extra_saved_iter = iter(extra_saved)
    saved = []
    for idx in bundle.training_info.c2k_return_mapping:
        if idx < visible_outputs:
            saved.append(output[idx])
        else:
            saved.append(next(extra_saved_iter))
    ctx.save_for_backward(*saved)


def _flexsn_training_backward(ctx, grad_out: List[Optional[torch.Tensor]]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    required_grads = bundle.training_info.num_outputs + bundle.training_info.num_states
    grad_inputs = [
        _grad_or_zeros(grad_out, i, ctx.output_template_specs[i])
        for i in range(required_grads)
    ]
    # ctx.input_template_specs mirrors the full flat argument list passed into the
    # training custom op: input sequences first, then any explicit initial states.
    arg_templates = [_materialize_template(spec) for spec in ctx.input_template_specs]
    try:
        if ctx._active_ref_finalizer.alive:
            ctx._active_ref_finalizer.detach()
        grads = list(
            flexsn_inductor_backward(
                ctx.handle,
                grad_inputs,
                list(ctx.saved_tensors),
                arg_templates,
            )
        )
        if len(grads) != len(ctx.input_template_specs):
            grads = grads[: len(ctx.input_template_specs)]
    finally:
        release_active_flexsn_kernel_handle(ctx.handle)
    return None, grads


def _flexsn_training_final_state_backward(ctx, grad_out: List[Optional[torch.Tensor]]):
    bundle = _lookup_kernel_handle(ctx.handle)
    if bundle.backward_kernel is None or bundle.training_info is None:
        raise RuntimeError("FlexSN backward kernel is unavailable for this handle.")

    output_grads = []
    for i in range(bundle.training_info.num_outputs):
        output_grads.append(_grad_or_zeros(grad_out, i, ctx.output_template_specs[i]))

    state_grads = []
    for i in range(bundle.training_info.num_states):
        state_seq_shape, state_seq_dtype, state_seq_device = (
            ctx.state_seq_template_specs[i]
        )
        final_grad_index = bundle.training_info.num_outputs + i
        final_grad = (
            grad_out[final_grad_index] if final_grad_index < len(grad_out) else None
        )
        seq_grad = torch.zeros(
            state_seq_shape,
            dtype=state_seq_dtype,
            device=state_seq_device,
        )
        if final_grad is not None and state_seq_shape[0] > 0:
            seq_grad[-1].copy_(final_grad)
        state_grads.append(seq_grad)

    # ctx.input_template_specs already includes both input-sequence and initial-state
    # templates, and flexsn_inductor_backward splits them back apart internally.
    arg_templates = [_materialize_template(spec) for spec in ctx.input_template_specs]
    try:
        if ctx._active_ref_finalizer.alive:
            ctx._active_ref_finalizer.detach()
        grads = list(
            flexsn_inductor_backward(
                ctx.handle,
                [*output_grads, *state_grads],
                list(ctx.saved_tensors),
                arg_templates,
            )
        )
        if len(grads) != len(ctx.input_template_specs):
            grads = grads[: len(ctx.input_template_specs)]
        explicit_state_start = bundle.training_info.num_inputs
        for i in range(bundle.training_info.num_states):
            final_grad_index = bundle.training_info.num_outputs + i
            final_grad = (
                grad_out[final_grad_index] if final_grad_index < len(grad_out) else None
            )
            state_seq_shape = ctx.state_seq_template_specs[i][0]
            grad_index = explicit_state_start + i
            if (
                final_grad is not None
                and state_seq_shape[0] == 0
                and grad_index < len(grads)
            ):
                grads[grad_index] = grads[grad_index] + final_grad
    finally:
        release_active_flexsn_kernel_handle(ctx.handle)
    return None, grads


torch.library.register_autograd(
    "sj::flexsn_inductor_training",
    _flexsn_training_backward,
    setup_context=_flexsn_training_setup_context,
)

torch.library.register_autograd(
    "sj::flexsn_inductor_training_final_state",
    _flexsn_training_final_state_backward,
    setup_context=_flexsn_training_final_state_setup_context,
)


def attach_flexsn_handle_finalizer(owner, handle: int):
    """
    **API Language:**
    :ref:`中文 <attach_flexsn_handle_finalizer-cn>` | :ref:`English <attach_flexsn_handle_finalizer-en>`

    ----

    .. _attach_flexsn_handle_finalizer-cn:

    * **中文**

    为指定 ``owner`` 绑定一个 ``weakref.finalize`` ，在对象销毁时自动释放对应的
    FlexSN kernel handle。

    :param owner: 句柄所有者对象
    :param handle: FlexSN kernel handle
    :type handle: int
    :return: 绑定好的 finalizer
    :rtype: weakref.finalize

    ----

    .. _attach_flexsn_handle_finalizer-en:

    * **English**

    Attach a ``weakref.finalize`` object to ``owner`` so the corresponding
    FlexSN kernel handle is released automatically when the owner is destroyed.

    :param owner: Owner object of the handle
    :param handle: FlexSN kernel handle
    :type handle: int
    :return: Attached finalizer
    :rtype: weakref.finalize
    """
    return weakref.finalize(owner, release_flexsn_kernel_handle, handle)
