import logging
import math
import threading

import torch

try:
    import cupy
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.cuda_kernel.neuron_kernel: {e}")
    cupy = None

from ... import surrogate
from ..cuda_utils import register_python_object
from .helpers import sg_registry_key as _sg_registry_key

__all__ = [
    "_decode_v_reset",
    "_sg_obj_id",
    "_CapturedAutogradCtx",
    "_stash_capture_ctx",
    "_take_capture_ctx",
    "_resolve_sg_cuda_code_fun",
    "save_cuda_codes",
    "multistep_if_ptt",
    "multistep_lif_ptt",
    "multistep_plif_ptt",
    "multistep_qif_ptt",
    "multistep_izhikevich_ptt",
    "multistep_eif_ptt",
]


def _decode_v_reset(v_reset_value: float):
    """Decode the v_reset parameter from a float value.

    In custom CUDA kernel calls, NaN is used as a sentinel for soft reset
    (equivalent to ``v_reset=None`` in Python).

    :param v_reset_value: The raw v_reset value (may be NaN)
    :type v_reset_value: float
    :return: ``None`` if NaN (soft reset), otherwise the original value
    :rtype: Optional[float]
    """
    return None if math.isnan(v_reset_value) else v_reset_value


def _sg_obj_id(sg) -> int:
    """Register a surrogate gradient function object and return its unique ID.

    The returned ID is used to look up the surrogate function during CUDA kernel
    code generation at runtime.

    :param sg: The surrogate gradient function object
    :type sg: surrogate.SurrogateFunctionBase
    :return: A unique integer ID for the surrogate function
    :rtype: int
    """
    return register_python_object(sg, _sg_registry_key(sg))


class _CapturedAutogradCtx:
    """A minimal autograd context for capturing saved tensors in CUDA kernels.

    This is used internally by the PTT (Python Truncated Taylor) CUDA kernel
    path to store tensors that need to be passed to the backward pass.

    .. admonition:: Note
        :class: note

        Unlike torch's autograd ``Function`` context, this class does not
        implement any gradient computation logic — it is purely a storage
        container.
    """
    saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


_CAPTURE_CTX_LOCK = threading.Lock()
_CAPTURE_CTX_BY_ID: dict[int, _CapturedAutogradCtx] = {}
_CAPTURE_CTX_NEXT_ID = 0


def _stash_capture_ctx(captured_ctx: _CapturedAutogradCtx) -> int:
    """Store a captured autograd context and return its lookup ID.

    The context is stored in a thread-safe global dictionary and can be
    retrieved later via :func:`_take_capture_ctx`.

    :param captured_ctx: The captured autograd context to store
    :type captured_ctx: _CapturedAutogradCtx
    :return: A unique integer ID for later retrieval
    :rtype: int
    """
    global _CAPTURE_CTX_NEXT_ID
    with _CAPTURE_CTX_LOCK:
        _CAPTURE_CTX_NEXT_ID += 1
        capture_id = _CAPTURE_CTX_NEXT_ID
        _CAPTURE_CTX_BY_ID[capture_id] = captured_ctx
    return capture_id


def _should_stash_capture_ctx(inputs) -> bool:
    for item in inputs:
        if isinstance(item, torch.Tensor) and item.requires_grad:
            return True
    return False


def _take_capture_ctx(capture_id: int) -> _CapturedAutogradCtx:
    """Retrieve and remove a previously stored autograd context by its ID.

    :param capture_id: The ID returned by :func:`_stash_capture_ctx`
    :type capture_id: int
    :return: The stored autograd context
    :rtype: _CapturedAutogradCtx
    :raises RuntimeError: If no context is found for the given ID
    """
    with _CAPTURE_CTX_LOCK:
        captured_ctx = _CAPTURE_CTX_BY_ID.pop(capture_id, None)
    if captured_ctx is None:
        raise RuntimeError(f"Unknown capture context id={capture_id}")
    return captured_ctx


def _resolve_sg_cuda_code_fun(sg):
    """Resolve the ``cuda_code`` function from a surrogate gradient object.

    :param sg: The surrogate gradient function object
    :type sg: surrogate.SurrogateFunctionBase
    :return: The ``cuda_code`` callable of the surrogate function
    :rtype: Callable
    :raises RuntimeError: If the surrogate function does not implement ``cuda_code``
    """
    sg_cuda_code_fun = getattr(sg, "cuda_code", None)
    if sg_cuda_code_fun is None:
        raise RuntimeError(
            "The surrogate_function must have cuda_code function!"
            "Please check the implementation of surrogate function."
        )
    return sg_cuda_code_fun


def save_cuda_codes(cu_file_path: str = "./neuron_kernel_sample.cu"):
    """Save generated CUDA kernel source text for neuron kernels to a ``.cu`` file.
    **API Language:**
    :ref:`中文 <save_cuda_codes-cn>` | :ref:`English <save_cuda_codes-en>`

    ----

    .. _save_cuda_codes-cn:

    * **中文**

    TODO: add Chinese description

    :param cu_file_path: 输出的 CUDA 文件路径
    :type cu_file_path: str
    :param cu_file_path: Output CUDA file path
    :type cu_file_path: str
    :return: None
    :rtype: None

    ----

    .. _save_cuda_codes-en:

    * **English**

    TODO: add English description

    :param cu_file_path: 输出的 CUDA 文件路径
    :param cu_file_path: Output CUDA file path
    :type cu_file_path: str
    :type cu_file_path: str
    :return: None
    :rtype: None
    """
    from . import eif, integrate_and_fire, izhikevich, lif, plif, qif

    kernels = [
        (
            "MultiStepIFNodePTT",
            integrate_and_fire.create_fptt_kernel,
            integrate_and_fire.create_bptt_kernel,
        ),
        ("MultiStepLIFNodePTT", lif.create_fptt_kernel, lif.create_bptt_kernel),
        (
            "MultiStepParametricLIFNodePTT",
            plif.create_fptt_kernel,
            plif.create_bptt_kernel,
        ),
        ("MultiStepQIFNodePTT", qif.create_fptt_kernel, qif.create_bptt_kernel),
        (
            "MultiStepIzhikevichNodePTT",
            izhikevich.create_fptt_kernel,
            izhikevich.create_bptt_kernel,
        ),
        ("MultiStepEIFNodePTT", eif.create_fptt_kernel, eif.create_bptt_kernel),
    ]

    with open(cu_file_path, "w+") as cu_file:
        cu_file.write(
            "// This file is created by spikingjelly.activation_based.cuda_kernel.neuron_kernel.save_cuda_codes.\n"
        )
        cu_file.write(
            "// Note that codes in this file will not be executed This file is just created for reading.\n"
        )
        for name, create_fptt_kernel, create_bptt_kernel in kernels:
            cu_file.write("\n// " + name + "\n")
            for sg in surrogate._has_cuda_:
                for hard_reset in [True, False]:
                    for dtype in ["fp32", "fp16"]:
                        # IF/QIF/EIF/Izh fptt signatures: (hard_reset, dtype)
                        # LIF/PLIF signatures include decay_input.
                        if name in (
                            "MultiStepLIFNodePTT",
                            "MultiStepParametricLIFNodePTT",
                        ):
                            for decay_input in [True, False]:
                                cu_file.write(
                                    f"\n// {name} fptt {sg.__name__}, decay_input={decay_input}, hard_reset={hard_reset}, dtype={dtype}\n"
                                )
                                fp_codes = create_fptt_kernel(
                                    decay_input, hard_reset, dtype
                                ).code
                                cu_file.write(fp_codes)
                                for detach_reset in [True, False]:
                                    cu_file.write(
                                        f"\n// {name} bptt {sg.__name__}, decay_input={decay_input}, hard_reset={hard_reset}, dtype={dtype}, detach_reset={detach_reset}\n"
                                    )
                                    bp_codes = create_bptt_kernel(
                                        sg().cuda_code,
                                        decay_input,
                                        hard_reset,
                                        detach_reset,
                                        dtype,
                                    ).code
                                    cu_file.write(bp_codes)
                        else:
                            cu_file.write(
                                f"\n// {name} fptt {sg.__name__}, hard_reset={hard_reset}, dtype={dtype}\n"
                            )
                            fp_codes = create_fptt_kernel(hard_reset, dtype).code
                            cu_file.write(fp_codes)
                            for detach_reset in [True, False]:
                                cu_file.write(
                                    f"\n// {name} bptt {sg.__name__}, hard_reset={hard_reset}, dtype={dtype}, detach_reset={detach_reset}\n"
                                )
                                bp_codes = create_bptt_kernel(
                                    sg().cuda_code,
                                    hard_reset,
                                    detach_reset,
                                    dtype,
                                ).code
                                cu_file.write(bp_codes)


def multistep_if_ptt(*args, **kwargs):
    """Multi-step IF neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_if_ptt-cn>` | :ref:`English <multistep_if_ptt-en>`

    ----

    .. _multistep_if_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_if_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .integrate_and_fire import multistep_if_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_lif_ptt(*args, **kwargs):
    """Multi-step LIF neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_lif_ptt-cn>` | :ref:`English <multistep_lif_ptt-en>`

    ----

    .. _multistep_lif_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_lif_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .lif import multistep_lif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_plif_ptt(*args, **kwargs):
    """Multi-step Parametric LIF neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_plif_ptt-cn>` | :ref:`English <multistep_plif_ptt-en>`

    ----

    .. _multistep_plif_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_plif_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .plif import multistep_plif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_qif_ptt(*args, **kwargs):
    """Multi-step QIF neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_qif_ptt-cn>` | :ref:`English <multistep_qif_ptt-en>`

    ----

    .. _multistep_qif_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_qif_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .qif import multistep_qif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_izhikevich_ptt(*args, **kwargs):
    """Multi-step Izhikevich neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_izhikevich_ptt-cn>` | :ref:`English <multistep_izhikevich_ptt-en>`

    ----

    .. _multistep_izhikevich_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_izhikevich_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .izhikevich import multistep_izhikevich_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_eif_ptt(*args, **kwargs):
    """Multi-step EIF neuron forward/backward via PTT CUDA kernel.
    **API Language:**
    :ref:`中文 <multistep_eif_ptt-cn>` | :ref:`English <multistep_eif_ptt-en>`

    ----

    .. _multistep_eif_ptt-cn:

    * **中文**

    TODO: add Chinese description

    .. admonition:: Note
    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
        This is a re-exported wrapper. See
        for the full documentation.

    ----

    .. _multistep_eif_ptt-en:

    * **English**

    TODO: add English description

    :return: Forward spike and backward gradient tensors
    :rtype: Tuple[torch.Tensor, ...]
    """
    from .eif import multistep_eif_ptt as _impl

    return _impl(*args, **kwargs)
