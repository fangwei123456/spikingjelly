import logging
import math
import threading

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
    # NaN is used as a sentinel for soft reset (v_reset=None) in custom-op calls.
    return None if math.isnan(v_reset_value) else v_reset_value


def _sg_obj_id(sg) -> int:
    return register_python_object(sg, _sg_registry_key(sg))


class _CapturedAutogradCtx:
    saved_tensors = ()

    def save_for_backward(self, *tensors):
        self.saved_tensors = tensors


_CAPTURE_CTX_LOCK = threading.Lock()
_CAPTURE_CTX_BY_ID: dict[int, _CapturedAutogradCtx] = {}
_CAPTURE_CTX_NEXT_ID = 0


def _stash_capture_ctx(captured_ctx: _CapturedAutogradCtx) -> int:
    global _CAPTURE_CTX_NEXT_ID
    with _CAPTURE_CTX_LOCK:
        _CAPTURE_CTX_NEXT_ID += 1
        capture_id = _CAPTURE_CTX_NEXT_ID
        _CAPTURE_CTX_BY_ID[capture_id] = captured_ctx
    return capture_id


def _take_capture_ctx(capture_id: int) -> _CapturedAutogradCtx:
    with _CAPTURE_CTX_LOCK:
        captured_ctx = _CAPTURE_CTX_BY_ID.pop(capture_id, None)
    if captured_ctx is None:
        raise RuntimeError(f"Unknown capture context id={capture_id}")
    return captured_ctx


def _resolve_sg_cuda_code_fun(sg):
    sg_cuda_code_fun = getattr(sg, "cuda_code", None)
    if sg_cuda_code_fun is None:
        raise RuntimeError(
            "The surrogate_function must have cuda_code function!"
            "Please check the implementation of surrogate function."
        )
    return sg_cuda_code_fun


def save_cuda_codes(cu_file_path: str = "./neuron_kernel_sample.cu"):
    """
    Save generated CUDA kernel source text for neuron kernels to a ``.cu`` file.
    """
    from . import eif, integrate_and_fire, izhikevich, lif, plif, qif

    kernels = [
        ("MultiStepIFNodePTT", integrate_and_fire.create_fptt_kernel, integrate_and_fire.create_bptt_kernel),
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
            "// This file is created by spikingjelly.activation_based.neuron_kernel.save_cuda_codes.\n"
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
                        if name in ("MultiStepLIFNodePTT", "MultiStepParametricLIFNodePTT"):
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
    from .integrate_and_fire import multistep_if_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_lif_ptt(*args, **kwargs):
    from .lif import multistep_lif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_plif_ptt(*args, **kwargs):
    from .plif import multistep_plif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_qif_ptt(*args, **kwargs):
    from .qif import multistep_qif_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_izhikevich_ptt(*args, **kwargs):
    from .izhikevich import multistep_izhikevich_ptt as _impl

    return _impl(*args, **kwargs)


def multistep_eif_ptt(*args, **kwargs):
    from .eif import multistep_eif_ptt as _impl

    return _impl(*args, **kwargs)
