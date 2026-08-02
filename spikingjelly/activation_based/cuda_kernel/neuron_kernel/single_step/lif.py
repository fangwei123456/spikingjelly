from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F

from ..... import configure
from .... import surrogate
from ...cuda_utils import (
    DeviceEnvironment,
    cal_blocks,
    register_python_object,
    resolve_python_object,
)
from ..surrogate_registry import _cuda_codes, _resolve_cuda_code_id
from .base import (
    NeuronBPKernel,
    NeuronFPKernel,
    cfunction,
    cupy,
    _prepare_backward,
    _prepare_forward,
)


class LIFNodeFPKernel(NeuronFPKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f"const {dtype} &", cname="decay")

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(
                z=f"{self.dtype} LIFNodeFPKernel_temp_var",
                x="v[index]",
                y="v_reset",
                dtype=self.dtype,
            )
        else:
            codes = f"{self.dtype} LIFNodeFPKernel_temp_var = v[index];"

        if self.decay_input:
            codes += cfunction.sub(
                z="LIFNodeFPKernel_temp_var",
                x="x[index]",
                y="LIFNodeFPKernel_temp_var",
                dtype=self.dtype,
            )
            codes += cfunction.mul(
                z="LIFNodeFPKernel_temp_var",
                x="decay",
                y="LIFNodeFPKernel_temp_var",
                dtype=self.dtype,
            )
        else:
            codes += cfunction.mul(
                z="LIFNodeFPKernel_temp_var",
                x="decay",
                y="LIFNodeFPKernel_temp_var",
                dtype=self.dtype,
            )
            codes += cfunction.sub(
                z="LIFNodeFPKernel_temp_var",
                x="x[index]",
                y="LIFNodeFPKernel_temp_var",
                dtype=self.dtype,
            )

        codes += cfunction.add(
            z="h[index]", x="LIFNodeFPKernel_temp_var", y="v[index]", dtype=self.dtype
        )

        return codes


class LIFNodeBPKernel(NeuronBPKernel):
    def __init__(
        self,
        decay_input: bool,
        surrogate_cuda_codes: str,
        hard_reset: bool,
        detach_reset: bool,
        dtype: str,
    ):
        super().__init__(surrogate_cuda_codes, hard_reset, detach_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f"const {dtype} &", cname="decay")

    def grad_h_to_v(self) -> str:
        return cfunction.sub(
            z=f"const {self.dtype} grad_h_to_v",
            x=cfunction.constant(None, x=1.0, dtype=self.dtype),
            y="decay",
            dtype=self.dtype,
        )

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(
                y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
            )
        else:
            return f"const {self.dtype} grad_h_to_x = decay;"


_LIF_FWD_KERNEL_CACHE = {}
_LIF_BWD_KERNEL_CACHE = {}


def _get_lif_forward_kernel(
    *, device: int, decay_input: bool, hard_reset: bool, dtype: str
) -> LIFNodeFPKernel:
    key = (device, decay_input, hard_reset, dtype)
    kernel = _LIF_FWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = LIFNodeFPKernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
        _LIF_FWD_KERNEL_CACHE[key] = kernel
    return kernel


def _get_lif_backward_kernel(
    *,
    device: int,
    decay_input: bool,
    sg_cupy_id: int,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> LIFNodeBPKernel:
    key = (
        device,
        decay_input,
        sg_cupy_id,
        hard_reset,
        detach_reset,
        dtype,
    )
    kernel = _LIF_BWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = LIFNodeBPKernel(
            decay_input=decay_input,
            surrogate_cuda_codes=_cuda_codes(sg_cupy_id, dtype),
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
        _LIF_BWD_KERNEL_CACHE[key] = kernel
    return kernel


@torch.library.custom_op("sj::cupy_single_step_lif_forward", mutates_args=())
def cupy_single_step_lif_forward(
    x: torch.Tensor,
    v: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    decay: float,
    forward_kernel_id: int,
    backward_kernel_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_kernel = resolve_python_object(forward_kernel_id)
    py_dict = {
        "x": x,
        "v": v,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
        "decay": decay,
    }
    blocks, threads, py_dict = _prepare_forward(py_dict)
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    return py_dict["spike"], py_dict["v_next"], py_dict["h"]


@torch.library.register_fake("sj::cupy_single_step_lif_forward")
def _cupy_single_step_lif_forward_fake(
    x, v, v_th, v_reset, soft_reset, decay, forward_kernel_id, backward_kernel_id
):
    return x.new_empty(x.shape), x.new_empty(x.shape), x.new_empty(x.shape)


def _setup_single_step_lif_context(ctx, inputs, output):
    x, _, v_th, v_reset, soft_reset, decay, _, backward_kernel_id = inputs
    h = output[2]
    ctx.save_for_backward(h)
    ctx.backward_kernel = resolve_python_object(backward_kernel_id)
    ctx.blocks = cal_blocks(
        (x.numel() + 1) // 2 if x.dtype == torch.float16 else x.numel()
    )
    ctx.threads = configure.cuda_threads
    with DeviceEnvironment(x.get_device()):
        numel = x.numel()
        if x.dtype == torch.float16:
            numel = (numel + 1) // 2
        ctx.numel = cupy.asarray(numel, dtype=np.int32)
        if x.dtype == torch.float32:
            ctx.v_th = cupy.asarray(v_th, dtype=cupy.float32)
            ctx.v_reset = (
                None if soft_reset else cupy.asarray(v_reset, dtype=cupy.float32)
            )
            ctx.decay = cupy.asarray(decay, dtype=cupy.float32)
        elif x.dtype == torch.float16:
            ctx.v_th = cupy.asarray([v_th, v_th], dtype=cupy.float16)
            ctx.v_reset = (
                None
                if soft_reset
                else cupy.asarray([v_reset, v_reset], dtype=cupy.float16)
            )
            ctx.decay = cupy.asarray([decay, decay], dtype=cupy.float16)
        else:
            raise NotImplementedError(x.dtype)


def _single_step_lif_backward(ctx, grad_spike, grad_v_next, _grad_h):
    backward_kernel, blocks, threads, py_dict = _prepare_backward(
        ctx, grad_spike, grad_v_next
    )
    py_dict["decay"] = ctx.decay
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_single_step_lif_forward",
    _single_step_lif_backward,
    setup_context=_setup_single_step_lif_context,
)


def lif_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_th: float,
    v_reset: Optional[float],
    decay: float,
    decay_input: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    dtype = "float" if x.dtype == torch.float32 else "half2"
    if x.dtype not in (torch.float32, torch.float16):
        raise NotImplementedError(x.dtype)
    device = x.get_device()
    hard_reset = v_reset is not None
    forward_kernel = _get_lif_forward_kernel(
        device=device,
        decay_input=decay_input,
        hard_reset=hard_reset,
        dtype=dtype,
    )
    backward_kernel = _get_lif_backward_kernel(
        device=device,
        decay_input=decay_input,
        sg_cupy_id=_resolve_cuda_code_id(surrogate_function, dtype),
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    )
    need_unpad = x.dtype == torch.float16 and x.numel() % 2 != 0
    if need_unpad:
        x = F.pad(x, (0, 1))
        v = F.pad(v, (0, 1))
    fk = register_python_object(forward_kernel)
    bk = register_python_object(backward_kernel)
    vr = float("nan") if v_reset is None else float(v_reset)
    spike, v_next, _ = cupy_single_step_lif_forward(
        x, v, v_th, vr, v_reset is None, decay, fk, bk
    )
    if need_unpad:
        spike = spike[..., :-1]
        v_next = v_next[..., :-1]
    return spike, v_next
