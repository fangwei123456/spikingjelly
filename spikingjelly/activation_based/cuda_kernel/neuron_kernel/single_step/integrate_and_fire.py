from functools import lru_cache
from typing import Optional

import numpy as np

import torch
import torch.nn.functional as F

from ..... import configure
from .... import surrogate
from ...cuda_utils import DeviceEnvironment, cal_blocks
from ..surrogate_code import _decode_cuda_code, _surrogate_cuda_code
from .base import (
    NeuronBPKernel,
    NeuronFPKernel,
    cfunction,
    cupy,
    _prepare_backward,
    _prepare_forward,
)


class IFNodeFPKernel(NeuronFPKernel):
    def neuronal_charge(self) -> str:
        return cfunction.add(z="h[index]", x="x[index]", y="v[index]", dtype=self.dtype)


class IFNodeBPKernel(NeuronBPKernel):
    def grad_h_to_v(self) -> str:
        return cfunction.constant(
            y=f"const {self.dtype} grad_h_to_v", x=1.0, dtype=self.dtype
        )

    def grad_h_to_x(self) -> str:
        return cfunction.constant(
            y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
        )


@lru_cache(maxsize=128)
def _get_if_forward_kernel(*, hard_reset: bool, dtype: str) -> IFNodeFPKernel:
    return IFNodeFPKernel(hard_reset=hard_reset, dtype=dtype)


@lru_cache(maxsize=128)
def _get_if_backward_kernel(
    *,
    sg_cupy_code: str,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> IFNodeBPKernel:
    return IFNodeBPKernel(
        surrogate_cuda_codes=_decode_cuda_code(sg_cupy_code),
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    )


@torch.library.custom_op("sj::cupy_single_step_if_forward", mutates_args=())
def cupy_single_step_if_forward(
    x: torch.Tensor,
    v: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_cupy_code: str,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    dtype = "float" if x.dtype == torch.float32 else "half2"
    forward_kernel = _get_if_forward_kernel(hard_reset=not soft_reset, dtype=dtype)
    py_dict = {
        "x": x,
        "v": v,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
    }
    blocks, threads, py_dict = _prepare_forward(py_dict)
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    return py_dict["spike"], py_dict["v_next"], py_dict["h"]


@torch.library.register_fake("sj::cupy_single_step_if_forward")
def _cupy_single_step_if_forward_fake(
    x, v, v_th, v_reset, soft_reset, detach_reset, sg_cupy_code
):
    return x.new_empty(x.shape), x.new_empty(x.shape), x.new_empty(x.shape)


def _setup_single_step_if_context(ctx, inputs, output):
    x, _, v_th, v_reset, soft_reset, detach_reset, sg_cupy_code = inputs
    h = output[2]
    ctx.save_for_backward(h)
    dtype = "float" if x.dtype == torch.float32 else "half2"
    ctx.backward_kernel = _get_if_backward_kernel(
        sg_cupy_code=sg_cupy_code,
        hard_reset=not soft_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    )
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
        elif x.dtype == torch.float16:
            ctx.v_th = cupy.asarray([v_th, v_th], dtype=cupy.float16)
            ctx.v_reset = (
                None
                if soft_reset
                else cupy.asarray([v_reset, v_reset], dtype=cupy.float16)
            )
        else:
            raise NotImplementedError(x.dtype)


def _single_step_if_backward(ctx, grad_spike, grad_v_next, _grad_h):
    backward_kernel, blocks, threads, py_dict = _prepare_backward(
        ctx, grad_spike, grad_v_next
    )
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_single_step_if_forward",
    _single_step_if_backward,
    setup_context=_setup_single_step_if_context,
)


def if_step(
    x: torch.Tensor,
    v: torch.Tensor,
    v_th: float,
    v_reset: Optional[float],
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    if x.dtype not in (torch.float32, torch.float16):
        raise NotImplementedError(x.dtype)
    if not x.is_cuda:
        raise RuntimeError("if_step requires a CUDA tensor.")
    dtype = "float" if x.dtype == torch.float32 else "half2"
    sg_cupy_code = _surrogate_cuda_code(surrogate_function, dtype)
    need_unpad = x.dtype == torch.float16 and x.numel() % 2 != 0
    if need_unpad:
        x = F.pad(x, (0, 1))
        v = F.pad(v, (0, 1))
    vr = float("nan") if v_reset is None else float(v_reset)
    spike, v_next, _ = cupy_single_step_if_forward(
        x, v, v_th, vr, v_reset is None, detach_reset, sg_cupy_code
    )
    if need_unpad:
        spike = spike[..., :-1]
        v_next = v_next[..., :-1]
    return spike, v_next
