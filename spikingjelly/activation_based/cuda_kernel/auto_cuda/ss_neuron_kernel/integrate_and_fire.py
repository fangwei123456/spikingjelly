import numpy as np
import threading
import weakref

import torch
import torch.nn.functional as F

from ..... import configure
from ...cuda_utils import (
    DeviceEnvironment,
    cal_blocks,
    python_object_registry_key,
    register_python_object,
    resolve_python_object,
)
from .ss_neuron_kernel_base import (
    NeuronATGFBase,
    NeuronBPKernel,
    NeuronFPKernel,
    cfunction,
    cupy,
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


_KERNEL_OBJ_ID_LOCK = threading.Lock()
_KERNEL_OBJ_ID_CACHE: "weakref.WeakKeyDictionary[object, int]" = (
    weakref.WeakKeyDictionary()
)


def _cached_kernel_obj_id(kernel_obj) -> int:
    with _KERNEL_OBJ_ID_LOCK:
        obj_id = _KERNEL_OBJ_ID_CACHE.get(kernel_obj)
        if obj_id is not None:
            return obj_id
        obj_id = register_python_object(
            kernel_obj, python_object_registry_key(kernel_obj)
        )
        _KERNEL_OBJ_ID_CACHE[kernel_obj] = obj_id
        return obj_id


@torch.library.custom_op("sj::cupy_ss_if_forward", mutates_args=())
def cupy_ss_if_forward(
    x: torch.Tensor,
    v: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    forward_kernel_id: int,
    backward_kernel_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    forward_kernel = resolve_python_object(forward_kernel_id)
    py_dict = {
        "x": x,
        "v": v,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
    }
    _, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    return py_dict["spike"], py_dict["v_next"], py_dict["h"]


@torch.library.register_fake("sj::cupy_ss_if_forward")
def _cupy_ss_if_forward_fake(
    x, v, v_th, v_reset, soft_reset, forward_kernel_id, backward_kernel_id
):
    return x.new_empty(x.shape), x.new_empty(x.shape), x.new_empty(x.shape)


def _setup_ss_if_ctx(ctx, inputs, output):
    x, _, v_th, v_reset, soft_reset, _, backward_kernel_id = inputs
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
        elif x.dtype == torch.float16:
            ctx.v_th = cupy.asarray([v_th, v_th], dtype=cupy.float16)
            ctx.v_reset = (
                None
                if soft_reset
                else cupy.asarray([v_reset, v_reset], dtype=cupy.float16)
            )
        else:
            raise NotImplementedError(x.dtype)


def _ss_if_bw(ctx, grad_spike, grad_v_next):
    backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
        ctx, grad_spike, grad_v_next
    )
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_ss_if_forward",
    _ss_if_bw,
    setup_context=_setup_ss_if_ctx,
)


def ss_if_step(x, v, v_th, v_reset, forward_kernel, backward_kernel):
    need_unpad = x.dtype == torch.float16 and x.numel() % 2 != 0
    if need_unpad:
        x = F.pad(x, (0, 1))
        v = F.pad(v, (0, 1))
    fk = _cached_kernel_obj_id(forward_kernel)
    bk = _cached_kernel_obj_id(backward_kernel)
    vr = float("nan") if v_reset is None else float(v_reset)
    spike, v_next, _ = cupy_ss_if_forward(x, v, v_th, vr, v_reset is None, fk, bk)
    if need_unpad:
        spike = spike[..., :-1]
        v_next = v_next[..., :-1]
    return spike, v_next
