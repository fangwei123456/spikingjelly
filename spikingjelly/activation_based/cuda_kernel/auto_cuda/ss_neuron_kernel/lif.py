from typing import Callable

import torch

from ...cuda_utils import (
    DeviceEnvironment,
    cal_blocks,
    python_object_registry_key,
    register_python_object,
    resolve_python_object,
)
from ..... import configure
from .ss_neuron_kernel_base import (
    NeuronATGFBase,
    NeuronBPKernel,
    NeuronFPKernel,
    cfunction,
    cupy,
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
        surrogate_function: Callable,
        hard_reset: bool,
        detach_reset: bool,
        dtype: str,
    ):
        super().__init__(surrogate_function, hard_reset, detach_reset, dtype)
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



@torch.library.custom_op("sj::cupy_ss_lif_forward", mutates_args=())
def cupy_ss_lif_forward(
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
    _, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    return py_dict["spike"], py_dict["v_next"], py_dict["h"]


@torch.library.register_fake("sj::cupy_ss_lif_forward")
def _cupy_ss_lif_forward_fake(
    x, v, v_th, v_reset, soft_reset, decay, forward_kernel_id, backward_kernel_id
):
    return x.new_empty(x.shape), x.new_empty(x.shape), x.new_empty(x.shape)


def _setup_ss_lif_ctx(ctx, inputs, output):
    x, _, v_th, v_reset, soft_reset, decay, _, backward_kernel_id = inputs
    h = output[2]
    ctx.save_for_backward(h)
    ctx.backward_kernel = resolve_python_object(backward_kernel_id)
    ctx.blocks = cal_blocks((x.numel() + 1) // 2 if x.dtype == torch.float16 else x.numel())
    ctx.threads = configure.cuda_threads
    with DeviceEnvironment(x.get_device()):
        numel = x.numel()
        if x.dtype == torch.float16:
            numel = (numel + 1) // 2
        ctx.numel = cupy.asarray(numel)
        if x.dtype == torch.float32:
            ctx.v_th = cupy.asarray(v_th, dtype=cupy.float32)
            ctx.v_reset = (
                None
                if soft_reset
                else cupy.asarray(v_reset, dtype=cupy.float32)
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


def _ss_lif_bw(ctx, grad_spike, grad_v_next):
    backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
        ctx, grad_spike, grad_v_next
    )
    py_dict["decay"] = ctx.decay
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_ss_lif_forward",
    _ss_lif_bw,
    setup_context=_setup_ss_lif_ctx,
)

def ss_lif_step(x, v, v_th, v_reset, decay, forward_kernel, backward_kernel):
    fk = register_python_object(forward_kernel, python_object_registry_key(forward_kernel))
    bk = register_python_object(backward_kernel, python_object_registry_key(backward_kernel))
    vr = float("nan") if v_reset is None else float(v_reset)
    spike, v_next, _ = cupy_ss_lif_forward(x, v, v_th, vr, v_reset is None, decay, fk, bk)
    return spike, v_next
