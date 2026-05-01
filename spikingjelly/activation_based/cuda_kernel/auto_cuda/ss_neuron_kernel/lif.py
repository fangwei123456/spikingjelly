from typing import Callable, Optional

import torch

from ...cuda_utils import (
    register_python_object,
    resolve_python_object,
    use_cupy_custom_op,
)
from .integrate_and_fire import _replay_and_grad
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


class LIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        v: torch.Tensor,
        v_th: float,
        v_reset: Optional[float],
        decay: float,
        forward_kernel: LIFNodeFPKernel,
        backward_kernel: LIFNodeBPKernel,
    ):
        py_dict = {
            "x": x,
            "v": v,
            "v_th": v_th,
            "v_reset": v_reset,
            "decay": decay,
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        forward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        NeuronATGFBase.ctx_save(
            ctx,
            requires_grad,
            py_dict["h"],
            blocks=blocks,
            threads=threads,
            numel=py_dict["numel"],
            v_th=py_dict["v_th"],
            v_reset=py_dict["v_reset"],
            backward_kernel=backward_kernel,
            decay=py_dict["decay"],
        )

        return py_dict["spike"], py_dict["v_next"]

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
            ctx, grad_spike, grad_v_next
        )
        py_dict["decay"] = ctx.decay

        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        backward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None, None


if use_cupy_custom_op() and cupy is not None:

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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_kernel = resolve_python_object(forward_kernel_id)
        backward_kernel = resolve_python_object(backward_kernel_id)
        with torch.no_grad():
            return LIFNodeATGF.apply(
                x,
                v,
                v_th,
                None if soft_reset else v_reset,
                decay,
                forward_kernel,
                backward_kernel,
            )


    @torch.library.register_fake("sj::cupy_ss_lif_forward")
    def _cupy_ss_lif_forward_fake(
        x, v, v_th, v_reset, soft_reset, decay, forward_kernel_id, backward_kernel_id
    ):
        return x.new_empty(x.shape), x.new_empty(x.shape)


    def _setup_ss_lif_ctx(ctx, inputs, output):
        ctx.inputs = inputs


    def _ss_lif_bw(ctx, grad_spike, grad_v_next):
        (
            x,
            v,
            v_th,
            v_reset,
            soft_reset,
            decay,
            forward_kernel_id,
            backward_kernel_id,
        ) = ctx.inputs
        forward_kernel = resolve_python_object(forward_kernel_id)
        backward_kernel = resolve_python_object(backward_kernel_id)
        grads = _replay_and_grad(
            LIFNodeATGF.apply,
            (x, v),
            (
                v_th,
                None if soft_reset else v_reset,
                decay,
                forward_kernel,
                backward_kernel,
            ),
            (grad_spike, grad_v_next),
        )
        return grads[0], grads[1], None, None, None, None, None, None


    torch.library.register_autograd(
        "sj::cupy_ss_lif_forward",
        _ss_lif_bw,
        setup_context=_setup_ss_lif_ctx,
    )


def ss_lif_step(x, v, v_th, v_reset, decay, forward_kernel, backward_kernel):
    if use_cupy_custom_op() and cupy is not None:
        fk = register_python_object(forward_kernel, repr(forward_kernel))
        bk = register_python_object(backward_kernel, repr(backward_kernel))
        vr = float("nan") if v_reset is None else float(v_reset)
        return cupy_ss_lif_forward(x, v, v_th, vr, v_reset is None, decay, fk, bk)
    return LIFNodeATGF.apply(
        x, v, v_th, v_reset, decay, forward_kernel, backward_kernel
    )
