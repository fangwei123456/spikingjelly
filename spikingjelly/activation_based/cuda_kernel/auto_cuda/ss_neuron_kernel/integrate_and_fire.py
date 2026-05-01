from typing import Optional

import torch

from ...cuda_utils import (
    register_python_object,
    resolve_python_object,
    use_cupy_custom_op,
)
from .common import replay_and_grad
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


class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x: torch.Tensor,
        v: torch.Tensor,
        v_th: float,
        v_reset: Optional[float],
        forward_kernel: IFNodeFPKernel,
        backward_kernel: IFNodeBPKernel,
    ):
        py_dict = {"x": x, "v": v, "v_th": v_th, "v_reset": v_reset}
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
        )

        return py_dict["spike"], py_dict["v_next"]

    @staticmethod
    def backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
            ctx, grad_spike, grad_v_next
        )
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        backward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        return py_dict["grad_x"], py_dict["grad_v"], None, None, None, None


if use_cupy_custom_op() and cupy is not None:

    @torch.library.custom_op("sj::cupy_ss_if_forward", mutates_args=())
    def cupy_ss_if_forward(
        x: torch.Tensor,
        v: torch.Tensor,
        v_th: float,
        v_reset: float,
        soft_reset: bool,
        forward_kernel_id: int,
        backward_kernel_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        forward_kernel = resolve_python_object(forward_kernel_id)
        backward_kernel = resolve_python_object(backward_kernel_id)
        with torch.no_grad():
            return IFNodeATGF.apply(
                x,
                v,
                v_th,
                None if soft_reset else v_reset,
                forward_kernel,
                backward_kernel,
            )

    @torch.library.register_fake("sj::cupy_ss_if_forward")
    def _cupy_ss_if_forward_fake(
        x, v, v_th, v_reset, soft_reset, forward_kernel_id, backward_kernel_id
    ):
        return x.new_empty(x.shape), x.new_empty(x.shape)

    def _setup_ss_if_ctx(ctx, inputs, output):
        ctx.inputs = inputs
        _, _, _, _, _, forward_kernel_id, backward_kernel_id = inputs
        # Pin kernels until backward finishes to avoid weak-ref eviction.
        ctx.forward_kernel = resolve_python_object(forward_kernel_id)
        ctx.backward_kernel = resolve_python_object(backward_kernel_id)

    def _ss_if_bw(ctx, grad_spike, grad_v_next):
        x, v, v_th, v_reset, soft_reset, _forward_kernel_id, _backward_kernel_id = (
            ctx.inputs
        )
        forward_kernel = ctx.forward_kernel
        backward_kernel = ctx.backward_kernel
        grads = replay_and_grad(
            IFNodeATGF.apply,
            (x, v),
            (
                v_th,
                None if soft_reset else v_reset,
                forward_kernel,
                backward_kernel,
            ),
            (grad_spike, grad_v_next),
        )
        return grads[0], grads[1], None, None, None, None, None

    torch.library.register_autograd(
        "sj::cupy_ss_if_forward",
        _ss_if_bw,
        setup_context=_setup_ss_if_ctx,
    )


def ss_if_step(x, v, v_th, v_reset, forward_kernel, backward_kernel):
    if use_cupy_custom_op() and cupy is not None:
        fk = register_python_object(forward_kernel, repr(forward_kernel))
        bk = register_python_object(backward_kernel, repr(backward_kernel))
        vr = float("nan") if v_reset is None else float(v_reset)
        return cupy_ss_if_forward(x, v, v_th, vr, v_reset is None, fk, bk)
    return IFNodeATGF.apply(x, v, v_th, v_reset, forward_kernel, backward_kernel)
