from typing import Optional

import torch

from .common import (
    cfunction,
    cupy,
    cuda_utils,
    configure,
    math,
    surrogate,
    _CUPY_CUSTOM_OP_AVAILABLE,
    _dtype_to_cupy_kernel_dtype,
    _surrogate_cuda_codes_from_id,
    _use_cupy_custom_op,
    resolve_sg_cupy_id_and_key,
    NeuronATGFBase,
    NeuronBPTTKernel,
    NeuronFPTTKernel,
)


class IFNodeFPTTKernel(NeuronFPTTKernel):
    def neuronal_charge(self) -> str:
        return cfunction.add(
            z="h_seq[t]", x="x_seq[t]", y="v_v_seq[t]", dtype=self.dtype
        )


class IFNodeBPTTKernel(NeuronBPTTKernel):
    def grad_h_next_to_v(self) -> str:
        return cfunction.constant(
            y=f"const {self.dtype} grad_h_next_to_v", x=1.0, dtype=self.dtype
        )

    def grad_h_to_x(self) -> str:
        return cfunction.constant(
            y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
        )


class IFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: Optional[float],
        forward_kernel: IFNodeFPTTKernel,
        backward_kernel: IFNodeBPTTKernel,
    ):
        py_dict = {"x_seq": x_seq, "v_init": v_init, "v_th": v_th, "v_reset": v_reset}
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        forward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        NeuronATGFBase.ctx_save(
            ctx,
            requires_grad,
            py_dict["h_seq"],
            blocks=blocks,
            threads=threads,
            numel=py_dict["numel"],
            N=py_dict["N"],
            v_th=py_dict["v_th"],
            v_reset=py_dict["v_reset"],
            backward_kernel=backward_kernel,
        )

        return py_dict["spike_seq"], py_dict["v_v_seq"][1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
            ctx, grad_spike_seq, grad_v_seq
        )

        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        backward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        return py_dict["grad_x_seq"], py_dict["grad_v_init"], None, None, None, None


_IF_FWD_KERNEL_CACHE = {}
_IF_BWD_KERNEL_CACHE = {}


def _get_if_forward_kernel(*, hard_reset: bool, dtype: str) -> IFNodeFPTTKernel:
    key = (hard_reset, dtype)
    kernel = _IF_FWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = IFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)
        _IF_FWD_KERNEL_CACHE[key] = kernel
    return kernel


def _get_if_backward_kernel(
    *,
    sg_cupy_id: int,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> IFNodeBPTTKernel:
    key = (sg_cupy_id, hard_reset, detach_reset, dtype)
    kernel = _IF_BWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = IFNodeBPTTKernel(
            surrogate_function=_surrogate_cuda_codes_from_id(sg_cupy_id),
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
        _IF_BWD_KERNEL_CACHE[key] = kernel
    return kernel


def _legacy_multistep_if(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_th: float,
    v_reset: Optional[float],
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool,
    forward_kernel: Optional[IFNodeFPTTKernel],
    backward_kernel: Optional[IFNodeBPTTKernel],
):
    hard_reset = v_reset is not None
    dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    if forward_kernel is None:
        forward_kernel = _get_if_forward_kernel(hard_reset=hard_reset, dtype=dtype)
    if backward_kernel is None:
        if not hasattr(surrogate_function, "cuda_codes"):
            raise TypeError(
                "surrogate_function for CuPy legacy path must provide cuda_codes."
            )
        backward_kernel = IFNodeBPTTKernel(
            surrogate_function=surrogate_function.cuda_codes,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
    return IFNodeATGF.apply(
        x_seq,
        v_init,
        v_th,
        v_reset,
        forward_kernel,
        backward_kernel,
    )


if _CUPY_CUSTOM_OP_AVAILABLE:

    @torch.library.custom_op("sj::cupy_multistep_if_forward", mutates_args=())
    def cupy_multistep_if_forward(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: float,
        soft_reset: bool,
        detach_reset: bool,
        sg_cupy_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_seq = x_seq.contiguous()
        v_init = v_init.contiguous()
        dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
        hard_reset = not soft_reset
        forward_kernel = _get_if_forward_kernel(hard_reset=hard_reset, dtype=dtype)

        py_dict = {
            "x_seq": x_seq,
            "v_init": v_init,
            "v_th": v_th,
            "v_reset": None if soft_reset else v_reset,
        }
        _, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        forward_kernel((blocks,), (threads,), py_dict)
        return py_dict["spike_seq"], py_dict["v_v_seq"][1:,], py_dict["h_seq"]


    @torch.library.register_fake("sj::cupy_multistep_if_forward")
    def _cupy_multistep_if_forward_fake(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: float,
        soft_reset: bool,
        detach_reset: bool,
        sg_cupy_id: int,
    ):
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
        )


    def _setup_cupy_multistep_if_context(ctx, inputs, output):
        _, _, v_th, v_reset, soft_reset, detach_reset, sg_cupy_id = inputs
        h_seq = output[2]
        ctx.save_for_backward(h_seq)
        ctx.v_th = v_th
        ctx.v_reset = None if soft_reset else v_reset
        ctx.detach_reset = detach_reset
        ctx.sg_cupy_id = sg_cupy_id


    def _cupy_multistep_if_backward(ctx, grad_spike_seq, grad_v_seq, grad_h_seq):
        del grad_h_seq
        (h_seq,) = ctx.saved_tensors
        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()
        h_seq = h_seq.contiguous()
        dtype = _dtype_to_cupy_kernel_dtype(grad_spike_seq.dtype)
        hard_reset = ctx.v_reset is not None
        backward_kernel = _get_if_backward_kernel(
            sg_cupy_id=ctx.sg_cupy_id,
            hard_reset=hard_reset,
            detach_reset=ctx.detach_reset,
            dtype=dtype,
        )

        numel = grad_spike_seq.numel()
        N = grad_spike_seq.shape[1]
        if grad_spike_seq.dtype == torch.float16:
            N = math.ceil(N / 2)
            numel = N * grad_spike_seq.shape[0]
        blocks = cuda_utils.cal_blocks(N)
        threads = configure.cuda_threads
        with cuda_utils.DeviceEnvironment(grad_spike_seq.get_device()):
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)
        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(
            zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype
        )
        py_dict = {
            "numel": numel,
            "N": N,
            "grad_spike_seq": grad_spike_seq,
            "grad_v_seq": grad_v_seq,
            "h_seq": h_seq,
            "grad_x_seq": zero_data[0:-1],
            "grad_v_init": zero_data[-1],
            "v_th": ctx.v_th,
            "v_reset": ctx.v_reset,
        }
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        backward_kernel((blocks,), (threads,), py_dict)
        return py_dict["grad_x_seq"], py_dict["grad_v_init"], None, None, None, None, None


    torch.library.register_autograd(
        "sj::cupy_multistep_if_forward",
        _cupy_multistep_if_backward,
        setup_context=_setup_cupy_multistep_if_context,
    )


def multistep_if(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    forward_kernel: Optional[IFNodeFPTTKernel] = None,
    backward_kernel: Optional[IFNodeBPTTKernel] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    if _use_cupy_custom_op():
        try:
            sg_cupy_id, _ = resolve_sg_cupy_id_and_key(surrogate_function)
            soft_reset = v_reset is None
            v_reset_value = 0.0 if v_reset is None else float(v_reset)
            s_seq, v_seq, _ = cupy_multistep_if_forward(
                x_seq,
                v_init,
                v_threshold,
                v_reset_value,
                soft_reset,
                detach_reset,
                sg_cupy_id,
            )
            return s_seq, v_seq
        except Exception:
            pass

    return _legacy_multistep_if(
        x_seq=x_seq,
        v_init=v_init,
        v_th=v_threshold,
        v_reset=v_reset,
        surrogate_function=surrogate_function,
        detach_reset=detach_reset,
        forward_kernel=forward_kernel,
        backward_kernel=backward_kernel,
    )
