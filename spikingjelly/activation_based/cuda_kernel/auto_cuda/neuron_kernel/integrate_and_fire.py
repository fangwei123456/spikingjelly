from typing import Optional

import torch
import torch.nn.functional as F

from .common import (
    cfunction,
    cupy,
    cuda_utils,
    configure,
    math,
    surrogate,
    _dtype_to_cupy_kernel_dtype,
    scalar_to_cupy,
    prepare_forward_meta,
    _surrogate_cuda_codes_from_id,
    resolve_sg_cupy_id_and_key,
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
    orig_n = x_seq.shape[1]
    need_unpad = False
    if x_seq.dtype == torch.float16 and orig_n % 2 != 0:
        # Keep legacy behavior: pad odd neuron count for half2 kernels.
        x_seq = F.pad(x_seq, (0, 1))
        v_init = F.pad(v_init, (0, 1))
        need_unpad = True
    dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    hard_reset = not soft_reset
    forward_kernel = _get_if_forward_kernel(hard_reset=hard_reset, dtype=dtype)

    py_dict = {
        "x_seq": x_seq,
        "v_init": v_init,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
    }
    blocks, threads, py_dict = prepare_forward_meta(py_dict)
    py_dict["spike_seq"] = torch.empty_like(x_seq)
    py_dict["h_seq"] = torch.empty_like(x_seq)
    py_dict["v_v_seq"] = x_seq.new_empty((x_seq.shape[0] + 1, *x_seq.shape[1:]))
    py_dict["v_v_seq"][0].copy_(py_dict.pop("v_init"))
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    spike_seq = py_dict["spike_seq"]
    v_seq = py_dict["v_v_seq"][1:,]
    h_seq = py_dict["h_seq"]
    if need_unpad:
        spike_seq = spike_seq[:, :orig_n]
        v_seq = v_seq[:, :orig_n]
        h_seq = h_seq[:, :orig_n]
    return spike_seq, v_seq, h_seq


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


@torch.library.custom_op("sj::cupy_multistep_if_backward", mutates_args=())
def cupy_multistep_if_backward(
    grad_spike_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_cupy_id: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    grad_spike_seq = grad_spike_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    dtype = _dtype_to_cupy_kernel_dtype(grad_spike_seq.dtype)
    hard_reset = not soft_reset
    backward_kernel = _get_if_backward_kernel(
        sg_cupy_id=sg_cupy_id,
        hard_reset=hard_reset,
        detach_reset=detach_reset,
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
    grad_x_seq = torch.empty_like(grad_spike_seq)
    grad_v_init = torch.empty_like(grad_v_seq[0])
    py_dict = {
        "numel": numel,
        "N": N,
        "grad_spike_seq": grad_spike_seq,
        "grad_v_seq": grad_v_seq,
        "h_seq": h_seq,
        "grad_x_seq": grad_x_seq,
        "grad_v_init": grad_v_init,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
    }
    scalar_to_cupy(py_dict, ref="grad_spike_seq")
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x_seq"], py_dict["grad_v_init"]


@torch.library.register_fake("sj::cupy_multistep_if_backward")
def _cupy_multistep_if_backward_fake(
    grad_spike_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    sg_cupy_id: int,
):
    return torch.empty_like(grad_spike_seq), torch.empty_like(grad_v_seq[0])


def _cupy_multistep_if_backward_autograd(ctx, grad_spike_seq, grad_v_seq, grad_h_seq):
    del grad_h_seq
    (h_seq,) = ctx.saved_tensors
    soft_reset = ctx.v_reset is None
    v_reset = 0.0 if soft_reset else float(ctx.v_reset)
    grad_x_seq, grad_v_init = cupy_multistep_if_backward(
        grad_spike_seq,
        grad_v_seq,
        h_seq,
        ctx.v_th,
        v_reset,
        soft_reset,
        ctx.detach_reset,
        ctx.sg_cupy_id,
    )
    return grad_x_seq, grad_v_init, None, None, None, None, None


torch.library.register_autograd(
    "sj::cupy_multistep_if_forward",
    _cupy_multistep_if_backward_autograd,
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
    del forward_kernel, backward_kernel
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
