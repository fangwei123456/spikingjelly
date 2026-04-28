from typing import Optional, Callable

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


class LIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f"const {dtype} &", cname="decay")

    def neuronal_charge(self) -> str:
        if self.hard_reset:
            codes = cfunction.sub(
                z=f"{self.dtype} LIFNodeFPTTKernel_temp_var",
                x="v_v_seq[t]",
                y="v_reset",
                dtype=self.dtype,
            )
        else:
            codes = f"{self.dtype} LIFNodeFPTTKernel_temp_var = v_v_seq[t];"

        if self.decay_input:
            codes += cfunction.sub(
                z="LIFNodeFPTTKernel_temp_var",
                x="x_seq[t]",
                y="LIFNodeFPTTKernel_temp_var",
                dtype=self.dtype,
            )
            codes += cfunction.mul(
                z="LIFNodeFPTTKernel_temp_var",
                x="decay",
                y="LIFNodeFPTTKernel_temp_var",
                dtype=self.dtype,
            )
        else:
            codes += cfunction.mul(
                z="LIFNodeFPTTKernel_temp_var",
                x="decay",
                y="LIFNodeFPTTKernel_temp_var",
                dtype=self.dtype,
            )
            codes += cfunction.sub(
                z="LIFNodeFPTTKernel_temp_var",
                x="x_seq[t]",
                y="LIFNodeFPTTKernel_temp_var",
                dtype=self.dtype,
            )

        codes += cfunction.add(
            z="h_seq[t]",
            x="LIFNodeFPTTKernel_temp_var",
            y="v_v_seq[t]",
            dtype=self.dtype,
        )

        return codes


class LIFNodeBPTTKernel(NeuronBPTTKernel):
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

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(
            z=f"const {self.dtype} grad_h_next_to_v",
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
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: Optional[float],
        decay: float,
        forward_kernel: LIFNodeFPTTKernel,
        backward_kernel: LIFNodeBPTTKernel,
    ):
        py_dict = {
            "x_seq": x_seq,
            "v_init": v_init,
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
            py_dict["h_seq"],
            blocks=blocks,
            threads=threads,
            numel=py_dict["numel"],
            N=py_dict["N"],
            v_th=py_dict["v_th"],
            v_reset=py_dict["v_reset"],
            backward_kernel=backward_kernel,
            decay=py_dict["decay"],
        )

        return py_dict["spike_seq"], py_dict["v_v_seq"][1:,]

    @staticmethod
    def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(
            ctx, grad_spike_seq, grad_v_seq
        )
        py_dict["decay"] = ctx.decay

        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")

        backward_kernel((blocks,), (threads,), py_dict)

        if "v_reset" not in py_dict:
            py_dict["v_reset"] = None

        return (
            py_dict["grad_x_seq"],
            py_dict["grad_v_init"],
            None,
            None,
            None,
            None,
            None,
        )


_LIF_FWD_KERNEL_CACHE = {}
_LIF_BWD_KERNEL_CACHE = {}


def _get_lif_forward_kernel(
    *, decay_input: bool, hard_reset: bool, dtype: str
) -> LIFNodeFPTTKernel:
    key = (decay_input, hard_reset, dtype)
    kernel = _LIF_FWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = LIFNodeFPTTKernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
        _LIF_FWD_KERNEL_CACHE[key] = kernel
    return kernel


def _get_lif_backward_kernel(
    *,
    decay_input: bool,
    sg_cupy_id: int,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> LIFNodeBPTTKernel:
    key = (decay_input, sg_cupy_id, hard_reset, detach_reset, dtype)
    kernel = _LIF_BWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = LIFNodeBPTTKernel(
            decay_input=decay_input,
            surrogate_function=_surrogate_cuda_codes_from_id(sg_cupy_id),
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
        _LIF_BWD_KERNEL_CACHE[key] = kernel
    return kernel


def _legacy_multistep_lif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_th: float,
    v_reset: Optional[float],
    decay: float,
    decay_input: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool,
    forward_kernel: Optional[LIFNodeFPTTKernel],
    backward_kernel: Optional[LIFNodeBPTTKernel],
):
    hard_reset = v_reset is not None
    dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    if forward_kernel is None:
        forward_kernel = _get_lif_forward_kernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
    if backward_kernel is None:
        if not hasattr(surrogate_function, "cuda_codes"):
            raise TypeError(
                "surrogate_function for CuPy legacy path must provide cuda_codes."
            )
        backward_kernel = LIFNodeBPTTKernel(
            decay_input=decay_input,
            surrogate_function=surrogate_function.cuda_codes,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
    return LIFNodeATGF.apply(
        x_seq,
        v_init,
        v_th,
        v_reset,
        decay,
        forward_kernel,
        backward_kernel,
    )


if _CUPY_CUSTOM_OP_AVAILABLE:

    @torch.library.custom_op("sj::cupy_multistep_lif_forward", mutates_args=())
    def cupy_multistep_lif_forward(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: float,
        soft_reset: bool,
        detach_reset: bool,
        decay: float,
        decay_input: bool,
        sg_cupy_id: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        x_seq = x_seq.contiguous()
        v_init = v_init.contiguous()
        dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
        hard_reset = not soft_reset
        forward_kernel = _get_lif_forward_kernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
        py_dict = {
            "x_seq": x_seq,
            "v_init": v_init,
            "v_th": v_th,
            "v_reset": None if soft_reset else v_reset,
            "decay": decay,
        }
        _, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        forward_kernel((blocks,), (threads,), py_dict)
        return py_dict["spike_seq"], py_dict["v_v_seq"][1:,], py_dict["h_seq"]


    @torch.library.register_fake("sj::cupy_multistep_lif_forward")
    def _cupy_multistep_lif_forward_fake(
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: float,
        soft_reset: bool,
        detach_reset: bool,
        decay: float,
        decay_input: bool,
        sg_cupy_id: int,
    ):
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
        )


    def _setup_cupy_multistep_lif_context(ctx, inputs, output):
        _, _, v_th, v_reset, soft_reset, detach_reset, decay, decay_input, sg_cupy_id = inputs
        h_seq = output[2]
        ctx.save_for_backward(h_seq)
        ctx.v_th = v_th
        ctx.v_reset = None if soft_reset else v_reset
        ctx.detach_reset = detach_reset
        ctx.decay = decay
        ctx.decay_input = decay_input
        ctx.sg_cupy_id = sg_cupy_id


    def _cupy_multistep_lif_backward(ctx, grad_spike_seq, grad_v_seq, grad_h_seq):
        del grad_h_seq
        (h_seq,) = ctx.saved_tensors
        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()
        h_seq = h_seq.contiguous()
        dtype = _dtype_to_cupy_kernel_dtype(grad_spike_seq.dtype)
        hard_reset = ctx.v_reset is not None
        backward_kernel = _get_lif_backward_kernel(
            decay_input=ctx.decay_input,
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
            "decay": ctx.decay,
        }
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        backward_kernel((blocks,), (threads,), py_dict)
        return py_dict["grad_x_seq"], py_dict["grad_v_init"], None, None, None, None, None, None, None


    torch.library.register_autograd(
        "sj::cupy_multistep_lif_forward",
        _cupy_multistep_lif_backward,
        setup_context=_setup_cupy_multistep_lif_context,
    )


def multistep_lif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay_input: bool,
    tau: float,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    forward_kernel: Optional[LIFNodeFPTTKernel] = None,
    backward_kernel: Optional[LIFNodeBPTTKernel] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    decay = 1.0 / tau
    if _use_cupy_custom_op():
        try:
            sg_cupy_id, _ = resolve_sg_cupy_id_and_key(surrogate_function)
        except TypeError:
            return _legacy_multistep_lif(
                x_seq=x_seq,
                v_init=v_init,
                v_th=v_threshold,
                v_reset=v_reset,
                decay=decay,
                decay_input=decay_input,
                surrogate_function=surrogate_function,
                detach_reset=detach_reset,
                forward_kernel=forward_kernel,
                backward_kernel=backward_kernel,
            )

        soft_reset = v_reset is None
        v_reset_value = 0.0 if v_reset is None else float(v_reset)
        s_seq, v_seq, _ = cupy_multistep_lif_forward(
            x_seq,
            v_init,
            v_threshold,
            v_reset_value,
            soft_reset,
            detach_reset,
            decay,
            decay_input,
            sg_cupy_id,
        )
        return s_seq, v_seq

    return _legacy_multistep_lif(
        x_seq=x_seq,
        v_init=v_init,
        v_th=v_threshold,
        v_reset=v_reset,
        decay=decay,
        decay_input=decay_input,
        surrogate_function=surrogate_function,
        detach_reset=detach_reset,
        forward_kernel=forward_kernel,
        backward_kernel=backward_kernel,
    )
