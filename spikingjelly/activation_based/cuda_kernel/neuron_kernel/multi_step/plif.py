import math
from functools import cache
from typing import Callable, Optional

import torch

from ..... import configure
from .... import surrogate
from ... import cuda_utils
from ...auto_cuda import base as auto_cuda_base, cfunction
from .base import (
    _dtype_to_cupy_kernel_dtype,
    cupy,
    prepare_forward_meta,
    NeuronBPTTKernel,
    NeuronFPTTKernel,
)
from ..surrogate_registry import (
    _cuda_codes_callable,
    _resolve_cuda_code_id,
)


class ParametricLIFNodeFPTTKernel(NeuronFPTTKernel):
    def __init__(self, decay_input: bool, hard_reset: bool, dtype: str):
        super().__init__(hard_reset, dtype)
        self.decay_input = decay_input
        self.add_param(ctype=f"const {dtype} *", cname="decay")

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
                x="decay[0]",
                y="LIFNodeFPTTKernel_temp_var",
                dtype=self.dtype,
            )
        else:
            codes += cfunction.mul(
                z="LIFNodeFPTTKernel_temp_var",
                x="decay[0]",
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


class ParametricLIFNodeBPTTKernel(NeuronBPTTKernel):
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
        self.add_param(ctype=f"const {dtype} *", cname="decay")
        self.add_param(ctype="float *", cname="grad_decay")
        self.add_param(ctype=f"const {dtype} *", cname="v_v_seq")

    def grad_h_next_to_v(self) -> str:
        return cfunction.sub(
            z=f"const {self.dtype} grad_h_next_to_v",
            x=cfunction.constant(None, x=1.0, dtype=self.dtype),
            y="decay[0]",
            dtype=self.dtype,
        )

    def grad_h_to_x(self) -> str:
        if not self.decay_input:
            return cfunction.constant(
                y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
            )
        else:
            return f"const {self.dtype} grad_h_to_x = decay[0];"

    @property
    def head(self):
        codes = """
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
        """
        codes += rf"""
            __shared__ float sdata[{configure.cuda_threads}];
        """
        codes += """
            if (index < N)
            {
                const int dt = N;
        """

        codes += self.pre_core

        if self.reverse:
            codes += """
                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            """
        else:
            codes += """
                for(int t = index; t < numel; t += dt)
                {
            """
        return codes

    @property
    def pre_core(self):
        codes = auto_cuda_base.CodeTyper(16)
        codes.append("sdata[threadIdx.x] = 0.0f;")
        return super().pre_core + "\n" + codes.codes

    @property
    def core(self):
        core_codes = auto_cuda_base.CodeTyper(18)
        with auto_cuda_base.CodeBlock(core_codes):
            if self.decay_input:
                core_codes.append(
                    cfunction.sub(
                        z=f"{self.dtype} temp_var",
                        x="h_seq[t]",
                        y="v_v_seq[t]",
                        dtype=self.dtype,
                    )
                )
                core_codes.append(
                    cfunction.mul(
                        z="temp_var", x="temp_var", y="grad_h", dtype=self.dtype
                    )
                )
                core_codes.append(
                    cfunction.div(
                        z="temp_var", x="temp_var", y="decay[0]", dtype=self.dtype
                    )
                )

            else:
                if self.hard_reset:
                    core_codes.append(
                        cfunction.sub(
                            z=f"{self.dtype} temp_var",
                            x="v_reset",
                            y="v_v_seq[t]",
                            dtype=self.dtype,
                        )
                    )
                    core_codes.append(
                        cfunction.mul(
                            z="temp_var", x="temp_var", y="grad_h", dtype=self.dtype
                        )
                    )
                else:
                    core_codes.append(
                        cfunction.mul(
                            z=f"{self.dtype} temp_var",
                            x="grad_h",
                            y="v_v_seq[t]",
                            dtype=self.dtype,
                        )
                    )
                    core_codes.append(
                        cfunction.neg(y="temp_var", x="temp_var", dtype=self.dtype)
                    )

            if self.dtype == "float":
                core_codes.append("sdata[threadIdx.x] += temp_var;")
            elif self.dtype == "half2":
                core_codes.append(
                    "sdata[threadIdx.x] += __half2float(__hadd(__low2half(temp_var), __high2half(temp_var)));"
                )
            else:
                raise NotImplementedError(self.dtype)

        return super().core + "\n" + core_codes.codes

    @property
    def tail(self):
        codes = """
                }
        """

        codes += self.post_core

        codes += """
            }
            else
            {
                sdata[threadIdx.x] = 0.0f;
            }
            int threadx = blockDim.x;
            #pragma unroll
            for (int stride = threadx >> 1; stride > 0; stride = stride >> 1)
            {
            // Synchronize all thread before next loop
            __syncthreads();
            if (threadIdx.x < stride)
            {
                sdata[threadIdx.x] += sdata[threadIdx.x + stride];
            }
            }
            __syncthreads();
            if (threadIdx.x == 0)
            {
            atomicAdd(grad_decay, sdata[0]);
            }
        }
        """
        return codes


def _ensure_power_of_two_threads() -> None:
    threads = configure.cuda_threads
    if threads <= 0 or (threads & (threads - 1)) != 0:
        raise ValueError(
            "PLIF backward reduction kernel requires configure.cuda_threads to be "
            f"a positive power of two, but got {threads}."
        )


@cache
def _get_plif_forward_kernel(
    *, decay_input: bool, hard_reset: bool, dtype: str
) -> ParametricLIFNodeFPTTKernel:
    return ParametricLIFNodeFPTTKernel(
        decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
    )


@cache
def _get_plif_backward_kernel(
    *,
    decay_input: bool,
    sg_cupy_id: int,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> ParametricLIFNodeBPTTKernel:
    return ParametricLIFNodeBPTTKernel(
        decay_input=decay_input,
        surrogate_function=_cuda_codes_callable(sg_cupy_id, dtype),
        hard_reset=hard_reset,
        detach_reset=detach_reset,
        dtype=dtype,
    )


@torch.library.custom_op("sj::cupy_multistep_plif_forward", mutates_args=())
def cupy_multistep_plif_forward(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    decay: torch.Tensor,
    decay_input: bool,
    sg_cupy_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
        raise ValueError(
            "When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!"
        )
    x_seq = x_seq.contiguous()
    v_init = v_init.contiguous()
    decay = decay.contiguous()
    dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    hard_reset = not soft_reset
    forward_kernel = _get_plif_forward_kernel(
        decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
    )
    py_dict = {
        "x_seq": x_seq,
        "v_init": v_init,
        "v_th": v_th,
        "v_reset": None if soft_reset else v_reset,
        "decay": decay,
    }
    blocks, threads, py_dict = prepare_forward_meta(py_dict)
    py_dict["spike_seq"] = torch.empty_like(x_seq)
    py_dict["h_seq"] = torch.empty_like(x_seq)
    py_dict["v_v_seq"] = x_seq.new_empty((x_seq.shape[0] + 1, *x_seq.shape[1:]))
    py_dict["v_v_seq"][0].copy_(py_dict.pop("v_init"))
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    forward_kernel((blocks,), (threads,), py_dict)
    return (
        py_dict["spike_seq"],
        py_dict["v_v_seq"][1:,],
        py_dict["h_seq"],
    )


@torch.library.register_fake("sj::cupy_multistep_plif_forward")
def _cupy_multistep_plif_forward_fake(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    decay: torch.Tensor,
    decay_input: bool,
    sg_cupy_id: int,
):
    return (
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
        x_seq.new_empty(x_seq.shape),
    )


def _recover_v_v_seq_view(v_seq: torch.Tensor):
    base_offset = v_seq.storage_offset() - v_seq.stride()[0]
    if base_offset < 0:
        return None
    full_shape = (v_seq.shape[0] + 1, *v_seq.shape[1:])
    try:
        return v_seq.as_strided(full_shape, v_seq.stride(), storage_offset=base_offset)
    except RuntimeError:
        return None


def _setup_cupy_multistep_plif_context(ctx, inputs, output):
    (
        _,
        v_init,
        v_th,
        v_reset,
        soft_reset,
        detach_reset,
        decay,
        decay_input,
        sg_cupy_id,
    ) = inputs
    h_seq = output[2]
    v_seq = output[1]
    v_v_seq = _recover_v_v_seq_view(v_seq)
    if v_v_seq is None:
        v_v_seq = torch.cat((v_init.unsqueeze(0), v_seq), dim=0).contiguous()
    ctx.save_for_backward(h_seq, v_v_seq, decay)
    ctx.v_th = v_th
    ctx.v_reset = None if soft_reset else v_reset
    ctx.detach_reset = detach_reset
    ctx.decay_input = decay_input
    ctx.sg_cupy_id = sg_cupy_id


@torch.library.custom_op("sj::cupy_multistep_plif_backward", mutates_args=())
def cupy_multistep_plif_backward(
    grad_spike_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_v_seq: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    decay: torch.Tensor,
    decay_input: bool,
    sg_cupy_id: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    grad_spike_seq = grad_spike_seq.contiguous()
    grad_v_seq = grad_v_seq.contiguous()
    h_seq = h_seq.contiguous()
    dtype = _dtype_to_cupy_kernel_dtype(grad_spike_seq.dtype)
    hard_reset = not soft_reset
    backward_kernel = _get_plif_backward_kernel(
        decay_input=decay_input,
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
    _ensure_power_of_two_threads()
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
        "decay": decay,
        "grad_decay": torch.zeros_like(decay, dtype=torch.float),
        "v_v_seq": v_v_seq,
    }
    cuda_utils._scalar_to_cupy(py_dict, ref="grad_spike_seq")
    if py_dict["v_reset"] is None:
        py_dict.pop("v_reset")
    backward_kernel((blocks,), (threads,), py_dict)
    return py_dict["grad_x_seq"], py_dict["grad_v_init"], py_dict["grad_decay"]


@torch.library.register_fake("sj::cupy_multistep_plif_backward")
def _cupy_multistep_plif_backward_fake(
    grad_spike_seq: torch.Tensor,
    grad_v_seq: torch.Tensor,
    h_seq: torch.Tensor,
    v_v_seq: torch.Tensor,
    v_th: float,
    v_reset: float,
    soft_reset: bool,
    detach_reset: bool,
    decay: torch.Tensor,
    decay_input: bool,
    sg_cupy_id: int,
):
    return (
        torch.empty_like(grad_spike_seq),
        torch.empty_like(grad_v_seq[0]),
        torch.empty_like(decay, dtype=torch.float),
    )


def _cupy_multistep_plif_backward_autograd(ctx, grad_spike_seq, grad_v_seq, grad_h_seq):
    del grad_h_seq
    h_seq, v_v_seq, decay = ctx.saved_tensors
    soft_reset = ctx.v_reset is None
    v_reset = 0.0 if soft_reset else float(ctx.v_reset)
    grad_x_seq, grad_v_init, grad_decay = cupy_multistep_plif_backward(
        grad_spike_seq,
        grad_v_seq,
        h_seq,
        v_v_seq,
        ctx.v_th,
        v_reset,
        soft_reset,
        ctx.detach_reset,
        decay,
        ctx.decay_input,
        ctx.sg_cupy_id,
    )
    return grad_x_seq, grad_v_init, None, None, None, None, grad_decay, None, None


torch.library.register_autograd(
    "sj::cupy_multistep_plif_forward",
    _cupy_multistep_plif_backward_autograd,
    setup_context=_setup_cupy_multistep_plif_context,
)


def plif_multi_step(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> tuple[torch.Tensor, torch.Tensor]:
    sg_cupy_id = _resolve_cuda_code_id(
        surrogate_function, _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    )
    soft_reset = v_reset is None
    v_reset_value = 0.0 if v_reset is None else float(v_reset)
    s_seq, v_seq, _ = cupy_multistep_plif_forward(
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
