from typing import Optional, Callable

import logging
import torch

from .common import (
    base,
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
        codes = base.CodeTyper(16)
        codes.append("sdata[threadIdx.x] = 0.0f;")
        return super().pre_core + "\n" + codes.codes

    @property
    def core(self):
        core_codes = base.CodeTyper(18)
        with base.CodeBlock(core_codes):
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


class ParametricLIFNodeATGF(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        x_seq: torch.Tensor,
        v_init: torch.Tensor,
        v_th: float,
        v_reset: Optional[float],
        decay: torch.Tensor,
        forward_kernel: ParametricLIFNodeFPTTKernel,
        backward_kernel: ParametricLIFNodeBPTTKernel,
    ):
        if x_seq.dtype == torch.float16 and v_init.numel() % 2 != 0:
            raise ValueError(
                "When using the the PLIF neuron with half2 cupy backend, the numer of neurons should be even to avoid the wrong gradient of tau caused by padding!"
            )
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
            py_dict["v_v_seq"],
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
        py_dict["grad_decay"] = torch.zeros_like(ctx.decay, dtype=torch.float)
        py_dict["v_v_seq"] = ctx.saved_tensors[1]

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
            py_dict["grad_decay"],
            None,
            None,
        )


_PLIF_FWD_KERNEL_CACHE = {}
_PLIF_BWD_KERNEL_CACHE = {}


def _get_plif_forward_kernel(
    *, decay_input: bool, hard_reset: bool, dtype: str
) -> ParametricLIFNodeFPTTKernel:
    key = (decay_input, hard_reset, dtype)
    kernel = _PLIF_FWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = ParametricLIFNodeFPTTKernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
        _PLIF_FWD_KERNEL_CACHE[key] = kernel
    return kernel


def _get_plif_backward_kernel(
    *,
    decay_input: bool,
    sg_cupy_id: int,
    hard_reset: bool,
    detach_reset: bool,
    dtype: str,
) -> ParametricLIFNodeBPTTKernel:
    key = (decay_input, sg_cupy_id, hard_reset, detach_reset, dtype)
    kernel = _PLIF_BWD_KERNEL_CACHE.get(key)
    if kernel is None:
        kernel = ParametricLIFNodeBPTTKernel(
            decay_input=decay_input,
            surrogate_function=_surrogate_cuda_codes_from_id(sg_cupy_id),
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
        _PLIF_BWD_KERNEL_CACHE[key] = kernel
    return kernel


def _legacy_multistep_plif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    v_th: float,
    v_reset: Optional[float],
    decay: torch.Tensor,
    decay_input: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    detach_reset: bool,
    forward_kernel: Optional[ParametricLIFNodeFPTTKernel],
    backward_kernel: Optional[ParametricLIFNodeBPTTKernel],
):
    hard_reset = v_reset is not None
    dtype = _dtype_to_cupy_kernel_dtype(x_seq.dtype)
    if forward_kernel is None:
        forward_kernel = _get_plif_forward_kernel(
            decay_input=decay_input, hard_reset=hard_reset, dtype=dtype
        )
    if backward_kernel is None:
        if not hasattr(surrogate_function, "cuda_codes"):
            raise TypeError(
                "surrogate_function for CuPy legacy path must provide cuda_codes."
            )
        backward_kernel = ParametricLIFNodeBPTTKernel(
            decay_input=decay_input,
            surrogate_function=surrogate_function.cuda_codes,
            hard_reset=hard_reset,
            detach_reset=detach_reset,
            dtype=dtype,
        )
    return ParametricLIFNodeATGF.apply(
        x_seq,
        v_init,
        v_th,
        v_reset,
        decay,
        forward_kernel,
        backward_kernel,
    )


if _CUPY_CUSTOM_OP_AVAILABLE:

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
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
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
        _, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        forward_kernel((blocks,), (threads,), py_dict)
        return (
            py_dict["spike_seq"],
            py_dict["v_v_seq"][1:,],
            py_dict["h_seq"],
            py_dict["v_v_seq"],
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
        T = x_seq.shape[0]
        return (
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty(x_seq.shape),
            x_seq.new_empty((T + 1, *x_seq.shape[1:])),
        )


    def _setup_cupy_multistep_plif_context(ctx, inputs, output):
        _, _, v_th, v_reset, soft_reset, detach_reset, decay, decay_input, sg_cupy_id = inputs
        h_seq = output[2]
        v_v_seq = output[3]
        ctx.save_for_backward(h_seq, v_v_seq, decay)
        ctx.v_th = v_th
        ctx.v_reset = None if soft_reset else v_reset
        ctx.detach_reset = detach_reset
        ctx.decay_input = decay_input
        ctx.sg_cupy_id = sg_cupy_id


    def _cupy_multistep_plif_backward(
        ctx, grad_spike_seq, grad_v_seq, grad_h_seq, grad_v_v_seq
    ):
        del grad_h_seq
        del grad_v_v_seq
        h_seq, v_v_seq, decay = ctx.saved_tensors
        grad_spike_seq = grad_spike_seq.contiguous()
        grad_v_seq = grad_v_seq.contiguous()
        h_seq = h_seq.contiguous()
        dtype = _dtype_to_cupy_kernel_dtype(grad_spike_seq.dtype)
        hard_reset = ctx.v_reset is not None
        backward_kernel = _get_plif_backward_kernel(
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
            "decay": decay,
            "grad_decay": torch.zeros_like(decay, dtype=torch.float),
            "v_v_seq": v_v_seq,
        }
        if py_dict["v_reset"] is None:
            py_dict.pop("v_reset")
        backward_kernel((blocks,), (threads,), py_dict)
        return (
            py_dict["grad_x_seq"],
            py_dict["grad_v_init"],
            None,
            None,
            None,
            None,
            py_dict["grad_decay"],
            None,
            None,
        )


    torch.library.register_autograd(
        "sj::cupy_multistep_plif_forward",
        _cupy_multistep_plif_backward,
        setup_context=_setup_cupy_multistep_plif_context,
    )


def multistep_plif(
    x_seq: torch.Tensor,
    v_init: torch.Tensor,
    decay: torch.Tensor,
    decay_input: bool,
    v_threshold: float,
    v_reset: Optional[float],
    detach_reset: bool,
    surrogate_function: surrogate.SurrogateFunctionBase,
    forward_kernel: Optional[ParametricLIFNodeFPTTKernel] = None,
    backward_kernel: Optional[ParametricLIFNodeBPTTKernel] = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    r"""
    **API Language:**
    :ref:`中文 <multistep_plif-cn>` | :ref:`English <multistep_plif-en>`

    ----

    .. _multistep_plif-cn:

    * **中文**

    使用 CuPy 后端执行多步 Parametric LIF 神经元前向与反向计算，返回脉冲序列与膜电位序列。
    该接口会优先走 ``torch.library`` 自定义算子路径；当代理函数不支持 CuPy ID 解析时，自动回退到 legacy 路径。

    ``decay`` 为可训练参数，通常满足 :math:`\tau = 1 / decay`。当 ``decay_input=True`` 时，充电项对输入衰减；
    否则按标准 PLIF 形式更新膜电位。``v_reset=None`` 表示 soft reset，否则为 hard reset。

    :param x_seq: 输入序列，shape 通常为 ``[T, N, *]``
    :type x_seq: torch.Tensor

    :param v_init: 初始膜电位，shape 通常为 ``[N, *]``
    :type v_init: torch.Tensor

    :param decay: 衰减参数张量（可训练）
    :type decay: torch.Tensor

    :param decay_input: 是否对输入项进行衰减
    :type decay_input: bool

    :param v_threshold: 放电阈值
    :type v_threshold: float

    :param v_reset: 重置电位。``None`` 表示 soft reset
    :type v_reset: Optional[float]

    :param detach_reset: 反向传播时是否截断 reset 分支梯度
    :type detach_reset: bool

    :param surrogate_function: 反向传播使用的替代梯度函数
    :type surrogate_function: surrogate.SurrogateFunctionBase

    :param forward_kernel: 可选，复用前向 CUDA kernel 实例
    :type forward_kernel: Optional[ParametricLIFNodeFPTTKernel]

    :param backward_kernel: 可选，复用反向 CUDA kernel 实例
    :type backward_kernel: Optional[ParametricLIFNodeBPTTKernel]

    :return: ``(s_seq, v_seq)``，分别为脉冲序列与每步膜电位
    :rtype: tuple[torch.Tensor, torch.Tensor]

    ----

    .. _multistep_plif-en:

    * **English**

    Run multi-step Parametric LIF forward/backward computation with the CuPy backend and return
    spike sequences and membrane potentials. This API prefers the ``torch.library`` custom-op path;
    if the surrogate function cannot be resolved to a CuPy id, it falls back to the legacy path.

    ``decay`` is a learnable decay parameter and is usually related to :math:`\tau` by
    :math:`\tau = 1 / decay`. When ``decay_input=True``, the input term is decayed during charging;
    otherwise the standard PLIF update rule is used. ``v_reset=None`` indicates soft reset, otherwise
    hard reset is applied.

    :param x_seq: Input sequence, typically with shape ``[T, N, *]``
    :type x_seq: torch.Tensor

    :param v_init: Initial membrane potential, typically with shape ``[N, *]``
    :type v_init: torch.Tensor

    :param decay: Decay parameter tensor (learnable)
    :type decay: torch.Tensor

    :param decay_input: Whether to decay the input term during charging
    :type decay_input: bool

    :param v_threshold: Firing threshold
    :type v_threshold: float

    :param v_reset: Reset potential. ``None`` means soft reset
    :type v_reset: Optional[float]

    :param detach_reset: Whether to detach reset-branch gradients in backward
    :type detach_reset: bool

    :param surrogate_function: Surrogate gradient function used in backward
    :type surrogate_function: surrogate.SurrogateFunctionBase

    :param forward_kernel: Optional pre-built forward CUDA kernel instance
    :type forward_kernel: Optional[ParametricLIFNodeFPTTKernel]

    :param backward_kernel: Optional pre-built backward CUDA kernel instance
    :type backward_kernel: Optional[ParametricLIFNodeBPTTKernel]

    :return: ``(s_seq, v_seq)``, spike sequence and per-step membrane potential sequence
    :rtype: tuple[torch.Tensor, torch.Tensor]
    """
    if _use_cupy_custom_op():
        try:
            sg_cupy_id, _ = resolve_sg_cupy_id_and_key(surrogate_function)
            soft_reset = v_reset is None
            v_reset_value = 0.0 if v_reset is None else float(v_reset)
            s_seq, v_seq, _, _ = cupy_multistep_plif_forward(
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
        except Exception as e:
            logging.debug("multistep_plif custom-op fallback: %s", e)

    return _legacy_multistep_plif(
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
