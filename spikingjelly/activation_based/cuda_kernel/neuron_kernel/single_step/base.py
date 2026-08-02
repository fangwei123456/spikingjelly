import logging
import math

import numpy as np
import torch

try:
    import cupy
except BaseException as e:
    logging.info(
        f"spikingjelly.activation_based.cuda_kernel.neuron_kernel.single_step: {e}"
    )
    cupy = None

from ..... import configure
from ... import cuda_utils
from ...auto_cuda import base as auto_cuda_base, cfunction
from ..cuda_code import _neuronal_fire, _neuronal_hard_reset, _neuronal_soft_reset


def scalar_to_cupy(py_dict: dict, ref: str = "x"):
    device = py_dict[ref].get_device()
    dtype = py_dict[ref].dtype

    with cuda_utils.DeviceEnvironment(device):
        for key, value in py_dict.items():
            if isinstance(value, float):
                if dtype == torch.float32:
                    value = cupy.asarray(value, dtype=np.float32)
                elif dtype == torch.float16:
                    value = cupy.asarray([value, value], dtype=np.float16)
                else:
                    raise NotImplementedError(dtype)
                py_dict[key] = value

            elif isinstance(value, int):
                py_dict[key] = cupy.asarray(value, dtype=np.int32)


class NeuronFPKernel(auto_cuda_base.CKernel1D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}"
        )
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f"const {dtype} *", cname="x")
        self.add_param(ctype=f"const {dtype} *", cname="v")
        self.add_param(ctype=f"{dtype} *", cname="h")
        self.add_param(ctype=f"{dtype} *", cname="v_next")
        self.add_param(ctype=f"{dtype} *", cname="spike")
        self.add_param(ctype=f"{dtype} &", cname="v_th")
        if hard_reset:
            self.add_param(ctype=f"{dtype} &", cname="v_reset")

    def neuronal_charge(self) -> str:
        return "// neuronal_charge should be defined here!"

    @property
    def core(self):
        core_codes = auto_cuda_base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(
            _neuronal_fire(
                spike="spike[index]", v="h[index]", v_th="v_th", dtype=self.dtype
            )
        )

        if self.hard_reset:
            core_codes.append(
                _neuronal_hard_reset(
                    v_next="v_next[index]",
                    h="h[index]",
                    spike="spike[index]",
                    v_reset="v_reset",
                    dtype=self.dtype,
                )
            )
        else:
            core_codes.append(
                _neuronal_soft_reset(
                    v_next="v_next[index]",
                    h="h[index]",
                    spike="spike[index]",
                    v_th="v_th",
                    dtype=self.dtype,
                )
            )

        self._core = core_codes.codes
        return self._core


class NeuronBPKernel(auto_cuda_base.CKernel1D):
    def __init__(
        self,
        surrogate_cuda_codes: str,
        hard_reset: bool,
        detach_reset: bool,
        dtype: str,
    ):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}_{'detach_reset' if detach_reset else 'nodetach_reset'}"
        )
        self.surrogate_cuda_codes = surrogate_cuda_codes
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f"const {dtype} *", cname="grad_spike")
        self.add_param(ctype=f"const {dtype} *", cname="grad_v_next")
        self.add_param(ctype=f"const {dtype} *", cname="h")
        self.add_param(ctype=f"{dtype} *", cname="grad_x")
        self.add_param(ctype=f"{dtype} *", cname="grad_v")
        self.add_param(ctype=f"{dtype} &", cname="v_th")
        if hard_reset:
            self.add_param(ctype=f"{dtype} &", cname="v_reset")

    def grad_h_to_v(self) -> str:
        return "// grad_h_to_v should be defined here!"

    def grad_h_to_x(self) -> str:
        return "// grad_h_to_x should be defined here!"

    @property
    def core(self):
        core_codes = auto_cuda_base.CodeTyper(18)

        core_codes.append(
            cfunction.sub(
                z=f"const {self.dtype} over_th",
                x="h[index]",
                y="v_th",
                dtype=self.dtype,
            )
        )
        core_codes.append(
            cfunction.heaviside(
                y=f"const {self.dtype} spike", x="over_th", dtype=self.dtype
            )
        )
        core_codes.append(self.surrogate_cuda_codes)

        if self.hard_reset:
            core_codes.append(
                cfunction.sub(
                    z=f"{self.dtype} grad_v_next_to_h",
                    x=cfunction.constant(y=None, x=1.0, dtype=self.dtype),
                    y="spike",
                    dtype=self.dtype,
                )
            )

            if not self.detach_reset:
                with auto_cuda_base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(
                            z=f"{self.dtype} temp_var",
                            x="v_reset",
                            y="h[index]",
                            dtype=self.dtype,
                        )
                    )
                    core_codes.append(
                        cfunction.mul(
                            z="temp_var",
                            x="temp_var",
                            y="grad_s_to_h",
                            dtype=self.dtype,
                        )
                    )
                    core_codes.append(
                        cfunction.add(
                            z="grad_v_next_to_h",
                            x="temp_var",
                            y="grad_v_next_to_h",
                            dtype=self.dtype,
                        )
                    )

        else:
            core_codes.append(
                f"{self.dtype} grad_v_next_to_h = {cfunction.constant(None, 1.0, dtype=self.dtype)}"
            )

            if not self.detach_reset:
                with auto_cuda_base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.mul(
                            z=f"{self.dtype} temp_var",
                            x="v_th",
                            y="grad_s_to_h",
                            dtype=self.dtype,
                        )
                    )
                    core_codes.append(
                        cfunction.sub(
                            z="grad_v_next_to_h",
                            x="grad_v_next_to_h",
                            y="temp_var",
                            dtype=self.dtype,
                        )
                    )

        core_codes.append(
            cfunction.mul(
                z=f"{self.dtype} grad_h",
                x="grad_s_to_h",
                y="grad_spike[index]",
                dtype=self.dtype,
            )
        )
        core_codes.append(
            cfunction.add(
                z="grad_h",
                x=cfunction.mul(
                    z=None,
                    x="grad_v_next[index]",
                    y="grad_v_next_to_h",
                    dtype=self.dtype,
                ),
                y="grad_h",
                dtype=self.dtype,
            )
        )

        core_codes.append(self.grad_h_to_v())
        core_codes.append(
            cfunction.mul(
                z="grad_v[index]", x="grad_h", y="grad_h_to_v", dtype=self.dtype
            )
        )

        core_codes.append(self.grad_h_to_x())
        core_codes.append(
            cfunction.mul(
                z="grad_x[index]", x="grad_h", y="grad_h_to_x", dtype=self.dtype
            )
        )

        self._core = core_codes.codes
        return self._core


def _prepare_forward(py_dict: dict):
    device = py_dict["x"].get_device()
    scalar_to_cupy(py_dict)

    for name in ("h", "spike", "v_next"):
        py_dict[name] = torch.empty_like(py_dict["x"])
    numel = py_dict["x"].numel()
    threads = configure.cuda_threads
    if py_dict["x"].dtype == torch.float16:
        # half2 kernels process two neurons per CUDA element.
        numel = math.ceil(numel / 2)

    blocks = cuda_utils.cal_blocks(numel)
    with cuda_utils.DeviceEnvironment(device):
        py_dict["numel"] = cupy.asarray(numel, dtype=np.int32)

    return blocks, threads, py_dict


def _prepare_backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
    py_dict = {
        "numel": ctx.numel,
        "grad_spike": grad_spike,
        "grad_v_next": grad_v_next,
        "h": ctx.saved_tensors[0],
        "grad_x": torch.empty_like(grad_spike),
        "grad_v": torch.empty_like(grad_spike),
        "v_th": ctx.v_th,
        "v_reset": ctx.v_reset,
    }
    return ctx.backward_kernel, ctx.blocks, ctx.threads, py_dict
