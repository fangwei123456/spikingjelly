import logging
import math
from typing import Callable, Iterable

import numpy as np
import torch

try:
    import cupy
except BaseException as e:
    logging.info(
        f"spikingjelly.activation_based.cuda_kernel.auto_cuda.ss_neuronal_kernel: {e}"
    )
    cupy = None

from ..... import configure
from ... import cuda_utils
from .. import base, cfunction


def if_requires_grad(items: Iterable):
    requires_grad = False
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                requires_grad = True
                break

    return requires_grad


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


def new_tensors(news: tuple, py_dict: dict, ref: str = "x"):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(
        torch.split(
            torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0]
        )
    ):
        py_dict[news[i]] = item


def neuronal_hard_reset(
    v_next: str, h: str, spike: str, v_reset: str, dtype: str = "float"
):
    if dtype == "float":
        return f"{v_next} = {h} * (1.0f - {spike}) + {v_reset} * {spike};"
    elif dtype == "half2":
        return f"{v_next} = __hfma2({h}, __hsub2(__float2half2_rn(1.0f), {spike}), __hmul2(v_reset, {spike}));"
    else:
        raise NotImplementedError(dtype)


def neuronal_soft_reset(
    v_next: str, h: str, spike: str, v_th: str, dtype: str = "float"
):
    if dtype == "float":
        return f"{v_next} = {h} - {v_th} * {spike};"
    elif dtype == "half2":
        return f"{v_next} = __hsub2({h}, __hmul2({v_th}, {spike}));"
    else:
        raise NotImplementedError(dtype)


def neuronal_fire(spike: str, v: str, v_th: str, dtype: str = "float"):
    if dtype == "float":
        return cfunction.heaviside(y=spike, x=f"({v} - {v_th})", dtype=dtype)
    elif dtype == "half2":
        return cfunction.heaviside(y=spike, x=f"__hsub2({v}, {v_th})", dtype=dtype)
    else:
        raise NotImplementedError(dtype)


class NeuronFPKernel(base.CKernel1D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}"
        )
        self.hard_reset = hard_reset
        self.dtype = dtype
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
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`H = f(X, V, ...)`.

        This function should define how ``h`` is calculated by ``x[index], v[index]`` and other params if
        the neuron needs.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def neuronal_charge(self) -> str:
                return cfunction.add(
                    z="h[index]", x="x[index]", y="v[index]", dtype=self.dtype
                )
        """
        return "// neuronal_charge should be defined here!"

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(
            neuronal_fire(
                spike="spike[index]", v="h[index]", v_th="v_th", dtype=self.dtype
            )
        )

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(
                    v_next="v_next[index]",
                    h="h[index]",
                    spike="spike[index]",
                    v_reset="v_reset",
                    dtype=self.dtype,
                )
            )
        else:
            core_codes.append(
                neuronal_soft_reset(
                    v_next="v_next[index]",
                    h="h[index]",
                    spike="spike[index]",
                    v_th="v_th",
                    dtype=self.dtype,
                )
            )

        self._core = core_codes.codes
        return self._core


class NeuronBPKernel(base.CKernel1D):
    def __init__(
        self,
        surrogate_function: Callable,
        hard_reset: bool,
        detach_reset: bool,
        dtype: str,
    ):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}_{'detach_reset' if detach_reset else 'nodetach_reset'}"
        )
        self.surrogate_function = surrogate_function
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

    @property
    def post_core(self):
        codes = base.CodeTyper(16)
        codes.append(self.grad_h_next_to_v())
        codes.append(
            cfunction.mul(
                z="grad_v[index]", x="grad_h", y="grad_h_next_to_v", dtype=self.dtype
            )
        )
        self._post_core = codes.codes
        return self._post_core

    def grad_h_to_v(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H}{\\mathrm{d} V}`.

        This function should define how ``grad_h_to_v`` is calculated. Note that ``grad_h_to_v`` has not been
        declared. Thus, this function should also declare ``grad_h_to_v``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_to_v(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_to_v", x=1.0, dtype=self.dtype
                )
        """
        return "// grad_h_to_v should be defined here!"

    def grad_h_to_x(self) -> str:
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t]}{\\mathrm{d} X[t]}`.

        This function should define how ``grad_h_to_x`` is calculated. Note that ``grad_h_to_x`` has not been
        declared. Thus, this function should also declare ``grad_h_to_x``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_to_x(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
                )
        """
        return "// grad_h_to_x should be defined here!"

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

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
        core_codes.append(
            self.surrogate_function(
                y=f"const {self.dtype} grad_s_to_h", x="over_th", dtype=self.dtype
            )
        )

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
                with base.CodeBlock(core_codes):
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
                with base.CodeBlock(core_codes):
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


class NeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        r"""
        **API Language:**
        :ref:`中文 <ss_neuron_kernel_pre_forward-cn>` | :ref:`English <ss_neuron_kernel_pre_forward-en>`

        ----

        .. _ss_neuron_kernel_pre_forward-cn:

        * **中文**

        对单步神经元 ``autograd.Function`` 的前向输入字典做预处理，返回 CUDA kernel
        调用所需的参数。

        :param py_dict: 由神经元前向函数构建的字典，至少应包含 ``x``、``v``、``v_reset``
        :type py_dict: dict

        :return: ``(requires_grad, blocks, threads, py_dict)``

            - ``requires_grad``: 是否存在需要梯度的张量
            - ``blocks``: CUDA kernel 启动参数 ``blocks``
            - ``threads``: CUDA kernel 启动参数 ``threads``，
              默认来自 ``spikingjelly.configure.cuda_threads``
            - ``py_dict``: 预处理后的字典。相较输入字典会：

              1) 将 ``float/int`` 标量转换为 ``cupy.ndarray``；
              2) 新增 ``h``、``spike``、``v_next``（与 ``x`` 或 ``v`` 同形状的零张量）；
              3) 新增 ``numel``（``cupy.ndarray``）。当 ``x.dtype == torch.half`` 时，
              kernel 按 half2 路径计算，``numel = math.ceil(numel / 2)``。
        :rtype: tuple

        ----

        .. _ss_neuron_kernel_pre_forward-en:

        * **English**

        Preprocess the forward input dictionary of single-step neuron
        ``autograd.Function`` and return runtime parameters required by the CUDA
        kernel launch.

        :param py_dict: A dict built from the neuron's forward function. It should
            at least contain ``x``, ``v``, and ``v_reset``.
        :type py_dict: dict

        :return: ``(requires_grad, blocks, threads, py_dict)``

            - ``requires_grad``: whether any tensor in ``py_dict`` requires grad
            - ``blocks``: CUDA launch parameter ``blocks``
            - ``threads``: CUDA launch parameter ``threads``; default value is
              ``spikingjelly.configure.cuda_threads``
            - ``py_dict``: processed dict. Compared with the input, it will:

              1) convert ``float/int`` scalars to ``cupy.ndarray``;
              2) add ``h``, ``spike``, ``v_next`` (zero tensors with shape matching
              ``x`` or ``v``);
              3) add ``numel`` (as ``cupy.ndarray``). If ``x.dtype ==
              torch.half``, the half2 path is used and ``numel = math.ceil(numel /
              2)``.
        :rtype: tuple
        """
        device = py_dict["x"].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        scalar_to_cupy(py_dict)

        new_tensors(("h", "spike", "v_next"), py_dict)
        numel = py_dict["x"].numel()
        threads = configure.cuda_threads
        if py_dict["x"].dtype == torch.float16:
            # we will take two neurons to calculate as one neuron in cuda half2
            # pad will be implemented by the kernel.__call__
            numel = math.ceil(numel / 2)

        blocks = cuda_utils.cal_blocks(numel)

        with cuda_utils.DeviceEnvironment(device):
            numel = cupy.asarray(numel, dtype=np.int32)

        py_dict["numel"] = numel

        return requires_grad, blocks, threads, py_dict

    @staticmethod
    def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param requires_grad: if any tensor in forward params requires grad
        :type requires_grad: bool
        :param args: tensors that need to be saved by ``ctx.save_for_backward``
        :param kwargs: items that need to be saved by ``ctx.xx = xx``

        Saves ``*args, **kwargs`` in ``ctx`` by ``ctx.save_for_backward(*args)`` and ``ctx.xx = xx`` for all ``xx`` in ``kwargs.items()``.
        """
        if requires_grad:
            ctx.save_for_backward(*args)
            for key, value in kwargs.items():
                ctx.__setattr__(key, value)

    @staticmethod
    def pre_backward(ctx, grad_spike: torch.Tensor, grad_v_next: torch.Tensor):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param grad_spike: gradients of ``spike``
        :type grad_spike: torch.Tensor
        :param grad_v_next: gradients of ``v_next``
        :type grad_v_next: torch.Tensor
        :return: backward_kernel, blocks, threads, py_dict

            backward_kernel: NeuronBPTTKernel
                The CUDA kernel used for backward. It should be provided in ``ctx.backward_kernel``

            blocks: int
                CUDA param used in calling CUDA kernel. It should be provided in ``ctx.blocks``

            threads: int
                CUDA param used in calling CUDA kernel. It should be provided in ``ctx.threads``
        :rtype: tuple
        """
        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads

        h = ctx.saved_tensors[0]
        numel = ctx.numel
        v_th = ctx.v_th
        v_reset = ctx.v_reset

        zero_shape = list(grad_spike.shape)
        zero_shape[0] *= 2
        zero_data = torch.zeros(
            zero_shape, device=grad_spike.device, dtype=grad_spike.dtype
        )

        # For fp16, ctx.numel will be divided by 2 later. Here is a reliable way to divide tensor equally
        real_numel = grad_spike.size(0)
        grad_x = zero_data[:real_numel]
        grad_v = zero_data[real_numel:]

        py_dict = {
            "numel": numel,
            "grad_spike": grad_spike,
            "grad_v_next": grad_v_next,
            "h": h,
            "grad_x": grad_x,
            "grad_v": grad_v,
            "v_th": v_th,
            "v_reset": v_reset,
        }

        return backward_kernel, blocks, threads, py_dict
