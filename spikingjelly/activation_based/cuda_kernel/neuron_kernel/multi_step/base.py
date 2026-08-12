from spikingjelly.logger import logger
import math
from typing import Callable, Iterable

import torch

try:
    import cupy
except (ImportError, OSError) as e:
    logger.debug("Optional CuPy dependency unavailable: {}", e)
    cupy = None


from ..... import configure
from ... import cuda_utils
from ...auto_cuda import base as auto_cuda_base, cfunction
from ..cuda_code import _neuronal_fire, _neuronal_hard_reset, _neuronal_soft_reset


def _dtype_to_cupy_kernel_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float16:
        return "half2"
    raise NotImplementedError(dtype)


class NeuronFPTTKernel(auto_cuda_base.CKernel2D):
    def __init__(self, hard_reset: bool, dtype: str):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}",
            reverse=False,
        )
        self.hard_reset = hard_reset
        self.dtype = dtype
        self.add_param(ctype=f"const {dtype} *", cname="x_seq")
        self.add_param(ctype=f"{dtype} *", cname="v_v_seq")
        self.add_param(ctype=f"{dtype} *", cname="h_seq")
        self.add_param(ctype=f"{dtype} *", cname="spike_seq")
        self.add_param(ctype=f"{dtype} &", cname="v_th")
        if hard_reset:
            self.add_param(ctype=f"{dtype} &", cname="v_reset")

    def neuronal_charge(self) -> str:
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronfpttkernel-neuronal_charge-cn>` |
        :ref:`English <neuronfpttkernel-neuronal_charge-en>`

        ----

        .. _neuronfpttkernel-neuronal_charge-cn:

        * **中文**

        返回用于计算 :math:`H[t] = f(X[t], V[t-1], \ldots)` 的 CUDA 代码字符串。
        子类应在该函数中定义 ``h_seq[t]`` 如何由 ``x_seq[t]``、``v_v_seq[t]`` 以及其他参数计算。

        示例（IF 神经元）：

        .. code-block:: python

            def neuronal_charge(self) -> str:
                # v_v_seq[t] 对应 v_seq[t - dt]
                return cfunction.add(
                    z="h_seq[t]", x="x_seq[t]", y="v_v_seq[t]", dtype=self.dtype
                )

        :return: CUDA 代码字符串
        :rtype: str

        ----

        .. _neuronfpttkernel-neuronal_charge-en:

        * **English**

        Return CUDA code that computes :math:`H[t] = f(X[t], V[t-1], \ldots)`.
        Subclasses should implement how ``h_seq[t]`` is computed from ``x_seq[t]``,
        ``v_v_seq[t]``, and other neuron-specific parameters.

        Example (IF neuron):

        .. code-block:: python

            def neuronal_charge(self) -> str:
                # v_v_seq[t] is v_seq[t - dt]
                return cfunction.add(
                    z="h_seq[t]", x="x_seq[t]", y="v_v_seq[t]", dtype=self.dtype
                )

        :return: CUDA code string
        :rtype: str
        """
        return "// neuronal_charge should be defined here!"

    @property
    def core(self):
        core_codes = auto_cuda_base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(
            _neuronal_fire(
                spike="spike_seq[t]", v="h_seq[t]", v_th="v_th", dtype=self.dtype
            )
        )

        if self.hard_reset:
            core_codes.append(
                _neuronal_hard_reset(
                    v_next="v_v_seq[t + dt]",
                    h="h_seq[t]",
                    spike="spike_seq[t]",
                    v_reset="v_reset",
                    dtype=self.dtype,
                )
            )
        else:
            core_codes.append(
                _neuronal_soft_reset(
                    v_next="v_v_seq[t + dt]",
                    h="h_seq[t]",
                    spike="spike_seq[t]",
                    v_th="v_th",
                    dtype=self.dtype,
                )
            )

        self._core = core_codes.codes
        return self._core


class NeuronBPTTKernel(auto_cuda_base.CKernel2D):
    def __init__(
        self,
        surrogate_function: Callable,
        hard_reset: bool,
        detach_reset: bool,
        dtype: str,
    ):
        super().__init__(
            kernel_name=f"{self.__class__.__name__}_{dtype}_{'hard_reset' if hard_reset else 'soft_reset'}_{'detach_reset' if detach_reset else 'nodetach_reset'}",
            reverse=True,
        )
        self.surrogate_function = surrogate_function
        self.hard_reset = hard_reset
        self.detach_reset = detach_reset
        self.dtype = dtype
        self.add_param(ctype=f"const {dtype} *", cname="grad_spike_seq")
        self.add_param(ctype=f"const {dtype} *", cname="grad_v_seq")
        self.add_param(ctype=f"const {dtype} *", cname="h_seq")
        self.add_param(ctype=f"{dtype} *", cname="grad_x_seq")
        self.add_param(ctype=f"{dtype} *", cname="grad_v_init")
        self.add_param(ctype=f"{dtype} &", cname="v_th")
        if hard_reset:
            self.add_param(ctype=f"{dtype} &", cname="v_reset")

    @property
    def pre_core(self):
        codes = auto_cuda_base.CodeTyper(16)
        if self.dtype == "float":
            codes.append("float grad_h = 0.0f;")
        elif self.dtype == "half2":
            codes.append(cfunction.float2half2(y="half2 grad_h", x="0.0f"))
        else:
            raise NotImplementedError(self.dtype)

        self._pre_core = codes.codes
        return self._pre_core

    @property
    def post_core(self):
        codes = auto_cuda_base.CodeTyper(16)
        codes.append(self.grad_h_next_to_v())
        codes.append(
            cfunction.mul(
                z="grad_v_init[index]",
                x="grad_h",
                y="grad_h_next_to_v",
                dtype=self.dtype,
            )
        )
        self._post_core = codes.codes
        return self._post_core

    def grad_h_next_to_v(self) -> str:
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronbpttkernel-grad_h_next_to_v-cn>` |
        :ref:`English <neuronbpttkernel-grad_h_next_to_v-en>`

        ----

        .. _neuronbpttkernel-grad_h_next_to_v-cn:

        * **中文**

        返回计算 :math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}` 的 CUDA 代码字符串。
        子类应在此函数中给出 ``grad_h_next_to_v`` 的计算，并同时完成其声明。

        示例（IF 神经元）：

        .. code-block:: python

            def grad_h_next_to_v(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_next_to_v", x=1.0, dtype=self.dtype
                )

        :return: CUDA 代码字符串
        :rtype: str

        ----

        .. _neuronbpttkernel-grad_h_next_to_v-en:

        * **English**

        Return CUDA code that computes :math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}`.
        Subclasses should define and declare ``grad_h_next_to_v`` in this method.

        Example (IF neuron):

        .. code-block:: python

            def grad_h_next_to_v(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_next_to_v", x=1.0, dtype=self.dtype
                )

        :return: CUDA code string
        :rtype: str
        """
        return "// grad_h_next_to_v should be defined here!"

    def grad_h_to_x(self) -> str:
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronbpttkernel-grad_h_to_x-cn>` |
        :ref:`English <neuronbpttkernel-grad_h_to_x-en>`

        ----

        .. _neuronbpttkernel-grad_h_to_x-cn:

        * **中文**

        返回计算 :math:`\frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}` 的 CUDA 代码字符串。
        子类应在此函数中给出 ``grad_h_to_x`` 的计算，并同时完成其声明。

        示例（IF 神经元）：

        .. code-block:: python

            def grad_h_to_x(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
                )

        :return: CUDA 代码字符串
        :rtype: str

        ----

        .. _neuronbpttkernel-grad_h_to_x-en:

        * **English**

        Return CUDA code that computes :math:`\frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}`.
        Subclasses should define and declare ``grad_h_to_x`` in this method.

        Example (IF neuron):

        .. code-block:: python

            def grad_h_to_x(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_to_x", x=1.0, dtype=self.dtype
                )

        :return: CUDA code string
        :rtype: str
        """
        return "// grad_h_to_x should be defined here!"

    @property
    def core(self):
        core_codes = auto_cuda_base.CodeTyper(18)

        core_codes.append(
            cfunction.sub(
                z=f"const {self.dtype} over_th",
                x="h_seq[t]",
                y="v_th",
                dtype=self.dtype,
            )
        )
        core_codes.append(
            cfunction.heaviside(
                y=f"const {self.dtype} spike_seq_t", x="over_th", dtype=self.dtype
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
                    z=f"{self.dtype} grad_v_to_h",
                    x=cfunction.constant(y=None, x=1.0, dtype=self.dtype),
                    y="spike_seq_t",
                    dtype=self.dtype,
                )
            )

            if not self.detach_reset:
                with auto_cuda_base.CodeBlock(core_codes):
                    core_codes.append(
                        cfunction.sub(
                            z=f"{self.dtype} temp_var",
                            x="v_reset",
                            y="h_seq[t]",
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
                            z="grad_v_to_h",
                            x="temp_var",
                            y="grad_v_to_h",
                            dtype=self.dtype,
                        )
                    )

        else:
            core_codes.append(
                f"{self.dtype} grad_v_to_h = {cfunction.constant(None, 1.0, dtype=self.dtype)}"
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
                            z="grad_v_to_h",
                            x="grad_v_to_h",
                            y="temp_var",
                            dtype=self.dtype,
                        )
                    )

        core_codes.append(self.grad_h_next_to_v())
        core_codes.append(
            cfunction.mul(
                z="grad_h", x="grad_h", y="grad_h_next_to_v", dtype=self.dtype
            )
        )
        core_codes.append(
            cfunction.add(z="grad_h", x="grad_v_seq[t]", y="grad_h", dtype=self.dtype)
        )
        core_codes.append(
            cfunction.mul(z="grad_h", x="grad_h", y="grad_v_to_h", dtype=self.dtype)
        )
        with auto_cuda_base.CodeBlock(core_codes):
            core_codes.append(
                cfunction.mul(
                    z=f"{self.dtype} temp_var",
                    x="grad_spike_seq[t]",
                    y="grad_s_to_h",
                    dtype=self.dtype,
                )
            )
            core_codes.append(
                cfunction.add(z="grad_h", x="grad_h", y="temp_var", dtype=self.dtype)
            )

        core_codes.append(self.grad_h_to_x())
        core_codes.append(
            cfunction.mul(
                z="grad_x_seq[t]", x="grad_h", y="grad_h_to_x", dtype=self.dtype
            )
        )

        self._core = core_codes.codes
        return self._core


def if_requires_grad(items: Iterable):
    return any(isinstance(item, torch.Tensor) and item.requires_grad for item in items)


def prepare_forward_meta(py_dict: dict, ref: str = "x_seq"):
    device = py_dict[ref].get_device()
    cuda_utils._scalar_to_cupy(py_dict, ref=ref)

    numel = py_dict[ref].numel()
    N = py_dict[ref].shape[1]
    threads = configure.cuda_threads
    if py_dict[ref].dtype == torch.float16:
        # Use half2 path: two neurons packed as one lane.
        N = math.ceil(N / 2)
        numel = N * py_dict[ref].shape[0]
    blocks = cuda_utils.cal_blocks(N)

    with cuda_utils.DeviceEnvironment(device):
        py_dict["numel"] = cuda_utils._as_cupy_int32(numel, "numel")
        py_dict["N"] = cuda_utils._as_cupy_int32(N, "N")

    return blocks, threads, py_dict


def new_tensors(news: tuple, py_dict: dict, ref: str = "x_seq"):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= len(news)
    for i, item in enumerate(
        torch.split(
            torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0]
        )
    ):
        py_dict[news[i]] = item


class NeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronatgfbase-pre_forward-cn>` |
        :ref:`English <neuronatgfbase-pre_forward-en>`

        ----

        .. _neuronatgfbase-pre_forward-cn:

        * **中文**

        为神经元前向 CUDA kernel 执行准备参数与中间张量。

        :param py_dict: 从神经元 ``forward`` 自动求导函数构建的字典，至少应包含
            ``x_seq``、``v_init``、``v_reset``。
        :type py_dict: dict

        :return: ``(requires_grad, blocks, threads, py_dict)``

            ``requires_grad``: ``bool``，若 ``py_dict`` 中任一张量需要梯度则为 ``True``。

            ``blocks``: ``int``，调用 CUDA kernel 的 ``blocks`` 参数。

            ``threads``: ``int``，调用 CUDA kernel 的 ``threads`` 参数，默认取
            ``spikingjelly.configure.cuda_threads``。

            ``py_dict``: ``dict``，返回字典相较输入会：
            1) 将 ``float/int`` 标量转换为 ``cupy.ndarray``；
            2) 新增 ``h_seq``、``spike_seq``、``v_v_seq``；
            3) 新增 ``N``、``numel``（均为 ``cupy.ndarray``）。当
            ``x_seq.dtype == torch.float16`` 时，按 half2 规则调整 ``N`` 和 ``numel``。

        :rtype: tuple

        ----

        .. _neuronatgfbase-pre_forward-en:

        * **English**

        Prepare parameters and intermediate tensors for the forward CUDA kernel.

        :param py_dict: A dict built from the neuron's forward autograd function.
            It should at least contain ``x_seq``, ``v_init``, and ``v_reset``.
        :type py_dict: dict

        :return: ``(requires_grad, blocks, threads, py_dict)``

            ``requires_grad``: ``bool``. ``True`` if any tensor in ``py_dict`` requires grad.

            ``blocks``: ``int``. CUDA ``blocks`` argument for kernel launch.

            ``threads``: ``int``. CUDA ``threads`` argument for kernel launch.
            The default is ``spikingjelly.configure.cuda_threads``.

            ``py_dict``: ``dict``. Compared with the input dict, it:
            1) converts ``float/int`` scalars to ``cupy.ndarray``;
            2) adds ``h_seq``, ``spike_seq``, and ``v_v_seq``;
            3) adds ``N`` and ``numel`` (both ``cupy.ndarray``). When
            ``x_seq.dtype == torch.float16``, ``N`` and ``numel`` are adjusted for
            half2 execution.

        :rtype: tuple
        """
        device = py_dict["x_seq"].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        cuda_utils._scalar_to_cupy(py_dict, ref="x_seq")

        new_tensors(("h_seq", "spike_seq", "v_seq"), py_dict)
        py_dict["v_v_seq"] = torch.cat(
            (py_dict.pop("v_init").unsqueeze(0), py_dict.pop("v_seq"))
        )
        numel = py_dict["x_seq"].numel()
        N = py_dict["x_seq"].shape[1]
        threads = configure.cuda_threads
        if py_dict["x_seq"].dtype == torch.float16:
            # we will take two neurons to calculate as one neuron in cuda half2
            # pad will be implemented by the kernel.__call__
            N = math.ceil(N / 2)
            numel = N * py_dict["x_seq"].shape[0]

        blocks = cuda_utils.cal_blocks(N)

        with cuda_utils.DeviceEnvironment(device):
            numel = cuda_utils._as_cupy_int32(numel, "numel")
            N = cuda_utils._as_cupy_int32(N, "N")

        py_dict["numel"] = numel
        py_dict["N"] = N

        return requires_grad, blocks, threads, py_dict

    @staticmethod
    def ctx_save(ctx, requires_grad: bool, *args, **kwargs):
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronatgfbase-ctx_save-cn>` |
        :ref:`English <neuronatgfbase-ctx_save-en>`

        ----

        .. _neuronatgfbase-ctx_save-cn:

        * **中文**

        当 ``requires_grad`` 为 ``True`` 时，将前向所需上下文保存到 ``ctx``。

        :param ctx: :class:`torch.autograd.Function` 的上下文对象。
        :type ctx: Any
        :param requires_grad: 前向输入中是否存在需要梯度的张量。
        :type requires_grad: bool
        :param args: 使用 ``ctx.save_for_backward`` 保存的张量。
        :type args: tuple
        :param kwargs: 通过 ``ctx.xx = xx`` 保存的附加字段。
        :type kwargs: dict

        ----

        .. _neuronatgfbase-ctx_save-en:

        * **English**

        Save forward context into ``ctx`` when ``requires_grad`` is ``True``.

        :param ctx: Context object in :class:`torch.autograd.Function`.
        :type ctx: Any
        :param requires_grad: Whether any forward input tensor requires grad.
        :type requires_grad: bool
        :param args: Tensors saved by ``ctx.save_for_backward``.
        :type args: tuple
        :param kwargs: Extra fields saved via ``ctx.xx = xx`` assignments.
        :type kwargs: dict
        """
        if requires_grad:
            ctx.save_for_backward(*args)
            for key, value in kwargs.items():
                ctx.__setattr__(key, value)

    @staticmethod
    def pre_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        r"""
        .. rubric:: API Language

        :ref:`中文 <neuronatgfbase-pre_backward-cn>` |
        :ref:`English <neuronatgfbase-pre_backward-en>`

        ----

        .. _neuronatgfbase-pre_backward-cn:

        * **中文**

        为反向 CUDA kernel 执行准备参数与输出梯度缓冲区。

        :param ctx: :class:`torch.autograd.Function` 的上下文对象。
        :type ctx: Any
        :param grad_spike_seq: ``spike_seq`` 的梯度。
        :type grad_spike_seq: torch.Tensor
        :param grad_v_seq: ``v_seq`` 的梯度。
        :type grad_v_seq: torch.Tensor
        :return: ``(backward_kernel, blocks, threads, py_dict)``

            ``backward_kernel``: ``NeuronBPTTKernel``，反向使用的 CUDA kernel
            （来自 ``ctx.backward_kernel``）。

            ``blocks``: ``int``，kernel 启动参数（来自 ``ctx.blocks``）。

            ``threads``: ``int``，kernel 启动参数（来自 ``ctx.threads``）。

            ``py_dict``: ``dict``，包含反向 kernel 计算所需全部输入输出张量。

        :rtype: tuple

        ----

        .. _neuronatgfbase-pre_backward-en:

        * **English**

        Prepare parameters and output gradient buffers for the backward CUDA kernel.

        :param ctx: Context object in :class:`torch.autograd.Function`.
        :type ctx: Any
        :param grad_spike_seq: Gradient of ``spike_seq``.
        :type grad_spike_seq: torch.Tensor
        :param grad_v_seq: Gradient of ``v_seq``.
        :type grad_v_seq: torch.Tensor
        :return: ``(backward_kernel, blocks, threads, py_dict)``

            ``backward_kernel``: ``NeuronBPTTKernel`` used in backward
            (from ``ctx.backward_kernel``).

            ``blocks``: ``int`` kernel launch parameter (from ``ctx.blocks``).

            ``threads``: ``int`` kernel launch parameter (from ``ctx.threads``).

            ``py_dict``: ``dict`` containing all tensor inputs/outputs for backward
            kernel execution.

        :rtype: tuple
        """
        backward_kernel = ctx.backward_kernel
        blocks = ctx.blocks
        threads = ctx.threads

        h_seq = ctx.saved_tensors[0]
        numel = ctx.numel
        N = ctx.N
        v_th = ctx.v_th
        v_reset = ctx.v_reset

        zero_shape = list(grad_spike_seq.shape)
        zero_shape[0] += 1
        zero_data = torch.zeros(
            zero_shape, device=grad_spike_seq.device, dtype=grad_spike_seq.dtype
        )
        grad_x_seq = zero_data[0:-1]
        grad_v_init = zero_data[-1]

        py_dict = {
            "numel": numel,
            "N": N,
            "grad_spike_seq": grad_spike_seq,
            "grad_v_seq": grad_v_seq,
            "h_seq": h_seq,
            "grad_x_seq": grad_x_seq,
            "grad_v_init": grad_v_init,
            "v_th": v_th,
            "v_reset": v_reset,
        }

        return backward_kernel, blocks, threads, py_dict
