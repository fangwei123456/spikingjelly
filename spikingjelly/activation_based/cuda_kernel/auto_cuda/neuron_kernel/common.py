from typing import Callable, Iterable
import torch
import numpy as np
import logging
import math
import os
import threading

try:
    import cupy
except BaseException as e:
    logging.info(
        f"spikingjelly.activation_based.cuda_kernel.auto_cuda.neuronal_kernel: {e}"
    )
    cupy = None


from ... import cuda_utils
from ..... import surrogate
from ..... import configure
from .. import base, cfunction

try:
    _CUPY_CUSTOM_OP_AVAILABLE = all(
        hasattr(torch.library, name)
        for name in ("custom_op", "register_fake", "register_autograd")
    )
except BaseException:
    _CUPY_CUSTOM_OP_AVAILABLE = False


def _env_flag_enabled(var_name: str) -> bool:
    v = os.getenv(var_name)
    if v is None:
        return True
    return v.strip().lower() not in ("0", "false", "off", "no")


def _use_cupy_custom_op() -> bool:
    return (
        _CUPY_CUSTOM_OP_AVAILABLE
        and cupy is not None
        and _env_flag_enabled("SJ_USE_CUPY_OP")
    )


_SURROGATE_CUPY_REGISTRY_LOCK = threading.Lock()
_SURROGATE_CUPY_NEXT_ID = 0
_SURROGATE_CUPY_ID_TO_CODES: dict[int, Callable[[str, str, str], str]] = {}
_SURROGATE_CUPY_KEY_TO_ID: dict[str, int] = {}


def _normalize_sg_value(value):
    if isinstance(value, torch.Tensor):
        if value.numel() == 1:
            return value.item()
        return tuple(value.detach().cpu().reshape(-1).tolist())
    if isinstance(value, (bool, int, float, str)):
        return value
    return repr(value)


def _surrogate_registry_key(
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> str:
    param_items = ()
    if hasattr(surrogate_function, "_sg_params"):
        params = surrogate_function._sg_params
        param_items = tuple(
            sorted((k, _normalize_sg_value(v)) for k, v in params.items())
        )
    return repr(
        (
            surrogate_function.__class__.__module__,
            surrogate_function.__class__.__qualname__,
            _normalize_sg_value(getattr(surrogate_function, "spiking", True)),
            param_items,
        )
    )


def resolve_sg_cupy_id_and_key(
    surrogate_function: surrogate.SurrogateFunctionBase,
) -> tuple[int, str]:
    if not hasattr(surrogate_function, "cuda_codes"):
        raise TypeError(
            "CuPy backend requires surrogate_function.cuda_codes for custom_op path."
        )

    sg_codes = surrogate_function.cuda_codes
    if not callable(sg_codes):
        raise TypeError(
            "surrogate_function.cuda_codes must be callable for CuPy custom_op path."
        )

    sg_key = _surrogate_registry_key(surrogate_function)

    global _SURROGATE_CUPY_NEXT_ID
    with _SURROGATE_CUPY_REGISTRY_LOCK:
        sg_id = _SURROGATE_CUPY_KEY_TO_ID.get(sg_key)
        if sg_id is None:
            sg_id = _SURROGATE_CUPY_NEXT_ID
            _SURROGATE_CUPY_NEXT_ID += 1
            _SURROGATE_CUPY_KEY_TO_ID[sg_key] = sg_id
            _SURROGATE_CUPY_ID_TO_CODES[sg_id] = sg_codes
    return sg_id, sg_key


def _dtype_to_cupy_kernel_dtype(dtype: torch.dtype) -> str:
    if dtype == torch.float32:
        return "float"
    if dtype == torch.float16:
        return "half2"
    raise NotImplementedError(dtype)


def _surrogate_cuda_codes_from_id(sg_cupy_id: int) -> Callable[[str, str, str], str]:
    sg_codes = _SURROGATE_CUPY_ID_TO_CODES.get(sg_cupy_id)
    if sg_codes is None:
        raise RuntimeError(
            f"Unknown sg_cupy_id={sg_cupy_id} in custom_op backward. "
            "This usually means surrogate registry was not initialized before execution."
        )
    return sg_codes


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


class NeuronFPTTKernel(base.CKernel2D):
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
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`H[t] = f(X[t], V[t-1], ...)`.

        This function should define how ``h_seq[t]`` is calculated by ``x_seq[t], v_v_seq[t]`` and other params if
        the neuron needs.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def neuronal_charge(self) -> str:
                # note that v_v_seq[t] is v_seq[t - dt]
                return cfunction.add(
                    z="h_seq[t]", x="x_seq[t]", y="v_v_seq[t]", dtype=self.dtype
                )
        """
        return "// neuronal_charge should be defined here!"

    @property
    def core(self):
        core_codes = base.CodeTyper(18)

        core_codes.append(self.neuronal_charge())

        core_codes.append(
            neuronal_fire(
                spike="spike_seq[t]", v="h_seq[t]", v_th="v_th", dtype=self.dtype
            )
        )

        if self.hard_reset:
            core_codes.append(
                neuronal_hard_reset(
                    v_next="v_v_seq[t + dt]",
                    h="h_seq[t]",
                    spike="spike_seq[t]",
                    v_reset="v_reset",
                    dtype=self.dtype,
                )
            )
        else:
            core_codes.append(
                neuronal_soft_reset(
                    v_next="v_v_seq[t + dt]",
                    h="h_seq[t]",
                    spike="spike_seq[t]",
                    v_th="v_th",
                    dtype=self.dtype,
                )
            )

        self._core = core_codes.codes
        return self._core


class NeuronBPTTKernel(base.CKernel2D):
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
        codes = base.CodeTyper(16)
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
        codes = base.CodeTyper(16)
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
        """
        :return: CUDA code
        :rtype: str

        Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t+1]}{\\mathrm{d} V[t]}`.

        This function should define how ``grad_h_next_to_v`` is calculated. Note that ``grad_h_next_to_v`` has not been
        declared. Thus, this function should also declare ``grad_h_next_to_v``.

        For example, the IF neuron define this function as:

        .. code-block:: python

            def grad_h_next_to_v(self) -> str:
                return cfunction.constant(
                    y=f"const {self.dtype} grad_h_next_to_v", x=1.0, dtype=self.dtype
                )
        """
        return "// grad_h_next_to_v should be defined here!"

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
                with base.CodeBlock(core_codes):
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
        with base.CodeBlock(core_codes):
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
    requires_grad = False
    for item in items:
        if isinstance(item, torch.Tensor):
            if item.requires_grad:
                requires_grad = True
                break

    return requires_grad


def scalar_to_cupy(py_dict: dict, ref: str = "x_seq"):
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
                py_dict[key] = cupy.asarray(value)


def new_tensors(news: tuple, py_dict: dict, ref: str = "x_seq"):
    ref = py_dict[ref]
    zero_shape = list(ref.shape)
    zero_shape[0] *= news.__len__()
    for i, item in enumerate(
        torch.split(
            torch.zeros(zero_shape, device=ref.device, dtype=ref.dtype), ref.shape[0]
        )
    ):
        py_dict[news[i]] = item


class NeuronATGFBase:
    @staticmethod
    def pre_forward(py_dict: dict):
        """
        :param py_dict: a dict built from the neuron's forward autograd function. It should at least contain ``x_seq, v_init, v_reset``
        :type py_dict: dict
        :return: requires_grad, blocks, threads, py_dict

            requires_grad: bool
                if any tensor in ``py_dict`` requires grad, then ``requires_grad = True``;else ``requires_grad = False``

            blocks: int
                CUDA param used in calling CUDA kernel

            threads: int
                CUDA param used in calling CUDA kernel. The default value is ``spikingjelly.configure.cuda_threads``

            py_dict: dict
                Compared with the input ``py_dict``, the returned ``py_dict`` will:

                    * convert all ``float/int`` scalars in ``py_dict`` to ``cupy.ndarray``

                    * add ``h_seq, spike_seq, v_v_seq`` to ``py_dict``. ``h_seq, spike_seq`` are zero tensors
                      with the same shape with ``x_seq``. ``v_v_seq`` is concatenated from ``v_init`` and
                      ``v_seq``, which is zero tensors with the same shape with ``x_seq``

                    * add ``N, numel`` to ``py_dict``. Note that ``x_seq.shape = [T, N]`` and ``numel = T * N``.
                      A specific case is that ``x_seq.dtype == torch.half``, then ``N = math.ceil(N / 2)``, and
                      ``numel = N * x_seq.shape[0]``.
                      Note that ``N, numel`` in the returned ``py_dict`` are ``cupy.ndarray``


        :rtype: tuple
        """
        device = py_dict["x_seq"].get_device()
        requires_grad = if_requires_grad(py_dict.values())
        scalar_to_cupy(py_dict)

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
            numel = cupy.asarray(numel)
            N = cupy.asarray(N)

        py_dict["numel"] = numel
        py_dict["N"] = N

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
    def pre_backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):
        """
        :param ctx: ``ctx`` in :class:`torch.autograd.Function`
        :param grad_spike_seq: gradients of ``spike_seq``
        :type grad_spike_seq: torch.Tensor
        :param grad_v_seq: gradients of ``v_seq``
        :type grad_v_seq: torch.Tensor
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
