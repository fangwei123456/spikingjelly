"""Strided addressing for the existing point-neuron CUDA formulas."""

import math
import re
from functools import lru_cache

import torch

from .... import configure
from ..._neuron_layout import _layout_args
from .. import cuda_utils
from ..auto_cuda.base import CKernel2D, _get_raw_kernel


@lru_cache(maxsize=512)
def _strided_code(code, name, sizes, layouts):
    # Only pointer parameters belonging to neuron arrays are rewritten. Scalars
    # and parameter-gradient reductions retain their original CUDA types.
    count = math.prod(sizes)
    dense_strides = []
    span = 1
    for size in sizes:
        dense_strides.append(span)
        span *= size
    if all(
        strides[0] == (count if is_sequence else 0)
        and all(
            size == 1 or stride == dense
            for size, stride, dense in zip(sizes, strides[1:], dense_strides)
        )
        and (not half or (count % 2 == 0 and aligned))
        for _, strides, half, aligned, is_sequence in layouts
    ):
        return code
    signature = re.search(r"void\s+" + name + r"\s*\((.*?)\)\s*\{", code, re.S)
    parameters = signature.group(1)
    declarations = []
    definitions = []
    for tensor_name, strides, half, _, _ in layouts:
        parameter = re.search(
            r"(const\s+)?(float|half2)\s*\*\s*" + tensor_name + r"\b", parameters
        )
        if parameter is None:
            raise ValueError(
                f"{name}: expected a float* or half2* neuron array for {tensor_name}"
            )
        readonly = bool(parameter.group(1))
        array_type = "SJ_" + tensor_name
        packed_count = (count + 1) // 2 if half else count
        expression = f"(i / {packed_count}) * {strides[0]}LL"
        if all(
            size == 1 or stride == dense
            for size, stride, dense in zip(sizes, strides[1:], dense_strides)
        ):
            expression += " + n"
        else:
            divisor = 1
            for size, stride in zip(sizes, strides[1:]):
                expression += f" + ((n / {divisor}) % {size}) * {stride}LL"
                divisor *= size
        definitions.append(
            f"struct {array_type} {{\n"
            f"  {'const ' if readonly else ''}{'half' if half else 'float'}* p;\n"
            f"  __device__ __forceinline__ long long address(unsigned int i, unsigned int n) const {{ return {expression}; }}\n"
        )
        if half:
            # The packed lane number is independent of physical adjacency.
            # Kernel indices are nonnegative int32; unpacked half lanes can use
            # the full uint32 range. Address products retain their LL suffix.
            definitions.append(
                f"  __device__ __forceinline__ {'half2' if readonly else 'SJHalfRef'} operator[](unsigned int i) const {{\n"
                f"    unsigned int n = (i % {packed_count}) * 2;\n"
                "    auto a = p + address(i, n);\n"
                f"    auto b = n + 1 < {count} ? p + address(i, n + 1) : nullptr;\n"
            )
            if readonly:
                definitions.append(
                    "    if (b == a + 1 && (reinterpret_cast<unsigned long long>(a) & 3) == 0) return *reinterpret_cast<const half2*>(a);\n"
                    "    return __halves2half2(*a, b ? *b : __float2half(0));\n"
                )
            else:
                definitions.append("    return {a, b};\n")
            definitions.append("  }\n};\n")
        else:
            definitions.append(
                f"  __device__ __forceinline__ {'const ' if readonly else ''}float& operator[](unsigned int i) const {{ return p[address(i, i % {count})]; }}\n}};\n"
            )
        parameters = re.sub(
            r"\b" + tensor_name + r"\b", "raw_" + tensor_name, parameters
        )
        declarations.append(
            f"{array_type} {tensor_name}{{reinterpret_cast<{'const ' if readonly else ''}{'half' if half else 'float'}*>(raw_{tensor_name})}};"
        )
    prefix = r"""
#include <cuda_fp16.h>
struct SJHalfRef {
    half* a;
    half* b;
    __device__ __forceinline__ operator half2() const {
        if (b == a + 1 && (reinterpret_cast<unsigned long long>(a) & 3) == 0)
            return *reinterpret_cast<const half2*>(a);
        return __halves2half2(*a, b ? *b : __float2half(0));
    }
    __device__ __forceinline__ void operator=(half2 value) const {
        if (b == a + 1 && (reinterpret_cast<unsigned long long>(a) & 3) == 0) {
            *reinterpret_cast<half2*>(a) = value;
        } else {
            *a = __low2half(value);
            if (b) *b = __high2half(value);
        }
    }
    __device__ __forceinline__ void operator=(const SJHalfRef& value) const {
        operator=(static_cast<half2>(value));
    }
};
"""
    return (
        prefix
        + "".join(definitions)
        + code[: signature.start(1)]
        + parameters
        + code[signature.end(1) : signature.end()]
        + "\n".join(declarations)
        + code[signature.end() :]
    )


@lru_cache(maxsize=128)
def _parameter_names(code, name):
    signature = re.search(r"void\s+" + name + r"\s*\((.*?)\)", code, re.S)
    return tuple(
        re.findall(r"\w+", parameter)[-1] for parameter in signature.group(1).split(",")
    )


@lru_cache(maxsize=128)
def _initialize_states(code, name, state_names, count, half):
    signature = re.search(r"void\s+" + name + r"\s*\((.*?)\)\s*\{", code, re.S)
    ctype = "half2" if half else "float"
    parameters = signature.group(1) + "".join(
        f", const {ctype}* {state}" for state in state_names
    )
    lanes = (count + 1) // 2 if half else count
    initialization = f"\nconst int initial_index = blockIdx.x * blockDim.x + threadIdx.x;\nif (initial_index < {lanes}) {{\n"
    for state in state_names:
        buffer = state[0] + "_" + state[0] + "_seq"
        initialization += f"{buffer}[initial_index] = {state}[initial_index];\n"
    return (
        code[: signature.start(1)]
        + parameters
        + code[signature.end(1) : signature.end()]
        + initialization
        + "}\n"
        + code[signature.end() :]
    )


def _launch_strided(
    code,
    name,
    grid,
    block,
    arguments,
    *,
    sequence=True,
    initial_states=None,
):
    # The CUDA signature owns parameter order, regardless of dictionary order.
    arguments = {n: arguments[n] for n in _parameter_names(code, name)}
    reference = next(
        arguments[n] for n in ("x_seq", "h_seq", "x", "h") if n in arguments
    )
    if initial_states:
        initial_states = {
            n: x.to(dtype=reference.dtype) for n, x in initial_states.items()
        }
        code = _initialize_states(
            code,
            name,
            tuple(initial_states),
            math.prod(reference.shape[1:]),
            reference.dtype == torch.float16,
        )
        arguments.update(initial_states)
    arrays = [
        (n, x)
        for n, x in arguments.items()
        if isinstance(x, torch.Tensor) and n not in ("decay", "grad_decay")
    ]
    if any(x.device != reference.device for _, x in arrays):
        raise ValueError("Point-neuron tensors must be on the same CUDA device.")
    count = math.prod(reference.shape[1:]) if sequence else reference.numel()
    dense = all(x.is_contiguous() for _, x in arrays)
    if reference.dtype == torch.float16:
        dense = (
            dense and count % 2 == 0 and all(x.data_ptr() % 4 == 0 for _, x in arrays)
        )
    if not dense:
        sizes, strides = _layout_args(
            reference, *(x for _, x in arrays), sequence=sequence
        )
        layouts = tuple(
            (
                n,
                stride,
                x.dtype == torch.float16,
                x.data_ptr() % 4 == 0,
                sequence and x.ndim == reference.ndim,
            )
            for (n, x), stride in zip(arrays, strides)
        )
        code = _strided_code(code, name, sizes, layouts)
    kernel = _get_raw_kernel(
        code,
        name,
        tuple(configure.cuda_compiler_options),
        configure.cuda_compiler_backend,
    )
    with cuda_utils.DeviceEnvironment(reference.get_device()):
        kernel(
            grid,
            block,
            tuple(
                x.data_ptr() if isinstance(x, torch.Tensor) else x
                for x in arguments.values()
            ),
        )


def _launch_generated(kernel, grid, block, py_dict):
    sequence = isinstance(kernel, CKernel2D)
    initial_states = {n: py_dict[n] for n in ("v_init", "w_init") if n in py_dict}
    py_dict = {n: value for n, value in py_dict.items() if n not in initial_states}
    device = kernel.get_device(py_dict)
    kernel.check_device(device, py_dict)
    kernel.check_ctypes(py_dict)
    kernel.check_keys(py_dict)
    _launch_strided(
        kernel.full_codes,
        kernel.kernel_name,
        grid,
        block,
        py_dict,
        sequence=sequence,
        initial_states=initial_states,
    )
