from typing import Tuple
import tempfile
from pathlib import Path
import hashlib

import torch
import torch.fx as fx

try:
    import triton
    import triton.language as tl
except BaseException as e:
    import logging
    from .. import dummy

    logging.info(
        f"spikingjelly.activation_based.triton_kernel.torch2triton.graph2triton: {e}"
    )
    triton = dummy.DummyTriton
    tl = None

from ..triton_utils import type_str_dict
from ..triton_utils import ensure_cleanup_tmp_python_files


__all__ = [
    "generate_triton_code_str",
    "compile_triton_code_str",
]


def _generate_hash(s: str, w: int = 8) -> str:
    hasher = hashlib.sha256(s.encode("utf-8"))
    return hasher.hexdigest()[:w]


def _uw(arg) -> str:
    """Unwrap an argument to its string representation for Triton code generation."""
    if isinstance(arg, fx.Node):
        return arg.name
    elif isinstance(arg, torch.dtype):
        return type_str_dict[arg]
    return str(arg)


# code generation rules
FX_TO_TRITON = {
    "add": lambda args, kwargs: f"{_uw(args[0])} + {_uw(args[1])}",
    "add.Scalar": lambda args, kwargs: f"{_uw(args[0])} + {_uw(args[1])}",
    "add.Tensor": lambda args, kwargs: f"{_uw(args[0])} + {_uw(args[1])}",
    "sub": lambda args, kwargs: f"{_uw(args[0])} - {_uw(args[1])}",
    "sub.Tensor": lambda args, kwargs: f"{_uw(args[0])} - {_uw(args[1])}",
    "rsub.Scalar": lambda args, kwargs: f"{_uw(args[1])} - {_uw(args[0])}",
    "mul": lambda args, kwargs: f"{_uw(args[0])} * {_uw(args[1])}",
    "mul.Tensor": lambda args, kwargs: f"{_uw(args[0])} * {_uw(args[1])}",
    "mul.Scalar": lambda args, kwargs: f"{_uw(args[0])} * {_uw(args[1])}",
    "div": lambda args, kwargs: f"{_uw(args[0])} / {_uw(args[1])}",
    "div.Tensor": lambda args, kwargs: f"{_uw(args[0])} / {_uw(args[1])}",
    "div.Scalar": lambda args, kwargs: f"{_uw(args[0])} / {_uw(args[1])}",
    "bitwise_and.Tensor": lambda args, kwargs: f"{_uw(args[0])} & {_uw(args[1])}",
    "bitwise_or.Tensor": lambda args, kwargs: f"{_uw(args[0])} | {_uw(args[1])}",
    "bitwise_not.default": lambda args, kwargs: f"~{_uw(args[0])}",
    "eq.Tensor": lambda args, kwargs: f"{_uw(args[0])} == {_uw(args[1])}",
    "eq.Scalar": lambda args, kwargs: f"{_uw(args[0])} == {_uw(args[1])}",
    "ge.Tensor": lambda args, kwargs: f"{_uw(args[0])} >= {_uw(args[1])}",
    "ge.Scalar": lambda args, kwargs: f"{_uw(args[0])} >= {_uw(args[1])}",
    "le.Tensor": lambda args, kwargs: f"{_uw(args[0])} <= {_uw(args[1])}",
    "le.Scalar": lambda args, kwargs: f"{_uw(args[0])} <= {_uw(args[1])}",
    "gt.Tensor": lambda args, kwargs: f"{_uw(args[0])} > {_uw(args[1])}",
    "gt.Scalar": lambda args, kwargs: f"{_uw(args[0])} > {_uw(args[1])}",
    "lt.Tensor": lambda args, kwargs: f"{_uw(args[0])} < {_uw(args[1])}",
    "lt.Scalar": lambda args, kwargs: f"{_uw(args[0])} < {_uw(args[1])}",
    "reciprocal.default":  # may result in change of dtype!!!
    lambda args, kwargs: f"(1. / {_uw(args[0])}).to({_uw(args[0])}.dtype)",
    "neg.default": lambda args, kwargs: f"-{_uw(args[0])}",
    "spike_fn.default": lambda args,
    kwargs: f"({_uw(args[0])} >= 0.).to({_uw(args[0])}.dtype)",
    "detach.default": lambda args, kwargs: f"{_uw(args[0])}",
    "sigmoid.default":  # triton does not support exponential operations on fp16
    lambda args,
    kwargs: f"tl.sigmoid({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "sigmoid_backward.default":  # args[1] is the output of sigmoid
    lambda args, kwargs: (f"{_uw(args[0])} * {_uw(args[1])} * (1 - {_uw(args[1])})"),
    "_to_copy.default": lambda args,
    kwargs: f"{_uw(args[0])}.to({_uw(kwargs['dtype'])})",
    "scalar_tensor.default": lambda args,
    kwargs: f"tl.full([], {_uw(args[0])}, {_uw(kwargs['dtype'])})",
    "where.self": lambda args,
    kwargs: f"tl.where({_uw(args[0])}.to(tl.int1), {_uw(args[1])}, {_uw(args[2])})",
}

INDENTATION = " " * 4  # four spaces


def generate_triton_code_str(
    graph: fx.Graph,
    fn_name: str,
    verbose: bool = False,
) -> Tuple[str, str]:
    """Given a fx.Graph, generate its corresponding Triton code string.

    Args:
        graph (fx.Graph)
        fn_name (str): name of the original PyTorch function. For generating the Triton kernel name.
        verbose (bool, optional): Defaults to False.

    Returns:
        Tuple[str, str]: the generated Triton code string and the name of the Triton function.
    """
    if verbose:
        print(graph)

    inputs = []
    triton_code_lines = []
    for node in graph.nodes:
        if node.op == "placeholder":  # function inputs
            inputs.append(node.name)
        elif node.op in ["call_function", "call_method"]:
            op_name = (
                node.target.__name__ if node.op == "call_function" else node.target
            )  # e.g. mul.Tensor, spike_fn.default, rsub.Scalar, ...
            if op_name in FX_TO_TRITON:  # apply the transpile rule
                rhs = FX_TO_TRITON[op_name](node.args, node.kwargs)
                triton_code_lines.append(f"{node.name} = {rhs}")
            else:
                raise NotImplementedError(
                    f"{node.op} {op_name} has not yet been implemented "
                    f"in FX_TO_TRITON mapping."
                )
        elif node.op == "output":
            if isinstance(node.args[0], fx.Node):
                # only one return value
                things = node.args[0].name
            else:
                # multiple return values
                things = ", ".join(arg.name for arg in node.args[0])
            triton_code_lines.append(f"return {things}")
        else:
            raise NotImplementedError(
                f"Operation {node.op} has not yet been implemented."
            )

    triton_code_lines = f"{INDENTATION}" + f"\n{INDENTATION}".join(triton_code_lines)
    fn_name = f"{fn_name}_{_generate_hash(triton_code_lines)}"
    signature = ", ".join(inputs)
    signature = f"@triton.jit\ndef {fn_name}({signature}):"
    prefix = "import triton\nimport triton.language as tl"
    return f"{prefix}\n\n{signature}\n{triton_code_lines}", fn_name


@ensure_cleanup_tmp_python_files
def compile_triton_code_str(
    triton_code: str,
    kernel_name: str,
    verbose: bool = False,
    name_space: dict = {},
):
    """Compile a Triton code string into a runnable Triton JIT function.

    Writes the Triton code to a temporary file, compiles it, and extracts the
    JIT function from the compiled namespace.

    Args:
        triton_code (str): The Triton code string to compile.
        kernel_name (str): The name of the Triton function to extract.
        verbose (bool, optional): If True, print the path to the temporary file. Defaults to False.
        name_space (dict, optional): A namespace dictionary to use for `exec`. Defaults to {}.

    Returns:
        triton.JITFunction: The compiled Triton JIT function.
    """
    # create a temporary file
    with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
        f.write(triton_code)
        fpath = Path(f.name)
        if verbose:
            print(f"Triton code `{kernel_name}` written to {fpath}")
        # the file will not be deleted until the end of the program

    name_space.update(
        {
            "triton": triton,
            "tl": tl,
            "__name__": "spikingjelly.activation_based.triton_kernel.codegen",
        }
    )
    with open(fpath, "r") as f:
        code = compile(f.read(), fpath, "exec")
        exec(code, name_space)  # name_space will be updated

    if kernel_name in name_space:
        return name_space[kernel_name]
    else:
        raise ValueError(f"Function {kernel_name} not found in compiled namespace")
