from typing import Optional, Tuple
import errno
import importlib.util
import linecache
import os
from pathlib import Path
import hashlib
import stat
import sys
import tempfile
import threading
import types

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
    triton = dummy.DummyImport()
    tl = dummy.DummyImport()

from ..triton_utils import type_str_dict


__all__ = [
    "generate_triton_code_str",
    "compile_triton_code_str",
]


_MODULE_CACHE_LOCK_GUARD = threading.Lock()
_MODULE_CACHE_LOCKS = {}
_CODEGEN_CACHE_DIR = None
_CODEGEN_CACHE_DIR_LOCK = threading.Lock()
_NAMESPACE_METADATA_KEYS = {
    "__name__",
    "__spec__",
    "__loader__",
    "__package__",
    "__file__",
    "__cached__",
    "__builtins__",
    "__doc__",
}


def _get_module_cache_lock(module_name: str) -> threading.Lock:
    with _MODULE_CACHE_LOCK_GUARD:
        return _MODULE_CACHE_LOCKS.setdefault(module_name, threading.Lock())


def _generate_hash(s: str, w: int = 8) -> str:
    hasher = hashlib.sha256(s.encode("utf-8"))
    return hasher.hexdigest()[:w]


def _has_real_triton_runtime() -> bool:
    return isinstance(triton, types.ModuleType) and isinstance(tl, types.ModuleType)


def _codegen_cache_dir() -> Path:
    global _CODEGEN_CACHE_DIR
    if _CODEGEN_CACHE_DIR is not None:
        return _CODEGEN_CACHE_DIR
    with _CODEGEN_CACHE_DIR_LOCK:
        if _CODEGEN_CACHE_DIR is not None:
            return _CODEGEN_CACHE_DIR
        cache_dir = _resolve_codegen_cache_dir()
        _CODEGEN_CACHE_DIR = cache_dir
        return cache_dir


def _resolve_codegen_cache_dir() -> Path:
    candidates = []
    uid = getattr(os, "getuid", lambda: None)()
    try:
        candidates.append(Path.home() / ".spikingjelly" / "triton_codegen")
    except RuntimeError:
        pass
    temp_suffix = f"_{uid}" if uid is not None else ""
    candidates.append(
        Path(tempfile.gettempdir()) / f"spikingjelly_triton_codegen{temp_suffix}"
    )
    last_error = None
    for cache_dir in candidates:
        try:
            cache_dir.mkdir(parents=True, exist_ok=True, mode=0o700)
            if uid is not None:
                st = cache_dir.stat()
                if st.st_uid == uid:
                    os.chmod(cache_dir, 0o700)
                    st = cache_dir.stat()
                mode = stat.S_IMODE(st.st_mode)
                if (
                    st.st_uid != uid
                    or not (mode & stat.S_IWUSR)
                    or (mode & 0o077)
                ):
                    continue
            with tempfile.NamedTemporaryFile(dir=cache_dir, delete=True):
                pass
            return cache_dir
        except OSError as e:
            last_error = e
            if e.errno not in (errno.EACCES, errno.EROFS, errno.EPERM):
                raise
    if last_error is not None:
        raise last_error
    raise RuntimeError("Failed to initialize Triton codegen cache directory")


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
    "add.Tensor": lambda args, kwargs: (
        f"{_uw(args[0])} + {_uw(args[1])}"
        if kwargs.get("alpha", 1.0) == 1.0
        else f"{_uw(args[0])} + ({kwargs['alpha']} * {_uw(args[1])})"
    ),
    "sub": lambda args, kwargs: f"{_uw(args[0])} - {_uw(args[1])}",
    "sub.Tensor": lambda args, kwargs: (
        f"{_uw(args[0])} - {_uw(args[1])}"
        if kwargs.get("alpha", 1.0) == 1.0
        else f"{_uw(args[0])} - ({kwargs['alpha']} * {_uw(args[1])})"
    ),
    "sub.Scalar": lambda args, kwargs: f"{_uw(args[0])} - {_uw(args[1])}",
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
    # logical_* follow ATen truthiness: non-zero = True; bitwise ops would give
    # wrong results for numeric inputs (e.g. logical_not(2) → False, but ~2 = -3)
    "logical_and.default": lambda args,
    kwargs: f"({_uw(args[0])} != 0) & ({_uw(args[1])} != 0)",
    "logical_or.default": lambda args,
    kwargs: f"({_uw(args[0])} != 0) | ({_uw(args[1])} != 0)",
    "logical_not.default": lambda args, kwargs: f"({_uw(args[0])} == 0)",
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
    "tanh_backward.default":  # args[0]=grad_out, args[1]=tanh_output
    lambda args, kwargs: f"{_uw(args[0])} * (1 - {_uw(args[1])} * {_uw(args[1])})",
    "threshold_backward.default":  # args: grad, input, threshold
    lambda args,
    kwargs: f"tl.where({_uw(args[1])} > {_uw(args[2])}, {_uw(args[0])}, 0.0)",
    "_to_copy.default": lambda args,
    kwargs: f"{_uw(args[0])}.to({_uw(kwargs['dtype'])})",
    "scalar_tensor.default": lambda args,
    kwargs: f"tl.full([], {_uw(args[0])}, {_uw(kwargs['dtype'])})",
    "where.self": lambda args,
    kwargs: f"tl.where({_uw(args[0])}.to(tl.int1), {_uw(args[1])}, {_uw(args[2])})",
    # ---------- unary math (upcast fp16→fp32 for transcendentals) ----------
    "exp.default": lambda args,
    kwargs: f"tl.exp({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "log.default": lambda args,
    kwargs: f"tl.log({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "log2.default": lambda args,
    kwargs: f"tl.log2({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "sqrt.default": lambda args,
    kwargs: f"tl.sqrt({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "rsqrt.default": lambda args,
    kwargs: f"tl.rsqrt({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "abs.default": lambda args, kwargs: f"tl.abs({_uw(args[0])})",
    "tanh.default": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.tanh("
        f"{_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    "sin.default": lambda args,
    kwargs: f"tl.math.sin({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "cos.default": lambda args,
    kwargs: f"tl.math.cos({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    "erf.default": lambda args,
    kwargs: f"tl.math.erf({_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)",
    # ---------- rounding ----------
    "floor.default": lambda args, kwargs: f"tl.floor({_uw(args[0])})",
    "ceil.default": lambda args, kwargs: f"tl.ceil({_uw(args[0])})",
    "round.default": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.round("
        f"{_uw(args[0])}.to(tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    # ---------- activation ----------
    "relu.default": lambda args, kwargs: f"tl.maximum({_uw(args[0])}, 0.0)",
    "sign.default": lambda args,
    kwargs: (
        f"({_uw(args[0])} > 0.).to({_uw(args[0])}.dtype)"
        f" - ({_uw(args[0])} < 0.).to({_uw(args[0])}.dtype)"
    ),
    "sgn.default": lambda args,  # complex sign; for real tensors same as sign
    kwargs: (
        f"({_uw(args[0])} > 0.).to({_uw(args[0])}.dtype)"
        f" - ({_uw(args[0])} < 0.).to({_uw(args[0])}.dtype)"
    ),
    # ---------- binary element-wise ----------
    "minimum.default": lambda args, kwargs: f"tl.minimum({_uw(args[0])}, {_uw(args[1])})",
    "maximum.default": lambda args, kwargs: f"tl.maximum({_uw(args[0])}, {_uw(args[1])})",
    "ne.Scalar": lambda args, kwargs: f"{_uw(args[0])} != {_uw(args[1])}",
    "ne.Tensor": lambda args, kwargs: f"{_uw(args[0])} != {_uw(args[1])}",
    "fmod.Scalar": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.fmod("
        f"{_uw(args[0])}.to(tl.float32),"
        f" tl.full([], {_uw(args[1])}, tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    "fmod.Tensor": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.fmod("
        f"{_uw(args[0])}.to(tl.float32),"
        f" {_uw(args[1])}.to(tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    "pow.Tensor_Scalar": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.pow("
        f"{_uw(args[0])}.to(tl.float32),"
        f" tl.full([], {_uw(args[1])}, tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    "pow.Tensor_Tensor": lambda args,
    kwargs: (
        f"tl.extra.cuda.libdevice.pow("
        f"{_uw(args[0])}.to(tl.float32),"
        f" {_uw(args[1])}.to(tl.float32)).to({_uw(args[0])}.dtype)"
    ),
    # ---------- clamp ----------
    "clamp.default": lambda args,
    kwargs: (
        # args: (tensor, min_val, max_val) — both optional
        f"tl.minimum(tl.maximum({_uw(args[0])}, {_uw(args[1])}), {_uw(args[2])})"
        if len(args) >= 3 and args[1] is not None and args[2] is not None
        else f"tl.maximum({_uw(args[0])}, {_uw(args[1])})"
        if len(args) >= 2 and args[1] is not None
        else f"tl.minimum({_uw(args[0])}, {_uw(args[2])})"
        if len(args) >= 3 and args[2] is not None
        else _uw(args[0])
    ),
    "clamp_min.default": lambda args, kwargs: f"tl.maximum({_uw(args[0])}, {_uw(args[1])})",
    "clamp_max.default": lambda args, kwargs: f"tl.minimum({_uw(args[0])}, {_uw(args[1])})",
    # ---------- misc ----------
    "clone.default": lambda args, kwargs: f"{_uw(args[0])}",
    # Use tl.full to avoid propagating NaN/Inf from input values
    "zeros_like.default": lambda args,
    kwargs: f"tl.full({_uw(args[0])}.shape, 0, {_uw(args[0])}.dtype)",
    "ones_like.default": lambda args,
    kwargs: f"tl.full({_uw(args[0])}.shape, 1, {_uw(args[0])}.dtype)",
    # masked_fill(tensor, mask, value): fill where mask=True with value
    "masked_fill.Scalar": lambda args,
    kwargs: f"tl.where({_uw(args[1])}.to(tl.int1), {_uw(args[2])}, {_uw(args[0])})",
    "masked_fill.Tensor": lambda args,
    kwargs: f"tl.where({_uw(args[1])}.to(tl.int1), {_uw(args[2])}, {_uw(args[0])})",
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


def compile_triton_code_str(
    triton_code: str,
    kernel_name: str,
    verbose: bool = False,
    name_space: Optional[dict] = None,
):
    """Compile a Triton code string into a runnable Triton JIT function.

    Materializes the Triton code under the persistent codegen cache, loads or
    reuses the matching module object, and extracts the requested JIT function.

    Args:
        triton_code (str): The Triton code string to compile/cache.
        kernel_name (str): The name of the Triton function to extract.
        verbose (bool, optional): If True, print whether the cached source was
            written or reused, along with its path. Defaults to False.
        name_space (Optional[dict], optional): Optional globals injected before execution.
            When provided, it will be updated with symbols defined by the compiled module.
            Calls without ``name_space`` reuse a cached module keyed by the generated
            source hash; calls with ``name_space`` reload so injected symbols stay fresh.

    Returns:
        triton.JITFunction: The compiled Triton JIT function.
    """
    if not _has_real_triton_runtime():
        raise ImportError(
            "compile_triton_code_str requires a real Triton installation; "
            "the imported triton/tl modules are unavailable."
        )

    caller_namespace = name_space
    cacheable = caller_namespace is None
    if caller_namespace is None:
        module_globals = {"triton": triton, "tl": tl}
    else:
        module_globals = {
            key: value
            for key, value in caller_namespace.items()
            if key not in _NAMESPACE_METADATA_KEYS
        }
        module_globals.setdefault("triton", triton)
        module_globals.setdefault("tl", tl)

    module_hash = _generate_hash(f"{kernel_name}\n{triton_code}", w=16)
    module_name = (
        "spikingjelly.activation_based.triton_kernel.codegen."
        f"{kernel_name}_{module_hash}"
    )
    fpath = _codegen_cache_dir() / f"{kernel_name}_{module_hash}.py"

    needs_write = not fpath.exists()
    if needs_write:
        tmp_path = None
        try:
            with tempfile.NamedTemporaryFile(
                "w", encoding="utf-8", dir=fpath.parent, delete=False, suffix=".tmp"
            ) as tmp_file:
                tmp_path = Path(tmp_file.name)
                tmp_file.write(triton_code)
            os.replace(tmp_path, fpath)
            try:
                os.chmod(fpath, 0o600)
            except OSError:
                pass
        except Exception:
            if tmp_path is not None:
                try:
                    tmp_path.unlink()
                except FileNotFoundError:
                    pass
            raise
    if verbose:
        action = "written to" if needs_write else "loaded from cache"
        print(f"Triton code `{kernel_name}` {action} {fpath}")

    linecache.checkcache(str(fpath))

    with _get_module_cache_lock(module_name):
        module = sys.modules.get(module_name) if cacheable else None
        if module is None:
            spec = importlib.util.spec_from_file_location(module_name, fpath)
            if spec is None or spec.loader is None:
                raise ImportError(f"Could not create import spec for {fpath}")
            module = importlib.util.module_from_spec(spec)
            module.__dict__.update(module_globals)
            spec.loader.exec_module(module)
            if cacheable:
                sys.modules[module_name] = module
    if caller_namespace is not None:
        exported_symbols = {
            key: value
            for key, value in module.__dict__.items()
            if key not in _NAMESPACE_METADATA_KEYS
        }
        caller_namespace.update(exported_symbols)
    if kernel_name in module.__dict__:
        return module.__dict__[kernel_name]
    raise ValueError(f"Function {kernel_name} not found in compiled namespace")
