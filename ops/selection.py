import importlib
import os
import threading
from functools import partial
from typing import Callable, NamedTuple, Optional

import torch

from spikingjelly.logger import logger

from .native_loader import _check_native_device


# Offline complete-call rankings; see benchmark/benchmark_neuron_implementations.py.
_NEURON_NAMESPACES = (
    "sj_activation_aware_if",
    "sj_eif",
    "sj_if",
    "sj_ilif",
    "sj_izhikevich",
    "sj_lif",
    "sj_plif",
    "sj_qif",
    "sj_stbif",
)
_DEFAULT_CUDA_PRIORITY = ("cuda", "triton", "torch")
_CUDA_PRIORITIES = {
    capability: dict.fromkeys(_NEURON_NAMESPACES, _DEFAULT_CUDA_PRIORITY)
    for capability in ((8, 0), (8, 6), (12, 0))
}
_COMPILE_CUDA_PRIORITIES = dict.fromkeys(
    _NEURON_NAMESPACES, ("triton", "cuda", "torch")
)


def _require_automatic_torch(
    selection: "_CudaSelection", device: torch.device, reason: str
) -> None:
    if device.type != "cuda":
        return
    if selection._requested != "auto":
        raise RuntimeError(
            f"{selection._environment_variable}={selection._requested!r} cannot "
            f"run this execution profile: {reason}."
        )
    # Diagnostics must not introduce graph breaks into the Torch reference path.
    if torch.compiler.is_compiling():
        return
    index = device.index
    if index is None:
        index = torch.cuda.current_device()
    key = (index, reason)
    if key in selection._selections:
        return
    with selection._lock:
        if key not in selection._selections:
            selection._selections[key] = None
            logger.info(
                "ops selection operator={} device=cuda:{} ({}) implementation=torch-reference profile=fallback reason={}",
                selection._namespace,
                index,
                torch.cuda.get_device_name(index),
                reason,
            )


def _require_provider(
    selection: "_CudaSelection",
    device: torch.device,
    implementation: str,
    reason: str,
) -> None:
    if device.type != "cuda":
        raise RuntimeError(f"{reason} requires CUDA.")
    if selection._requested not in ("auto", implementation):
        raise RuntimeError(
            f"{selection._environment_variable}={selection._requested!r} cannot "
            f"run this execution profile: {reason} requires {implementation}."
        )


class _CudaImplementation(NamedTuple):
    name: str
    trace_forward: Callable
    unavailable: dict[str, str]
    trace_backward: Optional[Callable]
    eager_forward: Callable
    eager_backward: Optional[Callable]
    cache_tag: str = ""
    unpack: Optional[Callable] = None


class _CudaSelection:
    def __init__(
        self,
        package: str,
        namespace: str,
        environment_variable: str,
        cpu_forward: Callable,
        cpu_backward: Optional[Callable] = None,
        on_select=None,
    ):
        self._on_select = on_select
        self._package = package
        self._namespace = namespace
        self._environment_variable = environment_variable
        self._requested = os.environ.get(environment_variable, "auto")
        self._cpu_forward = cpu_forward
        self._cpu_backward = cpu_backward
        self._selections = {}
        self._compiled_selections = {}
        self._lock = threading.Lock()

    def _select(self, index: int, *, execution: str = "eager"):
        if self._requested not in ("auto", *_DEFAULT_CUDA_PRIORITY):
            raise ValueError(
                f"{self._environment_variable} must be auto, cuda, triton, or torch; "
                f"got {self._requested!r}"
            )
        if torch.version.hip:
            raise RuntimeError(
                "SpikingJelly neuron CUDA implementations require NVIDIA CUDA"
            )
        capability = torch.cuda.get_device_capability(index)
        priority = _CUDA_PRIORITIES.get(capability, {}).get(
            self._namespace, _DEFAULT_CUDA_PRIORITY
        )
        if execution == "compile":
            priority = _COMPILE_CUDA_PRIORITIES.get(self._namespace, priority)
        unavailable = {}
        candidates = priority if self._requested == "auto" else (self._requested,)
        with torch.cuda.device(index):
            for name in candidates:
                module_name = "native" if name == "cuda" else name
                if name == "torch":
                    module_name = "cpu"
                try:
                    module = importlib.import_module(f".{module_name}", self._package)
                    if name == "cuda":
                        _check_native_device(module._build_info, index)
                except (ImportError, OSError) as error:
                    unavailable[name] = str(error)
                    logger.info("{} {} unavailable: {}", self._namespace, name, error)
                    continue
                if name == "triton" and hasattr(module, "_forward_impl"):
                    forward_impl = module._forward_impl
                    backward_impl = getattr(module, "_backward_impl", None)
                    trace_forward = partial(
                        forward_impl, _kernel_wrapper=torch.library.wrap_triton
                    )
                    trace_backward = (
                        partial(
                            backward_impl, _kernel_wrapper=torch.library.wrap_triton
                        )
                        if backward_impl is not None
                        else None
                    )
                elif name == "torch":
                    trace_forward = forward_impl = module._forward_impl
                    trace_backward = backward_impl = getattr(
                        module, "_backward_impl", None
                    )
                else:
                    trace_forward = getattr(
                        getattr(torch.ops, self._namespace), f"{module_name}_forward"
                    ).default
                    trace_backward = getattr(
                        getattr(torch.ops, self._namespace),
                        f"{module_name}_backward",
                        None,
                    )
                    trace_backward = (
                        trace_backward.default if trace_backward is not None else None
                    )
                    forward_impl = getattr(module, "_forward_impl", trace_forward)
                    backward_impl = getattr(module, "_backward_impl", trace_backward)
                selected = _CudaImplementation(
                    name,
                    trace_forward,
                    unavailable,
                    trace_backward,
                    forward_impl,
                    backward_impl,
                    unpack=getattr(module, "_unpack", None),
                )
                if self._on_select is not None:
                    selected = self._on_select(
                        self._namespace,
                        index,
                        module,
                        selected,
                        capability=capability,
                        priority=candidates,
                        execution=execution,
                    )
                return selected
        reasons = "; ".join(f"{name}: {reason}" for name, reason in unavailable.items())
        raise RuntimeError(
            f"No available {self._namespace} CUDA implementation. {reasons}"
        )

    def get_trace_forward(self, device: torch.device):
        if device.type == "cpu":
            return self._cpu_forward
        if device.type != "cuda":
            raise RuntimeError(
                f"SpikingJelly neurons support CPU and NVIDIA CUDA, not {device}"
            )
        return self._get_compile_selection(device).trace_forward

    def get_trace_backward(self, device: torch.device):
        if device.type == "cpu":
            return self._cpu_backward
        return self._get_compile_selection(device).trace_backward

    def get_cuda_forward(self, device: torch.device):
        return self._get_cuda_selection(device).eager_forward

    def get_cuda_backward(self, device: torch.device):
        return self._get_cuda_selection(device).eager_backward

    def _get_cuda_selection(self, device: torch.device):
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        key = index
        if key not in self._selections:
            with self._lock:
                if key not in self._selections:
                    self._selections[key] = self._select(index)
        return self._selections[key]

    def _get_compile_selection(self, device: torch.device):
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        if index not in self._compiled_selections:
            with self._lock:
                if index not in self._compiled_selections:
                    self._compiled_selections[index] = self._select(
                        index, execution="compile"
                    )
        return self._compiled_selections[index]

    def diagnostics(
        self, device: torch.device, *, execution: str = "eager"
    ) -> dict[str, object]:
        if device.type != "cuda":
            raise ValueError("neuron_implementation expects a CUDA device")
        if execution not in ("eager", "compile"):
            raise ValueError("execution must be eager or compile")
        selected = (
            self._get_cuda_selection(device)
            if execution == "eager"
            else self._get_compile_selection(device)
        )
        return {
            "implementation": selected.name,
            "unavailable": dict(selected.unavailable),
        }
