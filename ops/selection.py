import importlib
import os
import threading
from functools import partial
from typing import Callable, NamedTuple, Optional

import torch

from spikingjelly.logger import logger

from .native_loader import _check_native_device


class _CudaImplementation(NamedTuple):
    name: str
    trace_forward: Callable
    unavailable: dict[str, str]
    trace_backward: Optional[Callable]
    eager_forward: Callable
    eager_backward: Optional[Callable]
    cache_tag: str = ""


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
        self._lock = threading.Lock()

    def _select(self, index: int):
        priority = ("cuda", "triton", "cupy")
        if self._requested not in ("auto", *priority):
            raise ValueError(
                f"{self._environment_variable} must be auto, cuda, triton, or cupy; "
                f"got {self._requested!r}"
            )
        if torch.version.hip:
            raise RuntimeError(
                "Experimental neuron CUDA implementations require NVIDIA CUDA"
            )
        unavailable = {}
        candidates = priority if self._requested == "auto" else (self._requested,)
        with torch.cuda.device(index):
            for name in candidates:
                module_name = "native" if name == "cuda" else name
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
                )
                if self._on_select is not None:
                    selected = self._on_select(self._namespace, index, module, selected)
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
                f"Experimental neurons support CPU and NVIDIA CUDA, not {device}"
            )
        return self._get_cuda_selection(device).trace_forward

    def get_trace_backward(self, device: torch.device):
        if device.type == "cpu":
            return self._cpu_backward
        return self._get_cuda_selection(device).trace_backward

    def get_cuda_forward(self, device: torch.device):
        return self._get_cuda_selection(device).eager_forward

    def get_cuda_backward(self, device: torch.device):
        return self._get_cuda_selection(device).eager_backward

    def _get_cuda_selection(self, device: torch.device):
        index = device.index
        if index is None:
            index = torch.cuda.current_device()
        if index not in self._selections:
            with self._lock:
                if index not in self._selections:
                    self._selections[index] = self._select(index)
        return self._selections[index]

    def diagnostics(self, device: torch.device) -> dict[str, object]:
        if device.type != "cuda":
            raise ValueError("get_cuda_implementation expects a CUDA device")
        self.get_trace_forward(device)
        index = (
            device.index if device.index is not None else torch.cuda.current_device()
        )
        selected = self._selections[index]
        return {
            "implementation": selected.name,
            "unavailable": dict(selected.unavailable),
        }
