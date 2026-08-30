from __future__ import annotations

import time
from collections.abc import Callable
from dataclasses import dataclass

import torch


def validate_cuda_graph_model(model: torch.nn.Module) -> None:
    for module in model.modules():
        if getattr(module, "backend", None) == "cupy":
            raise ValueError("CUDA Graph does not support the CuPy neuron backend.")
        if getattr(module, "store_v_seq", False):
            raise ValueError("CUDA Graph requires store_v_seq=False.")


@dataclass
class CudaGraphStats:
    warmup_runs: int = 0
    captures: int = 0
    replays: int = 0
    eager_fallbacks: int = 0
    capture_seconds: float = 0.0
    graph_memory_bytes: int = 0


class StaticCudaGraph:
    def __init__(
        self,
        function: Callable,
        warmup_steps: int = 3,
    ) -> None:
        if warmup_steps <= 0:
            raise ValueError("warmup_steps must be positive.")
        self.function = function
        self.warmup_steps = warmup_steps
        self.stats = CudaGraphStats()
        self._graph = None
        self._capture_stream = None
        self._inputs: tuple[torch.Tensor, ...] = ()
        self._input_metadata = ()

    def _warmup(self, inputs: tuple[torch.Tensor, ...]):
        device = inputs[0].device
        current_stream = torch.cuda.current_stream(device)
        if self._capture_stream is None:
            self._capture_stream = torch.cuda.Stream(device=device)
        stream = self._capture_stream
        stream.wait_stream(current_stream)
        with torch.cuda.stream(stream):
            output = self.function(*inputs)
        current_stream.wait_stream(stream)
        return output

    @staticmethod
    def _metadata(inputs: tuple[torch.Tensor, ...]) -> tuple:
        return tuple(
            (
                value.shape,
                value.stride(),
                value.dtype,
                value.device,
                value.requires_grad,
            )
            for value in inputs
        )

    def _capture(self, inputs: tuple[torch.Tensor, ...]):
        self._inputs = tuple(
            value.detach().clone().requires_grad_(value.requires_grad)
            for value in inputs
        )
        self._input_metadata = self._metadata(inputs)
        device = inputs[0].device
        graph = torch.cuda.CUDAGraph()
        torch.cuda.synchronize(device)
        allocated_before = torch.cuda.memory_allocated(device)
        started = time.perf_counter()
        with torch.cuda.graph(
            graph, stream=self._capture_stream, capture_error_mode="thread_local"
        ):
            self._outputs = self.function(*self._inputs)
        torch.cuda.current_stream(device).wait_stream(self._capture_stream)
        capture_seconds = time.perf_counter() - started
        # Capture records the work; the first replay materializes its outputs.
        graph.replay()
        self._graph = graph
        self.stats.captures += 1
        self.stats.capture_seconds = capture_seconds
        self.stats.graph_memory_bytes = max(
            0, torch.cuda.memory_allocated(device) - allocated_before
        )
        return self._outputs

    def __call__(self, *inputs: torch.Tensor):
        if not inputs or any(
            not isinstance(value, torch.Tensor) or not value.is_cuda for value in inputs
        ):
            raise RuntimeError("StaticCudaGraph requires one or more CUDA tensors.")
        if self._graph is None and self.stats.warmup_runs < self.warmup_steps:
            self.stats.warmup_runs += 1
            return self._warmup(inputs)
        if self._graph is None:
            return self._capture(inputs)
        if self._metadata(inputs) != self._input_metadata:
            self.stats.eager_fallbacks += 1
            return self.function(*inputs)
        for target, source in zip(self._inputs, inputs, strict=True):
            target.copy_(source)
        self._graph.replay()
        self.stats.replays += 1
        return self._outputs
