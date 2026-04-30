import warnings
from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from functorch.compile import aot_function, min_cut_rematerialization_partition

warnings.filterwarnings("ignore", category=UserWarning)


__all__ = [
    "generate_inference_graph",
    "generate_forward_and_backward_graph",
]


class GraphCollector:
    """Provide this class to aot_function to collect forward and backward graph."""

    def __init__(self):
        self.fwd_graph = None
        self.bwd_graph = None

    def get_forward_compiler(self):
        def _fw_compiler(fx_module, *args, **kwargs):
            self.fwd_graph = fx_module.graph
            return fx_module

        return _fw_compiler

    def get_backward_compiler(self):
        def _bw_compiler(fx_module, *args, **kwargs):
            self.bwd_graph = fx_module.graph
            return fx_module

        return _bw_compiler


class GraphOptimizer(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target.__name__ == "detach.default":
            # Remove `.detach()` operation.
            # We can safely remove it since the bwd graph has already been generated!
            return args[0]
        return super().call_function(target, args, kwargs)


def _optimize_graph(graph: fx.Graph):
    return GraphOptimizer(fx.GraphModule({}, graph)).transform().graph


def generate_inference_graph(fn: Callable, example_inputs: tuple) -> fx.Graph:
    """Generate an optimized inference FX graph.

    Chinese:
        为给定的 PyTorch 函数生成优化后的推理 FX 图。

    English:
        Generate an optimized inference FX graph for a PyTorch callable.

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :type fn: Callable
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :type example_inputs: tuple
    :return: EN: Optimized forward FX graph. Chinese: 优化后的前向 FX 图。
    :rtype: torch.fx.Graph
    :raises ValueError: EN: Raised when the traced callable fails to produce a forward graph. Chinese: 当被追踪函数未能产生前向 FX 图时抛出。
    """
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler(),
    )  # ahead-of-time autograd

    for i in example_inputs:
        if isinstance(i, torch.Tensor):
            i.requires_grad = False  # for inference

    # feed the fake inputs
    _ = f(*example_inputs)
    if collector.fwd_graph is None:
        raise ValueError(f"Failed to capture an inference graph for {fn}.")
    return _optimize_graph(collector.fwd_graph)


def generate_forward_and_backward_graph(
    fn: Callable,
    example_inputs: tuple,
    requires_grad: Optional[Sequence[bool]] = None,
) -> Tuple[fx.Graph, fx.Graph]:
    """Generate optimized forward/backward FX graphs.

    Chinese:
        为给定的 PyTorch 函数生成优化后的前向与反向 FX 图。

    English:
        Generate optimized forward and backward FX graphs for a PyTorch callable.

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :type fn: Callable
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :type example_inputs: tuple
    :param requires_grad: EN: Optional gradient-requirement flags for each example input. Chinese: 每个示例输入对应的可选求导标志。
    :type requires_grad: Optional[Sequence[bool]]
    :return: EN: Optimized forward and backward FX graphs. Chinese: 优化后的前向与反向 FX 图。
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    :raises ValueError: EN: Raised when ``requires_grad`` length mismatches ``example_inputs``, when the callable does not return a tensor/list/tuple, or when no differentiable output exists. Chinese: 当 ``requires_grad`` 长度与 ``example_inputs`` 不匹配、函数返回值不是张量/列表/元组、或不存在可求导输出时抛出。
    """
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler(),
        partition_fn=min_cut_rematerialization_partition,
    )

    if requires_grad is not None:
        if len(requires_grad) != len(example_inputs):
            raise ValueError(
                "requires_grad must have the same length as example_inputs"
            )
        for i, r in zip(example_inputs, requires_grad):
            if isinstance(i, torch.Tensor):
                i.requires_grad = r
    else:  # if not specified, assume that all tensors require gradients
        for i in example_inputs:
            if isinstance(i, torch.Tensor):
                i.requires_grad = True

    # feed the fake inputs
    ys = f(*example_inputs)
    # Normalise to tuple so iteration always walks outputs, not tensor dimensions
    if isinstance(ys, torch.Tensor):
        ys = (ys,)
    elif not isinstance(ys, (list, tuple)):
        raise ValueError(f"Expected {fn} to return a tuple/list of Tensors, got {type(ys)}")
    diff_outputs = [
        y
        for y in ys
        if isinstance(y, torch.Tensor) and (y.requires_grad or y.grad_fn is not None)
    ]
    if not diff_outputs:
        raise ValueError(
            f"No differentiable Tensor found in the output of the function {fn}"
        )
    torch.autograd.backward(
        diff_outputs,
        [torch.randn_like(y) for y in diff_outputs],
    )

    if collector.fwd_graph is None or collector.bwd_graph is None:
        raise ValueError(f"Failed to capture both forward and backward graphs for {fn}.")
    collector.bwd_graph.lint()

    return (
        _optimize_graph(collector.fwd_graph),
        _optimize_graph(collector.bwd_graph),
    )
