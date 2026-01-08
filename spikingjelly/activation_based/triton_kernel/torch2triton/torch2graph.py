from typing import Tuple, Optional, Callable
import warnings

import torch
import torch.fx as fx
from torch._functorch.aot_autograd import aot_function

warnings.filterwarnings("ignore", category=UserWarning)


__all__ = [
    "generate_inference_graph",
    "generate_forward_and_backward_graph",
]


class GraphCollector:
    """Provide this class to aot_function to collect forward and backward graph.
    """

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


def generate_inference_graph(fn: Callable, example_inputs: tuple):
    """Generate an optimized inference graph from a PyTorch function.

    For inference scenarios, all input tensors are set to `requires_grad=False`
    to disable gradient tracking.

    Args:
        fn (Callable): The PyTorch function to generate the inference graph for.
        example_inputs (tuple): A tuple of example inputs to trace the function's execution.

    Returns:
        fx.Graph: The optimized inference graph.
    """
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler()
    )  # ahead-of-time autograd

    for i in example_inputs:
        if isinstance(i, torch.Tensor):
            i.requires_grad = False  # for inference

    # feed the fake inputs
    ys = f(*example_inputs)
    return _optimize_graph(collector.fwd_graph)


def generate_forward_and_backward_graph(
    fn: Callable,
    example_inputs: tuple,
    requires_grad: Optional[Tuple[bool]] = None
):
    """Generate optimized forward and backward graphs from a PyTorch function.

    Allows specifying gradient requirements for input tensors; if not specified,
    all tensors default to `requires_grad=True`.

    Args:
        fn (Callable): The PyTorch function to generate the graphs for.
        example_inputs (tuple): A tuple of example inputs to trace the function's execution.
        requires_grad (Tuple[bool], optional): A tuple indicating whether each
            input tensor requires gradient. If given, must match the length of
            `example_inputs`. Defaults to None, where all input tensors are set
            to `requires_grad=True`.

    Returns:
        Tuple[fx.Graph, fx.Graph]: the optimized forward and backward graph.
    """
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler()
    )

    if requires_grad is not None:
        for i, r in zip(example_inputs, requires_grad):
            if isinstance(i, torch.Tensor):
                i.requires_grad = r
    else:  # if not specified, assume that all tensors require gradients
        for i in example_inputs:
            if isinstance(i, torch.Tensor):
                i.requires_grad = True

    # feed the fake inputs
    ys = f(*example_inputs)
    # choose a Tensor in ys as the starting point of .backward()
    o = None
    for y in ys:
        if isinstance(y, torch.Tensor):
            o = y
            break
    if o is None:
        raise ValueError(f"No Tensor found in the output of the function {fn}")
    # create a fake gradient
    g = torch.randn_like(o)
    # backward
    o.backward(g)

    collector.bwd_graph.lint()

    return (
        _optimize_graph(collector.fwd_graph),
        _optimize_graph(collector.bwd_graph),
    )
