import warnings
from typing import Any, Callable, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from functorch.compile import aot_function, min_cut_rematerialization_partition

warnings.filterwarnings("ignore", category=UserWarning)


__all__ = [
    "generate_inference_graph",
    "generate_forward_and_backward_graph",
]


class GraphCollector:
    """Provide this class to aot_function to collect forward and backward graph/module.
    **API Language:**
    :ref:`中文 <GraphCollector-cn>` | :ref:`English <GraphCollector-en>`

    ----

    .. _GraphCollector-cn:

    * **中文**

    TODO: add Chinese description

    :rtype: None
    We store both the raw GraphModule and the Graph to allow safe optimizations that
    preserve module attributes/constants when possible.

    ----

    .. _GraphCollector-en:

    * **English**

    TODO: add English description

    :return: None
    :rtype: None
    """

    def __init__(self):
        self.fwd_graph: Optional[fx.Graph] = None
        self.bwd_graph: Optional[fx.Graph] = None
        self.fwd_module: Optional[fx.GraphModule] = None
        self.bwd_module: Optional[fx.GraphModule] = None

    def get_forward_compiler(self):
        def _fw_compiler(fx_module: fx.GraphModule, *args, **kwargs):
            self.fwd_module = fx_module
            self.fwd_graph = fx_module.graph
            return fx_module

        return _fw_compiler

    def get_backward_compiler(self):
        def _bw_compiler(fx_module: fx.GraphModule, *args, **kwargs):
            self.bwd_module = fx_module
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


def _optimize_graph(graph_or_module: Any) -> fx.Graph:
    # Accept either an fx.GraphModule or an fx.Graph. Prefer operating on the
    # original GraphModule when available so module attributes/constants are
    # preserved; otherwise fall back to creating a temporary GraphModule.
    if isinstance(graph_or_module, fx.GraphModule):
        gm = graph_or_module
    else:
        gm = fx.GraphModule({}, graph_or_module)

    optimized = GraphOptimizer(gm).transform()
    return optimized.graph


def generate_inference_graph(fn: Callable, example_inputs: tuple) -> fx.Graph:
    """Generate an optimized inference FX graph.
    **API Language:**
    :ref:`中文 <generate_inference_graph-cn>` | :ref:`English <generate_inference_graph-en>`

    ----

    .. _generate_inference_graph-cn:

    * **中文**

    TODO: add Chinese description

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :type fn: Callable
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :type example_inputs: tuple
    :return: EN: Optimized forward FX graph. Chinese: 优化后的前向 FX 图。
    :rtype: torch.fx.Graph
    :raises ValueError: EN: Raised when the traced callable fails to produce a forward graph. Chinese: 当被追踪函数未能产生前向 FX 图时抛出。
    Chinese:
        为给定的 PyTorch 函数生成优化后的推理 FX 图。
    English:
        Generate an optimized inference FX graph for a PyTorch callable.

    ----

    .. _generate_inference_graph-en:

    * **English**

    TODO: add English description

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :type fn: Callable
    :type example_inputs: tuple
    :raises ValueError: EN: Raised when the traced callable fails to produce a forward graph. Chinese: 当被追踪函数未能产生前向 FX 图时抛出。
    :return: EN: Optimized forward FX graph. Chinese: 优化后的前向 FX 图。
    :rtype: torch.fx.Graph
    """
    collector = GraphCollector()

    # Build local inputs so we don't mutate callers' tensors. For inference
    # tracing we detach inputs so they do not require gradients.
    local_inputs = []
    for i in example_inputs:
        if isinstance(i, torch.Tensor):
            local_inputs.append(i.detach())
        else:
            local_inputs.append(i)

    # Capture the graph using aot_function which provides the flattened ATen-level graph
    # required for the Triton compiler.
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler(),
    )
    _ = f(*local_inputs)

    if collector.fwd_module is None and collector.fwd_graph is None:
        raise ValueError(f"Failed to capture an inference graph for {fn}.")

    return _optimize_graph(collector.fwd_module or collector.fwd_graph)


def generate_forward_and_backward_graph(
    fn: Callable,
    example_inputs: tuple,
    requires_grad: Optional[Sequence[bool]] = None,
) -> Tuple[fx.Graph, fx.Graph]:
    """Generate optimized forward/backward FX graphs.
    **API Language:**
    :ref:`中文 <generate_forward_and_backward_graph-cn>` | :ref:`English <generate_forward_and_backward_graph-en>`

    ----

    .. _generate_forward_and_backward_graph-cn:

    * **中文**

    TODO: add Chinese description

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :type fn: Callable
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :type example_inputs: tuple
    :param requires_grad: EN: Optional gradient-requirement flags for each example input. Chinese: 每个示例输入对应的可选求导标志。
    :type requires_grad: Optional[Sequence[bool]]
    :return: EN: Optimized forward and backward FX graphs. Chinese: 优化后的前向与反向 FX 图。
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    :raises ValueError: EN: Raised when ``requires_grad`` length mismatches ``example_inputs``, when the callable does not return a tensor/list/tuple, or when no differentiable output exists. Chinese: 当 ``requires_grad`` 长度与 ``example_inputs`` 不匹配、函数返回值不是张量/列表/元组、或不存在可求导输出时抛出。
    Chinese:
        为给定的 PyTorch 函数生成优化后的前向与反向 FX 图。
    English:
        Generate optimized forward and backward FX graphs for a PyTorch callable.

    ----

    .. _generate_forward_and_backward_graph-en:

    * **English**

    TODO: add English description

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :param example_inputs: EN: Example inputs used for tracing. Chinese: 用于追踪的示例输入。
    :param requires_grad: EN: Optional gradient-requirement flags for each example input. Chinese: 每个示例输入对应的可选求导标志。
    :type fn: Callable
    :type example_inputs: tuple
    :type requires_grad: Optional[Sequence[bool]]
    :raises ValueError: EN: Raised when ``requires_grad`` length mismatches ``example_inputs``, when the callable does not return a tensor/list/tuple, or when no differentiable output exists. Chinese: 当 ``requires_grad`` 长度与 ``example_inputs`` 不匹配、函数返回值不是张量/列表/元组、或不存在可求导输出时抛出。
    :return: EN: Optimized forward and backward FX graphs. Chinese: 优化后的前向与反向 FX 图。
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    """
    collector = GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.get_forward_compiler(),
        bw_compiler=collector.get_backward_compiler(),
        partition_fn=min_cut_rematerialization_partition,
    )

    # Build local inputs so we don't mutate callers' tensors. Set requires_grad
    # on detached clones as needed to avoid modifying non-leaf tensors.
    local_inputs = []
    if requires_grad is not None:
        if len(requires_grad) != len(example_inputs):
            raise ValueError(
                "requires_grad must have the same length as example_inputs"
            )
        for i, r in zip(example_inputs, requires_grad):
            if isinstance(i, torch.Tensor):
                if r:
                    local_inputs.append(i.detach().requires_grad_(True))
                else:
                    local_inputs.append(i.detach())
            else:
                local_inputs.append(i)
    else:  # if not specified, assume that all tensors require gradients
        for i in example_inputs:
            if isinstance(i, torch.Tensor):
                local_inputs.append(i.detach().requires_grad_(True))
            else:
                local_inputs.append(i)

    # feed the fake inputs
    ys = f(*local_inputs)
    # Normalise to tuple so iteration always walks outputs, not tensor dimensions
    if isinstance(ys, torch.Tensor):
        ys = (ys,)
    elif not isinstance(ys, (list, tuple)):
        raise ValueError(
            f"Expected {fn} to return a tuple/list of Tensors, got {type(ys)}"
        )
    diff_outputs = [
        y
        for y in ys
        if isinstance(y, torch.Tensor) and (y.requires_grad or y.grad_fn is not None)
    ]
    if not diff_outputs:
        raise ValueError(
            f"No differentiable Tensor found in the output of the function {fn}"
        )
    torch.autograd.backward(diff_outputs, [torch.randn_like(y) for y in diff_outputs])

    if (collector.fwd_module is None and collector.fwd_graph is None) or (
        collector.bwd_module is None and collector.bwd_graph is None
    ):
        raise ValueError(
            f"Failed to capture both forward and backward graphs for {fn}."
        )

    # Run lint on the backward graph (if available)
    if collector.bwd_graph is not None:
        collector.bwd_graph.lint()

    # Prefer modules when available so optimizations preserve attributes/constants.
    fwd_src = collector.fwd_module or collector.fwd_graph
    bwd_src = collector.bwd_module or collector.bwd_graph

    return (_optimize_graph(fwd_src), _optimize_graph(bwd_src))
