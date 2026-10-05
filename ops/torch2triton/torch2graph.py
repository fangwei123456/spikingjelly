from typing import Callable, Optional, Sequence, Tuple

import torch
import torch.fx as fx
from functorch.compile import (
    aot_function,
    make_boxed_func,
    min_cut_rematerialization_partition,
)

__all__ = [
    "generate_forward_and_backward_graph",
]


class _GraphCollector:
    def __init__(self):
        self.fwd_module: Optional[fx.GraphModule] = None
        self.bwd_module: Optional[fx.GraphModule] = None

    def forward_compiler(self, module: fx.GraphModule, _inputs):
        self.fwd_module = module
        return make_boxed_func(module)

    def backward_compiler(self, module: fx.GraphModule, _inputs):
        self.bwd_module = module
        return make_boxed_func(module)


class GraphOptimizer(fx.Transformer):
    def call_function(self, target, args, kwargs):
        if target.__name__ == "detach.default":
            # Remove `.detach()` operation.
            # We can safely remove it since the bwd graph has already been generated!
            return args[0]
        return super().call_function(target, args, kwargs)


def generate_forward_and_backward_graph(
    fn: Callable,
    example_inputs: tuple,
    requires_grad: Optional[Sequence[bool]] = None,
) -> Tuple[fx.Graph, fx.Graph]:
    """Generate optimized forward/backward FX graphs.

    **API Language** - :ref:`中文 <generate_forward_and_backward_graph-cn>` | :ref:`English <generate_forward_and_backward_graph-en>`

    ----

    .. _generate_forward_and_backward_graph-cn:

    * **中文**

    生成前向和反向计算图

    :param fn: EN: Callable to trace. Chinese: 待追踪的可调用对象。
    :type fn: ``Callable``
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

    Generate forward and backward graphs

    :type fn: ``Callable``
    :type example_inputs: tuple
    :type requires_grad: Optional[Sequence[bool]]
    :raises ValueError: EN: Raised when ``requires_grad`` length mismatches ``example_inputs``, when the callable does not return a tensor/list/tuple, or when no differentiable output exists. Chinese: 当 ``requires_grad`` 长度与 ``example_inputs`` 不匹配、函数返回值不是张量/列表/元组、或不存在可求导输出时抛出。
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    """
    collector = _GraphCollector()
    f = aot_function(
        fn,
        fw_compiler=collector.forward_compiler,
        bw_compiler=collector.backward_compiler,
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

    if collector.fwd_module is None or collector.bwd_module is None:
        raise ValueError(
            f"Failed to capture both forward and backward graphs for {fn}."
        )
    collector.bwd_module.graph.lint()
    return (
        GraphOptimizer(collector.fwd_module).transform().graph,
        GraphOptimizer(collector.bwd_module).transform().graph,
    )
