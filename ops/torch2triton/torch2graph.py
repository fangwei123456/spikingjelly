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
    r"""
    **API Language** - :ref:`中文 <generate_forward_and_backward_graph-cn>` | :ref:`English <generate_forward_and_backward_graph-en>`

    ----

    .. _generate_forward_and_backward_graph-cn:

    * **中文**

    为 PyTorch 函数生成优化后的前向与反向 FX 图。示例输入的求导设置只应用于
    局部张量，不修改调用者的张量。

    :param fn: 待追踪的可调用对象，返回张量或张量列表/元组。
    :type fn: Callable
    :param example_inputs: 用于追踪的示例输入。
    :type example_inputs: tuple
    :param requires_grad: 每个示例输入的求导标志；默认 ``None`` 时对所有张量输入求导。
    :type requires_grad: Optional[Sequence[bool]]
    :return: 优化后的前向与反向 FX 图。
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    :raises ValueError: 求导标志数量与输入数量不匹配、返回值不是张量/列表/元组，
        或未能捕获前向与反向图。
    :raises NotImplementedError: 没有可求导的输出，无法生成反向图。

    ----

    .. _generate_forward_and_backward_graph-en:

    * **English**

    Generate optimized forward and backward FX graphs for a PyTorch callable.
    Gradient flags apply to local tensors without modifying the caller's tensors.

    :param fn: Callable to trace, returning a tensor or a list/tuple of tensors.
    :type fn: Callable
    :param example_inputs: Example inputs used for tracing.
    :type example_inputs: tuple
    :param requires_grad: Per-input gradient flags. With ``None`` (the default),
        all tensor inputs require gradients.
    :type requires_grad: Optional[Sequence[bool]]
    :return: Optimized forward and backward FX graphs.
    :rtype: Tuple[torch.fx.Graph, torch.fx.Graph]
    :raises ValueError: Gradient flags and inputs differ in count, the return
        value is not a tensor/list/tuple, or both graphs could not be captured.
    :raises NotImplementedError: No output is differentiable, so a backward
        graph cannot be generated.
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
        raise NotImplementedError(
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
