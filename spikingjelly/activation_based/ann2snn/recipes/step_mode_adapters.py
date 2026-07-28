from __future__ import annotations

import builtins
import copy
import operator
from typing import Any, Callable, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import fx

from spikingjelly.activation_based import base, functional, layer


_STEP_MODULE_TYPES: Tuple[Tuple[Type[nn.Module], Type[nn.Module]], ...] = (
    (nn.Conv1d, layer.Conv1d),
    (nn.Conv2d, layer.Conv2d),
    (nn.Conv3d, layer.Conv3d),
    (nn.ConvTranspose1d, layer.ConvTranspose1d),
    (nn.ConvTranspose2d, layer.ConvTranspose2d),
    (nn.ConvTranspose3d, layer.ConvTranspose3d),
    (nn.Linear, layer.Linear),
    (nn.MaxPool1d, layer.MaxPool1d),
    (nn.MaxPool2d, layer.MaxPool2d),
    (nn.MaxPool3d, layer.MaxPool3d),
    (nn.AvgPool1d, layer.AvgPool1d),
    (nn.AvgPool2d, layer.AvgPool2d),
    (nn.AvgPool3d, layer.AvgPool3d),
    (nn.AdaptiveAvgPool1d, layer.AdaptiveAvgPool1d),
    (nn.AdaptiveAvgPool2d, layer.AdaptiveAvgPool2d),
    (nn.AdaptiveAvgPool3d, layer.AdaptiveAvgPool3d),
    (nn.BatchNorm1d, layer.BatchNorm1d),
    (nn.BatchNorm2d, layer.BatchNorm2d),
    (nn.BatchNorm3d, layer.BatchNorm3d),
    (nn.GroupNorm, layer.GroupNorm),
    (nn.Upsample, layer.Upsample),
    (nn.Flatten, layer.Flatten),
    (nn.Dropout, layer.Dropout),
    (nn.Dropout2d, layer.Dropout2d),
)

_RATE_CODING_STATELESS_MODULE_TYPES = tuple(
    source_type for source_type, _ in _STEP_MODULE_TYPES
)
_SHAPE_ONLY_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (nn.Flatten,)
_RATE_CODING_SAFE_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (nn.ReLU,)
_TRANSFORMER_SAFE_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout2d,
)


def _normalize_dim(
    dim: int, rank: int, module_name: str, *, allow_end: bool = False
) -> int:
    size = rank + int(allow_end)
    normalized = dim + size if dim < 0 else dim
    if not 0 <= normalized < size:
        raise ValueError(
            f"{module_name} got dim={dim}, which is out of range for "
            f"an ANN tensor rank of {rank}."
        )
    return normalized


def _check_dims_do_not_touch_batch(
    dim: Any, rank: int, module_name: str, *, allow_end: bool = False
) -> None:
    if isinstance(dim, int):
        dims = (dim,)
    elif isinstance(dim, (tuple, list)):
        dims = dim
    else:
        raise ValueError(
            f"{module_name} requires literal integer dimensions, but got {dim!r}."
        )
    if any(
        _normalize_dim(value, rank, module_name, allow_end=allow_end) == 0
        for value in dims
    ):
        raise ValueError(
            f"{module_name} in multi-step mode merges the time and batch "
            "dimensions. Operations that touch the original batch dimension "
            "are not sequence-preserving and are not supported."
        )


class _StepModeTensorOp(nn.Module, base.StepModule):
    def __init__(
        self,
        op_name: str,
        op_kwargs: Optional[dict[str, Any]] = None,
        step_mode: str = "s",
    ) -> None:
        super().__init__()
        self.op_name = op_name
        self.op_kwargs = op_kwargs or {}
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"op_name={self.op_name}, step_mode={self.step_mode}"

    def _module_name(self) -> str:
        return f"{self.__class__.__name__}({self.op_name})"

    def _single_step(self, *inputs):
        x = inputs[0]
        if self.op_name == "mean":
            return x.mean(
                dim=self.op_kwargs["dim"],
                keepdim=self.op_kwargs["keepdim"],
            )
        if self.op_name == "flatten":
            return torch.flatten(
                x,
                start_dim=self.op_kwargs["start_dim"],
                end_dim=self.op_kwargs["end_dim"],
            )
        if self.op_name == "transpose":
            return torch.transpose(
                x,
                self.op_kwargs["dim0"],
                self.op_kwargs["dim1"],
            )
        if self.op_name == "unsqueeze":
            return torch.unsqueeze(x, self.op_kwargs["dim"])
        if self.op_name == "unflatten":
            return x.unflatten(
                self.op_kwargs["dim"],
                self.op_kwargs["unflattened_size"],
            )
        if self.op_name == "permute":
            return x.permute(self.op_kwargs["dims"])
        if self.op_name == "cat":
            return torch.cat(inputs, dim=self.op_kwargs["dim"])
        if self.op_name == "shape":
            return x.shape
        if self.op_name == "dim":
            return x.dim()
        if self.op_name == "size":
            return x.size(inputs[1]) if len(inputs) == 2 else x.size()
        if self.op_name == "reshape":
            sizes = inputs[1:]
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            return x.reshape(*sizes)
        if self.op_name == "expand":
            sizes = inputs[1:]
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            return x.expand(*sizes)
        if self.op_name == "getitem":
            return x[self.op_kwargs["item"]]
        raise ValueError(f"Unsupported stateless tensor op {self.op_name!r}.")

    def _multi_step(self, *inputs):
        x = inputs[0]
        if self.op_name == "cat":
            rank = x.dim() - 1
            module_name = self._module_name()
            dim = _normalize_dim(self.op_kwargs["dim"], rank, module_name)
            if dim == 0:
                raise ValueError(
                    f"{module_name} in multi-step mode merges the time and batch "
                    "dimensions. Operations that touch the original batch dimension "
                    "are not sequence-preserving and are not supported."
                )
            return torch.cat(inputs, dim=dim + 1)
        if self.op_name == "reshape":
            sizes = inputs[1:]
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            if not sizes:
                raise ValueError("reshape requires at least one output size.")
            if sizes[0] != x.shape[1] and not (sizes[0] == -1 and len(sizes) > 1):
                raise ValueError(
                    f"{self.__class__.__name__} only supports reshapes that "
                    "preserve the original batch dimension in multi-step mode."
                )
            flat = x.flatten(0, 1)
            y = flat.reshape(flat.shape[0], *sizes[1:])
            return y.reshape(x.shape[0], x.shape[1], *y.shape[1:])
        if self.op_name == "expand":
            sizes = inputs[1:]
            if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
                sizes = tuple(sizes[0])
            if x.dim() != len(sizes) + 1:
                raise ValueError(
                    f"{self.__class__.__name__} requires a time-distributed "
                    "input tensor in multi-step mode."
                )
            if sizes[0] != x.shape[1] and sizes[0] != -1 and x.shape[1] != 1:
                raise ValueError(
                    f"{self.__class__.__name__} only supports expand that "
                    "preserves the original batch dimension in multi-step mode."
                )
            return x.expand(x.shape[0], *sizes)
        if self.op_name in {
            "mean",
            "flatten",
            "transpose",
            "unsqueeze",
            "unflatten",
            "permute",
        }:
            module_name = self._module_name()
            rank = x.dim() - 1
            if self.op_name == "mean":
                dims = self.op_kwargs["dim"]
            elif self.op_name == "flatten":
                start = _normalize_dim(self.op_kwargs["start_dim"], rank, module_name)
                end = _normalize_dim(self.op_kwargs["end_dim"], rank, module_name)
                if start > end:
                    raise ValueError(
                        f"{module_name} got start_dim={self.op_kwargs['start_dim']} "
                        f"and end_dim={self.op_kwargs['end_dim']}, which do not form "
                        "a valid flatten range."
                    )
                dims = self.op_kwargs["start_dim"]
            elif self.op_name == "transpose":
                dims = (self.op_kwargs["dim0"], self.op_kwargs["dim1"])
            elif self.op_name == "unsqueeze":
                _check_dims_do_not_touch_batch(
                    self.op_kwargs["dim"], rank, module_name, allow_end=True
                )
                return functional.seq_to_ann_forward(x, self._single_step)
            elif self.op_name == "unflatten":
                dims = self.op_kwargs["dim"]
            else:
                return functional.seq_to_ann_forward(x, self._single_step)
            _check_dims_do_not_touch_batch(dims, rank, module_name)
            return functional.seq_to_ann_forward(x, self._single_step)
        if self.op_name == "shape":
            return x.shape[1:]
        if self.op_name == "dim":
            return x.dim() - 1
        if self.op_name == "size":
            if len(inputs) == 1:
                return x.size()[1:]
            module_name = self._module_name()
            dim = _normalize_dim(inputs[1], x.dim() - 1, module_name)
            return x.size(dim + 1)
        if self.op_name == "getitem":
            return functional.seq_to_ann_forward(x, self._single_step)
        raise ValueError(f"Unsupported stateless tensor op {self.op_name!r}.")

    def forward(self, *inputs):
        if self.step_mode == "s":
            return self._single_step(*inputs)
        if self.step_mode == "m":
            return self._multi_step(*inputs)
        raise ValueError(self.step_mode)


def _make_step_module(module: nn.Module) -> nn.Module:
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        if module.return_indices:
            raise ValueError("Step-mode adapter does not support return_indices=True.")
    if (
        isinstance(module, nn.AvgPool1d)
        and getattr(module, "divisor_override", None) is not None
    ):
        raise ValueError(
            "Step-mode adapter does not support nn.AvgPool1d with "
            "divisor_override because layer.AvgPool1d does not expose "
            "that parameter."
        )
    if isinstance(module, (nn.Dropout, nn.Dropout2d)):
        if module.inplace:
            raise ValueError(
                f"Step-mode adapter does not support {type(module).__name__} with "
                "inplace=True because its step-mode equivalent does not expose that "
                "parameter."
            )
        replacement_type = (
            layer.Dropout2d if isinstance(module, nn.Dropout2d) else layer.Dropout
        )
        return replacement_type(p=module.p).train(module.training)
    for source_type, step_type in _STEP_MODULE_TYPES:
        if isinstance(module, source_type):
            replacement = copy.deepcopy(module)
            # Step-mode wrappers inherit the matching torch module and add only
            # step-mode behavior, so the copied parameters and state stay flat.
            replacement.__class__ = step_type
            replacement.step_mode = "s"
            return replacement
    raise TypeError(type(module).__name__)


def _check_literal_dim(dim: Any, context: str) -> None:
    if isinstance(dim, int):
        return
    if isinstance(dim, (tuple, list)) and all(isinstance(value, int) for value in dim):
        return
    raise ValueError(f"{context} requires literal integer dimensions, but got {dim!r}.")


def _getitem_preserves_batch_dim(item: Any) -> bool:
    if item is Ellipsis:
        return True
    if isinstance(item, slice):
        return item == slice(None, None, None)
    if isinstance(item, tuple):
        if not item:
            return False
        first = item[0]
        if first is Ellipsis:
            return True
        return isinstance(first, slice) and first == slice(None, None, None)
    return False


def _contains_fx_node(value: Any) -> bool:
    if isinstance(value, fx.Node):
        return True
    if isinstance(value, slice):
        return any(
            _contains_fx_node(part) for part in (value.start, value.stop, value.step)
        )
    if isinstance(value, tuple):
        return any(_contains_fx_node(part) for part in value)
    return False


def _is_non_tensor_getitem_input(node: Any) -> bool:
    return isinstance(node, fx.Node) and (
        (
            node.op == "call_function"
            and node.target is builtins.getattr
            and len(node.args) == 2
            and node.args[1] == "shape"
        )
        or (node.op == "call_method" and node.target == "size")
        or node.meta.get("ann2snn_step_mode_op") in ("shape", "size")
        or (
            node.op == "call_module"
            and any(
                user.op == "call_function"
                and user.target is operator.getitem
                and len(user.args) >= 2
                and user.args[0] is node
                and user.args[1] == 1
                and not user.users
                for user in node.users
            )
        )
    )


def _is_batch_size_node(node: Any, tensor_node: Any) -> bool:
    return isinstance(node, fx.Node) and (
        (
            node.op == "call_method"
            and node.target == "size"
            and node.args == (tensor_node, 0)
        )
        or (
            node.meta.get("ann2snn_step_mode_op") == "size"
            and node.args == (tensor_node, 0)
        )
    )


def _make_tensor_rewrite(
    node: fx.Node, context: str
) -> Optional[Tuple[_StepModeTensorOp, Tuple[Any, ...]]]:
    args = tuple(node.args)
    kwargs = node.kwargs
    target = node.target

    if target == "size":
        input_args = list(args)
        if "dim" in kwargs:
            if len(input_args) > 1:
                raise ValueError(f"{context} got size dim as both arg and keyword.")
            input_args.append(kwargs["dim"])
        if len(input_args) > 2:
            raise ValueError(f"{context} does not support size with extra arguments.")
        if len(input_args) == 2:
            _check_literal_dim(input_args[1], context)
        return _StepModeTensorOp("size"), tuple(input_args)
    if target == "dim":
        return _StepModeTensorOp("dim"), (args[0],)
    if target == "view":
        if len(args) == 3 and _is_batch_size_node(args[1], args[0]) and args[2] == -1:
            return (
                _StepModeTensorOp("flatten", {"start_dim": 1, "end_dim": -1}),
                (args[0],),
            )
        return _StepModeTensorOp("reshape"), args
    if target in ("reshape", torch.reshape):
        return _StepModeTensorOp("reshape"), args
    if target == "expand":
        return _StepModeTensorOp("expand"), args

    if target is builtins.getattr:
        if len(node.args) == 2 and node.args[1] == "shape":
            return _StepModeTensorOp("shape"), (node.args[0],)
        return None
    if target is operator.getitem:
        item = node.args[1]
        if isinstance(item, int):
            if not _is_non_tensor_getitem_input(node.args[0]):
                raise ValueError(
                    f"{context} does not support integer tensor getitem because "
                    "it is not sequence-preserving."
                )
            return None
        if not isinstance(item, (slice, tuple)) and item is not Ellipsis:
            raise ValueError(f"{context} does not support getitem index {item!r}.")
        if _contains_fx_node(item):
            raise ValueError(
                f"{context} requires tensor getitem indices to be static literals."
            )
        if not _getitem_preserves_batch_dim(item):
            raise ValueError(
                f"{context} requires tensor getitem to preserve the original "
                "batch dimension."
            )
        return _StepModeTensorOp("getitem", {"item": item}), (node.args[0],)

    tensor = args[0] if args else kwargs["input"]
    if target in ("mean", torch.mean):
        if len(args) > 1:
            dim = args[1]
        elif "dim" in kwargs:
            dim = kwargs["dim"]
        else:
            raise ValueError(
                f"{context} does not support mean without an explicit dim."
            )
        keepdim = args[2] if len(args) > 2 else kwargs.get("keepdim", False)
        if keepdim not in (False, True):
            raise ValueError(
                f"{context} requires literal bool keepdim, but got {keepdim!r}."
            )
        _check_literal_dim(dim, context)
        return (
            _StepModeTensorOp("mean", {"dim": dim, "keepdim": bool(keepdim)}),
            (tensor,),
        )
    if target in ("flatten", torch.flatten):
        start_dim = args[1] if len(args) > 1 else kwargs.get("start_dim", 0)
        end_dim = args[2] if len(args) > 2 else kwargs.get("end_dim", -1)
        _check_literal_dim(start_dim, context)
        _check_literal_dim(end_dim, context)
        return (
            _StepModeTensorOp("flatten", {"start_dim": start_dim, "end_dim": end_dim}),
            (tensor,),
        )
    if target in ("transpose", torch.transpose):
        dim0 = args[1] if len(args) > 1 else kwargs["dim0"]
        dim1 = args[2] if len(args) > 2 else kwargs["dim1"]
        _check_literal_dim(dim0, context)
        _check_literal_dim(dim1, context)
        return (
            _StepModeTensorOp("transpose", {"dim0": dim0, "dim1": dim1}),
            (tensor,),
        )
    if target in ("unsqueeze", torch.unsqueeze):
        if len(args) > 1:
            dim = args[1]
        elif "dim" in kwargs:
            dim = kwargs["dim"]
        else:
            raise ValueError(f"{context} does not support unsqueeze without a dim.")
        _check_literal_dim(dim, context)
        return _StepModeTensorOp("unsqueeze", {"dim": dim}), (tensor,)
    if target == "unflatten":
        if len(args) > 1:
            dim = args[1]
        elif "dim" in kwargs:
            dim = kwargs["dim"]
        else:
            raise ValueError(f"{context} does not support unflatten without a dim.")
        if len(args) > 2:
            sizes = args[2]
        else:
            sizes = kwargs.get("sizes", kwargs.get("unflattened_size"))
        _check_literal_dim(dim, context)
        if not isinstance(sizes, (tuple, list, torch.Size)):
            raise ValueError(
                f"{context} requires literal unflatten sizes, but got {sizes!r}."
            )
        return (
            _StepModeTensorOp(
                "unflatten",
                {"dim": dim, "unflattened_size": tuple(sizes)},
            ),
            (tensor,),
        )
    if target == "permute":
        dims = args[1:]
        if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
            dims = tuple(dims[0])
        for dim in dims:
            _check_literal_dim(dim, context)
        rank = len(dims)
        dims = tuple(_normalize_dim(dim, rank, context) for dim in dims)
        if not dims or dims[0] != 0:
            raise ValueError(
                f"{context} requires permute to preserve the original batch dimension."
            )
        return _StepModeTensorOp("permute", {"dims": dims}), (tensor,)
    if target is torch.cat:
        tensors = args[0] if args else kwargs["tensors"]
        dim = args[1] if len(args) > 1 else kwargs.get("dim", 0)
        _check_literal_dim(dim, context)
        return _StepModeTensorOp("cat", {"dim": int(dim)}), tuple(tensors)
    return None


def adapt_step_mode_graph(
    fx_model: fx.GraphModule,
    *,
    context: str,
    wrap_module_types: Tuple[Type[nn.Module], ...] = (),
    safe_module_types: Tuple[Type[nn.Module], ...] = (),
    safe_call_functions: Tuple[Callable[..., Any], ...] = (),
) -> fx.GraphModule:
    modules = dict(fx_model.named_modules())
    existing_modules = set(modules.keys())
    tensor_op_index = 0
    passthrough_call_functions = {
        operator.add,
        operator.eq,
        operator.floordiv,
        operator.neg,
        torch.add,
        torch.neg,
        torch.sub,
        torch._assert,
        operator.mul,
        operator.sub,
    }
    rewritten_call_functions = {
        operator.getitem,
        torch.cat,
        builtins.getattr,
        torch.flatten,
        torch.mean,
        torch.reshape,
        torch.transpose,
        torch.unsqueeze,
    }

    for node in list(fx_model.graph.nodes):
        if node.op == "call_module":
            module = modules[node.target]
            if isinstance(module, base.StepModule) or isinstance(
                module, base.MemoryModule
            ):
                continue
            if isinstance(module, nn.Identity):
                continue
            if isinstance(module, safe_module_types):
                continue
            if isinstance(module, wrap_module_types):
                replacement = _make_step_module(module)
                parent_name, _, child_name = node.target.rpartition(".")
                parent = (
                    fx_model.get_submodule(parent_name) if parent_name else fx_model
                )
                setattr(parent, child_name, replacement)
                modules[node.target] = replacement
                continue
            raise ValueError(
                f"{context} does not support module node {node.target!r} of "
                f"type {type(module).__name__}."
            )

        size_node = None
        if node.op == "call_method":
            if node.target == "size":
                if any(
                    user.op == "call_method"
                    and user.target == "view"
                    and len(user.args) == 3
                    and user.args[0] is node.args[0]
                    and user.args[1] is node
                    and user.args[2] == -1
                    for user in node.users
                ):
                    continue
            if node.target == "masked_fill":
                continue
            size_node = (
                node.args[1]
                if node.target == "view"
                and len(node.args) == 3
                and _is_batch_size_node(node.args[1], node.args[0])
                and node.args[2] == -1
                else None
            )
        elif node.op == "call_function":
            if hasattr(node.target, "_fields"):
                continue
            if node.target in safe_call_functions:
                continue
            if node.target in passthrough_call_functions:
                continue
            if node.target not in rewritten_call_functions:
                raise ValueError(
                    f"{context} does not support function node {node.target!r}."
                )
        else:
            continue

        rewrite = _make_tensor_rewrite(node, context)
        if rewrite is None:
            if node.op == "call_function":
                continue
            raise ValueError(
                f"{context} does not support tensor method {node.target!r}. "
                "Rewrite the model with supported sequence-preserving operations."
            )
        module, input_args = rewrite
        target = f"step_mode_tensor_op_{tensor_op_index}"
        while target in existing_modules:
            tensor_op_index += 1
            target = f"step_mode_tensor_op_{tensor_op_index}"
        tensor_op_index += 1
        fx_model.add_submodule(target, module)
        existing_modules.add(target)
        with fx_model.graph.inserting_after(node):
            replacement = fx_model.graph.call_module(target, args=input_args)
        replacement.meta["ann2snn_step_mode_op"] = module.op_name
        node.replace_all_uses_with(replacement)
        fx_model.graph.erase_node(node)
        if size_node is not None and not size_node.users:
            fx_model.graph.erase_node(size_node)

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model
