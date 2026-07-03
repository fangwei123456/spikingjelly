from __future__ import annotations

import builtins
import operator
from typing import Any, Callable, Dict, Iterable, Optional, Tuple, Type

import torch
import torch.nn as nn
from torch import fx

from spikingjelly.activation_based import base, functional, layer


_RATE_CODING_STATELESS_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Conv1d,
    nn.Conv2d,
    nn.Conv3d,
    nn.ConvTranspose1d,
    nn.ConvTranspose2d,
    nn.ConvTranspose3d,
    nn.Linear,
    nn.MaxPool1d,
    nn.MaxPool2d,
    nn.MaxPool3d,
    nn.AvgPool1d,
    nn.AvgPool2d,
    nn.AvgPool3d,
    nn.AdaptiveAvgPool1d,
    nn.AdaptiveAvgPool2d,
    nn.AdaptiveAvgPool3d,
    nn.BatchNorm1d,
    nn.BatchNorm2d,
    nn.BatchNorm3d,
    nn.GroupNorm,
    nn.Upsample,
    nn.Flatten,
    nn.Dropout,
    nn.Dropout2d,
)

_SHAPE_ONLY_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Flatten,
)
_RATE_CODING_SAFE_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (nn.ReLU,)
_TRANSFORMER_SAFE_MODULE_TYPES: Tuple[Type[nn.Module], ...] = (
    nn.Dropout,
    nn.Dropout2d,
)


def _normalize_dim(dim: int, rank: int, module_name: str) -> int:
    normalized = dim + rank if dim < 0 else dim
    if normalized < 0 or normalized >= rank:
        raise ValueError(
            f"{module_name} got dim={dim}, which is out of range for "
            f"an ANN tensor rank of {rank}."
        )
    return normalized


def _normalize_unsqueeze_dim(dim: int, rank: int, module_name: str) -> int:
    normalized = dim + rank + 1 if dim < 0 else dim
    if normalized < 0 or normalized > rank:
        raise ValueError(
            f"{module_name} got dim={dim}, which is out of range for "
            f"an ANN tensor rank of {rank}."
        )
    return normalized


def _dim_touches_batch(dim: Any, rank: int, module_name: str) -> bool:
    if isinstance(dim, int):
        return _normalize_dim(dim, rank, module_name) == 0
    if isinstance(dim, tuple):
        return any(_dim_touches_batch(d, rank, module_name) for d in dim)
    if isinstance(dim, list):
        return any(_dim_touches_batch(d, rank, module_name) for d in dim)
    raise ValueError(
        f"{module_name} requires literal integer dimensions, but got {dim!r}."
    )


def _check_does_not_touch_batch_dim(dim: Any, rank: int, module_name: str) -> None:
    if _dim_touches_batch(dim, rank, module_name):
        raise ValueError(
            f"{module_name} in multi-step mode merges the time and batch "
            "dimensions before applying the ANN tensor op. Operations that "
            "touch the original batch dimension are not sequence-preserving "
            "and are not supported."
        )


def _check_dims_do_not_touch_batch_dim(
    dims: Iterable[int],
    rank: int,
    module_name: str,
) -> None:
    for dim in dims:
        _check_does_not_touch_batch_dim(dim, rank, module_name)


def _check_flatten_does_not_touch_batch_dim(
    start_dim: int,
    end_dim: int,
    rank: int,
    module_name: str,
) -> None:
    start = _normalize_dim(start_dim, rank, module_name)
    end = _normalize_dim(end_dim, rank, module_name)
    if start > end:
        raise ValueError(
            f"{module_name} got start_dim={start_dim} and end_dim={end_dim}, "
            "which do not form a valid flatten range."
        )
    if start == 0:
        raise ValueError(
            f"{module_name} in multi-step mode merges the time and batch "
            "dimensions before applying flatten. Flatten ranges that include "
            "the original batch dimension are not sequence-preserving and "
            "are not supported."
        )


def _check_unsqueeze_does_not_touch_batch_dim(
    dim: int,
    rank: int,
    module_name: str,
) -> None:
    if _normalize_unsqueeze_dim(dim, rank, module_name) == 0:
        raise ValueError(
            f"{module_name} in multi-step mode merges the time and batch "
            "dimensions before applying unsqueeze. Inserting a dimension "
            "before the original batch dimension is not sequence-preserving "
            "and is not supported."
        )


class _StatelessTensorOp(nn.Module, base.StepModule):
    def __init__(
        self,
        op_name: str,
        op_kwargs: Optional[Dict[str, Any]] = None,
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

    def _validate_multi_step(self, x: torch.Tensor) -> None:
        module_name = self._module_name()
        rank = x.dim() - 1
        if self.op_name == "mean":
            _check_does_not_touch_batch_dim(self.op_kwargs["dim"], rank, module_name)
        elif self.op_name == "flatten":
            _check_flatten_does_not_touch_batch_dim(
                self.op_kwargs["start_dim"],
                self.op_kwargs["end_dim"],
                rank,
                module_name,
            )
        elif self.op_name == "transpose":
            _check_dims_do_not_touch_batch_dim(
                (self.op_kwargs["dim0"], self.op_kwargs["dim1"]),
                rank,
                module_name,
            )
        elif self.op_name == "unsqueeze":
            _check_unsqueeze_does_not_touch_batch_dim(
                self.op_kwargs["dim"],
                rank,
                module_name,
            )
        elif self.op_name == "unflatten":
            _check_does_not_touch_batch_dim(
                self.op_kwargs["dim"],
                rank,
                module_name,
            )
        elif self.op_name == "permute":
            return

    def _apply_op(self, x: torch.Tensor) -> torch.Tensor:
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
        raise ValueError(f"Unsupported stateless tensor op {self.op_name!r}.")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return self._apply_op(x)
        if self.step_mode == "m":
            self._validate_multi_step(x)
            return functional.seq_to_ann_forward(x, self._apply_op)
        raise ValueError(self.step_mode)


class _StatelessCat(nn.Module, base.StepModule):
    def __init__(self, dim: int = 0, step_mode: str = "s") -> None:
        super().__init__()
        self.dim = dim
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"dim={self.dim}, step_mode={self.step_mode}"

    def forward(self, *tensors: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return torch.cat(tensors, dim=self.dim)
        if self.step_mode == "m":
            if self.dim == 0:
                raise ValueError(
                    f"{self.__class__.__name__} does not support concatenating "
                    "along the original batch dimension in multi-step mode."
                )
            return torch.cat(tensors, dim=self.dim + 1)
        raise ValueError(self.step_mode)


class _StatelessShape(nn.Module, base.StepModule):
    def __init__(self, step_mode: str = "s") -> None:
        super().__init__()
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor) -> torch.Size:
        if self.step_mode == "s":
            return x.shape
        if self.step_mode == "m":
            return x.shape[1:]
        raise ValueError(self.step_mode)


class _StatelessDim(nn.Module, base.StepModule):
    def __init__(self, step_mode: str = "s") -> None:
        super().__init__()
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor) -> int:
        if self.step_mode == "s":
            return x.dim()
        if self.step_mode == "m":
            return x.dim() - 1
        raise ValueError(self.step_mode)


class _StatelessReshape(nn.Module, base.StepModule):
    def __init__(self, step_mode: str = "s") -> None:
        super().__init__()
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor, *sizes: int) -> torch.Tensor:
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        if self.step_mode == "s":
            return x.reshape(*sizes)
        if self.step_mode == "m":
            if not sizes:
                raise ValueError("reshape requires at least one output size.")
            if sizes[0] != x.shape[1]:
                raise ValueError(
                    f"{self.__class__.__name__} only supports reshapes that "
                    "preserve the original batch dimension in multi-step mode."
                )
            flat = x.flatten(0, 1)
            y = flat.reshape(flat.shape[0], *sizes[1:])
            return y.view(x.shape[0], x.shape[1], *y.shape[1:])
        raise ValueError(self.step_mode)


class _StatelessExpand(nn.Module, base.StepModule):
    def __init__(self, step_mode: str = "s") -> None:
        super().__init__()
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor, *sizes: int) -> torch.Tensor:
        if len(sizes) == 1 and isinstance(sizes[0], (tuple, list)):
            sizes = tuple(sizes[0])
        if self.step_mode == "s":
            return x.expand(*sizes)
        if self.step_mode == "m":
            if x.dim() != len(sizes) + 1:
                raise ValueError(
                    f"{self.__class__.__name__} requires a time-distributed "
                    "input tensor in multi-step mode."
                )
            return x.expand(x.shape[0], *sizes)
        raise ValueError(self.step_mode)


class _StatelessGetItem(nn.Module, base.StepModule):
    def __init__(self, item: Any, step_mode: str = "s") -> None:
        super().__init__()
        self.item = item
        self.step_mode = step_mode

    def extra_repr(self) -> str:
        return f"item={self.item!r}, step_mode={self.step_mode}"

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.step_mode == "s":
            return x[self.item]
        if self.step_mode == "m":
            return functional.seq_to_ann_forward(x, lambda value: value[self.item])
        raise ValueError(self.step_mode)


def _first_tensor_options(module: nn.Module) -> Dict[str, Any]:
    options: Dict[str, Any] = {}
    for tensor in list(module.parameters(recurse=False)) + list(
        module.buffers(recurse=False)
    ):
        options["device"] = tensor.device
        if tensor.is_floating_point():
            options["dtype"] = tensor.dtype
        return options
    return options


def _copy_module_state(source: nn.Module, target: nn.Module) -> nn.Module:
    options = _first_tensor_options(source)
    if options:
        target.to(**options)
    target.load_state_dict(source.state_dict(), strict=True)
    target.train(source.training)
    for name, parameter in source.named_parameters(recurse=False):
        target_parameter = getattr(target, name, None)
        if isinstance(target_parameter, nn.Parameter):
            target_parameter.requires_grad = parameter.requires_grad
    return target


def _make_step_module(module: nn.Module) -> Optional[nn.Module]:
    if isinstance(module, nn.Conv1d):
        return _copy_module_state(
            module,
            layer.Conv1d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.Conv2d):
        return _copy_module_state(
            module,
            layer.Conv2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.Conv3d):
        return _copy_module_state(
            module,
            layer.Conv3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                dilation=module.dilation,
                groups=module.groups,
                bias=module.bias is not None,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.ConvTranspose1d):
        return _copy_module_state(
            module,
            layer.ConvTranspose1d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                groups=module.groups,
                bias=module.bias is not None,
                dilation=module.dilation,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.ConvTranspose2d):
        return _copy_module_state(
            module,
            layer.ConvTranspose2d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                groups=module.groups,
                bias=module.bias is not None,
                dilation=module.dilation,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.ConvTranspose3d):
        return _copy_module_state(
            module,
            layer.ConvTranspose3d(
                module.in_channels,
                module.out_channels,
                module.kernel_size,
                stride=module.stride,
                padding=module.padding,
                output_padding=module.output_padding,
                groups=module.groups,
                bias=module.bias is not None,
                dilation=module.dilation,
                padding_mode=module.padding_mode,
            ),
        )
    if isinstance(module, nn.Linear):
        return _copy_module_state(
            module,
            layer.Linear(
                module.in_features,
                module.out_features,
                bias=module.bias is not None,
            ),
        )
    if isinstance(module, (nn.MaxPool1d, nn.MaxPool2d, nn.MaxPool3d)):
        if module.return_indices:
            raise ValueError("Step-mode adapter does not support return_indices=True.")
        if isinstance(module, nn.MaxPool1d):
            pool_cls = layer.MaxPool1d
        elif isinstance(module, nn.MaxPool2d):
            pool_cls = layer.MaxPool2d
        else:
            pool_cls = layer.MaxPool3d
        return pool_cls(
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            dilation=module.dilation,
            return_indices=False,
            ceil_mode=module.ceil_mode,
        )
    if isinstance(module, (nn.AvgPool1d, nn.AvgPool2d, nn.AvgPool3d)):
        if isinstance(module, nn.AvgPool1d):
            pool_cls = layer.AvgPool1d
        elif isinstance(module, nn.AvgPool2d):
            pool_cls = layer.AvgPool2d
        else:
            pool_cls = layer.AvgPool3d
        return pool_cls(
            module.kernel_size,
            stride=module.stride,
            padding=module.padding,
            ceil_mode=module.ceil_mode,
            count_include_pad=module.count_include_pad,
            divisor_override=module.divisor_override,
        )
    if isinstance(
        module,
        (nn.AdaptiveAvgPool1d, nn.AdaptiveAvgPool2d, nn.AdaptiveAvgPool3d),
    ):
        if isinstance(module, nn.AdaptiveAvgPool1d):
            pool_cls = layer.AdaptiveAvgPool1d
        elif isinstance(module, nn.AdaptiveAvgPool2d):
            pool_cls = layer.AdaptiveAvgPool2d
        else:
            pool_cls = layer.AdaptiveAvgPool3d
        return pool_cls(module.output_size)
    if isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d)):
        if isinstance(module, nn.BatchNorm1d):
            bn_cls = layer.BatchNorm1d
        elif isinstance(module, nn.BatchNorm2d):
            bn_cls = layer.BatchNorm2d
        else:
            bn_cls = layer.BatchNorm3d
        return _copy_module_state(
            module,
            bn_cls(
                module.num_features,
                eps=module.eps,
                momentum=module.momentum,
                affine=module.affine,
                track_running_stats=module.track_running_stats,
            ),
        )
    if isinstance(module, nn.GroupNorm):
        return _copy_module_state(
            module,
            layer.GroupNorm(
                module.num_groups,
                module.num_channels,
                eps=module.eps,
                affine=module.affine,
            ),
        )
    if isinstance(module, nn.Upsample):
        return layer.Upsample(
            size=module.size,
            scale_factor=module.scale_factor,
            mode=module.mode,
            align_corners=module.align_corners,
            recompute_scale_factor=module.recompute_scale_factor,
        )
    if isinstance(module, nn.Flatten):
        return layer.Flatten(module.start_dim, module.end_dim)
    if isinstance(module, nn.Dropout2d):
        replacement = layer.Dropout2d(p=module.p)
        replacement.train(module.training)
        return replacement
    if isinstance(module, nn.Dropout):
        replacement = layer.Dropout(p=module.p)
        replacement.train(module.training)
        return replacement
    return None


def _replace_submodule(
    fx_model: fx.GraphModule,
    target: str,
    module: nn.Module,
) -> None:
    parent_name, _, child_name = target.rpartition(".")
    parent = fx_model.get_submodule(parent_name) if parent_name else fx_model
    setattr(parent, child_name, module)


def _tensor_op_input_arg(node: fx.Node, context: str) -> fx.Node:
    if len(node.args) >= 1:
        return node.args[0]
    if "input" in node.kwargs:
        return node.kwargs["input"]
    raise ValueError(
        f"{context} requires tensor op {node.target!r} to have an input "
        "tensor argument."
    )


def _check_literal_dim(dim: Any, context: str) -> None:
    if isinstance(dim, int):
        return
    if isinstance(dim, tuple):
        for d in dim:
            _check_literal_dim(d, context)
        return
    if isinstance(dim, list):
        for d in dim:
            _check_literal_dim(d, context)
        return
    raise ValueError(
        f"{context} requires literal integer dimensions, but got {dim!r}."
    )


def _make_mean_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 1:
        dim = args[1]
    elif "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        raise ValueError(f"{context} does not support mean without an explicit dim.")
    keepdim = kwargs.get("keepdim", False)
    if len(args) > 2:
        keepdim = args[2]
    if keepdim not in (False, True):
        raise ValueError(
            f"{context} requires literal bool keepdim, but got {keepdim!r}."
        )
    _check_literal_dim(dim, context)
    return _StatelessTensorOp(
        op_name="mean",
        op_kwargs={"dim": dim, "keepdim": bool(keepdim)},
    )


def _make_flatten_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 1:
        start_dim = args[1]
    elif "start_dim" in kwargs:
        start_dim = kwargs["start_dim"]
    else:
        start_dim = 0
    if len(args) > 2:
        end_dim = args[2]
    elif "end_dim" in kwargs:
        end_dim = kwargs["end_dim"]
    else:
        end_dim = -1
    _check_literal_dim(start_dim, context)
    _check_literal_dim(end_dim, context)
    return _StatelessTensorOp(
        op_name="flatten",
        op_kwargs={"start_dim": start_dim, "end_dim": end_dim},
    )


def _make_transpose_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 2:
        dim0 = args[1]
        dim1 = args[2]
    else:
        dim0 = kwargs["dim0"]
        dim1 = kwargs["dim1"]
    _check_literal_dim(dim0, context)
    _check_literal_dim(dim1, context)
    return _StatelessTensorOp(
        op_name="transpose",
        op_kwargs={"dim0": dim0, "dim1": dim1},
    )


def _make_unsqueeze_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 1:
        dim = args[1]
    elif "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        raise ValueError(f"{context} does not support unsqueeze without a dim.")
    _check_literal_dim(dim, context)
    return _StatelessTensorOp(
        op_name="unsqueeze",
        op_kwargs={"dim": dim},
    )


def _make_unflatten_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if len(args) > 1:
        dim = args[1]
    elif "dim" in kwargs:
        dim = kwargs["dim"]
    else:
        raise ValueError(f"{context} does not support unflatten without a dim.")
    if len(args) > 2:
        unflattened_size = args[2]
    elif "sizes" in kwargs:
        unflattened_size = kwargs["sizes"]
    elif "unflattened_size" in kwargs:
        unflattened_size = kwargs["unflattened_size"]
    else:
        raise ValueError(f"{context} does not support unflatten without sizes.")
    _check_literal_dim(dim, context)
    if not isinstance(unflattened_size, (tuple, list, torch.Size)):
        raise ValueError(
            f"{context} requires literal unflatten sizes, but got "
            f"{unflattened_size!r}."
        )
    return _StatelessTensorOp(
        op_name="unflatten",
        op_kwargs={"dim": dim, "unflattened_size": tuple(unflattened_size)},
    )


def _make_permute_module(node: fx.Node, context: str) -> _StatelessTensorOp:
    args = list(node.args)
    if len(args) < 2:
        raise ValueError(f"{context} does not support permute without dims.")
    dims = args[1:]
    if len(dims) == 1 and isinstance(dims[0], (tuple, list)):
        dims = tuple(dims[0])
    for dim in dims:
        _check_literal_dim(dim, context)
    if not dims or dims[0] != 0:
        raise ValueError(
            f"{context} requires permute to preserve the original batch "
            "dimension."
        )
    return _StatelessTensorOp(op_name="permute", op_kwargs={"dims": tuple(dims)})


def _make_cat_module(node: fx.Node, context: str) -> Tuple[_StatelessCat, Tuple[Any, ...]]:
    args = list(node.args)
    kwargs = dict(node.kwargs)
    if args:
        tensors = args[0]
    elif "tensors" in kwargs:
        tensors = kwargs["tensors"]
    else:
        raise ValueError(f"{context} does not support cat without tensors.")
    dim = kwargs.get("dim", 0)
    if len(args) > 1:
        dim = args[1]
    _check_literal_dim(dim, context)
    if not isinstance(tensors, (tuple, list)):
        raise ValueError(f"{context} requires cat tensors to be a tuple or list.")
    return _StatelessCat(dim=int(dim)), tuple(tensors)


def _getitem_preserves_batch_dim(item: Any) -> bool:
    if isinstance(item, slice):
        return item == slice(None, None, None)
    if isinstance(item, tuple):
        if not item:
            return False
        first = item[0]
        return isinstance(first, slice) and first == slice(None, None, None)
    return False


def _is_non_tensor_getitem_input(node: Any) -> bool:
    return isinstance(node, fx.Node) and (
        (
            node.op == "call_function"
            and node.target is builtins.getattr
            and len(node.args) == 2
            and node.args[1] == "shape"
        )
        or (
            node.op == "call_method"
            and node.target == "size"
        )
        or node.meta.get("ann2snn_step_mode_module") is _StatelessShape
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


def _replace_tensor_op_node(
    fx_model: fx.GraphModule,
    node: fx.Node,
    module: nn.Module,
    existing_modules: set[str],
    tensor_op_index: int,
    context: str,
    input_args: Optional[Tuple[Any, ...]] = None,
) -> int:
    target = f"step_mode_tensor_op_{tensor_op_index}"
    while target in existing_modules:
        tensor_op_index += 1
        target = f"step_mode_tensor_op_{tensor_op_index}"
    tensor_op_index += 1
    fx_model.add_submodule(target, module)
    existing_modules.add(target)
    with fx_model.graph.inserting_after(node):
        new_node = fx_model.graph.call_module(
            target,
            args=input_args or (_tensor_op_input_arg(node, context),),
        )
    new_node.meta["ann2snn_step_mode_module"] = module.__class__
    node.replace_all_uses_with(new_node)
    fx_model.graph.erase_node(node)
    return tensor_op_index


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
        torch.add,
        torch._assert,
        operator.mul,
    }
    rewritten_call_functions = {
        operator.getitem,
        torch.cat,
        builtins.getattr,
        torch.flatten,
        torch.mean,
        torch.transpose,
        torch.unsqueeze,
    }

    for node in list(fx_model.graph.nodes):
        if node.op == "call_module":
            if not isinstance(node.target, str):
                raise ValueError(f"{context} got a non-string module target.")
            module = modules.get(node.target)
            if module is None:
                module = fx_model.get_submodule(node.target)
                modules[node.target] = module
            if isinstance(module, base.StepModule) or isinstance(
                module, base.MemoryModule
            ):
                continue
            if isinstance(module, nn.Identity):
                continue
            if safe_module_types and isinstance(module, safe_module_types):
                continue
            if wrap_module_types and isinstance(module, wrap_module_types):
                replacement = _make_step_module(module)
                if replacement is None:
                    raise ValueError(
                        f"{context} does not support wrapping module "
                        f"{node.target!r} of type {type(module).__name__}."
                    )
                _replace_submodule(fx_model, node.target, replacement)
                modules[node.target] = replacement
                continue
            raise ValueError(
                f"{context} does not support module node {node.target!r} of "
                f"type {type(module).__name__}."
            )

        if node.op == "call_method":
            if node.target == "size":
                continue
            if node.target == "dim":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _StatelessDim(),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "view":
                if (
                    len(node.args) == 3
                    and isinstance(node.args[1], fx.Node)
                    and node.args[1].op == "call_method"
                    and node.args[1].target == "size"
                    and node.args[1].args == (node.args[0], 0)
                    and node.args[2] == -1
                ):
                    size_node = node.args[1]
                    tensor_op_index = _replace_tensor_op_node(
                        fx_model,
                        node,
                        _StatelessTensorOp(
                            op_name="flatten",
                            op_kwargs={"start_dim": 1, "end_dim": -1},
                        ),
                        existing_modules,
                        tensor_op_index,
                        context,
                    )
                    if not size_node.users:
                        fx_model.graph.erase_node(size_node)
                    continue
                raise ValueError(
                    f"{context} does not support tensor method {node.target!r}. "
                    "Use a sequence-preserving flatten/view pattern."
                )
            if node.target == "flatten":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_flatten_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "reshape":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _StatelessReshape(),
                    existing_modules,
                    tensor_op_index,
                    context,
                    input_args=tuple(node.args),
                )
                continue
            if node.target == "permute":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_permute_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "expand":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _StatelessExpand(),
                    existing_modules,
                    tensor_op_index,
                    context,
                    input_args=tuple(node.args),
                )
                continue
            if node.target == "transpose":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_transpose_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "unsqueeze":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_unsqueeze_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "unflatten":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_unflatten_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "mean":
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_mean_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
                continue
            if node.target == "masked_fill":
                continue
            raise ValueError(
                f"{context} does not support tensor method {node.target!r}. "
                "Rewrite the model with supported sequence-preserving operations."
            )

        if node.op == "call_function":
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
            if node.target is builtins.getattr:
                if len(node.args) != 2 or node.args[1] != "shape":
                    raise ValueError(
                        f"{context} only supports getattr(tensor, 'shape')."
                    )
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _StatelessShape(),
                    existing_modules,
                    tensor_op_index,
                    context,
                    input_args=(node.args[0],),
                )
                continue
            if node.target is operator.getitem:
                if len(node.args) < 2:
                    raise ValueError(f"{context} got malformed getitem node.")
                item = node.args[1]
                if isinstance(item, int):
                    if not _is_non_tensor_getitem_input(node.args[0]):
                        raise ValueError(
                            f"{context} does not support integer tensor "
                            "getitem because it is not sequence-preserving."
                        )
                    continue
                if not isinstance(item, (slice, tuple)):
                    raise ValueError(
                        f"{context} does not support getitem index {item!r}."
                    )
                if not _getitem_preserves_batch_dim(item):
                    raise ValueError(
                        f"{context} requires tensor getitem to preserve the "
                        "original batch dimension."
                    )
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _StatelessGetItem(item),
                    existing_modules,
                    tensor_op_index,
                    context,
                    input_args=(node.args[0],),
                )
                continue
            if node.target is torch.flatten:
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_flatten_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
            elif node.target is torch.mean:
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_mean_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
            elif node.target is torch.transpose:
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_transpose_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
            elif node.target is torch.unsqueeze:
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    _make_unsqueeze_module(node, context),
                    existing_modules,
                    tensor_op_index,
                    context,
                )
            elif node.target is torch.cat:
                module, input_args = _make_cat_module(node, context)
                tensor_op_index = _replace_tensor_op_node(
                    fx_model,
                    node,
                    module,
                    existing_modules,
                    tensor_op_index,
                    context,
                    input_args=input_args,
                )

    fx_model.graph.lint()
    fx_model.delete_all_unused_submodules()
    fx_model.recompile()
    return fx_model
