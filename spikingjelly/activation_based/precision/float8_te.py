from __future__ import annotations

from contextlib import nullcontext
import importlib
import warnings

import torch
import torch.nn as nn

from .. import layer
from .capability import build_capability_report, validate_capability
from .float8_base import wrap_float8_linear_module
from .float8_conv import (
    is_supported_pointwise_conv1d,
    make_linear_from_pointwise_conv1d,
    wrap_float8_pointwise_conv1d_module,
)
from .policy import PrecisionPolicy

_SUPPORTED_TE_RECIPES = {"auto", "delayed", "current", "block", "mxfp8"}


def _import_te_pytorch():
    import transformer_engine.pytorch as te

    return te


def _replace_child(module: nn.Module, child_name: str, wrapped: nn.Module) -> None:
    if isinstance(module, nn.ModuleDict):
        module[child_name] = wrapped
    elif isinstance(module, (nn.ModuleList, nn.Sequential)):
        try:
            module[int(child_name)] = wrapped
        except ValueError:
            setattr(module, child_name, wrapped)
    else:
        setattr(module, child_name, wrapped)


def _copy_linear_parameters(source: nn.Module, target: nn.Module) -> None:
    with torch.no_grad():
        target.weight.copy_(source.weight)
        source_bias = getattr(source, "bias", None)
        target_bias = getattr(target, "bias", None)
        if source_bias is not None and target_bias is not None:
            target_bias.copy_(source_bias)


def _copy_to_first_existing(
    value: torch.Tensor | None,
    target: nn.Module,
    names: tuple[str, ...],
) -> bool:
    if value is None:
        return True
    for name in names:
        target_value = getattr(target, name, None)
        if target_value is not None and hasattr(target_value, "copy_"):
            with torch.no_grad():
                target_value.copy_(value)
            return True
    return False


def _first_existing_name(target: nn.Module, names: tuple[str, ...]) -> str:
    for name in names:
        if getattr(target, name, None) is not None:
            return name
    return names[0]


def _make_te_linear(source: nn.Module, TELinear: type) -> nn.Module:
    bias = getattr(source, "bias", None) is not None
    kwargs = {
        "bias": bias,
        "params_dtype": source.weight.dtype,
    }
    try:
        converted = TELinear(source.in_features, source.out_features, **kwargs)
    except TypeError:
        converted = TELinear(source.in_features, source.out_features, bias=bias)
    converted = converted.to(device=source.weight.device, dtype=source.weight.dtype)
    _copy_linear_parameters(source, converted)
    return converted


def _is_supported_layer_norm(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.LayerNorm)
        and module.elementwise_affine
        and module.weight is not None
        and module.bias is not None
        and len(tuple(module.normalized_shape)) == 1
    )


def _make_te_layer_norm(source: nn.LayerNorm, TELayerNorm: type) -> nn.Module:
    hidden_size = int(source.normalized_shape[0])
    kwargs = {
        "eps": source.eps,
        "params_dtype": source.weight.dtype,
    }
    try:
        converted = TELayerNorm(hidden_size, **kwargs)
    except TypeError:
        converted = TELayerNorm(hidden_size, eps=source.eps)
    converted = converted.to(device=source.weight.device, dtype=source.weight.dtype)
    if not _copy_to_first_existing(
        source.weight,
        converted,
        ("weight", "layer_norm_weight", "ln_weight"),
    ):
        raise RuntimeError(
            "TE LayerNorm does not expose a compatible weight parameter."
        )
    if not _copy_to_first_existing(
        source.bias,
        converted,
        ("bias", "layer_norm_bias", "ln_bias"),
    ):
        raise RuntimeError("TE LayerNorm does not expose a compatible bias parameter.")
    return converted


def _make_te_layer_norm_linear(
    norm: nn.LayerNorm,
    linear: nn.Linear,
    TELayerNormLinear: type,
) -> nn.Module:
    hidden_size = int(norm.normalized_shape[0])
    bias = getattr(linear, "bias", None) is not None
    kwargs = {
        "eps": norm.eps,
        "bias": bias,
        "params_dtype": linear.weight.dtype,
    }
    try:
        converted = TELayerNormLinear(hidden_size, linear.out_features, **kwargs)
    except TypeError:
        converted = TELayerNormLinear(hidden_size, linear.out_features, bias=bias)
    converted = converted.to(device=linear.weight.device, dtype=linear.weight.dtype)
    copies = (
        _copy_to_first_existing(
            norm.weight, converted, ("layer_norm_weight", "ln_weight")
        ),
        _copy_to_first_existing(norm.bias, converted, ("layer_norm_bias", "ln_bias")),
        _copy_to_first_existing(linear.weight, converted, ("weight",)),
        _copy_to_first_existing(linear.bias, converted, ("bias",)),
    )
    if not all(copies):
        raise RuntimeError(
            "TE LayerNormLinear does not expose compatible parameter names."
        )
    return converted


def _make_te_layer_norm_mlp(
    norm: nn.LayerNorm,
    fc1: nn.Linear,
    fc2: nn.Linear,
    TELayerNormMLP: type,
) -> nn.Module:
    hidden_size = int(norm.normalized_shape[0])
    bias = (
        getattr(fc1, "bias", None) is not None or getattr(fc2, "bias", None) is not None
    )
    kwargs = {
        "eps": norm.eps,
        "bias": bias,
        "activation": "gelu",
        "params_dtype": fc1.weight.dtype,
    }
    try:
        converted = TELayerNormMLP(hidden_size, fc1.out_features, **kwargs)
    except TypeError:
        try:
            converted = TELayerNormMLP(
                hidden_size,
                fc1.out_features,
                bias=bias,
                activation="gelu",
            )
        except TypeError:
            warnings.warn(
                "TE LayerNormMLP does not accept activation='gelu'; falling back "
                "to the TE default activation.",
                RuntimeWarning,
                stacklevel=2,
            )
            converted = TELayerNormMLP(hidden_size, fc1.out_features, bias=bias)
    converted = converted.to(device=fc1.weight.device, dtype=fc1.weight.dtype)
    copies = (
        _copy_to_first_existing(
            norm.weight, converted, ("layer_norm_weight", "ln_weight")
        ),
        _copy_to_first_existing(norm.bias, converted, ("layer_norm_bias", "ln_bias")),
        _copy_to_first_existing(fc1.weight, converted, ("fc1_weight", "weight1")),
        _copy_to_first_existing(fc1.bias, converted, ("fc1_bias", "bias1")),
        _copy_to_first_existing(fc2.weight, converted, ("fc2_weight", "weight2")),
        _copy_to_first_existing(fc2.bias, converted, ("fc2_bias", "bias2")),
    )
    if not all(copies):
        raise RuntimeError(
            "TE LayerNormMLP does not expose compatible parameter names."
        )
    return converted


class _Float8TEPatternModule(nn.Module):
    def __init__(
        self,
        wrapped: nn.Module,
        state_key_map: dict[str, str],
    ):
        super().__init__()
        self.wrapped = wrapped
        self._state_key_map = state_key_map

    def _unwrap_output(self, output):
        return output[0] if isinstance(output, tuple) else output

    def forward(self, x):
        return self._unwrap_output(self.wrapped(x))

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__.get("_modules", {}).get("wrapped")
            if wrapped is None:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                ) from None
            return getattr(wrapped, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        if destination is None:
            destination = {}
        wrapped_state = self.wrapped.state_dict(keep_vars=keep_vars)
        mapped_wrapped_keys = set(self._state_key_map.values())
        for public_key, wrapped_key in self._state_key_map.items():
            if wrapped_key in wrapped_state:
                destination[prefix + public_key] = wrapped_state[wrapped_key]
        for wrapped_key, value in wrapped_state.items():
            if wrapped_key not in mapped_wrapped_keys:
                destination[prefix + "wrapped." + wrapped_key] = value
        return destination

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ):
        wrapped_prefix = prefix + "wrapped."
        for public_key, wrapped_key in self._state_key_map.items():
            key = prefix + public_key
            if key in state_dict:
                state_dict[wrapped_prefix + wrapped_key] = state_dict.pop(key)


class Float8TELayerNormLinearModule(_Float8TEPatternModule):
    pass


class Float8TELayerNormMLPModule(_Float8TEPatternModule):
    pass


def _is_layer_norm_linear_pattern(module: nn.Module) -> bool:
    children = list(module.children())
    if not (
        isinstance(module, nn.Sequential)
        and len(children) == 2
        and _is_supported_layer_norm(children[0])
        and isinstance(children[1], (nn.Linear, layer.Linear))
    ):
        return False
    hidden_size = int(children[0].normalized_shape[0])
    linear = children[1]
    return linear.in_features == hidden_size


def _is_layer_norm_mlp_shape_compatible(
    norm: nn.LayerNorm,
    fc1: nn.Linear,
    fc2: nn.Linear,
) -> bool:
    hidden_size = int(norm.normalized_shape[0])
    return (
        fc1.in_features == hidden_size
        and fc2.in_features == fc1.out_features
        and fc2.out_features == hidden_size
        and (fc1.bias is None) == (fc2.bias is None)
    )


def _is_layer_norm_mlp_pattern(module: nn.Module) -> bool:
    children = list(module.children())
    if not (
        isinstance(module, nn.Sequential)
        and len(children) == 4
        and _is_supported_layer_norm(children[0])
        and isinstance(children[1], (nn.Linear, layer.Linear))
        and isinstance(children[2], nn.GELU)
        and getattr(children[2], "approximate", "none") in (None, "none")
        and isinstance(children[3], (nn.Linear, layer.Linear))
    ):
        return False
    return _is_layer_norm_mlp_shape_compatible(children[0], children[1], children[3])


def _convert_layer_norm_linear_pattern(module: nn.Sequential, te) -> nn.Module | None:
    TELayerNormLinear = getattr(te, "LayerNormLinear", None)
    if TELayerNormLinear is None:
        return None
    norm, linear = list(module.children())
    converted = _make_te_layer_norm_linear(norm, linear, TELayerNormLinear)
    return Float8TELayerNormLinearModule(
        converted,
        {
            "0.weight": _first_existing_name(
                converted, ("layer_norm_weight", "ln_weight")
            ),
            "0.bias": _first_existing_name(converted, ("layer_norm_bias", "ln_bias")),
            "1.weight": _first_existing_name(converted, ("weight",)),
            "1.bias": _first_existing_name(converted, ("bias",)),
        },
    )


def _convert_layer_norm_mlp_pattern(module: nn.Sequential, te) -> nn.Module | None:
    TELayerNormMLP = getattr(te, "LayerNormMLP", None)
    if TELayerNormMLP is None:
        return None
    norm, fc1, _, fc2 = list(module.children())
    converted = _make_te_layer_norm_mlp(norm, fc1, fc2, TELayerNormMLP)
    return Float8TELayerNormMLPModule(
        converted,
        {
            "0.weight": _first_existing_name(
                converted, ("layer_norm_weight", "ln_weight")
            ),
            "0.bias": _first_existing_name(converted, ("layer_norm_bias", "ln_bias")),
            "1.weight": _first_existing_name(converted, ("fc1_weight", "weight1")),
            "1.bias": _first_existing_name(converted, ("fc1_bias", "bias1")),
            "3.weight": _first_existing_name(converted, ("fc2_weight", "weight2")),
            "3.bias": _first_existing_name(converted, ("fc2_bias", "bias2")),
        },
    )


def _convert_fused_pattern(module: nn.Module, te) -> tuple[nn.Module, str] | None:
    if _is_layer_norm_mlp_pattern(module):
        converted = _convert_layer_norm_mlp_pattern(module, te)
        if converted is not None:
            return converted, "LayerNormMLP"
    if _is_layer_norm_linear_pattern(module):
        converted = _convert_layer_norm_linear_pattern(module, te)
        if converted is not None:
            return converted, "LayerNormLinear"
    return None


def _te_recursive_convert(root: nn.Module, TELinear: type, report) -> None:
    memo: dict[nn.Module, nn.Module] = {}
    visited: set[int] = set()
    te = _import_te_pytorch()

    def _walk(module: nn.Module, prefix: str = "") -> None:
        for child_name, child in list(module._modules.items()):
            if child is None:
                continue
            child_fqn = f"{prefix}.{child_name}" if prefix else child_name
            if child in memo:
                _replace_child(module, child_name, memo[child])
                report.converted_modules.append(child_fqn)
                continue
            pattern = _convert_fused_pattern(child, te)
            if pattern is not None:
                wrapped, pattern_name = pattern
                memo[child] = wrapped
                _replace_child(module, child_name, wrapped)
                report.converted_modules.append(child_fqn)
                report.converted_patterns.append(
                    {"module": child_fqn, "pattern": pattern_name, "backend": "te"}
                )
            elif (
                isinstance(child, (nn.Linear, layer.Linear))
                or is_supported_pointwise_conv1d(child)
                or _is_supported_layer_norm(child)
            ):
                if is_supported_pointwise_conv1d(child):
                    source = make_linear_from_pointwise_conv1d(child)
                    converted = _make_te_linear(source, TELinear)
                    wrapped = wrap_float8_pointwise_conv1d_module(child, converted)
                elif _is_supported_layer_norm(child):
                    TELayerNorm = getattr(te, "LayerNorm", None)
                    if TELayerNorm is None:
                        report.skipped_modules.append(child_fqn)
                        continue
                    wrapped = _make_te_layer_norm(child, TELayerNorm)
                else:
                    converted = _make_te_linear(child, TELinear)
                    wrapped = wrap_float8_linear_module(child, converted)
                memo[child] = wrapped
                _replace_child(module, child_name, wrapped)
                report.converted_modules.append(child_fqn)
            else:
                report.skipped_modules.append(child_fqn)
                child_id = id(child)
                if child_id in visited:
                    continue
                visited.add(child_id)
                _walk(child, child_fqn)

    _walk(root)


class Float8TransformerEnginePolicy(PrecisionPolicy):
    name = "fp8-te"
    supports_layer_norm_conversion = True

    def __init__(
        self,
        device_type: str = "cuda",
        strict: str = "warn",
        fp8_recipe: str = "auto",
    ):
        super().__init__()
        self.device_type = device_type
        self.strict = strict
        self.fp8_recipe = fp8_recipe
        self._resolved_recipe_name: str | None = None
        self._resolved_recipe = None
        self._recipe_resolved = False

    def describe(self) -> dict:
        return {
            "name": self.name,
            "backend": "transformer-engine",
            "device_type": self.device_type,
            "strict": self.strict,
            "fp8_recipe": self.fp8_recipe,
            "resolved_fp8_recipe": self._resolved_recipe_name,
            "autocast": True,
            "grad_scaler": False,
        }

    def check_capability(self, model, device) -> None:
        report = build_capability_report(model, device, self.name)
        self._capability_report = report
        self._annotate_recipe_report(report)
        validate_capability(report)
        self._ensure_model_on_device(model, device)

    def _annotate_recipe_report(self, report: dict) -> None:
        recipe_name = str(self.fp8_recipe or "auto").lower()
        report["te_recipe_requested"] = recipe_name
        if recipe_name not in _SUPPORTED_TE_RECIPES:
            report["te_recipe_resolved"] = None
            report["te_recipe_fallback_reason"] = (
                f"unsupported Transformer Engine FP8 recipe {recipe_name!r}"
            )
            raise RuntimeError(report["te_recipe_fallback_reason"])
        availability = report.get("te_recipe_availability") or {}
        if recipe_name in {"block", "mxfp8"} and availability.get(recipe_name) is False:
            report["te_recipe_resolved"] = None
            report["te_recipe_fallback_reason"] = (
                f"Transformer Engine FP8 recipe {recipe_name!r} is unavailable"
            )
            raise RuntimeError(report["te_recipe_fallback_reason"])
        report["te_recipe_resolved"] = recipe_name
        report["te_recipe_fallback_reason"] = None
        self._resolved_recipe_name = recipe_name

    def _resolve_recipe(self, te):
        recipe_name = str(self.fp8_recipe or "auto").lower()
        if recipe_name not in _SUPPORTED_TE_RECIPES:
            raise RuntimeError(
                f"unsupported Transformer Engine FP8 recipe {recipe_name!r}"
            )
        if recipe_name == "auto":
            get_default_recipe = getattr(te, "get_default_recipe", None)
            self._resolved_recipe_name = "auto"
            self._resolved_recipe = get_default_recipe() if get_default_recipe else None
            self._recipe_resolved = True
            return self._resolved_recipe

        try:
            recipe_module = importlib.import_module("transformer_engine.common.recipe")
        except ImportError as exc:
            raise RuntimeError(
                "transformer-engine recipe module is unavailable."
            ) from exc

        constructors = {
            "delayed": ("DelayedScaling",),
            "current": ("Float8CurrentScaling", "CurrentScaling"),
            "block": ("Float8BlockScaling",),
            "mxfp8": ("MXFP8BlockScaling",),
        }
        for class_name in constructors[recipe_name]:
            recipe_cls = getattr(recipe_module, class_name, None)
            if recipe_cls is not None:
                self._resolved_recipe_name = recipe_name
                self._resolved_recipe = recipe_cls()
                self._recipe_resolved = True
                return self._resolved_recipe
        raise RuntimeError(
            f"transformer-engine does not expose a recipe class for {recipe_name!r}."
        )

    def _ensure_model_on_device(self, model, device) -> None:
        model_devices = {p.device for p in model.parameters()} | {
            b.device for b in model.buffers()
        }
        target_device = torch.device(device)
        if target_device.type == "cuda" and target_device.index is None:
            target_device = torch.device("cuda", torch.cuda.current_device())
        if model_devices and any(d != target_device for d in model_devices):
            raise RuntimeError(
                f"All model parameters and buffers must be moved to the target CUDA device "
                f"'{target_device}' (e.g. model.to('{target_device}')) before "
                "calling prepare_model_for_precision() for 'fp8-te'."
            )

    def autocast_context(self):
        try:
            te = _import_te_pytorch()
        except ImportError:
            return nullcontext()
        recipe = (
            self._resolved_recipe if self._recipe_resolved else self._resolve_recipe(te)
        )
        autocast = getattr(te, "autocast", None)
        if autocast is not None:
            if recipe is None:
                try:
                    return autocast(enabled=True, device=self.device_type)
                except TypeError:
                    return autocast(enabled=True)
            try:
                return autocast(
                    enabled=True,
                    recipe=recipe,
                    device=self.device_type,
                )
            except TypeError:
                return autocast(enabled=True, recipe=recipe)
        fp8_autocast = getattr(te, "fp8_autocast", None)
        if fp8_autocast is not None:
            warnings.warn(
                "transformer_engine.pytorch.fp8_autocast is deprecated; "
                "falling back because te.autocast is unavailable.",
                RuntimeWarning,
                stacklevel=2,
            )
            if recipe is None:
                try:
                    return fp8_autocast(enabled=True, device=self.device_type)
                except TypeError:
                    return fp8_autocast(enabled=True)
            try:
                return fp8_autocast(
                    enabled=True,
                    fp8_recipe=recipe,
                    device=self.device_type,
                )
            except TypeError:
                try:
                    return fp8_autocast(enabled=True, fp8_recipe=recipe)
                except TypeError:
                    try:
                        return fp8_autocast(
                            enabled=True,
                            recipe=recipe,
                            device=self.device_type,
                        )
                    except TypeError:
                        try:
                            return fp8_autocast(enabled=True, recipe=recipe)
                        except TypeError:
                            warnings.warn(
                                "transformer_engine.pytorch.fp8_autocast does not "
                                "accept an FP8 recipe argument; using its default "
                                "recipe.",
                                RuntimeWarning,
                                stacklevel=2,
                            )
                            return fp8_autocast(enabled=True)
        return nullcontext()

    def _convert_modules(self, model, report):
        te = _import_te_pytorch()
        TELinear = te.Linear

        if isinstance(model, (nn.Linear, layer.Linear)):
            converted = _make_te_linear(model, TELinear)
            report.converted_modules.append("<root>")
            return wrap_float8_linear_module(model, converted)
        if is_supported_pointwise_conv1d(model):
            source = make_linear_from_pointwise_conv1d(model)
            converted = _make_te_linear(source, TELinear)
            report.converted_modules.append("<root>")
            return wrap_float8_pointwise_conv1d_module(model, converted)
        if _is_supported_layer_norm(model):
            TELayerNorm = getattr(te, "LayerNorm", None)
            if TELayerNorm is not None:
                report.converted_modules.append("<root>")
                return _make_te_layer_norm(model, TELayerNorm)
            report.skipped_modules.append("<root>")
            return model
        pattern = _convert_fused_pattern(model, te)
        if pattern is not None:
            wrapped, pattern_name = pattern
            report.converted_modules.append("<root>")
            report.converted_patterns.append(
                {"module": "<root>", "pattern": pattern_name, "backend": "te"}
            )
            return wrapped

        _te_recursive_convert(model, TELinear, report)
        return model


__all__ = [
    "Float8TELayerNormLinearModule",
    "Float8TELayerNormMLPModule",
    "Float8TransformerEnginePolicy",
]
