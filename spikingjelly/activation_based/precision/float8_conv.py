from __future__ import annotations

import torch
import torch.nn as nn

from .. import layer


def is_supported_pointwise_conv1d(module: nn.Module) -> bool:
    if not isinstance(module, (nn.Conv1d, layer.Conv1d)):
        return False
    return (
        module.kernel_size == (1,)
        and module.stride == (1,)
        and module.padding == (0,)
        and module.dilation == (1,)
        and module.groups == 1
        and module.padding_mode == "zeros"
    )


def make_linear_from_pointwise_conv1d(conv: nn.Conv1d) -> nn.Linear:
    linear = nn.Linear(
        conv.in_channels,
        conv.out_channels,
        bias=conv.bias is not None,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )
    with torch.no_grad():
        linear.weight.copy_(conv.weight.squeeze(-1))
        if conv.bias is not None:
            linear.bias.copy_(conv.bias)
    return linear


class Float8PointwiseConv1dStepModule(nn.Module):
    def __init__(self, wrapped: nn.Module, original: nn.Conv1d, step_mode: str = "s"):
        super().__init__()
        self.wrapped = wrapped
        self.step_mode = step_mode
        self.in_channels = original.in_channels
        self.out_channels = original.out_channels
        self.kernel_size = original.kernel_size
        self.stride = original.stride
        self.padding = original.padding
        self.dilation = original.dilation
        self.groups = original.groups
        self.padding_mode = original.padding_mode

    @property
    def weight(self):
        # Conv1d-compatible view; the actual Parameter is self.wrapped.weight.
        return self.wrapped.weight.unsqueeze(-1)

    @property
    def bias(self):
        return getattr(self.wrapped, "bias", None)

    def set_step_mode(self, step_mode: str):
        self.step_mode = step_mode

    def _forward_single_step(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3:
            raise ValueError(
                f"expected x with shape [N, C, L], but got x with shape {x.shape}!"
            )
        y = self.wrapped(x.transpose(1, 2).contiguous())
        return y.transpose(1, 2).contiguous()

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self._forward_single_step(x)
        if self.step_mode == "m":
            if x.dim() != 4:
                raise ValueError(
                    f"expected x with shape [T, N, C, L], but got x with shape {x.shape}!"
                )
            y = self.wrapped(x.permute(0, 1, 3, 2).contiguous())
            return y.permute(0, 1, 3, 2).contiguous()
        raise ValueError(f"Unsupported step_mode {self.step_mode!r}.")

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
        destination = self.wrapped.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )
        key = prefix + "weight"
        if key in destination and destination[key].dim() == 2:
            destination[key] = destination[key].unsqueeze(-1)
        return destination

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.wrapped._save_to_state_dict(destination, prefix, keep_vars)
        key = prefix + "weight"
        if key in destination and destination[key].dim() == 2:
            destination[key] = destination[key].unsqueeze(-1)

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
        for suffix in self.wrapped.state_dict().keys():
            key = prefix + suffix
            if key not in state_dict:
                continue
            value = state_dict.pop(key)
            if suffix == "weight" and value.dim() == 3:
                value = value.squeeze(-1)
            state_dict[wrapped_prefix + suffix] = value


def wrap_float8_pointwise_conv1d_module(
    original: nn.Conv1d, converted: nn.Module
) -> nn.Module:
    step_mode = original.step_mode if isinstance(original, layer.Conv1d) else "s"
    return Float8PointwiseConv1dStepModule(converted, original, step_mode=step_mode)


__all__ = [
    "Float8PointwiseConv1dStepModule",
    "is_supported_pointwise_conv1d",
    "make_linear_from_pointwise_conv1d",
    "wrap_float8_pointwise_conv1d_module",
]
