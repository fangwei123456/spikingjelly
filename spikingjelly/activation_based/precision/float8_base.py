from __future__ import annotations

import torch
import torch.nn as nn

from .. import functional, layer


class Float8LinearStepModule(nn.Module):
    """Step-mode wrapper around a float8 linear module.

    Note:
        `_load_from_state_dict` rewrites the provided state_dict keys in place
        from `prefix + key` to `prefix + "wrapped." + key` so that PyTorch's
        recursive loading can populate the registered `wrapped` child module.
        Callers that intend to reuse the same state dict object elsewhere should
        clone or copy it first.
    """

    def __init__(self, wrapped: nn.Module, step_mode: str = "s"):
        super().__init__()
        self.wrapped = wrapped
        self.step_mode = step_mode

    def set_step_mode(self, step_mode: str):
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.wrapped(x)
        if self.step_mode == "m":
            return functional.seq_to_ann_forward(x, self.wrapped)
        raise ValueError(f"Unsupported step_mode {self.step_mode!r}.")

    def __getattr__(self, name: str):
        try:
            return super().__getattr__(name)
        except AttributeError:
            wrapped = self.__dict__.get("_modules", {}).get("wrapped")
            if wrapped is None:
                raise AttributeError(
                    f"'{type(self).__name__}' object has no attribute '{name}'"
                )
            return getattr(wrapped, name)

    def state_dict(self, destination=None, prefix="", keep_vars=False):
        return self.wrapped.state_dict(
            destination=destination, prefix=prefix, keep_vars=keep_vars
        )

    def _save_to_state_dict(self, destination, prefix, keep_vars):
        self.wrapped._save_to_state_dict(destination, prefix, keep_vars)

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
        wrapped_keys = set(self.wrapped.state_dict().keys())
        keys_to_rename = [
            k
            for k in list(state_dict.keys())
            if k.startswith(prefix) and not k.startswith(wrapped_prefix)
        ]
        for k in keys_to_rename:
            suffix = k[len(prefix) :]
            if suffix in wrapped_keys:
                state_dict[wrapped_prefix + suffix] = state_dict.pop(k)


def wrap_float8_linear_module(original: nn.Module, converted: nn.Module) -> nn.Module:
    if isinstance(original, layer.Linear):
        return Float8LinearStepModule(converted, step_mode=original.step_mode)
    if isinstance(original, nn.Linear):
        return Float8LinearStepModule(converted, step_mode="s")
    return converted

__all__ = ["Float8LinearStepModule", "wrap_float8_linear_module"]
