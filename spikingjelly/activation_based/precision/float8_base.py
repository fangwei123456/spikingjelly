from __future__ import annotations

import torch
import torch.nn as nn

from .. import functional, layer


class Float8LinearStepModule(nn.Module):
    def __init__(self, wrapped: nn.Module, step_mode: str = "s"):
        super().__init__()
        self.wrapped = wrapped
        self.step_mode = step_mode

    def forward(self, x: torch.Tensor):
        if self.step_mode == "s":
            return self.wrapped(x)
        if self.step_mode == "m":
            return functional.seq_to_ann_forward(x, self.wrapped)
        raise ValueError(f"Unsupported step_mode {self.step_mode!r}.")


def wrap_float8_linear_module(original: nn.Module, converted: nn.Module) -> nn.Module:
    if isinstance(original, layer.Linear):
        return Float8LinearStepModule(converted, step_mode=original.step_mode)
    return converted

__all__ = ["Float8LinearStepModule", "wrap_float8_linear_module"]
