from __future__ import annotations

import warnings
from typing import Any, Callable

import torch.nn as nn

from ..memory_residency import MemoryResidencyCounter, _access_convolution_backward


class NeuroMCMemoryResidencyCounter(MemoryResidencyCounter):
    def __init__(
        self,
        *,
        memory_config: Any | None = None,
        config: Any | None = None,
        capacity_bits: dict[str, float] | None = None,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        if memory_config is not None and config is None:
            warnings.warn(
                "memory_config is deprecated; use config instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            config = memory_config
        super().__init__(
            config=config,
            capacity_bits=capacity_bits,
            extra_rules=extra_rules,
            extra_ignore_modules=extra_ignore_modules,
        )


__all__ = [
    "NeuroMCMemoryResidencyCounter",
    "_access_convolution_backward",
]
