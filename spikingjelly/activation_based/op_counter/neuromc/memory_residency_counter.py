from __future__ import annotations

import warnings
from typing import Any, Callable

import torch.nn as nn

from ..memory_residency import MemoryResidencyCounter, _access_convolution_backward


class NeuroMCMemoryResidencyCounter(MemoryResidencyCounter):
    """Counter for memory residency in the NeuroMC framework.

    Tracks the memory residency (number of elements alive) at different
    memory hierarchy levels during model execution.
    """

    def __init__(
        self,
        memory_config: Any | None = None,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
        *,
        config: Any | None = None,
        capacity_bits: dict[str, float] | None = None,
    ):
        """
        :param memory_config: (Deprecated) Use ``config`` instead.
        :type memory_config: Any | None
        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        :param config: Memory hierarchy configuration
        :type config: Any | None
        :param capacity_bits: Capacity in bits per memory level
        :type capacity_bits: dict[str, float] | None
        :return: None
        :rtype: None
        """
        if memory_config is not None:
            if config is not None:
                raise TypeError("Pass only one of memory_config or config.")
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
