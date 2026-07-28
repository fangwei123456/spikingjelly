from __future__ import annotations

from typing import Any, Callable

import torch.nn as nn

from ..memory_residency import MemoryResidencyCounter, _access_convolution_backward


class NeuroMCMemoryResidencyCounter(MemoryResidencyCounter):
    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
        *,
        config: Any | None = None,
        capacity_bits: dict[str, float] | None = None,
    ):
        """
        Counter for memory residency in the NeuroMC framework.
        **API Language** - :ref:`中文 <NeuroMCMemoryResidencyCounter-cn>` | :ref:`English <NeuroMCMemoryResidencyCounter-en>`

        ----

        .. _NeuroMCMemoryResidencyCounter-cn:

        * **中文**

        NeuroMC内存驻留计数器

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        :param config: Memory hierarchy configuration
        :type config: Any | None
        :param capacity_bits: Capacity in bits per memory level
        :type capacity_bits: dict[str, float] | None

        ----

        .. _NeuroMCMemoryResidencyCounter-en:

        * **English**

        NeuroMC memory residency counter

        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        :param config: Memory hierarchy configuration
        :type config: Any | None
        :param capacity_bits: Capacity in bits per memory level
        :type capacity_bits: dict[str, float] | None
        """
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
