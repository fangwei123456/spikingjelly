from __future__ import annotations

from typing import Any, Callable

import torch
import torch.nn as nn

from .base_counter import NeuroMCBaseCounter

aten = torch.ops.aten

__all__ = ["NeuroMCSqrtCounter"]


def _sqrt_op(args, kwargs, out):
    return int(out.numel())


def _sqrt_native_batch_norm(args, kwargs, out):
    x = args[0]
    c = x.shape[1]
    return int(c)


class NeuroMCSqrtCounter(NeuroMCBaseCounter):
    """Counter for square root operations in the NeuroMC framework.
    **API Language:**
    :ref:`中文 <NeuroMCSqrtCounter-cn>` | :ref:`English <NeuroMCSqrtCounter-en>`

    ----

    .. _NeuroMCSqrtCounter-cn:

    * **中文**

    TODO: add Chinese description

    :rtype: None
    Tracks the number of square root and inverse square root operations
    performed during model execution.

    ----

    .. _NeuroMCSqrtCounter-en:

    * **English**

    TODO: add English description

    :return: None
    :rtype: None
    """

    def __init__(
        self,
        extra_rules: dict[Any, Callable] | None = None,
        extra_ignore_modules: list[nn.Module] | None = None,
    ):
        """
        :param extra_rules: Additional counting rules keyed by ATen operation
        :type extra_rules: dict[Any, Callable] | None
        :param extra_ignore_modules: Additional module types to ignore during counting
        :type extra_ignore_modules: list[nn.Module] | None
        :return: None
        :rtype: None
        """
        if extra_rules is None:
            extra_rules = {}
        super().__init__(extra_rules, extra_ignore_modules)
        self.rules = {
            aten.sqrt.default: _sqrt_op,
            aten.sqrt_.default: _sqrt_op,
            aten.rsqrt.default: _sqrt_op,
            aten.native_batch_norm.default: _sqrt_native_batch_norm,
        }
        self.rules.update(extra_rules)
