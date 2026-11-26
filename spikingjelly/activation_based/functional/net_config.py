from __future__ import annotations

import warnings

import torch.nn as nn

from ...logger import logger
from .. import base


__all__ = [
    "collect_reset_modules",
    "detach_net",
    "invalidate_reset_cache",
    "reset_collected_modules",
    "reset_net",
    "set_step_mode",
]


def collect_reset_modules(net: nn.Module) -> tuple[nn.Module, ...]:
    r"""
    **API Language** - :ref:`中文 <collect_reset_modules-cn>` | :ref:`English <collect_reset_modules-en>`

    ----

    .. _collect_reset_modules-cn:

    * **中文**

    遍历当前 ``net.modules()``，按遍历顺序收集具有可调用 ``reset`` 的模块，
    包括符合条件的 ``net`` 自身；共享模块按 ``modules()`` 的规则去重。
    收集不重置状态，也不记录非 ``MemoryModule`` 告警。

    返回的元组是由调用方持有的强引用快照，可交给
    :func:`reset_collected_modules` 重复使用。增删或替换模块、改变 ``reset``
    的可调用性后，调用方必须重新收集；旧快照仍引用旧模块。

    :param net: 要收集的网络；不限制其张量的形状、dtype、device 或后端。
    :type net: torch.nn.Module
    :return: 可重置模块的有序元组；没有符合条件的模块时为空元组。
    :rtype: tuple[torch.nn.Module, ...]

    ----

    .. _collect_reset_modules-en:

    * **English**

    Traverse the current ``net.modules()`` and collect modules with a callable
    ``reset`` in traversal order, including ``net`` itself when eligible. Shared
    modules are deduplicated by ``modules()``. Collection neither resets state
    nor emits warnings for non-``MemoryModule`` objects.

    The returned tuple is a caller-owned snapshot of strong references that can
    be reused with :func:`reset_collected_modules`. Recollect after adding,
    removing or replacing modules, or changing whether ``reset`` is callable;
    an old snapshot still references the old modules.

    :param net: Network to collect from; no tensor shape, dtype, device or
        backend restrictions are imposed.
    :type net: torch.nn.Module
    :return: Ordered tuple of resettable modules, empty if none are eligible.
    :rtype: tuple[torch.nn.Module, ...]
    """
    return tuple(m for m in net.modules() if callable(getattr(m, "reset", None)))


def reset_collected_modules(modules: tuple[nn.Module, ...]) -> None:
    r"""
    **API Language** - :ref:`中文 <reset_collected_modules-cn>` | :ref:`English <reset_collected_modules-en>`

    ----

    .. _reset_collected_modules-cn:

    * **中文**

    按元组顺序调用每个模块的 ``reset()``，由各模块恢复其状态。
    此函数直接使用 :func:`collect_reset_modules` 返回的快照，不遍历网络或
    检查结构变化；空元组不执行任何操作。调用方须在模型变化后重新收集。
    对非 :class:`~spikingjelly.activation_based.base.MemoryModule` 模块，每次
    调用其 ``reset()`` 前都会记录告警。异常原样传播，后续模块不会被重置。

    :param modules: 收集得到的模块元组；其中每个模块必须仍具有可调用的
        ``reset``。张量约束和状态变化由各模块的 ``reset()`` 决定。
    :type modules: tuple[torch.nn.Module, ...]
    :raises Exception: 任一模块的 ``reset()`` 抛出异常时原样传播。

    ----

    .. _reset_collected_modules-en:

    * **English**

    Call each module's ``reset()`` in tuple order to restore its state.
    This function directly reuses the snapshot from :func:`collect_reset_modules`
    without traversing the network or checking for structural changes; an empty
    tuple does nothing. The caller must recollect after model changes.
    Each call warns before resetting a module that is not a
    :class:`~spikingjelly.activation_based.base.MemoryModule`. Exceptions propagate
    unchanged and subsequent modules are not reset.

    :param modules: Collected module tuple; each module must still have a
        callable ``reset``. Tensor constraints and state changes are determined
        by each module's ``reset()``.
    :type modules: tuple[torch.nn.Module, ...]
    :raises Exception: Any exception from a module's ``reset()`` propagates unchanged.
    """
    for m in modules:
        if not isinstance(m, base.MemoryModule):
            logger.warning(
                "Trying to call `reset()` of {}, which is not a MemoryModule",
                m,
            )
        m.reset()


def invalidate_reset_cache(net: nn.Module) -> None:
    r"""
    **API Language** - :ref:`中文 <invalidate_reset_cache-cn>` | :ref:`English <invalidate_reset_cache-en>`

    ----

    .. _invalidate_reset_cache-cn:

    * **中文**

    此接口已弃用，将在 SpikingJelly 3.0 删除。调用仅发出
    ``DeprecationWarning``，不修改网络或任何快照。
    :func:`reset_net` 每次遍历当前网络，已不需要清除缓存；请删除此调用。
    使用显式快照时，改为 ``modules = collect_reset_modules(net)`` 并替换旧元组。

    :param net: 仅为兼容已发布接口保留的网络参数；不会被读取或修改。
    :type net: torch.nn.Module

    ----

    .. _invalidate_reset_cache-en:

    * **English**

    Deprecated; this interface will be removed in SpikingJelly 3.0. Calling it
    only emits a ``DeprecationWarning`` and changes neither the network nor any
    snapshot. :func:`reset_net` traverses the current network on every call,
    so cache invalidation is unnecessary; remove this call. For an explicit
    snapshot, replace the old tuple with ``modules = collect_reset_modules(net)``.

    :param net: Network argument retained for compatibility with the released
        interface; it is neither read nor modified.
    :type net: torch.nn.Module
    """
    warnings.warn(
        "invalidate_reset_cache() is deprecated and will be removed in SpikingJelly "
        "3.0. reset_net() now traverses the current network; remove this call. "
        "For an explicit snapshot, recollect it with collect_reset_modules(net).",
        DeprecationWarning,
        stacklevel=2,
    )


def reset_net(net: nn.Module) -> None:
    r"""
    **API Language** - :ref:`中文 <reset_net-cn>` | :ref:`English <reset_net-en>`

    ----

    .. _reset_net-cn:

    * **中文**

    重置 ``net`` 中所有可重置模块的状态。

    该函数会遍历 ``net.modules()`` 中的所有子模块；若某个子模块实现了
    可调用的 ``reset``，则调用该方法，包括符合条件的 ``net`` 自身。
    共享模块按 ``modules()`` 的规则去重。对于不是
    :class:`~spikingjelly.activation_based.base.MemoryModule` 但实现了
    ``reset()`` 的模块，此函数仍会调用 ``reset()``，同时记录告警。

    每次调用都重新收集，因此自动包含新增或替换的模块，并跳过已移除的模块。
    不维护隐式缓存。固定结构的循环可用 :func:`collect_reset_modules` 收集一次，
    再用 :func:`reset_collected_modules` 快速重置；模型变化后须由调用方重新收集。

    :param net: 要重置的网络；张量约束和状态变化由各模块的 ``reset()`` 决定。
    :type net: torch.nn.Module


    :raises Exception: 任何子模块 ``reset()`` 在执行过程中抛出的异常都会原样向上传播

    ----

    .. _reset_net-en:

    * **English**

    Reset the states of all resettable modules in ``net``.

    This function iterates over ``net.modules()`` and calls ``reset()`` on each
    module with a callable ``reset``, including ``net`` itself when eligible.
    Shared modules are deduplicated by ``modules()``. If a module is not an instance of
    :class:`~spikingjelly.activation_based.base.MemoryModule` but still defines
    ``reset()``, the function will still call it and emit a warning.

    Every call recollects modules, automatically including added or replaced
    modules and skipping removed ones. There is no implicit cache. For loops
    with a fixed structure, collect once with :func:`collect_reset_modules` and
    reset quickly with :func:`reset_collected_modules`; the caller must recollect
    after model changes.

    :param net: Network to reset; tensor constraints and state changes are
        determined by each module's ``reset()``.
    :type net: torch.nn.Module


    :raises Exception: Any exception raised by a submodule ``reset()`` call is propagated unchanged
    """
    reset_collected_modules(collect_reset_modules(net))


def set_step_mode(net: nn.Module, step_mode: str):
    r"""
    **API Language** - :ref:`中文 <set_step_mode-cn>` | :ref:`English <set_step_mode-en>`

    ----

    .. _set_step_mode-cn:

    * **中文**

    将 ``net`` 中所有具有 ``step_mode`` 属性的模块的步进模式设置为
    ``step_mode`` 。

    .. note::

        :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`,
        :class:`MultiStepContainer <spikingjelly.activation_based.layer.container.MultiStepContainer>`,
        :class:`SeqToANNContainer <spikingjelly.activation_based.layer.container.SeqToANNContainer>`,
        :class:`TLastMultiStepContainer <spikingjelly.activation_based.layer.container.TLastMultiStepContainer>`,
        :class:`TLastSeqToANNContainer <spikingjelly.activation_based.layer.container.TLastSeqToANNContainer>`,
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        的子模块（不包含包装器本身）的 ``step_mode`` 不会被改变。

    若某个模块具有 ``step_mode`` 属性但不是
    :class:`~spikingjelly.activation_based.base.StepModule`，则该函数仍会尝试赋值，
    同时记录告警。此时不会执行 ``StepModule`` 的步进模式校验，但模块自身的 setter
    仍可能校验该值。若希望该模块遵循步进模式约定，请让其继承
    :class:`~spikingjelly.activation_based.base.StepModule`（或
    :class:`~spikingjelly.activation_based.base.MemoryModule`），并在调用本函数前初始化
    有效的 ``step_mode``。

    :param net: 一个神经网络
    :type net: torch.nn.Module

    :param step_mode: 's' (单步模式) 或 'm' (多步模式)
    :type step_mode: str


    :raises ValueError: 若某个模块的 ``step_mode`` setter 不接受给定的 ``step_mode``，则该异常会原样向上传播

    ----

    .. _set_step_mode-en:

    * **English**

    Set ``step_mode`` to ``step_mode`` for all modules in ``net`` that expose a
    ``step_mode`` attribute.

    .. admonition:: Note
        :class: note

        The submodule (not including the container itself) of
        :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`,
        :class:`MultiStepContainer <spikingjelly.activation_based.layer.container.MultiStepContainer>`,
        :class:`SeqToANNContainer <spikingjelly.activation_based.layer.container.SeqToANNContainer>`,
        :class:`TLastMultiStepContainer <spikingjelly.activation_based.layer.container.TLastMultiStepContainer>`,
        :class:`TLastSeqToANNContainer <spikingjelly.activation_based.layer.container.TLastSeqToANNContainer>`,
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        will not be changed.

    If a module has a ``step_mode`` attribute but is not an instance of
    :class:`~spikingjelly.activation_based.base.StepModule`, the function still
    attempts to assign the new value and emits a warning. In that case, the
    ``StepModule`` validation is not applied, although the module's own setter
    may still validate the value. Inherit from
    :class:`~spikingjelly.activation_based.base.StepModule` (or
    :class:`~spikingjelly.activation_based.base.MemoryModule`) and initialize a
    valid ``step_mode`` before calling this function if the module should follow
    the step-mode contract.

    :param net: a network
    :type net: nn.Module

    :param step_mode: 's' (single-step) or 'm' (multi-step)
    :type step_mode: str


    :raises ValueError: Propagated if a module rejects the provided ``step_mode`` in its setter
    """
    from ..layer import (
        ElementWiseRecurrentContainer,
        LinearRecurrentContainer,
        MultiStepContainer,
        SeqToANNContainer,
        StepModeContainer,
        TLastMultiStepContainer,
        TLastSeqToANNContainer,
    )

    keep_step_mode_instance = (
        StepModeContainer,
        MultiStepContainer,
        SeqToANNContainer,
        TLastMultiStepContainer,
        TLastSeqToANNContainer,
        ElementWiseRecurrentContainer,
        LinearRecurrentContainer,
    )
    # step_mode of sub-modules in keep_step_mode_instance will not be changed

    modules = [net]
    while modules:
        m = modules.pop()
        if hasattr(m, "step_mode"):
            if not isinstance(m, base.StepModule):
                logger.warning(
                    "Trying to set the step mode for {}, which is not a "
                    "StepModule; step_mode={} is not validated by StepModule. "
                    "Inherit from "
                    "spikingjelly.activation_based.base.StepModule (or "
                    "MemoryModule) if this module should follow the step-mode "
                    "contract",
                    m,
                    step_mode,
                )
            m.step_mode = step_mode
        if not isinstance(m, keep_step_mode_instance):
            modules.extend(reversed(tuple(m.children())))


def detach_net(net: nn.Module):
    r"""
    **API Language** - :ref:`中文 <detach_net-cn>` | :ref:`English <detach_net-en>`

    ----

    .. _detach_net-cn:

    * **中文**

    将 ``net`` 中各有状态模块与之前时间步的计算图断开。

    该函数会遍历 ``net.modules()`` 中的所有子模块；若某个子模块实现了
    ``detach()`` 方法，则调用该方法。对于不是
    :class:`~spikingjelly.activation_based.base.MemoryModule` 但实现了
    ``detach()`` 的模块，此函数仍会调用 ``detach()``，同时记录告警。

    :param net: 任何属于 ``nn.Module`` 子类的网络
    :type net: torch.nn.Module


    :raises Exception: 任何子模块 ``detach()`` 在执行过程中抛出的异常都会原样向上传播

    ----

    .. _detach_net-en:

    * **English**

    Detach stateful modules in ``net`` from the computation graphs of previous
    time steps.

    This function iterates over ``net.modules()`` and calls ``detach()`` on each
    submodule that implements it. If a submodule is not an instance of
    :class:`~spikingjelly.activation_based.base.MemoryModule` but still defines
    ``detach()``, the function will still call it and emit a warning.

    :param net: Any network inherits from ``nn.Module``
    :type net: torch.nn.Module


    :raises Exception: Any exception raised by a submodule ``detach()`` call is propagated unchanged
    """
    for m in net.modules():
        if hasattr(m, "detach"):
            if not isinstance(m, base.MemoryModule):
                logger.warning(
                    "Trying to call `detach()` of {}, which is not a MemoryModule",
                    m,
                )
            m.detach()
