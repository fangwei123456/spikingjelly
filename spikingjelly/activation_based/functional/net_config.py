from __future__ import annotations

import logging
from typing import Optional, Union
from weakref import ReferenceType, WeakKeyDictionary, ref

import torch.nn as nn

from .. import base

__all__ = [
    "collect_reset_modules",
    "detach_net",
    "invalidate_reset_cache",
    "reset_collected_modules",
    "reset_net",
    "set_backend",
    "set_step_mode",
]

_RESET_MODULE_CACHE: WeakKeyDictionary[nn.Module, tuple[ReferenceType[nn.Module], ...]] = (
    WeakKeyDictionary()
)


def collect_reset_modules(net: nn.Module) -> tuple[nn.Module, ...]:
    return tuple(m for m in net.modules() if callable(getattr(m, "reset", None)))


def _resolve_cached_reset_modules(
    net: nn.Module,
) -> Optional[tuple[nn.Module, ...]]:
    cached = _RESET_MODULE_CACHE.get(net)
    if cached is None:
        return None

    modules = []
    for cached_module in cached:
        module = cached_module()
        if module is None:
            invalidate_reset_cache(net)
            return None
        modules.append(module)
    return tuple(modules)


def reset_collected_modules(modules: tuple[nn.Module, ...]) -> None:
    for m in modules:
        if not isinstance(m, base.MemoryModule):
            logging.warning(
                f"Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base"
                f".MemoryModule"
            )
        m.reset()


def invalidate_reset_cache(net: nn.Module) -> None:
    r"""
    **API Language:**
    :ref:`中文 <invalidate_reset_cache-cn>` | :ref:`English <invalidate_reset_cache-en>`

    ----

    .. _invalidate_reset_cache-cn:
    * **中文**

    清除 ``net`` 的 reset 模块缓存。当模型结构发生变化后调用
    （如 ``torch.compile`` 或模块替换）。

    :param net: 目标网络
    :type net: torch.nn.Module

    ----

    .. _invalidate_reset_cache-en:
    * **English**

    Clear the reset module cache for ``net``.
    Call after model structure changes (e.g. ``torch.compile`` or module swaps).

    :param net: Target network
    :type net: torch.nn.Module
    """
    _RESET_MODULE_CACHE.pop(net, None)


def reset_net(net: nn.Module):
    r"""
    **API Language:**
    :ref:`中文 <reset_net-cn>` | :ref:`English <reset_net-en>`

    ----

    .. _reset_net-cn:
    * **中文**

    重置 ``net`` 中所有可重置模块的状态。

    该函数会遍历 ``net.modules()`` 中的所有子模块；若某个子模块实现了
    ``reset()`` 方法，则调用该方法。对于不是
    :class:`~spikingjelly.activation_based.base.MemoryModule` 但实现了
    ``reset()`` 的模块，此函数仍会调用 ``reset()``，同时记录告警。

    首次调用时收集并缓存可重置模块列表，后续调用直接复用缓存。
    若模型结构发生变化（如 ``torch.compile`` 或模块替换），
    请调用 :func:`invalidate_reset_cache` 清除缓存。

    :param net: 任何属于 ``nn.Module`` 子类的网络
    :type net: torch.nn.Module

    :return: ``None``
    :rtype: None

    :raises Exception: 任何子模块 ``reset()`` 在执行过程中抛出的异常都会原样向上传播

    ----

    .. _reset_net-en:
    * **English**

    Reset the states of all resettable modules in ``net``.

    This function iterates over ``net.modules()`` and calls ``reset()`` on each
    submodule that implements it. If a submodule is not an instance of
    :class:`~spikingjelly.activation_based.base.MemoryModule` but still defines
    ``reset()``, the function will still call it and emit a warning.

    On the first call, the resettable module list is collected and cached;
    subsequent calls reuse the cache.
    If the model structure changes (e.g. ``torch.compile`` or module swaps),
    call :func:`invalidate_reset_cache` to clear the cache.

    :param net: Any network inherits from ``nn.Module``
    :type net: torch.nn.Module

    :return: ``None``
    :rtype: None

    :raises Exception: Any exception raised by a submodule ``reset()`` call is propagated unchanged
    """
    cached = _resolve_cached_reset_modules(net)
    if cached is not None:
        reset_collected_modules(cached)
        return
    modules = collect_reset_modules(net)
    _RESET_MODULE_CACHE[net] = tuple(ref(module) for module in modules)
    reset_collected_modules(modules)


def set_step_mode(net: nn.Module, step_mode: str):
    r"""
    **API Language:**
    :ref:`中文 <set_step_mode-cn>` | :ref:`English <set_step_mode-en>`

    ----

    .. _set_step_mode-cn:
    * **中文**

    将 ``net`` 中所有具有 ``step_mode`` 属性的模块的步进模式设置为
    ``step_mode`` 。

    .. note::

        :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`,
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        的子模块（不包含包装器本身）的 ``step_mode`` 不会被改变。

    若某个模块具有 ``step_mode`` 属性但不是
    :class:`~spikingjelly.activation_based.base.StepModule`，则该函数仍会尝试赋值，
    同时记录告警。

    :param net: 一个神经网络
    :type net: torch.nn.Module

    :param step_mode: 's' (单步模式) 或 'm' (多步模式)
    :type step_mode: str

    :return: ``None``
    :rtype: None

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
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        will not be changed.

    If a module has a ``step_mode`` attribute but is not an instance of
    :class:`~spikingjelly.activation_based.base.StepModule`, the function still
    attempts to assign the new value and emits a warning.

    :param net: a network
    :type net: nn.Module

    :param step_mode: 's' (single-step) or 'm' (multi-step)
    :type step_mode: str

    :return: ``None``
    :rtype: None

    :raises ValueError: Propagated if a module rejects the provided ``step_mode`` in its setter
    """
    from ..layer import (
        ElementWiseRecurrentContainer,
        LinearRecurrentContainer,
        StepModeContainer,
    )

    keep_step_mode_instance = (
        StepModeContainer,
        ElementWiseRecurrentContainer,
        LinearRecurrentContainer,
    )
    # step_mode of sub-modules in keep_step_mode_instance will not be changed

    keep_step_mode_containers = []
    for m in net.modules():
        if isinstance(m, keep_step_mode_instance):
            keep_step_mode_containers.append(m)

    for m in net.modules():
        if hasattr(m, "step_mode"):
            is_contained = False
            for container in keep_step_mode_containers:
                if (
                    not isinstance(m, keep_step_mode_instance)
                    and m in container.modules()
                ):
                    is_contained = True
                    break
            if is_contained:
                # this function should not change step_mode of submodules in keep_step_mode_containers
                pass
            else:
                if not isinstance(m, (base.StepModule)):
                    logging.warning(
                        f"Trying to set the step mode for {m}, which is not spikingjelly.activation_based"
                        f".base.StepModule"
                    )
                m.step_mode = step_mode


def set_backend(
    net: nn.Module,
    backend: str,
    instance: Optional[Union[nn.Module, tuple[nn.Module]]] = None,
):
    r"""
    **API Language:**
    :ref:`中文 <set_backend-cn>` | :ref:`English <set_backend-en>`

    ----

    .. _set_backend-cn:

    * **中文**

    将 ``net`` 中所有满足 ``isinstance(m, instance)`` 且具有 ``backend``
    属性的模块后端设置为 ``backend``。

    仅当目标模块的 ``supported_backends`` 包含给定 ``backend`` 时才会实际更新；
    否则会记录告警并保留原有后端。若 ``instance`` 为 ``None``，则会检查所有具有
    ``backend`` 属性的模块。

    :param net: 一个神经网络
    :type net: torch.nn.Module

    :param backend: 使用哪个后端
    :type backend: str

    :param instance: 传给 ``isinstance`` 的筛选类型。满足该筛选且具有 ``backend`` 属性的模块后端会被检查。
        若为 ``None`` ，则所有具有 ``backend`` 属性的模块都会被检查
    :type instance: Optional[Union[nn.Module, tuple[nn.Module]]]

    :return: ``None``
    :rtype: None

    :raises Exception: 若目标模块在访问 ``supported_backends`` 或设置 ``backend`` 时抛出异常，则该异常会原样向上传播

    ----

    .. _set_backend-en:

    * **English**

    Set ``backend`` for all modules in ``net`` whose type matches ``instance``
    and that expose a ``backend`` attribute.

    The backend is updated only when ``backend`` is listed in the module's
    ``supported_backends``. Otherwise, a warning is logged and the existing
    backend is kept unchanged. If ``instance`` is ``None``, all modules with a
    ``backend`` attribute are checked.

    :param net: a network
    :type net: torch.nn.Module

    :param backend: the backend to be set
    :type backend: str

    :param instance: the type filter passed to ``isinstance``. Modules that
        match this filter and have a ``backend`` attribute will be checked. If
        ``None``, all modules with a ``backend`` attribute will be checked
    :type instance: Optional[Union[nn.Module, tuple[nn.Module]]]

    :return: ``None``
    :rtype: None

    :raises Exception: Propagated if a target module raises while exposing ``supported_backends`` or assigning ``backend``
    """
    instance = (nn.Module,) if instance is None else instance
    for m in net.modules():
        if isinstance(m, instance):
            if hasattr(m, "backend"):
                if not isinstance(m, base.MemoryModule):
                    logging.warning(
                        f"Trying to set the backend for {m}, which is not spikingjelly.activation_based.base.MemoryModule"
                    )
                if backend in m.supported_backends:
                    m.backend = backend
                else:
                    logging.warning(
                        f"{m} does not supports for backend={backend}. It will still use backend={m.backend}."
                    )


def detach_net(net: nn.Module):
    r"""
    **API Language:**
    :ref:`中文 <detach_net-cn>` | :ref:`English <detach_net-en>`

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

    :return: ``None``
    :rtype: None

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

    :return: ``None``
    :rtype: None

    :raises Exception: Any exception raised by a submodule ``detach()`` call is propagated unchanged
    """
    for m in net.modules():
        if hasattr(m, "detach"):
            if not isinstance(m, base.MemoryModule):
                logging.warning(
                    f"Trying to call `detach()` of {m}, which is not spikingjelly.activation_based.base"
                    f".MemoryModule"
                )
            m.detach()
