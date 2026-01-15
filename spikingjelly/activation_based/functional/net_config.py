import logging
from typing import Union, Optional

import torch.nn as nn

from .. import base


__all__ = ["reset_net", "set_step_mode", "set_backend", "detach_net"]


def reset_net(net: nn.Module):
    """
    **API Language:**
    :ref:`中文 <reset_net-cn>` | :ref:`English <reset_net-en>`

    ----

    .. _reset_net-cn:

    * **中文**

    将网络的状态重置。做法是遍历网络中的所有 ``Module``，若 ``m `` 为
    ``base.MemoryModule`` 函数或者是拥有 ``reset()`` 方法，则调用 ``m.reset()``。

    :param net: 任何属于 ``nn.Module`` 子类的网络
    :type net: torch.nn.Module

    :return: None

    ----

    .. _reset_net-en:

    * **English**

    Reset the whole network.  Walk through every ``Module`` as ``m``, and call
    ``m.reset()`` if this ``m`` is ``base.MemoryModule`` or ``m`` has ``reset()``.

    :param net: Any network inherits from ``nn.Module``
    :type net: torch.nn.Module

    :return: None
    """
    for m in net.modules():
        if hasattr(m, "reset"):
            if not isinstance(m, base.MemoryModule):
                logging.warning(
                    f"Trying to call `reset()` of {m}, which is not spikingjelly.activation_based.base"
                    f".MemoryModule"
                )
            m.reset()


def set_step_mode(net: nn.Module, step_mode: str):
    """
    **API Language:**
    :ref:`中文 <set_step_mode-cn>` | :ref:`English <set_step_mode-en>`

    ----

    .. _set_step_mode-cn:

    * **中文**

    将 ``net`` 中所有模块的步进模式设置为 ``step_mode`` 。

    .. note::

        :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`,
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        的子模块（不包含包装器本身）的 ``step_mode`` 不会被改变。

    :param net: 一个神经网络
    :type net: torch.nn.Module

    :param step_mode: 's' (单步模式) 或 'm' (多步模式)
    :type step_mode: str

    :return: None

    ----

    .. _set_step_mode-en:

    * **English**

    Set ``step_mode`` for all modules in ``net``.

    .. admonition:: Note
        :class: note

        The submodule (not including the container itself) of
        :class:`StepModeContainer <spikingjelly.activation_based.layer.container.StepModeContainer>`,
        :class:`ElementWiseRecurrentContainer <spikingjelly.activation_based.layer.container.ElementWiseRecurrentContainer>`,
        :class:`LinearRecurrentContainer <spikingjelly.activation_based.layer.container.LinearRecurrentContainer>`
        will not be changed.base.MemoryModule`` or ``m`` has ``reset()``.

    :param net: a network
    :type net: nn.Module

    :param step_mode: 's' (single-step) or 'm' (multi-step)
    :type step_mode: str

    :return: None
    """
    from ..layer import (
        StepModeContainer,
        ElementWiseRecurrentContainer,
        LinearRecurrentContainer,
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
    """
    **API Language:**
    :ref:`中文 <set_backend-cn>` | :ref:`English <set_backend-en>`

    ----

    .. _set_backend-cn:

    * **中文**

    将 ``net`` 中 所有类型为 ``instance`` 的模块后端更改为 ``backend`` 。

    :param net: 一个神经网络
    :type net: torch.nn.Module

    :param backend: 使用哪个后端
    :type backend: str

    :param instance: 类型为 ``instance`` 的模块后端会被改变。若为 ``None`` ，
        则所有模块的后端都会被改变
    :type instance: Optional[Union[nn.Module, tuple[nn.Module]]]

    :return: None

    ----

    .. _set_backend-en:

    * **English**

    Sets backends of all modules whose instance is ``instance`` in ``net`` to ``backend``.

    :param net: a network
    :type net: torch.nn.Module

    :param backend: the backend to be set
    :type backend: str

    :param instance: the backend of which instance will be changed. If ``None`` ,
        all modules' backend will be changed
    :type instance: Optional[Union[nn.Module, tuple[nn.Module]]]

    :return: None
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
    """
    **API Language:**
    :ref:`中文 <detach_net-cn>` | :ref:`English <detach_net-en>`

    ----

    .. _detach_net-cn:

    * **中文**

    将网络与之前的时间步的计算图断开。做法是遍历网络中的所有 ``Module``，若 ``m`` 为
    ``base.MemoryModule`` 函数或者是拥有 ``detach()`` 方法，则调用 ``m.detach()``。

    :param net: 任何属于 ``nn.Module`` 子类的网络
    :type net: torch.nn.Module

    :return: None

    ----

    .. _detach_net-en:

    * **English**

    Detach the computation graph of the whole network from previous time-steps.
    Walk through every ``Module`` as ``m``, and call ``m.detach()`` if this
    ``m`` is ``base.MemoryModule`` or ``m`` has ``detach()``.

    :param net: Any network inherits from ``nn.Module``
    :type net: torch.nn.Module

    :return: None
    """
    for m in net.modules():
        if hasattr(m, "detach"):
            if not isinstance(m, base.MemoryModule):
                logging.warning(
                    f"Trying to call `detach()` of {m}, which is not spikingjelly.activation_based.base"
                    f".MemoryModule"
                )
            m.detach()
