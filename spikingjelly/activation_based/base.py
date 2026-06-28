import copy
import logging
from abc import abstractmethod
from typing import Tuple, Generator, Optional, Callable, Any

import torch
import torch.nn as nn

try:
    import cupy
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.base: {e}")
    cupy = None

try:
    import triton
except BaseException as e:
    logging.info(f"spikingjelly.activation_based.base: {e}")
    triton = None

try:
    import lava.lib.dl.slayer as slayer
except BaseException:
    slayer = None


def check_backend_library(backend: str):
    r"""
    **API Language** - :ref:`中文 <check_backend_library-cn>` | :ref:`English <check_backend_library-en>`

    ----

    .. _check_backend_library-cn:

    * **中文**

    检查某个后端的python库是否已经安装。若未安装则此函数会报 ``ImportError`` 。

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'`` 或 ``'lava'``
    :type backend: str

    :raises ImportError: 若所请求后端依赖的 Python 库未安装，则抛出 ``ImportError``

    ----

    .. _check_backend_library-en:

    * **English**

    Check whether the python lib for backend is installed.
    If not, this function will raise an ``ImportError`` .

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'`` or ``'lava'``
    :type backend: str

    :raises ImportError: Raised when the Python package required by ``backend`` is not installed
    """
    if backend == "torch":
        return
    elif backend == "cupy":
        if cupy is None:
            raise ImportError(
                "CuPy is not installed! "
                'You can install it from "https://github.com/cupy/cupy".'
            )
    elif backend == "triton":
        if triton is None:
            raise ImportError(
                "Triton is not installed! "
                'You can install it from "https://github.com/openai/triton".'
            )
    elif backend == "lava":
        if slayer is None:
            raise ImportError(
                "Lava-DL is not installed! You can install it from "
                '"https://github.com/lava-nc/lava-dl". '
            )
    else:
        pass


class StepModule:
    r"""
    **API Language** - :ref:`中文 <StepModule-cn>` | :ref:`English <StepModule-en>`

    ----

    .. _StepModule-cn:

    * **中文**

    步进模式接口基类。

    实现该接口的模块通过 ``step_mode`` 区分单步模式 ``"s"`` 与多步模式 ``"m"``。

    ----

    .. _StepModule-en:

    * **English**

    Base interface for step-mode aware modules.

    Modules implementing this interface distinguish single-step mode ``"s"``
    from multi-step mode ``"m"`` through ``step_mode``.
    """

    def supported_step_mode(self) -> Tuple[str]:
        r"""
        **API Language** - :ref:`中文 <StepModule.supported_step_mode-cn>` | :ref:`English <StepModule.supported_step_mode-en>`

        ----

        .. _StepModule.supported_step_mode-cn:

        * **中文**

        :return: 包含支持的步进模式的tuple。``"s"`` 代表单步模式， ``"m"`` 代表多步模式。
        :rtype: Tuple[str]

        ----

        .. _StepModule.supported_step_mode-en:

        * **English**

        :return: a tuple that contains the supported step mode(s).  ``"s"`` is for
            single-step mode, and ``"m"`` is for multi-step mode.
        :rtype: Tuple[str]
        """
        return ("s", "m")

    @property
    def step_mode(self) -> str:
        r"""
        **API Language** - :ref:`中文 <StepModule.step_mode-cn>` | :ref:`English <StepModule.step_mode-en>`

        ----

        .. _StepModule.step_mode-cn:

        * **中文**

        :return: 模块当前使用的步进模式
        :rtype: str

        ----

        .. _StepModule.step_mode-en:

        * **English**

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        r"""
        **API Language** - :ref:`中文 <StepModule.step_mode-setter-cn>` | :ref:`English <StepModule.step_mode-setter-en>`

        ----

        .. _StepModule.step_mode-setter-cn:

        * **中文**

        将本模块的步进模式设置为 ``value`` 。

        :param value: 步进模式
        :type value: str

        :raises ValueError: 当 ``value`` 不在 ``self.supported_step_mode()`` 中时抛出

        ----

        .. _StepModule.step_mode-setter-en:

        * **English**

        Set the step mode of this module to be ``value`` .

        :param value: the step mode
        :type value: str

        :raises ValueError: Raised when ``value`` is not included in ``self.supported_step_mode()``
        """
        if value not in self.supported_step_mode():
            raise ValueError(
                f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!'
            )
        self._step_mode = value


class SingleStepModule(StepModule):
    r"""
    **API Language** - :ref:`中文 <SingleStepModule-cn>` | :ref:`English <SingleStepModule-en>`

    ----

    .. _SingleStepModule-cn:

    * **中文**

    单步模式模块的接口基类。

    实现该接口的模块仅支持单步模式 ``"s"``。

    ----

    .. _SingleStepModule-en:

    * **English**

    Base interface for single-step mode modules.

    Modules implementing this interface only support single-step mode ``"s"``.
    """

    def supported_step_mode(self):
        r"""
        **API Language** - :ref:`中文 <SingleStepModule.supported_step_mode-cn>` | :ref:`English <SingleStepModule.supported_step_mode-en>`

        ----

        .. _SingleStepModule.supported_step_mode-cn:

        * **中文**

        :return: 仅包含 ``"s"`` 的 tuple
        :rtype: Tuple[str]

        ----

        .. _SingleStepModule.supported_step_mode-en:

        * **English**

        :return: A tuple containing only ``"s"``
        :rtype: Tuple[str]
        """
        return ("s",)


class MultiStepModule(StepModule):
    r"""
    **API Language** - :ref:`中文 <MultiStepModule-cn>` | :ref:`English <MultiStepModule-en>`

    ----

    .. _MultiStepModule-cn:

    * **中文**

    多步模式模块的接口基类。

    实现该接口的模块仅支持多步模式 ``"m"``。

    ----

    .. _MultiStepModule-en:

    * **English**

    Base interface for multi-step mode modules.

    Modules implementing this interface only support multi-step mode ``"m"``.
    """

    def supported_step_mode(self):
        r"""
        **API Language** - :ref:`中文 <MultiStepModule.supported_step_mode-cn>` | :ref:`English <MultiStepModule.supported_step_mode-en>`

        ----

        .. _MultiStepModule.supported_step_mode-cn:

        * **中文**

        :return: 仅包含 ``"m"`` 的 tuple
        :rtype: Tuple[str]

        ----

        .. _MultiStepModule.supported_step_mode-en:

        * **English**

        :return: A tuple containing only ``"m"``
        :rtype: Tuple[str]
        """
        return ("m",)


class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.__init__-cn>` | :ref:`English <MemoryModule.__init__-en>`

        ----

        .. _MemoryModule.__init__-cn:

        * **中文**

        SpikingJelly 中所有有状态模块的基类。

        ``MemoryModule`` 通过 ``register_memory`` 注册内部状态变量，并提供
        ``reset``、``detach``、显式 memory 提取与恢复等通用能力。

        ----

        .. _MemoryModule.__init__-en:

        * **English**

        Base class of all stateful modules in SpikingJelly.

        ``MemoryModule`` registers internal state variables via
        ``register_memory`` and provides common utilities such as ``reset``,
        ``detach``, and explicit memory extraction / restoration.
        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = "torch"
        self._step_mode = "s"

    @property
    def supported_backends(self) -> Tuple[str]:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.supported_backends-cn>` | :ref:`English <MemoryModule.supported_backends-en>`

        ----

        .. _MemoryModule.supported_backends-cn:

        * **中文**

        :return: 支持的后端
        :rtype: Tuple[str]

        ----

        .. _MemoryModule.supported_backends-en:

        * **English**

        :return: supported backends
        :rtype: Tuple[str]
        """
        return ("torch",)

    @property
    def backend(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.backend-cn>` | :ref:`English <MemoryModule.backend-en>`

        ----

        .. _MemoryModule.backend-cn:

        * **中文**

        :return: 当前后端名称
        :rtype: str

        ----

        .. _MemoryModule.backend-en:

        * **English**

        :return: the name of the current backend
        :rtype: str
        """
        return self._backend

    @backend.setter
    def backend(self, value: str):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.backend-setter-cn>` | :ref:`English <MemoryModule.backend-setter-en>`

        ----

        .. _MemoryModule.backend-setter-cn:

        * **中文**

        设置当前模块的后端。

        只有当 ``value`` 属于 :meth:`supported_backends` 且对应依赖库已安装时，
        赋值才会成功。

        :param value: 目标后端名称
        :type value: str
        :raises NotImplementedError: 当 ``value`` 不在 ``supported_backends`` 中时抛出
        :raises ImportError: 当 ``value`` 对应的后端库未安装时抛出

        ----

        .. _MemoryModule.backend-setter-en:

        * **English**

        Set the backend of the current module.

        The assignment succeeds only when ``value`` is listed in
        :meth:`supported_backends` and the corresponding backend library is
        installed.

        :param value: Target backend name
        :type value: str
        :raises NotImplementedError: Raised when ``value`` is not listed in ``supported_backends``
        :raises ImportError: Raised when the backend library required by ``value`` is not installed
        """
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.single_step_forward-cn>` | :ref:`English <MemoryModule.single_step_forward-en>`

        ----

        .. _MemoryModule.single_step_forward-cn:

        * **中文**

        本模块的单步的前向传播函数。

        :param x: 输入张量，约定 ``shape = [N, *]``，其中 ``N`` 通常为 batch 维
        :type x: torch.Tensor

        :return: 单步前向传播的输出
        :rtype: Any

        ----

        .. _MemoryModule.single_step_forward-en:

        * **English**

        The single-step forward function for this module.

        :param x: Input tensor, conventionally with ``shape = [N, *]`` where ``N`` is usually the batch dimension
        :type x: torch.Tensor

        :return: Output of the single-step forward pass
        :rtype: Any
        """
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.multi_step_forward-cn>` | :ref:`English <MemoryModule.multi_step_forward-en>`

        ----

        .. _MemoryModule.multi_step_forward-cn:

        * **中文**

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现

        :param x_seq: 输入序列张量，约定 ``shape = [T, N, *]``，其中第 0 维为时间维
        :type x_seq: torch.Tensor

        :return: 按时间堆叠的输出序列
        :rtype: torch.Tensor

        :raises RuntimeError: 若某个时间步返回值无法被 ``torch.stack`` 堆叠，则底层异常会原样向上传播

        ----

        .. _MemoryModule.multi_step_forward-en:

        * **English**

        The multi-step forward function for this module, which is implemented by
        calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` time steps.

        :param x_seq: Input sequence tensor, conventionally with ``shape = [T, N, *]`` and the time axis at dimension 0
        :type x_seq: torch.Tensor

        :return: Output sequence stacked along the time dimension
        :rtype: torch.Tensor

        :raises RuntimeError: Any stacking failure raised by ``torch.stack`` is propagated unchanged
        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y)

        return torch.stack(y_seq, dim=0)

    def forward(self, *args, **kwargs):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.forward-cn>` | :ref:`English <MemoryModule.forward-en>`

        ----

        .. _MemoryModule.forward-cn:

        * **中文**

        若为单步模式 ``step_mode == "s"``，则调用 ``self.single_step_forward(...)`` 。
        若为多步模式 ``step_mode == "m"``，则调用 ``self.multi_step_forward(...)`` 。

        :return: 与当前 ``step_mode`` 对应的前向传播结果
        :rtype: Any

        :raises ValueError: 当 ``self.step_mode`` 既不是 ``"s"`` 也不是 ``"m"`` 时抛出

        ----

        .. _MemoryModule.forward-en:

        * **English**

        Call ``self.single_step_forward(...)`` if ``step_mode == "s"``.
        Call ``self.multi_step_forward(...)`` if ``step_mode == "m"``.

        :return: Forward result selected according to the current ``step_mode``
        :rtype: Any

        :raises ValueError: Raised when ``self.step_mode`` is neither ``"s"`` nor ``"m"``
        """
        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.extra_repr-cn>` | :ref:`English <MemoryModule.extra_repr-en>`

        ----

        .. _MemoryModule.extra_repr-cn:

        * **中文**

        返回附加到 ``nn.Module.__repr__`` 输出中的摘要字符串。

        :return: 附加到模块字符串表示中的摘要，包含 ``step_mode`` 与 ``backend``
        :rtype: str

        ----

        .. _MemoryModule.extra_repr-en:

        * **English**

        Return the summary string appended to ``nn.Module.__repr__``.

        :return: Summary appended to the module string representation, including ``step_mode`` and ``backend``
        :rtype: str
        """
        return f"step_mode={self.step_mode}, backend={self.backend}"

    def register_memory(self, name: str, value):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.register_memory-cn>` | :ref:`English <MemoryModule.register_memory-en>`

        ----

        .. _MemoryModule.register_memory-cn:

        * **中文**

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量将被初始化为 ``value`` 。
        每次调用 ``self.reset()`` 函数后， ``self.name`` 都会被重置为 ``value`` 。

        .. warning::

            若状态变量是个 ``torch.Tensor`` ，则 **不应对其做原地修改操作** 。

        :param name: 状态变量的名字
        :type name: str
        :param value: 状态变量的初始与重制值
        :type value: Any

        :raises AssertionError: 当 ``name`` 已经是模块现有成员属性时抛出

        ----

        .. _MemoryModule.register_memory-en:

        * **English**

        Register the state variable to memory dict, which saves stateful variables (e.g.,
        the membrane potential of a spiking neuron). The variable will be initialized as
        ``value`` . ``self.name`` will be set to ``value`` after calling ``self.reset()`` .

        .. warning::

            **Do not modify the state variable in-place** if it's a ``torch.Tensor`` .

        :param name: state variable's name
        :type name: str
        :param value: state variable's initial and reset value
        :type value: Any

        :raises AssertionError: Raised when ``name`` already exists as an attribute of the module
        """
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.reset-cn>` | :ref:`English <MemoryModule.reset-en>`

        ----

        .. _MemoryModule.reset-cn:

        * **中文**

        重置所有有状态变量为重制值。

        若当前状态与重制值均为同形状、同 dtype、同 device 的张量，则优先原地恢复；
        否则使用复制或重新赋值恢复。


        ----

        .. _MemoryModule.reset-en:

        * **English**

        Reset all stateful variables to their reset values.

        If both the current state and the reset value are tensors with the same
        shape, dtype, and device, the state is restored in-place whenever
        possible; otherwise it falls back to copy or reassignment.
        """
        for key in self._memories.keys():
            cur = self._memories[key]
            rv = self._memories_rv[key]
            if (
                isinstance(cur, torch.Tensor)
                and isinstance(rv, torch.Tensor)
                and cur.shape == rv.shape
                and cur.dtype == rv.dtype
                and cur.device == rv.device
            ):
                # detach_() breaks stale autograd graphs before in-place copy.
                # Falls back to deepcopy when cur is a view tensor
                # (detach_() raises RuntimeError on views).
                try:
                    cur.detach_().copy_(rv)
                except RuntimeError:
                    self._memories[key] = rv.detach().clone()
            elif isinstance(cur, torch.Tensor) and isinstance(rv, (int, float)):
                # Preserve Python-scalar sentinel semantics so the next forward
                # can materialize a fresh tensor with the new runtime shape.
                self._memories[key] = rv
            else:
                self._memories[key] = copy.deepcopy(rv)

    def set_reset_value(self, name: str, value):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.set_reset_value-cn>` | :ref:`English <MemoryModule.set_reset_value-en>`

        ----

        .. _MemoryModule.set_reset_value-cn:

        * **中文**

        设置状态变量 ``self.name`` 的重制值。

        :param name: 状态变量名称
        :type name: str
        :param value: 新的重制值
        :type value: Any

        ----

        .. _MemoryModule.set_reset_value-en:

        * **English**

        Set the reset value of state variable ``self.name`` .

        :param name: Name of the state variable
        :type name: str
        :param value: New reset value
        :type value: Any
        """
        self._memories_rv[name] = copy.deepcopy(value)

    def get_reset_value(self, name: str) -> Any:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.get_reset_value-cn>` | :ref:`English <MemoryModule.get_reset_value-en>`

        ----

        .. _MemoryModule.get_reset_value-cn:

        * **中文**

        获取状态变量 ``self.name`` 的重置值。

        :param name: 状态变量名称
        :type name: str
        :return: 状态变量的重置值
        :rtype: Any
        :raises KeyError: 当 ``name`` 不是已注册的状态变量，或没有设置重置值时抛出

        ----

        .. _MemoryModule.get_reset_value-en:

        * **English**

        Get the reset value of state variable ``self.name``.

        :param name: Name of the state variable
        :type name: str
        :return: Reset value of the state variable
        :rtype: Any
        :raises KeyError: Raised when ``name`` is not a registered state variable, or has no reset value
        """
        if name not in self._memories:
            raise KeyError(f"{name} is not a registered memory.")
        if name not in self._memories_rv:
            raise KeyError(f"{name} has no reset value. Call set_reset_value first.")
        return self._memories_rv[name]

    def __getattr__(self, name: str):
        #! use self.__dict__ instead of direct access to avoid infinite recursion
        if "_memories" in self.__dict__:
            memories = self.__dict__["_memories"]
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        #! use self.__dict__ instead of direct access to avoid infinite recursion
        _memories = self.__dict__.get("_memories")
        if _memories is not None and name in _memories:
            _memories[name] = value
        else:
            super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in self._memories:
            del self._memories[name]
            del self._memories_rv[name]
        else:
            return super().__delattr__(name)

    def __dir__(self):
        module_attrs = dir(self.__class__)
        attrs = list(self.__dict__.keys())
        parameters = list(self._parameters.keys())
        modules = list(self._modules.keys())
        buffers = list(self._buffers.keys())
        memories = list(self._memories.keys())
        keys = module_attrs + attrs + parameters + modules + buffers + memories

        # Eliminate attrs that are not legal Python variable names
        keys = [key for key in keys if not key[0].isdigit()]

        return sorted(keys)

    def memories(self) -> Generator:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.memories-cn>` | :ref:`English <MemoryModule.memories-en>`

        ----

        .. _MemoryModule.memories-cn:

        * **中文**

        :return: 返回一个所有状态变量的生成器
        :rtype: Generator

        ----

        .. _MemoryModule.memories-en:

        * **English**

        :return: a generator over all stateful variables
        :rtype: Generator
        """
        for _, value in self._memories.items():
            yield value

    def named_memories(self) -> Generator:
        r"""
        **API Language** - :ref:`中文 <MemoryModule.named_memories-cn>` | :ref:`English <MemoryModule.named_memories-en>`

        ----

        .. _MemoryModule.named_memories-cn:

        * **中文**

        :return: 返回一个所有状态变量名称及其值的生成器
        :rtype: Generator

        ----

        .. _MemoryModule.named_memories-en:

        * **English**

        :return: a generator over all stateful variables' names and values
        :rtype: Generator
        """
        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        r"""
        **API Language** - :ref:`中文 <MemoryModule.detach-cn>` | :ref:`English <MemoryModule.detach-en>`

        ----

        .. _MemoryModule.detach-cn:

        * **中文**

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT (Truncated Back Propagation Through Time)。


        ----

        .. _MemoryModule.detach-en:

        * **English**

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT (Truncated Back Propagation Through Time).
        """
        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica


def named_memories(module: nn.Module, prefix: str = "") -> Generator:
    r"""
    **API Language** - :ref:`中文 <named_memories-cn>` | :ref:`English <named_memories-en>`

    ----

    .. _named_memories-cn:

    * **中文**

    递归地生成模块树中的所有状态变量。类似于 ``named_parameters()`` 方法。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param prefix: 名称前缀
    :type prefix: str

    :return: 状态变量名称和值的生成器
    :rtype: Generator

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _named_memories-en:

    * **English**

    Recursively yield all memory variables in a module tree. Similar to ``named_parameters()`` .

    :param module: the target module
    :type module: torch.nn.Module

    :param prefix: name prefix
    :type prefix: str

    :return: a generator of memory variable names and values
    :rtype: Generator

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    if isinstance(module, MemoryModule):
        for name, value in module.named_memories():
            full_name = f"{prefix}.{name}" if prefix else name
            yield full_name, value

    for child_name, child in module.named_children():
        child_prefix = f"{prefix}.{child_name}" if prefix else child_name
        yield from named_memories(child, prefix=child_prefix)


def memories(module: nn.Module) -> Generator:
    r"""
    **API Language** - :ref:`中文 <memories-cn>` | :ref:`English <memories-en>`

    ----

    .. _memories-cn:

    * **中文**

    递归地生成模块树中的所有状态变量值。类似于 ``parameters()`` 方法。

    :param module: 目标模块
    :type module: nn.Module

    :return: 状态变量值的生成器
    :rtype: Generator

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _memories-en:

    * **English**

    Recursively yield all memory variables in a module tree. Similar to ``parameters()`` .

    :param module: the target module
    :type module: nn.Module

    :return: a generator of memory variable values
    :rtype: Generator

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    for _, value in named_memories(module):
        yield value


def extract_memories(module: nn.Module) -> list:
    r"""
    **API Language** - :ref:`中文 <extract_memories-cn>` | :ref:`English <extract_memories-en>`

    ----

    .. _extract_memories-cn:

    * **中文**

    提取模块中所有的状态变量值并返回列表。

    :param module: 目标模块
    :type module: torch.nn.Module

    :return: 状态变量值的列表
    :rtype: list

    :raises RecursionError: 若模块树存在异常递归结构，Python 递归遍历时会抛出异常

    ----

    .. _extract_memories-en:

    * **English**

    Extract all memory variable values from the module and return as a list.

    :param module: the target module
    :type module: torch.nn.Module

    :return: a list of memory variable values
    :rtype: list

    :raises RecursionError: Raised if traversing the module tree exceeds Python recursion limits
    """
    return [m for m in memories(module)]


def load_memories(module: nn.Module, memory_list: list):
    r"""
    **API Language** - :ref:`中文 <load_memories-cn>` | :ref:`English <load_memories-en>`

    ----

    .. _load_memories-cn:

    * **中文**

    将状态变量列表加载到模块中。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param memory_list: 状态变量值列表
    :type memory_list: list

    :raises ValueError: 当 ``memory_list`` 的长度与 ``module`` 当前状态变量数量不一致时抛出

    ----

    .. _load_memories-en:

    * **English**

    Load memory variables from a list into the module.

    :param module: the target module
    :type module: torch.nn.Module

    :param memory_list: list of memory variable values
    :type memory_list: list

    :raises ValueError: Raised when the length of ``memory_list`` does not match the number of current memory variables in ``module``
    """

    def _assign_memory_by_name(module: nn.Module, name: str, value):
        parts = name.split(".")
        obj = module
        for p in parts[:-1]:
            obj = getattr(obj, p)
        setattr(obj, parts[-1], value)

    named = list(named_memories(module))

    if len(named) != len(memory_list):
        raise ValueError(
            f"Memory length mismatch: model has {len(named)} memories "
            f"but list contains {len(memory_list)}"
        )

    for (name, _), value in zip(named, memory_list):
        _assign_memory_by_name(module, name, value)


class _FunctionalForward:
    def __init__(self, module: nn.Module, fn: Optional[Callable] = None):
        self.module = module
        self.fn = fn if fn is not None else module.forward
        self.num_states = len(list(named_memories(module)))

    def __call__(self, *args):
        if self.num_states == 0:  # stateless
            return self.fn(*args)

        inputs = args[: -self.num_states]
        states = args[-self.num_states :]
        original_states = extract_memories(self.module)
        load_memories(self.module, states)

        try:
            outputs = self.fn(*inputs)
            new_states = extract_memories(self.module)
        finally:
            load_memories(self.module, original_states)

        if not isinstance(outputs, tuple):
            outputs = (outputs,)
        return (*outputs, *new_states)


def to_functional_forward(module: nn.Module, fn: Optional[Callable] = None):
    r"""
    **API Language** - :ref:`中文 <to_functional_forward-cn>` | :ref:`English <to_functional_forward-en>`

    ----

    .. _to_functional_forward-cn:

    * **中文**

    给定一个可能包含隐式状态变量（记忆，memory）的模块，获取其显式状态的前向传播函数。

    对于包含状态的模块，返回的函数签名为 ``(*inputs, *states) -> (*outputs, *new_states)`` ，
    其中：

    - ``inputs`` 为原始 ``forward`` 所需的常规输入参数；
    - ``states`` 为当前模块中所有状态变量的值，其顺序与 ``extract_memories(module)`` 一致；
    - ``outputs`` 为原始 ``forward`` 的输出结果；
    - ``new_states`` 为执行前向传播后更新得到的状态变量。

    若模块中不存在任何状态变量，则直接返回 ``module.forward`` 本身。

    .. note::

        该函数通过在调用过程中 **临时替换模块内部状态** 的方式实现功能转换，
        并在执行结束后 **恢复原始状态** ，
        因此对模块本身不产生副作用。

    .. warning::

        如果某个状态变量为 ``torch.Tensor`` ，则其不应在 ``module.forward`` 中被原地修改。否则，
        会导致输入给前向传播函数的状态变量被修改，导致意想不到的错误。

    :param module: 目标模块
    :type module: torch.nn.Module

    :param fn: 含隐式状态的前向传播函数。若为 ``None`` ，则默认使用 ``module.forward`` 。
           该参数可用于指定特殊的前向传播函数（如， ``module`` 的父类的 ``forward`` ）。默认值
           为 ``None`` 。
    :type fn: Optional[Callable]

    :return: 带有显式输入输出状态的前向传播函数
    :rtype: Callable

    :raises ValueError: 若后续调用时提供的显式状态数量与 ``module`` 当前 memory 布局不一致，则相关 helper 可能抛出异常

    ----

    .. _to_functional_forward-en:

    * **English**

    Given a module that may contain implicit state variables, get the forward function
    with explicit state variables.

    For a stateful module, the returned function has the following signature
    ``(*inputs, *states) -> (*outputs, *new_states)``
    where:

    - ``inputs`` are the regular input arguments required by the original ``forward``;
    - ``states`` are the current memory variable values, in the same order as
      returned by ``extract_memories(module)``;
    - ``outputs`` are the outputs of the original ``forward`` method;
    - ``new_states`` are the updated memory variables after the forward pass.

    If the module does not contain any memory variables, ``module.forward`` is returned directly.

    .. note::

        The conversion is implemented by **temporarily loading the provided states** into
        the module, executing the original forward pass, extracting the updated states,
        and finally **restoring the original internal states**. Therefore, this operation
        has no side effects on the module itself.

    .. warning::

        If a state variable is a ``torch.Tensor``, it should not be modified in-place
        in ``module.forward``. Otherwise, the provided states will be modified,
        which may lead to unexpected errors.

    :param module: the target module
    :type module: torch.nn.Module

    :param fn: the forward function to be used. If ``None``, ``module.forward`` is used
           by default. This argument can be used to explicitly specify another forward
           function (e.g., the ``forward`` method of ``module``'s parent class).
           Defaults to ``None``.
    :type fn: Optional[Callable]

    :return: a functional-style forward function with explicit and flattened states
    :rtype: Callable

    ----

    * **代码示例 | Example**

    .. code:: python

        import torch
        import torch.nn as nn
        from spikingjelly.activation_based import base

        class StatefulModule(base.MemoryModule):
            def __init__(self):
                super().__init__()
                self.register_memory("counter", torch.tensor(0.0))
                self.linear = nn.Linear(10, 5)

            def single_step_forward(self, x):
                self.counter = self.counter + 1.0
                return self.linear(x)

        module = StatefulModule()
        f_forward = base.to_functional_forward(module)
        x = torch.randn(3, 10)
        initial_state = torch.tensor(0.0)
        output, new_state = f_forward(x, initial_state)

        assert torch.equal(output, module.linear(x))
        assert torch.equal(new_state, initial_state + 1.0)
    """
    return _FunctionalForward(module, fn)
