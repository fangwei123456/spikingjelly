import copy
import logging
from abc import abstractmethod
from typing import Tuple, Generator

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
except BaseException as e:
    slayer = None


def check_backend_library(backend: str):
    """
    **API Language:**
    :ref:`中文 <check_backend_library-cn>` | :ref:`English <check_backend_library-en>`

    ----

    .. _check_backend_library-cn:

    * **中文**

    检查某个后端的python库是否已经安装。若未安装则此函数会报 ``ImportError`` 。

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'``或 ``'lava'``
    :type backend: str

    ----

    .. _check_backend_library-en:

    * **English**

    Check whether the python lib for backend is installed.
    If not, this function will raise an ``ImportError`` .

    :param backend: ``'torch'``, ``'cupy'``, ``'triton'`` or ``'lava'``
    :type backend: str
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
    def supported_step_mode(self) -> Tuple[str]:
        """
        **API Language:**
        :ref:`中文 <StepModule.supported_step_mode-cn>` | :ref:`English <StepModule.supported_step_mode-en>`

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
        """
        **API Language;**
        :ref:`中文 <StepModule.step_mode-cn>` | :ref:`English <StepModule.step_mode-en>`

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
        """
        **API Language:**
        :ref:`中文 <StepModule.step_mode-setter-cn>` | :ref:`English <StepModule.step_mode-setter-en>`

        ----

        .. _StepModule.step_mode-setter-cn:

        * **中文**

        将本模块的步进模式设置为 ``value`` 。

        :param value: 步进模式
        :type value: str

        ----

        .. _StepModule.step_mode-setter-en:

        * **English**

        Set the step mode of this module to be ``value`` .

        :param value: the step mode
        :type value: str
        """
        if value not in self.supported_step_mode():
            raise ValueError(
                f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!'
            )
        self._step_mode = value


class SingleStepModule(StepModule):
    """
    **API Language:**
    :ref:`中文 <SingleStepModule-cn>` | :ref:`English <SingleStepModule-en>`

    ----

    .. _SingleStepModule-cn:

    * **中文**

    只支持单步的模块 ( ``step_mode == 's'`` )。

    ----

    .. _SingleStepModule-en:

    * **English**

    The module that only supports for single-step ( ``step_mode == 's'`` ).
    """

    def supported_step_mode(self):
        return ("s",)


class MultiStepModule(StepModule):
    """
    **API Language:**
    :ref:`中文 <MultiStepModule-cn>` | :ref:`English <MultiStepModule-en>`

    ----

    .. _MultiStepModule-cn:

    * **中文**

    只支持多步的模块 ( ``step_mode == 'm'`` )。

    ----

    .. _MultiStepModule-en:

    * **English**

    The module that only supports for multi-step ( ``step_mode == 'm'`` ).
    """

    def supported_step_mode(self):
        return ("m",)


class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.__init__-cn>` | :ref:`English <MemoryModule.__init__-en>`

        ----

        .. _MemoryModule.__init__-cn:

        * **中文**

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        ----

        .. _MemoryModule.__init__-en:

        * **English**

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.
        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = "torch"
        self._step_mode = "s"

    @property
    def supported_backends(self) -> Tuple[str]:
        """
        **API Language;**
        :ref:`中文 <MemoryModule.supported_backends-cn>` | :ref:`English <MemoryModule.supported_backends-en>`

        ----

        .. _MemoryModule.supported_backends-cn:

        * **中文**

        返回支持的后端，默认情况下只有 ``('torch', )`` 。

        :return: 支持的后端
        :rtype: Tuple[str]

        ----

        .. _MemoryModule.supported_backends-en:

        * **English**

        Return the supported backends. The default return value is ``('torch', )`` .

        :return: supported backends
        :rtype: Tuple[str]
        """
        return ("torch",)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(
                f"{value} is not a supported backend of {self._get_name()}!"
            )
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.single_step_forward-cn>` | :ref:`English <MemoryModule.single_step_forward-en>`

        ----

        .. _MemoryModule.single_step_forward-cn:

        * **中文**

        本模块的单步的前向传播函数。

        :param x: input tensor with ``shape = [N, *]``
        :type x: torch.Tensor

        ----

        .. _MemoryModule.single_step_forward-en:

        * **English**

        The single-step forward function for this module.

        :param x: input tensor with ``shape = [N, *]``
        :type x: torch.Tensor
        """
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.multi_step_forward-cn>` | :ref:`English <MemoryModule.multi_step_forward-en>`

        ----

        .. _MemoryModule.multi_step_forward-cn:

        * **中文**

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现

        :param x_seq: input tensor with ``shape = [T, N, *]``
        :type x_seq: torch.Tensor

        ----

        .. _MemoryModule.multi_step_forward-en:

        * **English**

        The multi-step forward function for this module, which is implemented by
        calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` time steps.

        :param x_seq: input tensor with ``shape = [T, N, *]``
        :type x_seq: torch.Tensor
        """
        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y)

        return torch.stack(y_seq, dim=0)

    def forward(self, *args, **kwargs):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.forward-cn>` | :ref:`English <MemoryModule.forward-en>`

        ----

        .. _MemoryModule.forward-cn:

        * **中文**

        若为单步模式 ``step_mode == "s"``，则调用 ``self.single_step_forward(...)`` 。
        若为多步模式 ``step_mode == "m"``，则调用 ``self.multi_step_forward(...)`` 。

        ----

        .. _MemoryModule.forward-en:

        * **English**

        Call ``self.single_step_forward(...)`` if ``step_mode == "s"``.
        Call ``self.multi_step_forward(...)`` if ``step_mode == "m"``.
        """
        if self.step_mode == "s":
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == "m":
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f"step_mode={self.step_mode}, backend={self.backend}"

    def register_memory(self, name: str, value):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.register_memory-cn>` | :ref:`English <MemoryModule.register_memory-en>`

        ----

        .. _MemoryModule.register_memory-cn:

        * **中文**

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量将被初始化为 ``value`` 。
        每次调用 ``self.reset()`` 函数后， ``self.name`` 都会被重置为 ``value`` 。

        :param name: 状态变量的名字
        :type name: str
        :param value: 状态变量的初始与重制值
        :type value: Any

        ----

        .. _MemoryModule.register_memory-en:

        * **English**

        Register the state variable to memory dict, which saves stateful variables (e.g.,
        the membrane potential of a spiking neuron). The variable will be initialized as
        ``value`` . ``self.name`` will be set to ``value`` after calling ``self.reset()`` .

        :param name: state variable's name
        :type name: str
        :param value: state variable's initial and reset value
        :type value: Any
        """
        assert not hasattr(self, name), f"{name} has been set as a member variable!"
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.reset-cn>` | :ref:`English <MemoryModule.reset-en>`

        ----

        .. _MemoryModule.reset-cn:

        * **中文**

        重置所有有状态变量为重制值。

        ----

        .. _MemoryModule.reset-en:

        * **English**

        Reset all stateful variables to their reset values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        """
        **API Language:**
        :ref:`中文 <MemoryModule.set_reset_value-cn>` | :ref:`English <MemoryModule.set_reset_value-en>`

        ----

        .. _MemoryModule.set_reset_value-cn:

        * **中文**

        设置状态变量 ``self.name`` 的重制值。

        ----

        .. _MemoryModule.set_reset_value-en:

        * **English**

        Set the reset value of state variable ``self.name`` .
        """
        self._memories_rv[name] = copy.deepcopy(value)

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
        """
        **API Language:**
        :ref:`中文 <MemoryModule.memories-cn>` | :ref:`English <MemoryModule.memories-en>`

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
        """
        **API Language:**
        :ref:`中文 <MemoryModule.named_memories-cn>` | :ref:`English <MemoryModule.named_memories-en>`

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
        """
        **API Language:**
        :ref:`中文 <MemoryModule.detach-cn>` | :ref:`English <MemoryModule.detach-en>`

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
