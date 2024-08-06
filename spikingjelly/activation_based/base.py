import torch
import torch.nn as nn
import copy
import logging
from abc import abstractmethod

try:
    import cupy
except BaseException as e:
    logging.info(f'spikingjelly.activation_based.base: {e}')
    cupy = None

try:
    import lava.lib.dl.slayer as slayer
except BaseException as e:
    slayer = None


def check_backend_library(backend: str):
    """
    * :ref:`API in English <check_backend_library-en>`

    .. _check_backend_library-cn:

    :param backend: ``'torch'``, ``'cupy'`` 或 ``'lava'``
    :type backend: str

    检查某个后端的python库是否已经安装。若未安装则此函数会报错。

    * :ref:`中文 API <check_backend_library-cn>`

    .. _check_backend_library-en:

    :param backend: ``'torch'``, ``'cupy'`` or ``'lava'``
    :type backend: str

    Check whether the python lib for backend is installed. If not, this function will raise an error.
    """
    if backend == 'torch':
        return
    elif backend == 'cupy':
        if cupy is None:
            raise ImportError('CuPy is not installed! You can install it from "https://github.com/cupy/cupy".')
    elif backend == 'lava':
        if slayer is None:
            raise ImportError('Lava-DL is not installed! You can install it from ' \
                              '"https://github.com/lava-nc/lava-dl". ')
    else:
        pass


class StepModule:
    def supported_step_mode(self):
        """
        * :ref:`API in English <StepModule.supported_step_mode-en>`

        .. _StepModule.supported_step_mode-cn:

        :return: 包含支持的后端的tuple
        :rtype: tuple[str]

        返回此模块支持的步进模式。

        * :ref:`中文 API <StepModule.supported_step_mode-cn>`

        .. _StepModule.supported_step_mode-en:

        :return: a tuple that contains the supported backends
        :rtype: tuple[str]

        """
        return ('s', 'm')

    @property
    def step_mode(self):
        """
        * :ref:`API in English <StepModule.step_mode-en>`

        .. _StepModule.step_mode-cn:

        :return: 模块当前使用的步进模式
        :rtype: str

        * :ref:`中文 API <StepModule.step_mode-cn>`

        .. _StepModule.step_mode-en:

        :return: the current step mode of this module
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        """
        * :ref:`API in English <StepModule.step_mode-setter-en>`

        .. _StepModule.step_mode-setter-cn:

        :param value: 步进模式
        :type value: str

        将本模块的步进模式设置为 ``value``

        * :ref:`中文 API <StepModule.step_mode-setter-cn>`

        .. _StepModule.step_mode-setter-en:

        :param value: the step mode
        :type value: str

        Set the step mode of this module to be ``value``

        """
        if value not in self.supported_step_mode():
            raise ValueError(f'step_mode can only be {self.supported_step_mode()}, but got "{value}"!')
        self._step_mode = value

class SingleModule(StepModule):
    """
    * :ref:`API in English <SingleModule-en>`

    .. _SingleModule-cn:

    只支持单步的模块 (``step_mode == 's'``)。

    * :ref:`中文 API <SingleModule-cn>`

    .. _SingleModule-en:

    The module that only supports for single-step (``step_mode == 's'``)
    """
    def supported_step_mode(self):
        return ('s', )

class MultiStepModule(StepModule):
    """
    * :ref:`API in English <MultiStepModule-en>`

    .. _MultiStepModule-cn:

    只支持多步的模块 (``step_mode == 'm'``)。

    * :ref:`中文 API <MultiStepModule-cn>`

    .. _MultiStepModule-en:

    The module that only supports for multi-step (``step_mode == 'm'``)
    """
    def supported_step_mode(self):
        return ('m', )

class MemoryModule(nn.Module, StepModule):
    def __init__(self):
        """
        * :ref:`API in English <MemoryModule.__init__-en>`

        .. _MemoryModule.__init__-cn:

        ``MemoryModule`` 是SpikingJelly中所有有状态（记忆）模块的基类。

        * :ref:`中文API <MemoryModule.__init__-cn>`

        .. _MemoryModule.__init__-en:

        ``MemoryModule`` is the base class of all stateful modules in SpikingJelly.

        """
        super().__init__()
        self._memories = {}
        self._memories_rv = {}
        self._backend = 'torch'
        self.step_mode = 's'

    @property
    def supported_backends(self):
        """
        * :ref:`API in English <MemoryModule.supported_backends-en>`

        .. _MemoryModule.supported_backends-cn:

        返回支持的后端，默认情况下只有 `('torch', )`

        :return: 支持的后端
        :rtype: tuple[str]

        * :ref:`中文API <MemoryModule.supported_backends-cn>`

        .. _MemoryModule.supported_backends-en:

        Return the supported backends. The default return value is `('torch', )`

        :return: supported backends
        :rtype: tuple[str]

        """
        return ('torch',)

    @property
    def backend(self):
        return self._backend

    @backend.setter
    def backend(self, value: str):
        if value not in self.supported_backends:
            raise NotImplementedError(f'{value} is not a supported backend of {self._get_name()}!')
        check_backend_library(value)
        self._backend = value

    @abstractmethod
    def single_step_forward(self, x: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.single_step_forward-en>`

        .. _MemoryModule.single_step_forward-cn:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        本模块的单步的前向传播函数


        * :ref:`中文 API <MemoryModule.single_step_forward-cn>`

        .. _MemoryModule.single_step_forward-en:

        :param x: input tensor with ``shape = [N, *] ``
        :type x: torch.Tensor

        The single-step forward function for this module

        """
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        """
        * :ref:`API in English <MemoryModule.multi_step_forward-en>`

        .. _MemoryModule.multi_step_forward-cn:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        本模块的多步的前向传播函数，通过调用 ``T`` 次 ``single_step_forward(x[t], *args, **kwargs)`` 实现


        * :ref:`中文 API <MemoryModule.multi_step_forward-cn>`

        .. _MemoryModule.multi_step_forward-en:

        :param x_seq: input tensor with ``shape = [T, N, *] ``
        :type x_seq: torch.Tensor

        The multi-step forward function for this module, which is implemented by calling ``single_step_forward(x[t], *args, **kwargs)`` over ``T`` times

        """

        T = x_seq.shape[0]
        y_seq = []
        for t in range(T):
            y = self.single_step_forward(x_seq[t], *args, **kwargs)
            y_seq.append(y.unsqueeze(0))

        return torch.cat(y_seq, 0)

    def forward(self, *args, **kwargs):
        if self.step_mode == 's':
            return self.single_step_forward(*args, **kwargs)
        elif self.step_mode == 'm':
            return self.multi_step_forward(*args, **kwargs)
        else:
            raise ValueError(self.step_mode)

    def extra_repr(self):
        return f'step_mode={self.step_mode}, backend={self.backend}'

    def register_memory(self, name: str, value):
        """
        * :ref:`API in English <MemoryModule.register_memory-en>`

        .. _MemoryModule.register_memory-cn:

        :param name: 变量的名字
        :type name: str
        :param value: 变量的值
        :type value: any

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。每次调用 ``self.reset()``
        函数后， ``self.name`` 都会被重置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``. ``self.name`` will be set to ``value`` after
        each calling of ``self.reset()``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量为默认值。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables to their default values.
        """
        for key in self._memories.keys():
            self._memories[key] = copy.deepcopy(self._memories_rv[key])

    def set_reset_value(self, name: str, value):
        self._memories_rv[name] = copy.deepcopy(value)

    def __getattr__(self, name: str):
        if '_memories' in self.__dict__:
            memories = self.__dict__['_memories']
            if name in memories:
                return memories[name]

        return super().__getattr__(name)

    def __setattr__(self, name: str, value) -> None:
        _memories = self.__dict__.get('_memories')
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

    def memories(self):
        """
        * :ref:`API in English <MemoryModule.memories-en>`

        .. _MemoryModule.memories-cn:

        :return: 返回一个所有状态变量的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.memories-cn>`

        .. _MemoryModule.memories-en:

        :return: an iterator over all stateful variables
        :rtype: Iterator
        """
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
        """
        * :ref:`API in English <MemoryModule.named_memories-en>`

        .. _MemoryModule.named_memories-cn:

        :return: 返回一个所有状态变量及其名称的迭代器
        :rtype: Iterator

        * :ref:`中文API <MemoryModule.named_memories-cn>`

        .. _MemoryModule.named_memories-en:

        :return: an iterator over all stateful variables and their names
        :rtype: Iterator
        """

        for name, value in self._memories.items():
            yield name, value

    def detach(self):
        """
        * :ref:`API in English <MemoryModule.detach-en>`

        .. _MemoryModule.detach-cn:

        从计算图中分离所有有状态变量。

        .. tip::

            可以使用这个函数实现TBPTT(Truncated Back Propagation Through Time)。


        * :ref:`中文API <MemoryModule.detach-cn>`

        .. _MemoryModule.detach-en:

        Detach all stateful variables.

        .. admonition:: Tip
            :class: tip

            We can use this function to implement TBPTT(Truncated Back Propagation Through Time).

        """

        for key in self._memories.keys():
            if isinstance(self._memories[key], torch.Tensor):
                self._memories[key].detach_()

    def _apply(self, fn):
        for key, value in self._memories.items():
            if isinstance(value, torch.Tensor):
                self._memories[key] = fn(value)
        # do not apply on default values
        # for key, value in self._memories_rv.items():
        #     if isinstance(value, torch.Tensor):
        #         self._memories_rv[key] = fn(value)
        return super()._apply(fn)

    def _replicate_for_data_parallel(self):
        replica = super()._replicate_for_data_parallel()
        replica._memories = self._memories.copy()
        return replica




