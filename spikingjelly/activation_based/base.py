import torch
import torch.nn as nn
import copy
from abc import abstractmethod

try:
    import cupy
except BaseException as e:
    cupy = None

try:
    import lava.lib.dl.slayer as slayer
except BaseException as e:
    slayer = None


def check_backend_library(backend: str):
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
        raise NotImplementedError(backend)

class MemoryModule(nn.Module):
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
        self._step_mode = 's'

    @property
    def step_mode(self):
        """
        * :ref:`API in English <MemoryModule.step_mode-en>`

        .. _MemoryModule.step_mode-cn:

        单步模式的输入输出均为 `shape = [N, *]` ，表示在 `1` 个时间步的数据。
        多步模式的输入输出均为 `shape = [T, N, *]` ，表示在 `T` 个时间步的数据。

        .. code-block:: python

            # note that this is just an example and can not run because `MemoryModule` has not defined forward function

            net = MemoryModule()
            T = 4
            N = 2
            x = torch.rand([T, N])
            ys = []
            net.step_mode = 's'
            for t in range(T):
                ys.append(net(x[t]).unsqueeze(0))
            ys = torch.cat(ys)
            net.reset()

            net.step_mode = 'm'
            ym = net(x)
            error = (ys - ym).abs().max()
            # error is 0

        :return: 单步还是多步，`s`表示单步，`m`表示多步。
        :rtype: str

        * :ref:`中文API <MemoryModule.step_mode-cn>`

        .. _MemoryModule.step_mode-en:

        The single-step mode uses data with `shape = [N, *]`, which is the data at one time-step.
        The multi-step mode uses data with `shape = [T, N, *]`, which is the data at `T` time-steps.

        .. code-block:: python

            # note that this is just an example and can not run because `MemoryModule` has not defined forward function

            net = MemoryModule()
            T = 4
            N = 2
            x = torch.rand([T, N])
            ys = []
            net.step_mode = 's'
            for t in range(T):
                ys.append(net(x[t]).unsqueeze(0))
            ys = torch.cat(ys)
            net.reset()

            net.step_mode = 'm'
            ym = net(x)
            error = (ys - ym).abs().max()
            # error is 0

        :return: the step mode. `s` means single-step, and `m` means multi-step
        :rtype: str
        """
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        if value not in ('s', 'm'):
            raise ValueError(f'step_mode can only be "s" or "m", but got "{value}"!')
        self._step_mode = value

    @property
    def supported_backends(self):
        """
        * :ref:`API in English <MemoryModule.supported_backends-en>`

        .. _MemoryModule.supported_backends-cn:

        返回支持的后端，默认情况下只有 `('torch', )`。如果继承者支持了其他后端，需要覆盖这个函数

        :return: 支持的后端，str组成的tuple
        :rtype: tuple

        * :ref:`中文API <MemoryModule.supported_backends-cn>`

        .. _MemoryModule.supported_backends-en:

        Return the supported backends. The default return value is `('torch', )`.
        If the child module supports other backends, it should override this function

        :return: supported backends in the form of `(str, ...)`
        :rtype: tuple

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
        pass

    def multi_step_forward(self, x_seq: torch.Tensor, *args, **kwargs):
        # x_seq.shape = [T, *]
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

        将变量存入用于保存有状态变量（例如脉冲神经元的膜电位）的字典中。这个变量的重置值会被设置为 ``value``。

        * :ref:`中文API <MemoryModule.register_memory-cn>`

        .. _MemoryModule.register_memory-en:

        :param name: variable's name
        :type name: str
        :param value: variable's value
        :type value: any

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a
        spiking neuron). The reset value of this variable will be ``value``.

        """
        assert not hasattr(self, name), f'{name} has been set as a member variable!'
        self._memories[name] = value
        self.set_reset_value(name, value)

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables.
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
        for name, value in self._memories.items():
            yield value

    def named_memories(self):
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


class StatelessModule:
    @property
    def step_mode(self):
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        if value not in ('s', 'm'):
            raise ValueError(f'step_mode can only be "s" or "m", but got "{value}"!')
        self._step_mode = value

class MultiStepStatelessModule(StatelessModule):
    @property
    def step_mode(self):
        return self._step_mode

    @step_mode.setter
    def step_mode(self, value: str):
        if value != 'm':
            raise ValueError(f'{self.__class__.__name__} only supports for step_mode = s (multi-step mode)!')
        self._step_mode = value