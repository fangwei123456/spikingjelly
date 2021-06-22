import torch
import torch.nn as nn


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
        self._memory = {}
        self._memory_rv = {}

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

        Register the variable to memory dict, which saves stateful variables (e.g., the membrane potential of a spiking neuron). The reset value of this variable will be ``value``.

        """
        self._memory[name] = value
        self._memory_rv[name] = value

    def reset(self):
        """
        * :ref:`API in English <MemoryModule.reset-en>`

        .. _MemoryModule.reset-cn:

        重置所有有状态变量。

        * :ref:`中文API <MemoryModule.reset-cn>`

        .. _MemoryModule.reset-en:

        Reset all stateful variables.
        """
        for key in self._memory.keys():
            self._memory[key] = self._memory_rv[key]
            
    def __getattr__(self, name: str):
        if name in self._memory:
            return self._memory[name]
        else:
            return super().__getattr__(name)
        
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

        for key in self._memory.keys():
            if isinstance(self._memory[key], torch.Tensor):
                self._memory[key].detach_()

    def _apply(self, fn):
        for key, value in self._memory:
            if isinstance(value, torch.Tensor):
                self._memory[key] = fn(value)

        for key, value in self._memory_rv:
            if isinstance(value, torch.Tensor):
                self._memory_rv[key] = fn(value)
        return super()._apply(fn)





