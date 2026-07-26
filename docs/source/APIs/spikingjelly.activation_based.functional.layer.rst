Stateful Layer State Transition Functions
+++++++++++++++++++++++++++++++++++++++++

这些函数显式接收并返回 stateful layer 的局部状态，不读取
``MemoryModule`` 的隐式 memory，也不负责 module 生命周期。

----

These functions receive and return local state for stateful layers explicitly.
They do not read implicit ``MemoryModule`` memory and do not manage the module
lifecycle.

.. automodule:: spikingjelly.activation_based.functional.layer
   :members:
   :undoc-members:
   :show-inheritance:
