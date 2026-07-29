Learning State Updates
++++++++++++++++++++++

这些函数显式接收 STDP/mSTDP/mSTDP-ET 的局部状态和 raw tensor 参数，不读取
``MemoryModule`` 的隐式 memory、monitor 缓存，也不负责 ``step_mode``、
``training/eval`` 或梯度写入。

----

These functions receive STDP/mSTDP/mSTDP-ET local state and raw tensor
parameters explicitly. They do not read implicit ``MemoryModule`` memory or
monitor buffers, and do not manage ``step_mode``, ``training/eval``, or gradient
writes.

.. automodule:: spikingjelly.activation_based.functional.learning
   :members:
   :undoc-members:
   :show-inheritance:
