Neuron State Transition Functions
++++++++++++++++++++++++++++++++++++++++

这些函数显式接收并返回神经元状态，不读取 ``MemoryModule`` 的隐式 memory，
也不负责 ``training/eval`` 或 backend dispatch。

----

These functions receive and return neuron states explicitly. They do not read
implicit ``MemoryModule`` memory and do not handle ``training/eval`` or backend
dispatch.

.. automodule:: spikingjelly.activation_based.functional.neuron
   :members:
   :undoc-members:
   :show-inheritance:
