Neuron State Updates
++++++++++++++++++++

``*_step`` 表示一次完整状态更新。``*_multi_step`` 仅表示具有独立序列实现的路径；
backend 专用路径在函数名中标明 backend。

----

``*_step`` denotes one complete state update. ``*_multi_step`` is reserved for an
independently implemented sequence path, with backend-specific paths naming the
backend explicitly.

.. automodule:: spikingjelly.activation_based.functional.neuron
   :members:
   :undoc-members:
   :show-inheritance:
