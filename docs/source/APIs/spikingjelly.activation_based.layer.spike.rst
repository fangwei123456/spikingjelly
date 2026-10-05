Binary spike layers / 二值脉冲层
=================================

通过 ``spikingjelly.activation_based.layer`` 导入 ``SpikeLinear`` 和
``SpikeConv1d/2d/3d``。这些层保持对应 PyTorch 层的参数、state_dict 和 padding
语义，输入须为二值 0/1；无时间状态。数值计算调用已注册的 functional spike 接口。

Import ``SpikeLinear`` and ``SpikeConv1d/2d/3d`` from
``spikingjelly.activation_based.layer``. These layers retain the corresponding
PyTorch parameter, state_dict and padding semantics, require binary 0/1 inputs,
and have no temporal state. They call the registered functional spike interfaces.

.. automodule:: spikingjelly.activation_based.layer.spike
   :members:
   :show-inheritance:
