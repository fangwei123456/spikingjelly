Conv-BN Fusion Functions
+++++++++++++++++++++++++++++

SpikingJelly 提供了将 **卷积层和批归一化层融合** 的工具。这些函数可以计算融合后的权重和偏置，加速卷积 SNN 的推理，使硬件部署成为可能。

----

SpikingJelly provides tools for **fusing convolution with batch normalization**. These functions can compute the fused weights and biases, accelerating the inference process of Convolutional SNNs, and make hardware deployment possible.

.. automodule:: spikingjelly.activation_based.functional.conv_bn_fuse
   :members:
   :undoc-members:
   :show-inheritance:
   :exclude-members: supported_backends, extra_repr, jit_*
