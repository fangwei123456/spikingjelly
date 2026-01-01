Spiking Transformer Construction, Training, and Improvements
===============================================================

Tutorial author: `Zhou Zhaokun <https://github.com/ZK-Zhou>`_ , `Yifan Huang (AllenYolk) <https://github.com/AllenYolk>`_

中文版： :doc:`../cn/spikformer`

This tutorial mainly introduces the construction of the Spiking Transformer (Spiking Transformer, Spikformer [#spikformer]_ ) model based on Spikingjelly, the details of training the Spiking Transformer, and the key points of improving the Spiking Transformer architecture.
Compared to SEW ResNet, the structure and stacking method of Spikformer are relatively simple, specifically consisting of three main components: Spiking Patch Splitting (SPS), Spiking Self Attention (SSA), and Multi-Layer Perceptron (MLP).
The stacking method is one SPS followed by multiple SSA-MLP combination blocks. The specific SSA and Spikformer are shown in the figure:

.. image:: ../../_static/tutorials/spikformer/spikformer-overview.png
    :width: 100%


Building a Spiking Transformer
---------------------------------------

First, import the relevant modules:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.activation_based import neuron


In the Spiking Self Attention mechanism, Query, Key, and Value are all spike sequences. The specific approach is to add spike neurons to the output of the three tensors, coupling spiking neurons to avoid introducing negative values. The Softmax function is removed, constructing the Spiking Self Attention mechanism:

.. code-block:: python

    class SSA(nn.Module):
        def __init__(self, dim, num_heads=8):
            super().__init__()
            assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
            self.dim = dim
            self.num_heads = num_heads
            self.scale = 0.125

            self.q_linear = nn.Linear(dim, dim)
            self.q_bn = nn.BatchNorm1d(dim)
            self.q_lif = neuron.LIFNode(step_mode="m")

            self.k_linear = nn.Linear(dim, dim)
            self.k_bn = nn.BatchNorm1d(dim)
            self.k_lif = neuron.LIFNode(step_mode="m")

            self.v_linear = nn.Linear(dim, dim)
            self.v_bn = nn.BatchNorm1d(dim)
            self.v_lif = neuron.LIFNode(step_mode="m")

            self.attn_lif = neuron.LIFNode(step_mode="m")

            self.proj_linear = nn.Linear(dim, dim)
            self.proj_bn = nn.BatchNorm1d(dim)
            self.proj_lif = neuron.LIFNode(step_mode="m")

        def forward(self, x):
            T,B,N,C = x.shape

            x_for_qkv = x.flatten(0, 1)  # TB, N, C
            q_linear_out = self.q_linear(x_for_qkv)  # [TB, N, C]
            q_linear_out = self.q_bn(q_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            q_linear_out = self.q_lif(q_linear_out)
            q = q_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            k_linear_out = self.k_linear(x_for_qkv)
            k_linear_out = self.k_bn(k_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            k_linear_out = self.k_lif(k_linear_out)
            k = k_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            v_linear_out = self.v_linear(x_for_qkv)
            v_linear_out = self.v_bn(v_linear_out. transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C).contiguous()
            v_linear_out = self.v_lif(v_linear_out)
            v = v_linear_out.reshape(T, B, N, self.num_heads, C//self.num_heads).permute(0, 1, 3, 2, 4).contiguous()

            attn = (q @ k.transpose(-2, -1)) * self.scale

            x = attn @ v
            x = x.transpose(2, 3).reshape(T, B, N, C).contiguous()
            x = self.attn_lif(x)
            x = x.flatten(0, 1)
            x = self.proj_lif(self.proj_bn(self.proj_linear(x).transpose(-1, -2)).transpose(-1, -2).reshape(T, B, N, C))

            return x

Based on SSA and MLP, construct the Spiking Transformer Block. Note that SEW-style residuals are used here. If MS-style residuals are used, the position of spike neurons in SSA and MLP needs to be changed:

.. code-block:: python

    class Block(nn.Module):
        def __init__(self, dim, num_heads, mlp_ratio=4.):
            super().__init__()
            self.attn = SSA(dim, num_heads=num_heads)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim)

        def forward(self, x):
            x = x + self.attn(x)
            x = x + self.mlp(x)
            return x

Finally, add the feedforward module to form Spikformer. The reader can also design hierarchical Spikformer based on resolution and complexity of the task. Refer to QKformer.

Improved SSA Implementation
--------------------------------

SpikingJelly ``0.0.0.1.0`` provides an efficient implementation of SSA in :class:`SpikingSelfAttention <spikingjelly.activation_based.layer.attention.SpikingSelfAttention>`.
Compared with the ``SSA`` introduced in the previous section, :class:`SpikingSelfAttention <spikingjelly.activation_based.layer.attention.SpikingSelfAttention>` introduces the following improvements.

#. Assume that both the input and output have shape ``[T, B, C, N]`` instead of ``[T, B, N, C]`` (**token-last** rather than channel-last). ``Conv1d`` is used instead of ``Linear`` layers. As a result, no tensor transposition is required before or after ``BatchNorm1d``.
#. The three Conv-BN-LIF blocks for Q, K, and V are merged into a single block whose channel dimension is three times larger. This allows ``q``, ``k``, and ``v`` to be generated in a single forward pass.
#. The order of tensor multiplications is modified. In the original implementation, ``q``, ``k``, and ``v`` have shape ``[T, B, N, C]``, and the tensor multiplication is performed as ``q @ k.transpose(-2, -1) @ v``. In the improved implementation, ``q``, ``k``, and ``v`` have shape ``[T, B, C, N]``, and the tensor multiplication becomes ``v @ k.transpose(-2, -1) @ q``.

.. note::

    Denote the original ``q``, ``k``, and ``v`` tensors of shape ``[T, B, N, C]``  as :math:`Q`, :math:`K`, and :math:`V`. The tensor multiplication (i.e., batched matrix multiplication) in SSA can then be written as

    .. math::

        X = Q K^T V,

    where :math:`K^T` denotes transposing the last two dimensions of :math:`K`. In the improved implementation, the new ``q``, ``k``, and ``v`` tensors, as well as the resulting tensor ``x``, have shape ``[T, B, C, N]``, which corresponds to :math:`Q^T`, :math:`K^T`, :math:`V^T`, and :math:`X^T`. Denoting these token-last tensors as :math:`Q'`, :math:`K'`, :math:`V'`, and :math:`X'`, we have

    .. math::

        X' &= X^T \\
           &= (Q K^T V)^T \\
           &= V^T K Q^T \\
           &= V' K'^T Q'

    Therefore, for the token-last tensors, the correct order of tensor multiplication is ``v @ k.transpose(-2, -1) @ q``.

The resulting ``SpikingSelfAttention`` module is shown below (for illustration purposes only). For more implementation details, please refer to the documentation and source code of :class:`SpikingSelfAttention <spikingjelly.activation_based.layer.attention.SpikingSelfAttention>`.

.. code:: python

    from spikingjelly.activation_based.layer import SeqToANNContainer

    class SpikingSelfAttention(nn.Module):
        def __init__(self, dim, num_heads=8):
            super().__init__()
            assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
            self.dim = dim
            self.num_heads = num_heads
            self.head_dim = dim // num_heads
            self.scale = 0.125

            self.qkv_conv_bn = SeqToANNContainer(
                nn.Conv1d(dim, dim * 3, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(dim*3),
            )
            self.qkv_lif = neuron.LIFNode(step_mode="m")

            self.attn_lif = neuron.LIFNode(step_mode="m")

            self.proj_conv_bn = SeqToANNContainer(
                nn.Conv1d(dim, dim, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm1d(dim),
            )
            self.proj_lif = neuron.LIFNode(step_mode="m")

        def forward(self, x_seq: torch.Tensor):
            T, B, C, N = x_seq.shape

            qkv = self.qkv_conv_bn(x_seq)
            qkv = self.qkv_lif(qkv) # [T, B, 3*C, N]
            qkv = qkv.reshape(T, B, 3*self.num_heads, C // self.num_heads, N)

            qt, kt, vt = qkv.chunk(3, dim=2)
            # qt, kt, vt.shape = [T, B, NUM_HEADS, Cph, N]
            x_seq = vt @ kt.transpose(-2, -1)
            x_seq = (x_seq@qt) * self.scale # [T, B, NUM_HEADS, Cph, N]

            x_seq = self.attn_lif(x_seq).reshape(T, B, C, N)

            x_seq = self.proj_conv_bn(x_seq)
            x_seq = self.proj_lif(x_seq) # [T, B, C, N]
            return x_seq

.. note::

    If token-last format is adopted, the implementation of ``MLP`` should also be changed from ``Linear`` to ``Conv1d`` in order to avoid unnecessary ``reshape`` operations that involve data copying.


Training Spiking Transformer
-----------------------------
The training of Spiking Transformer is different from SEW ResNet, which requires fewer epochs and converges quickly, while Spikformer generally requires more epochs to converge.
Taking ImageNet as an example, SEW ResNet only needs 150 epochs, while Spikformer needs more than 200 epochs, with performance increasing as training epochs increase.
Additionally, the learning rate update method and data augmentation strategy also have a significant impact on Spikformer's performance.

Improving Spiking Transformer
-----------------------------
The modeling form of Spiking Self Attention mechanism is still in open exploration, and there are multiple improvements, including: improving the form and calculation method of QKV, enhancing spatial-temporal attention capability of QKV, designing spike position encoding, and accelerating SSA block splitting.
Readers can explore new mechanisms suitable for SNN based on actual task requirements and performance orientation. Furthermore, the improvement of MLP and SPS feedforward modules in Spiking Transformer will also significantly affect its performance.
Some Spikformer variants include: SpikingResformer [#spikingresformer]_, as shown in:

.. image:: ../../_static/tutorials/spikformer/spikingresformer.png
    :width: 100%


As well as Spike-driven Transformer V1, V2, and V3, and QKformer, etc. See `here <https://scholar.google.com.hk/scholar?oi=bibs&hl=en&cites=12209743464525142624&as_sdt=5>`_ for details.



.. [#spikformer] Zhou Zhaokun, Zhu Yuesheng, He Chao, Wang Yaowei, Yan Shuicheng, Tian Yonghong, Yuan Li. Spikformer: When Spiking Neural Network Meets Transformer [C]. Proceedings of International Conference on Learning Representations, 2023.
.. [#spikingresformer] Shi Xinyu, Hao Zecheng, Yu Zhaofei. SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks [C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024: 5610-5619.
