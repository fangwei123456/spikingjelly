Spiking Transformer Construction, Training, and Improvements
===============================================================
Tutorial author: `Zhou Zhaokun <https://github.com/ZK-Zhou>`_

This tutorial mainly introduces the construction of the Spiking Transformer (Spiking Transformer, Spikformer [#spikformer]_ ) model based on Spikingjelly, the details of training the Spiking Transformer, and the key points of improving the Spiking Transformer architecture.
Compared to SEW ResNet, the structure and stacking method of Spikformer are relatively simple, specifically consisting of three main components: Spiking Patch Splitting (SPS), Spiking Self Attention (SSA), and Multi-Layer Perceptron (MLP).
The stacking method is one SPS followed by multiple SSA-MLP combination blocks. The specific SSA and Spikformer are shown in the figure:

.. image:: ../_static/tutorials/spikformer/spikformer-overview.png
    :width: 100%


Building a Spiking Transformer
-----------------------------

First, import the relevant modules:

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.activation_based import neuron


In the Spiking Self Attention mechanism, Query, Key, and Value are all spike sequences. The specific approach is to add spike neurons to the output of the three tensors, coupling spiking neurons to avoid introducing negative values. The Softmax function is removed, constructing the Spiking Self Attention mechanism:

.. code-block:: python

    class SSA(nn.Module):
        def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=1):
            super().__init__()
            assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
            self.dim = dim
            self.num_heads = num_heads
            self.scale = 0.125
            self.q_linear = nn.Linear(dim, dim)
            self.q_bn = nn.BatchNorm1d(dim)
            self.q_lif = neuron.LIFNode()
            self.k_linear = nn.Linear(dim, dim)
            self.k_bn = nn.BatchNorm1d(dim)
            self.k_lif = neuron.LIFNode()

            self.v_linear = nn.Linear(dim, dim)
            self.v_bn = nn.BatchNorm1d(dim)
            self.v_lif = neuron.LIFNode()

            self.attn_lif = neuron.LIFNode()

            self.proj_linear = nn.Linear(dim, dim)
            self.proj_bn = nn.BatchNorm1d(dim)
            self.proj_lif = neuron.LIFNode()

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
        def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                    drop_path=0., norm_layer=nn.LayerNorm, sr_ratio=1):
            super().__init__()
            self.attn = SSA(dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
                                attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)
            mlp_hidden_dim = int(dim * mlp_ratio)
            self.mlp = MLP(in_features=dim, hidden_features=mlp_hidden_dim, drop=drop)

        def forward(self, x):
            x = x + self.attn(x)
            x = x + self.mlp(x)
            return x

Finally, add the feedforward module to form Spikformer. The reader can also design hierarchical Spikformer based on resolution and complexity of the task. Refer to QKformer.

Training Spiking Transformer
-----------------------------
The training of Spiking Transformer is different from SEW ResNet, which requires fewer epochs and converges quickly, while Spikformer generally requires more epochs to converge.
Taking ImageNet as an example, SEW ResNet only needs 150 epochs, while Spikformer needs more than 200 epochs, with performance increasing as training epochs increase.
Additionally, the learning rate update method and data augmentation strategy also have a significant impact on Spikformer's performance.

Improving Spiking Transformer
-----------------------------
The modeling form of Spiking Self Attention mechanism is still in open exploration, and there are multiple improvements, including: improving the form and calculation method of QKV, enhancing spatial-temporal attention capability of QKV, designing spike position encoding, and accelerating SSA block splitting.
Readers can explore new mechanisms suitable for SNN based on actual task requirements and performance orientation. Furthermore, the improvement of MLP and SPS feedforward modules in Spiking Transformer will also significantly affect its performance.
Some Spikformer variants include: SpikingResformer, as shown in:

.. image:: ../_static/tutorials/spikformer/spikingresformer.png
    :width: 100%


As well as Spike-driven Transformer V1, V2, and V3, and QKformer, etc. See `here <https://scholar.google.com.hk/scholar?oi=bibs&hl=en&cites=12209743464525142624&as_sdt=5>`_ for details.



.. [#spikformer] Zhou Zhaokun, Zhu Yuesheng, He Chao, Wang Yaowei, Yan Shuicheng, Tian Yonghong, Yuan Li. Spikformer: When Spiking Neural Network Meets Transformer [C]. Proceedings of International Conference on Learning Representations, 2023.
.. [#spikingresformer] Shi Xinyu, Hao Zecheng, Yu Zhaofei. SpikingResformer: Bridging ResNet and Vision Transformer in Spiking Neural Networks [C]. Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition, 2024: 5610-5619.
