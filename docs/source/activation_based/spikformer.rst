脉冲Transformer构建、训练和改进
=======================================
本教程作者： `周昭坤 <https://github.com/ZK-Zhou>`_

本节教程主要介绍基于Spikingjelly构建脉冲Transformer （Spiking Transformer，Spikformer）模型，训练脉冲Transformer的细节以及改进脉冲Transformer架构的关键点。
和SEW ResNet相比，Spikformer的结构和堆叠方式较为简单，具体来说由三个主要组件，即脉冲块分割前馈模块（Spiking Patch Splitting，SPS）、脉冲自注意力机制（Spiking Self Attention，SSA）和多层感知模块（Multi-Layer Perceptron，MLP）组成。
堆叠方式则为一个SPS加若干个SSA-MLP组合块。

构建脉冲Transformer
----------------

首先导入相关的模块：

.. code-block:: python

    import torch
    import torch.nn as nn
    import numpy as np
    from spikingjelly.activation_based import neuron


脉冲自注意力机制的Query、Key和Value均为脉冲序列，具体做法是在三个张量输出时添加脉冲神经元，耦合脉冲神经元来避免引入负值，取消了Softmax函数，构建脉冲自注意力机制：

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
基于SSA和MLP构建脉冲Transformer的Block，注意此处使用SEW形式残差，若使用MS形式残差则需在SSA和MLP中更改脉冲神经元的位置：

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

最后加入前馈模块，组成Spikformer，读者还可以根据处理任务的分辨率和复杂性设计分层的Spikformer，参考QKformer。

训练脉冲Transformer
----------------
脉冲Transformer的训练与SEW ResNet不同，后者需要的轮次较少且收敛较快，而Spikformer通常需要更多的轮次才能收敛。
以ImageNet为例，SEW ResNet只需150轮次，Spikformer需要训练200轮以上，性能随着训练轮次的增加而提升。
此外，学习率更新方式和数据增强策略也对Spikformer性能影响较大。

改进脉冲Transformer
----------------
脉冲自注意力机制的建模形式尚处于开放探索阶段，已有多种改进，改进点的具体位置有：改进QKV的形式和计算方式，增强QKV的时空关注能力，设计脉冲位置编码和SSA分块加速等。
读者可根据实际任务需求和性能导向探索符合SNN的新型机制。此外，针对脉冲Transformer中的MLP和SPS前馈模块的改进也会显著影响其性能。
一些Spikformer变体有：SpikingResformer、Spike-driven Transformer V1、V2和V3和QKformer等等，详见 `此处 <https://scholar.google.com.hk/scholar?oi=bibs&hl=zh-CN&cites=12209743464525142624&as_sdt=5>`_
