编写CUPY神经元
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本教程介绍如何编写CUPY后端的神经元。本教程需要如下的前置知识：

#. 了解CUDA，能够实现简单的逐元素CUDA内核

#. 能够使用 :class:`torch.autograd.Function` 实现自定义反向传播

#. 已经阅读了 :class:`spikingjelly.activation_based.auto_cuda.base` 的全部API文档，\
   能够使用 :class:`spikingjelly.activation_based.auto_cuda.base` 编写2D CUDA内核

实现IF神经元的CUDA多步前向传播
-----------------------------

假设我们要编写一个python函数用于神经元进行多步前向传播(FPTT)，则这个函数的输入应该至少包括：

* ``v_init``: ``shape = [N]``，表示神经元在当前时刻的电压。其中 ``N`` 为神经元的数量。\
  当神经元是多维时，``N`` 应该是神经元展平后的数量

* ``x_seq``: ``shape = [T, N]``，表示 ``T`` 个time-steps的输入

* ``v_th``: ``float``，表示阈值电压

如果使用 hard reset，则还需要增加一个参数：

* ``v_reset``: ``float``，表示重置电压

这个函数的输出应该包括：

* ``spike_seq``: ``shape = [T, N]``，表示输出的 ``T`` 个time-steps的脉冲

* ``v_seq``: ``shape = [T, N]``，表示 ``T`` 个time-steps的放电后的电压。我们需要输出所有时刻而不仅仅是最后时刻的电压，因为有时可能会用到这些数据


若将FPTT写成CUDA函数，则函数参数仍然包括上述参数，但还需要一些额外的参数。

:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronFPTTKernel` 继承自\
:class:`spikingjelly.activation_based.auto_cuda.base.CKernel2D`。``NeuronFPTTKernel`` \
是神经元进行多步前向传播(FPTT)的CUDA内核基类。

我们可以查看其默认的CUDA参数声明：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronFPTTKernel(hard_reset=True, dtype='float')
    for key, value in base_kernel.cparams.items():
        print(f'key="{key}",'.ljust(20), f'value="{value}"'.ljust(20))


输出为：

.. code-block:: bash

    key="numel",         value="const int &" 
    key="N",             value="const int &" 
    key="x_seq",         value="const float *"
    key="v_v_seq",       value="float *"     
    key="h_seq",         value="float *"     
    key="spike_seq",     value="float *"     
    key="v_th",          value="float &"     
    key="v_reset",       value="float &" 

绝大多数参数与之前相同，不同的参数包括

* ``numel``: 元素总数，即 ``numel = T * N``

* ``N``: 神经元的数量

* ``v_v_seq``: ``shape = [T + 1, N]``，合并 ``v_init`` 和 ``v_seq`` 得到的

* ``h_seq``: ``shape = [T, N]``，充电后放电前的电压，反向传播时需要用到

``NeuronFPTTKernel`` 作为神经元FPTT的基类，类似于 :class:`spikingjelly.activation_based.neuron.BaseNode`，已经实现了\
放电和重置方程。我们在实现新神经元的FPTT CUDA内核时，只需要继承 ``NeuronFPTTKernel`` 并补充充电方程即可。

我们首先查看一下 ``NeuronFPTTKernel`` 的完整代码：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronFPTTKernel(hard_reset=True, dtype='float')
    

输出为：

.. code-block:: c++

        #include <cuda_fp16.h>
        extern "C" __global__
        void NeuronFPTTKernel_float_hard_reset(
        const int & numel, const int & N, const float * x_seq, float * v_v_seq, float * h_seq, float * spike_seq, float & v_th, float & v_reset
        )
        
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        
                for(int t = index; t < numel; t += dt)
                {
            
                  // neuronal charge should be defined in this function!;
                  spike_seq[t] = (h_seq[t] - v_th) >= 0.0f ? 1.0f: 0.0f;
                  v_v_seq[t + dt] = h_seq[t] * (1.0f - spike_seq[t]) + v_reset * spike_seq[t];

                }
        
            }
        }

可以发现，这个内核已经比较完善，仅需要我们补充部分代码。

``NeuronFPTTKernel`` 提供了 ``neuronal_charge`` 函数：

.. code-block:: python

    class NeuronFPTTKernel(base.CKernel2D):
        # ...
        @property
        def neuronal_charge(self) -> str:
            # e.g., for IFNode, this function shoule return:
            #   cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=dtype)
            return '// neuronal charge should be defined in this function!'


如果想要实现新的神经元，只需要重定义这个函数。现在以最简单的IF神经元为例，其充电方程为

.. math::
    
    H[t] = V[t - 1] + X[t]

则实现方式为：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):

        @property
        def neuronal_charge(self) -> str:
            # note that v_v_seq[t] is v_seq[t - dt]
            return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

    if_fptt_kernel = IFNodeFPTTKernel(hard_reset=True, dtype='float')
    print(if_fptt_kernel.full_codes)

输出为：

.. code-block:: c++

        #include <cuda_fp16.h>
        extern "C" __global__
        void IFNodeFPTTKernel_float_hard_reset(
        const int & numel, const int & N, const float * x_seq, float * v_v_seq, float * h_seq, float * spike_seq, float & v_th, float & v_reset
        )
        
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        
                for(int t = index; t < numel; t += dt)
                {
            
                  h_seq[t] = x_seq[t] + v_v_seq[t];
                  spike_seq[t] = (h_seq[t] - v_th) >= 0.0f ? 1.0f: 0.0f;
                  v_v_seq[t + dt] = h_seq[t] * (1.0f - spike_seq[t]) + v_reset * spike_seq[t];

                }
        
            }
        }

这其实就是一个完整的CUDA内核了。可以发现，``NeuronFPTTKernel`` 给编写CUDA内核带来了极大的方便。

需要注意的是，这里我们使用：

.. code-block:: python

    @property
    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

而不是手动编写：

.. code-block:: python

    @property
    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return 'h_seq[t] = x_seq[t] + v_v_seq[t];'

原因在于 :class:`spikingjelly.activation_based.auto_cuda.cfunction` 提供的函数，通常包括 ``float``\
和 ``half2`` 两种数据类型的实现，比我们手动编写两种更便捷。

若设置 ``dtype='half2'``，可以直接得到半精度的内核：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):

        @property
        def neuronal_charge(self) -> str:
            # note that v_v_seq[t] is v_seq[t - dt]
            return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

    if_fptt_kernel = IFNodeFPTTKernel(hard_reset=True, dtype='half2')
    print(if_fptt_kernel.full_codes)

输出为：

.. code-block:: c++


        #include <cuda_fp16.h>
        extern "C" __global__
        void IFNodeFPTTKernel_half2_hard_reset(
        const int & numel, const int & N, const half2 * x_seq, half2 * v_v_seq, half2 * h_seq, half2 * spike_seq, half2 & v_th, half2 & v_reset
        )
        
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        
                for(int t = index; t < numel; t += dt)
                {
            
                  h_seq[t] = __hadd2(x_seq[t], v_v_seq[t]);
                  spike_seq[t] = __hgeu2(__hsub2(h_seq[t], v_th), __float2half2_rn(0.0f));
                  v_v_seq[t + dt] = __hfma2(h_seq[t], __hsub2(__float2half2_rn(1.0f), spike_seq[t]), __hmul2(v_reset, spike_seq[t]));

                }
        
            }
        }

实现IF神经元的CUDA多步反向传播
-----------------------------
多步反向传播要比多步前向传播更为复杂。我们首先回顾SpikingJelly中的前向传播定义：

.. math::

    \begin{align}
        H[t] &= f(V[t - 1], X[t])\\
        S[t] &= \Theta(H[t] - V_{th})\\
        V[t] &= \begin{cases}
        H[t]\left( 1 - S[t] \right) + V_{reset}S[t], &\text{Hard Reset}\\
        H[t] - V_{th}S[t], &\text{Soft Reset}\\
    \end{cases}
    \end{align}

我们在前文中实现的前向传播可以表示为：

.. math::

    S[1,2,...,T], V[1,2,...,T] = F_{fp}(X[1,2,...,T], V[0])

相应的，我们需要实现的反向传播为：

.. math::

    \frac{\mathrm{d} L}{\mathrm{d} X[1,2,...,T]},\frac{\mathrm{d} L}{\mathrm{d} V[0]} =
     F_{bp}(\frac{\partial L}{\partial S[1,2,...,T]},\frac{\partial L}{\partial V[1,2,...,T]})

根据前向传播，推出反向传播的计算式为：

.. math::

    \begin{align}
        \frac{\mathrm{d} L}{\mathrm{d} X[t]} &= \frac{\mathrm{d} L}{\mathrm{d} H[t]} \frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}\\
        \frac{\mathrm{d} L}{\mathrm{d} H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\mathrm{d} S[t]}{\mathrm{d} H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\mathrm{d} L}{\mathrm{d} H[t+1]}\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]})\frac{\mathrm{d} V[t]}{\mathrm{d} H[t]}\\
        \frac{\mathrm{d} S[t]}{\mathrm{d} H[t]} &= \Theta'(H[t] - V_{th})\\
        \frac{\mathrm{d} V[t]}{\mathrm{d} H[t]} &= 
        \begin{cases}
            1 - S[t] + (-H[t] + V_{reset})\frac{\partial S[t]}{\partial H[t]}(1-D_{reset}), &\text{Hard Reset}\\
            1 - V_{th}\frac{\partial S[t]}{\partial H[t]}(1-D_{reset}), &\text{Soft Reset}\\
        \end{cases}
    \end{align}

其中 :math:`D_{reset}` 表示是否detach reset：

.. math::

    D_{reset} = \begin{cases}
        1, &\text{Detach Reset}\\
        0, &\text{Not Detach Reset}\\
    \end{cases}

合并公式得到：

.. math::

    \begin{align}
    \frac{\mathrm{d} L}{\mathrm{d} H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\mathrm{d} S[t]}{\mathrm{d} H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\mathrm{d} L}{\mathrm{d} H[t+1]}\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]})\frac{\mathrm{d} V[t]}{\mathrm{d} H[t]}\\
    \frac{\mathrm{d} L}{\mathrm{d} X[t]} &= \frac{\mathrm{d} L}{\mathrm{d} H[t]}\frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}\\
    \frac{\mathrm{d} L}{\mathrm{d} V[0]} &= \frac{\mathrm{d} L}{\mathrm{d} H[1]}\frac{\mathrm{d} H[1]}{\mathrm{d} V[0]}
    \end{align}