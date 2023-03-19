编写CUPY神经元
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

本教程介绍如何编写CUPY后端的神经元。本教程需要如下的前置知识：

#. 了解CUDA，能够实现简单的逐元素CUDA内核

#. 能够使用 :class:`torch.autograd.Function` 实现自定义反向传播

#. 已经阅读了 :class:`spikingjelly.activation_based.auto_cuda.base` 的全部API文档，\
   能够使用 :class:`spikingjelly.activation_based.auto_cuda.base` 编写2D CUDA内核

实现IF神经元的CUDA多步前向传播
----------------------------------------------------------

假设我们要编写一个python函数用于神经元进行多步前向传播(FPTT)，则这个函数的输入应该至少包括：

* ``v_init``: ``shape = [N]``，表示神经元在当前时刻的初始电压（上一个时刻的放电后的电压）。其中 ``N`` 为神经元的数量。\
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
    print(base_kernel.full_codes)
    

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
            
                  // neuronal_charge should be defined here!;
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

        def neuronal_charge(self) -> str:
            """
            :return: CUDA code
            :rtype: str

            Returns CUDA code for calculating :math:`H[t] = f(X[t], V[t-1], ...)`.

            This function should define how ``h_seq[t]`` is calculated by ``x_seq[t], v_v_seq[t]`` and other params if
            the neuron needs.

            For example, the IF neuron define this function as:

            .. code-block:: python

                def neuronal_charge(self) -> str:
                    # note that v_v_seq[t] is v_seq[t - dt]
                    return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)
            """
            return '// neuronal_charge should be defined here!'


如果想要实现新的神经元，只需要重定义这个函数。现在以最简单的IF神经元为例，其充电方程为

.. math::
    
    H[t] = V[t - 1] + X[t]

则实现方式为：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):


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


    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

而不是手动编写：

.. code-block:: python


    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return 'h_seq[t] = x_seq[t] + v_v_seq[t];'

原因在于 :class:`spikingjelly.activation_based.auto_cuda.cfunction` 提供的函数，通常包括 ``float``\
和 ``half2`` 两种数据类型的实现，比我们手动编写两种更便捷。

若设置 ``dtype='half2'``，可以直接得到半精度的内核：

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):


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
----------------------------------------------------------
多步反向传播(BPTT)要比多步前向传播更为复杂。我们首先回顾SpikingJelly中的前向传播定义：

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


因而，BPTT函数所需要的输入为：

* ``grad_spike_seq``: ``shape = [T, N]``，表示损失对 ``T`` 个时刻的输出脉冲 ``spike_seq`` 的梯度

* ``grad_v_seq``: ``shape = [T, N]``，表示损失对 ``T`` 个时刻的放电后的电压 ``v_seq`` 的梯度

BPTT函数的输出为：

* ``grad_x_seq``: ``shape = [T, N]``，表示损失对 ``T`` 个时刻的输入 ``x_seq`` 的梯度

* ``grad_v_init``: ``shape = [N]``，表示损失对 ``v_init`` 的梯度

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

上述公式中，:math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}, \frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}` 是由神经元的充电方程\
:math:`H[t] = f(V[t - 1], X[t])` 决定，与特定的神经元相关；:math:`\frac{\mathrm{d} S[t]}{\mathrm{d} H[t]}` 由替代函数决定；\
其余部分则是通用的。

因而，:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronBPTTKernel` 也实现了通用的计算部分。我们首先查看其函数参数：


.. code-block:: python

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, hard_reset=True, detach_reset=False, dtype='float')
    for key, value in base_kernel.cparams.items():
        print(f'key="{key}",'.ljust(22), f'value="{value}"'.ljust(20))

输出为：

.. code-block:: bash

    key="numel",           value="const int &" 
    key="N",               value="const int &" 
    key="grad_spike_seq",  value="const float *"
    key="grad_v_seq",      value="const float *"
    key="h_seq",           value="const float *"
    key="grad_x_seq",      value="float *"     
    key="grad_v_init",     value="float *"     
    key="v_th",            value="float &"     
    key="v_reset",         value="float &"   

参数含义在前文中已经介绍过。

这里需要注意，我们设置 ``NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, ...``，因为在反向传播时需要指定替代函数。

在SpikingJelly的替代函数类中，提供了 ``cuda_codes`` 函数以生成反向传播的CUDA代码。以 :class:`spikingjelly.activation_based.surrogate.Sigmoid` \
为例，其定义为：

.. code-block:: python

    class Sigmoid(SurrogateFunctionBase):
        # ...
        def cuda_codes(self, y: str, x: str, dtype: str):
            return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

我们尝试打印出反向传播的代码：

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    print(surrogate.Sigmoid().cuda_codes(y='grad_s', x='over_th', dtype='float'))

输出为：

.. code-block:: c++

    const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (4.0f) * over_th));
    grad_s = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (4.0f);


如果我们要自行实现支持CUDA反向传播的替代函数，也应该遵循类似的规范，按照如下格式进行定义：

.. code-block:: python

    class CustomSurrogateFunction:
        # ...
        def cuda_codes(self, y: str, x: str, dtype: str):
            # ...


接下来查看 ``NeuronBPTTKernel`` 完整内核代码：

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, hard_reset=True, detach_reset=False, dtype='float')
    print(base_kernel.full_codes)

输出为：


.. code-block:: c++

        #include <cuda_fp16.h>
        extern "C" __global__
        void NeuronBPTTKernel_float_hard_reset_nodetach_reset(
        const int & N, const float * grad_spike_seq, float * grad_v_init, const float * grad_v_seq, float * grad_x_seq, const float * h_seq, const int & numel, float & v_reset, float & v_th
        )
        
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        
                float grad_h = 0.0f;

                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            
                  const float over_th = h_seq[t] - v_th;
                  const float spike_seq_t = over_th >= 0.0f ? 1.0f: 0.0f;
                  const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (4.0f) * over_th));
                  const float grad_s_to_h = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (4.0f);
                  float grad_v_to_h = (1.0f) - spike_seq_t;
                  {
                   float temp_var = v_reset - h_seq[t];
                   temp_var = temp_var * grad_s_to_h;
                   grad_v_to_h = temp_var + grad_v_to_h;
                  }
                  // grad_h_next_to_v should be defined here!;
                  grad_h = grad_h * grad_h_next_to_v;
                  grad_h = grad_v_seq[t] + grad_h;
                  grad_h = grad_h * grad_v_to_h;
                  {
                   float temp_var = grad_spike_seq[t] * grad_s_to_h;
                   grad_h = grad_h + temp_var;
                  }
                  // grad_h_to_x should be defined here!;
                  grad_x_seq[t] = grad_h * grad_h_to_x;

                }
        
                // grad_h_next_to_v should be defined here!;
                grad_v_init[index] = grad_h * grad_h_next_to_v;

            }
        }
        
        

上述代码中注释的位置，即提示我们需要补充的位置。它们在 ``NeuronBPTTKernel`` 中有对应的函数：

.. code-block:: python

    class NeuronBPTTKernel(base.CKernel2D):
        # ...
        def grad_h_next_to_v(self) -> str:
            """
            :return: CUDA code
            :rtype: str

            Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t+1]}{\\mathrm{d} V[t]}`.

            This function should define how ``grad_h_next_to_v`` is calculated. Note that ``grad_h_next_to_v`` has not been
            declared. Thus, this function should also declare ``grad_h_next_to_v``.

            For example, the IF neuron define this function as:

            .. code-block:: python

                def grad_h_next_to_v(self) -> str:
                    return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)
            """
            return '// grad_h_next_to_v should be defined here!'


        def grad_h_to_x(self) -> str:
            """
            :return: CUDA code
            :rtype: str

            Returns CUDA code for calculating :math:`\\frac{\\mathrm{d} H[t]}{\\mathrm{d} X[t]}`.

            This function should define how ``grad_h_to_x`` is calculated. Note that ``grad_h_to_x`` has not been
            declared. Thus, this function should also declare ``grad_h_to_x``.

            For example, the IF neuron define this function as:

            .. code-block:: python

                def grad_h_to_x(self) -> str:
                    return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
            """
            return '// grad_h_to_x should be defined here!'



对于IF神经元，:math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}=1, \frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}=1`。\
因此，可以很容易实现IF神经元的BPTT内核：


.. code-block:: python

    class IFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
        def grad_h_next_to_v(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

        def grad_h_to_x(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)

接下来，就可以打印出完整的IF神经元BPTT的CUDA内核：

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
        def grad_h_next_to_v(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

        def grad_h_to_x(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)

    kernel = IFNodeBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, hard_reset=True, detach_reset=False, dtype='float')
    print(kernel.full_codes)

.. code-block:: c++

        #include <cuda_fp16.h>
        extern "C" __global__
        void IFNodeBPTTKernel_float_hard_reset_nodetach_reset(
        const int & N, const float * grad_spike_seq, float * grad_v_init, const float * grad_v_seq, float * grad_x_seq, const float * h_seq, const int & numel, float & v_reset, float & v_th
        )
        
        {
            const int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index < N)
            {
                const int dt = N;
        
                float grad_h = 0.0f;

                for(int t = numel - N + index; t >= 0; t -= dt)
                {
            
                  const float over_th = h_seq[t] - v_th;
                  const float spike_seq_t = over_th >= 0.0f ? 1.0f: 0.0f;
                  const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (4.0f) * over_th));
                  const float grad_s_to_h = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (4.0f);
                  float grad_v_to_h = (1.0f) - spike_seq_t;
                  {
                   float temp_var = v_reset - h_seq[t];
                   temp_var = temp_var * grad_s_to_h;
                   grad_v_to_h = temp_var + grad_v_to_h;
                  }
                  const float grad_h_next_to_v = 1.0f;
                  grad_h = grad_h * grad_h_next_to_v;
                  grad_h = grad_v_seq[t] + grad_h;
                  grad_h = grad_h * grad_v_to_h;
                  {
                   float temp_var = grad_spike_seq[t] * grad_s_to_h;
                   grad_h = grad_h + temp_var;
                  }
                  const float grad_h_to_x = 1.0f;
                  grad_x_seq[t] = grad_h * grad_h_to_x;

                }
        
                const float grad_h_next_to_v = 1.0f;
                grad_v_init[index] = grad_h * grad_h_next_to_v;

            }
        }
        

Python包装
----------------------------------------------------------
接下来，使用 :class:`torch.autograd.Function` 对FPTT和BPTT进行包装。

:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase` 提供了一些通用的函数用来包装。我们将在实现IF神经元的\
autograd Function时进行使用。建议首先阅读 :class:`NeuronATGFBase <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase>` 的API文档，\
我们在下文中会默认读者已经了解其各个函数的使用。

首先需要确定输入。在SpikingJelly中，CUDA内核会被作为前向传播的输入，是由神经元的类去生成，而不是autograd Function生成（在0.0.0.0.12及之前的老版本中是这样做的）。前向传播的定义如下：

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                    forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):

接下来根据输入，生成 ``py_dict``，并交给 :class:`NeuronATGFBase.pre_forward <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase.pre_forward>` 处理：


.. code-block:: python

        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

接下来就可以直接调用前向传播了：

.. code-block:: python

    forward_kernel((blocks,), (threads,), py_dict)

接下来，我们需要保存反向传播所需的参数：

.. code-block:: python

    NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)
 
最后返回 ``T`` 个time-steps的脉冲和电压。不要忘了 ``v_v_seq[1:]`` 才是要返回的 ``v_seq``，因此返回值为：

.. code-block:: python

    return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

完整的前向传播代码为：

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                    forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):
            py_dict = {
                'x_seq': x_seq,
                'v_init': v_init,
                'v_th': v_th,
                'v_reset': v_reset
            }
            requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

            forward_kernel((blocks,), (threads,), py_dict)

            NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                            numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward_kernel=backward_kernel)


            return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

接下来实现反向传播。反向传播函数的输入，是前向传播函数的输出tensor的梯度tensor，因此输入是：

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

借助 :class:`NeuronATGFBase.pre_backward <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase.pre_backward>`，进行预处理，\
得到执行反向传播内核的参数：

.. code-block:: python

    backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)

然后直接执行反向传播内核：

.. code-block:: python

    backward_kernel((blocks,), (threads,), py_dict)

最后返回梯度。前向传播有几个输入，反向传播就有几个返回值：

.. code-block:: python

    return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None

完整的代码为：

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                    forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):
            py_dict = {
                'x_seq': x_seq,
                'v_init': v_init,
                'v_th': v_th,
                'v_reset': v_reset
            }
            requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

            forward_kernel((blocks,), (threads,), py_dict)

            NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                            numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                            backward_kernel=backward_kernel)


            return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

        @staticmethod
        def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

            backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)
            backward_kernel((blocks,), (threads,), py_dict)

            return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None


实现CUPY后端
-------------------------------------
利用之前我们已经定义好的 ``IFNodeFPTTKernel, IFNodeBPTTKernel, IFNodeATGF``，我们实现一个简化的IF神经元，并添加CUPY后端。

完整的代码如下：


.. code-block:: python

    from spikingjelly.activation_based.auto_cuda.neuron_kernel import IFNodeFPTTKernel, IFNodeBPTTKernel, IFNodeATGF

    # put sources of ``IFNodeFPTTKernel, IFNodeBPTTKernel, IFNodeATGF`` before the following codes

    import torch
    from typing import Callable
    from spikingjelly.activation_based import base, surrogate

    class CUPYIFNode(base.MemoryModule):
        def __init__(self, v_threshold: float = 1., v_reset: float or None = 0.,
                    surrogate_function: Callable = surrogate.Sigmoid(), detach_reset: bool = False):
            super().__init__()
            self.v_threshold = v_threshold
            self.v_reset = v_reset
            self.surrogate_function = surrogate_function
            self.detach_reset = detach_reset
            self.step_mode = 'm'
            if v_reset is not None:
                self.register_memory('v', v_reset)
            else:
                self.register_memory('v', 0.)

        def multi_step_forward(self, x_seq: torch.Tensor):

            if isinstance(self.v, float):
                self.v = torch.zeros_like(x_seq[0])

            hard_reset = self.v_reset is not None
            if x_seq.dtype == torch.float:
                dtype = 'float'
            elif x_seq.dtype == torch.half:
                dtype = 'half2'


            forward_kernel = IFNodeFPTTKernel(hard_reset=hard_reset, dtype=dtype)
            backward_kernel = IFNodeBPTTKernel(surrogate_function=self.surrogate_function.cuda_codes, hard_reset=hard_reset, detach_reset=self.detach_reset, dtype=dtype)

            # All tensors wil be regard as 2D or 1D. Thus, we use flatten
            spike_seq, v_seq = IFNodeATGF.apply(x_seq.flatten(1), self.v.flatten(), self.v_threshold, self.v_reset, forward_kernel, backward_kernel)

            spike_seq = spike_seq.view(x_seq.shape)
            self.v = v_seq[-1].view(x_seq.shape[1:])

            return spike_seq

接下来，让我们与纯pytorch实现对比输出误差：


.. code-block:: python

    from spikingjelly.activation_based import neuron

    @torch.no_grad()
    def max_error(x: torch.Tensor, y: torch.Tensor):
        return (x - y).abs().max()

    T = 8
    N = 64
    C = 32 * 32 * 32
    device = 'cuda:0'
    x_seq = torch.rand([T, N, C], device=device, requires_grad=True)

    net_cupy = CUPYIFNode()
    y_cupy = net_cupy(x_seq)
    y_cupy.sum().backward()
    x_grad_cupy = x_seq.grad.clone()
    x_seq.grad.zero_()

    net_torch = neuron.IFNode(backend='torch', step_mode='m')
    y_torch = net_torch(x_seq)
    y_torch.sum().backward()
    x_grad_torch = x_seq.grad.clone()

    print('max error of y_seq', max_error(y_cupy, y_torch))
    print('max error of x_seq.grad', max_error(x_grad_cupy, x_grad_torch))

输出为：

.. code-block:: bash

    max error of y_seq tensor(0., device='cuda:0')
    max error of x_seq.grad tensor(1.3113e-06, device='cuda:0')

可以发现，基本没有误差，我们的实现是正确的。

接下来对比速度。实验在 ``NVIDIA Quadro RTX 6000`` 上进行：


.. code-block:: python
        
    from spikingjelly.activation_based import neuron, cuda_utils, functional

    def forward_backward(net: torch.nn.Module, x_seq: torch.Tensor):
        y_seq = net(x_seq)
        y_seq.sum().backward()
        x_seq.grad.zero_()
        functional.reset_net(net)


    N = 64
    C = 32 * 32 * 32
    device = 'cuda:0'

    net_cupy = CUPYIFNode()
    net_torch = neuron.IFNode(backend='torch', step_mode='m')

    repeats = 16

    for dtype in [torch.float, torch.half]:
        for T in [2, 4, 8, 16, 32]:
            x_seq = torch.rand([T, N, C], device=device, requires_grad=True, dtype=dtype)

            t_cupy = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_cupy, x_seq)
            t_torch = cuda_utils.cal_fun_t(repeats, device, forward_backward, net_torch, x_seq)

            print(f'dtype={dtype}, T={T},'.ljust(30), f't_torch / t_cupy = {round(t_torch / t_cupy, 2)}')

输出为：

.. code-block:: bash

    dtype=torch.float32, T=2,      t_torch / t_cupy = 0.59
    dtype=torch.float32, T=4,      t_torch / t_cupy = 1.47
    dtype=torch.float32, T=8,      t_torch / t_cupy = 2.67
    dtype=torch.float32, T=16,     t_torch / t_cupy = 4.17
    dtype=torch.float32, T=32,     t_torch / t_cupy = 6.93
    dtype=torch.float16, T=2,      t_torch / t_cupy = 0.68
    dtype=torch.float16, T=4,      t_torch / t_cupy = 1.31
    dtype=torch.float16, T=8,      t_torch / t_cupy = 2.2
    dtype=torch.float16, T=16,     t_torch / t_cupy = 4.77
    dtype=torch.float16, T=32,     t_torch / t_cupy = 6.7

可以发现，在是使用梯度替代法训练时常用的 ``T >= 4`` 时，手动编写的 ``CUPY`` 内核拥有较大的加速效果。
当 ``T`` 较小时，由于SpikingJelly中的pytorch函数大多使用jit进行了封装，因此速度比手写CUPY快是正常的。因为手写的CUPY逐元素内核，\
速度慢于jit优化后的pytorch逐元素操作。