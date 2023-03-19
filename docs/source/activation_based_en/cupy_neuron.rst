Implement CUPY Neuron
=======================================
Author: `fangwei123456 <https://github.com/fangwei123456>`_

This tutorial will introduce how to implement the cupy backend for spiking neurons. We suppose the reader:

#. Can implement simple element-wise CUDA kernels

#. Can implement custom backward with :class:`torch.autograd.Function`

#. Has read all APIs doc in :class:`spikingjelly.activation_based.auto_cuda.base`, and can implement 2D CUDA kernel by :class:`spikingjelly.activation_based.auto_cuda.base` 


Implement Forward Propagation Through Time 
----------------------------------------------------------
If we want to implement Forward Propagation Through Time (FPTT) by a python function, then the function should \
use the following input args: 

* ``v_init``: ``shape = [N]``, which is the initial membrane potential at current time-step 
  (the membrane potential after neuronal firing at the last time-step), where ``N`` is the number
  of neurons. When the neurons are multidimensional, ``N`` should be the number of neurons after
  flattening

* ``x_seq``: ``shape = [T, N]``, the input of ``T`` time-steps

* ``v_th``: ``float``, the threshold potential

If we use hard reset, we need an extra arg:

* ``v_reset``: ``float``, the reset potential


The output of the python FPTT function should include:

* ``spike_seq``: ``shape = [T, N]``, the output spikes at ``T`` time-steps

* ``v_seq``: ``shape = [T, N]``, the membrane potential after neuronal firing at ``T`` time-steps. 
  We output the membrane potential of all time-steps rather than only the last time-step, because we may use this data


If we implement the FPTT by CUDA, we will use some extra args, which will be introduced later.

:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronFPTTKernel` is inherited from :class:`spikingjelly.activation_based.auto_cuda.base.CKernel2D`. \
``NeuronFPTTKernel`` is the base class for FPTT. Let us print its CUDA kernel declaration:


.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronFPTTKernel(hard_reset=True, dtype='float')
    for key, value in base_kernel.cparams.items():
        print(f'key="{key}",'.ljust(20), f'value="{value}"'.ljust(20))


The outputs are:

.. code-block:: bash

    key="numel",         value="const int &" 
    key="N",             value="const int &" 
    key="x_seq",         value="const float *"
    key="v_v_seq",       value="float *"     
    key="h_seq",         value="float *"     
    key="spike_seq",     value="float *"     
    key="v_th",          value="float &"     
    key="v_reset",       value="float &" 

Most args have been introduced before. The new args are:

* ``numel``: the number of elements in input/output tensors, which is ``numel = T * N``

* ``N``: the number of neurons

* ``v_v_seq``: ``shape = [T + 1, N]``, which is concatenated from ``v_init`` and ``v_seq``

* ``h_seq``: ``shape = [T, N]``, the membrane potential after neuronal charging but before 
  neuronal firing, which will be used in backward


``NeuronFPTTKernel`` is the base class of neurons' FPTT CUDA kernels. Similar to :class:`spikingjelly.activation_based.neuron.BaseNode`, \
it has implemented the neuronal fire and neuronal reset functions. When we want to implement a neuron FPTT kernel, we only \
need to inherit it and implement the neuronal charge function.

Firstly, let us check the full codes of ``NeuronFPTTKernel``:

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronFPTTKernel(hard_reset=True, dtype='float')
    print(base_kernel.full_codes)
    

The outputs are:

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

We can find that this kernel is almost finished. We only need to add the neuronal charge function.

The ``neuronal_charge`` function in ``NeuronFPTTKernel`` is:

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

            For example, the IF neuron defines this function as:

            .. code-block:: python

                def neuronal_charge(self) -> str:
                    # note that v_v_seq[t] is v_seq[t - dt]
                    return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)
            """
            return '// neuronal_charge should be defined here!'


To implement the new neuron, we only need to define the ``neuronal_charge`` function.
Take the IF neuron as the example, whose neuronal charge function is:

.. math::
    
    H[t] = V[t - 1] + X[t]

And we can implement it as:

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):


        def neuronal_charge(self) -> str:
            # note that v_v_seq[t] is v_seq[t - dt]
            return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

    if_fptt_kernel = IFNodeFPTTKernel(hard_reset=True, dtype='float')
    print(if_fptt_kernel.full_codes)

The outputs are:

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

The above codes have implemented a complete CUDA kernel. We can find that it is easy to implement the kernel \
with ``NeuronFPTTKernel``.

Note that we use ``cfunction.add``:

.. code-block:: python


    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

We do not write codes like:

.. code-block:: python


    def neuronal_charge(self) -> str:
        # note that v_v_seq[t] is v_seq[t - dt]
        return 'h_seq[t] = x_seq[t] + v_v_seq[t];'

The reason is functions in :class:`spikingjelly.activation_based.auto_cuda.cfunction` provide both ``float`` \
and ``half2`` implementation. Thus, it is more convenient than we write CUDA code with different data types manually.


If we set ``dtype='half2'``, we will get the kernel of ``half2``:

.. code-block:: python

    from spikingjelly.activation_based.auto_cuda import neuron_kernel, cfunction

    class IFNodeFPTTKernel(neuron_kernel.NeuronFPTTKernel):


        def neuronal_charge(self) -> str:
            # note that v_v_seq[t] is v_seq[t - dt]
            return cfunction.add(z='h_seq[t]', x='x_seq[t]', y='v_v_seq[t]', dtype=self.dtype)

    if_fptt_kernel = IFNodeFPTTKernel(hard_reset=True, dtype='half2')
    print(if_fptt_kernel.full_codes)

The outputs are:

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

Implement Back Propagation Through Time
----------------------------------------------------------

It is harder to implement Back Propagation Through Time (BPTT) than FPTT. Firstly, let us \
review how the forward of the neuron is defined in SpikingJelly:


.. math::

    \begin{align}
        H[t] &= f(V[t - 1], X[t])\\
        S[t] &= \Theta(H[t] - V_{th})\\
        V[t] &= \begin{cases}
        H[t]\left( 1 - S[t] \right) + V_{reset}S[t], &\text{Hard Reset}\\
        H[t] - V_{th}S[t], &\text{Soft Reset}\\
    \end{cases}
    \end{align}

The FPTT has the formulation:

.. math::

    S[1,2,...,T], V[1,2,...,T] = F_{fp}(X[1,2,...,T], V[0])

Correspondingly, the BPTT should use the formulation as:

.. math::

    \frac{\mathrm{d} L}{\mathrm{d} X[1,2,...,T]},\frac{\mathrm{d} L}{\mathrm{d} V[0]} =
     F_{bp}(\frac{\partial L}{\partial S[1,2,...,T]},\frac{\partial L}{\partial V[1,2,...,T]})


Thus, the input args for the BPTT function are:

* ``grad_spike_seq``: ``shape = [T, N]``, the gradients of ``spike_seq``

* ``grad_v_seq``: ``shape = [T, N]``, the gradients of ``v_seq``

The outputs of BPTT function are:

* ``grad_x_seq``: ``shape = [T, N]``, the gradients of ``x_seq``

* ``grad_v_init``: ``shape = [N]``, the gradients of ``v_init``

According to the forward, we can calculate the backward as:

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

where :math:`D_{reset}` denotes whether we detach the neuronal reset:

.. math::

    D_{reset} = \begin{cases}
        1, &\text{Detach Reset}\\
        0, &\text{Not Detach Reset}\\
    \end{cases}

Finally, we get the backward formulation:

.. math::

    \begin{align}
    \frac{\mathrm{d} L}{\mathrm{d} H[t]} &=\frac{\partial L}{\partial S[t]}\frac{\mathrm{d} S[t]}{\mathrm{d} H[t]} + (\frac{\partial L}{\partial V[t]}+\frac{\mathrm{d} L}{\mathrm{d} H[t+1]}\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]})\frac{\mathrm{d} V[t]}{\mathrm{d} H[t]}\\
    \frac{\mathrm{d} L}{\mathrm{d} X[t]} &= \frac{\mathrm{d} L}{\mathrm{d} H[t]}\frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}\\
    \frac{\mathrm{d} L}{\mathrm{d} V[0]} &= \frac{\mathrm{d} L}{\mathrm{d} H[1]}\frac{\mathrm{d} H[1]}{\mathrm{d} V[0]}
    \end{align}

where :math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}, \frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}` are determined by the \
neuron's charge function :math:`H[t] = f(V[t - 1], X[t])`. :math:`\frac{\mathrm{d} S[t]}{\mathrm{d} H[t]}` is determined \
by the surrogate function. While other gradients compilation is general and can be used for all kinds of neurons.

:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronBPTTKernel` has implemented the general compilation. Let us \
check its declaration:

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, hard_reset=True, detach_reset=False, dtype='float')
    for key, value in base_kernel.cparams.items():
        print(f'key="{key}",'.ljust(22), f'value="{value}"'.ljust(20))

The outputs are:

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

We have introduced these args before.

Note that we use ``NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, ...`` because we need to define the surrogate function before applying backward.

Surrogate functions in SpikingJelly provide the ``cuda_codes`` function to create CUDA codes for backward. Let us check this function in \
:class:`spikingjelly.activation_based.surrogate.Sigmoid`: 

.. code-block:: python

    class Sigmoid(SurrogateFunctionBase):
        # ...
        def cuda_codes(self, y: str, x: str, dtype: str):
            return cfunction.sigmoid_backward(y=y, x=x, alpha=self.alpha, dtype=dtype)

Now let us print its codes:

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    print(surrogate.Sigmoid().cuda_codes(y='grad_s', x='over_th', dtype='float'))

The outputs are:

.. code-block:: c++

    const float sigmoid_backward__sigmoid_ax = 1.0f / (1.0f + expf(- (4.0f) * over_th));
    grad_s = (1.0f - sigmoid_backward__sigmoid_ax) * sigmoid_backward__sigmoid_ax * (4.0f);


To implement the custom surrogate function with support for CUDA kernel, we need to define the ``cuda_codes`` function by the \
following formulation:

.. code-block:: python

    class CustomSurrogateFunction:
        # ...
        def cuda_codes(self, y: str, x: str, dtype: str):
            # ...


Now let us check the full codes of ``NeuronBPTTKernel``:

.. code-block:: python

    from spikingjelly.activation_based import surrogate
    from spikingjelly.activation_based.auto_cuda import neuron_kernel

    base_kernel = neuron_kernel.NeuronBPTTKernel(surrogate_function=surrogate.Sigmoid().cuda_codes, hard_reset=True, detach_reset=False, dtype='float')
    print(base_kernel.full_codes)

The outputs are:


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

The comments in the above codes are what we should complete. These functions to be completed are defined in ``NeuronBPTTKernel``:

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

            For example, the IF neuron defines this function as:

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

            For example, the IF neuron defines this function as:

            .. code-block:: python

                def grad_h_to_x(self) -> str:
                    return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)
            """
            return '// grad_h_to_x should be defined here!'


For the IF neuron, :math:`\frac{\mathrm{d} H[t+1]}{\mathrm{d} V[t]}=1, \frac{\mathrm{d} H[t]}{\mathrm{d} X[t]}=1`. \
Thus, we can implement the BPTT kernel easily:


.. code-block:: python

    class IFNodeBPTTKernel(neuron_kernel.NeuronBPTTKernel):
        def grad_h_next_to_v(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_next_to_v', x=1., dtype=self.dtype)

        def grad_h_to_x(self) -> str:
            return cfunction.constant(y=f'const {self.dtype} grad_h_to_x', x=1., dtype=self.dtype)

Then we can print the full codes of the BPTT kernel of the IF neuron:

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
        

Python Wrap
----------------------------------------------------------
Now we need to use :class:`torch.autograd.Function` to wrap the FPTT and BPTT CUDA kernel.

:class:`spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase` provides some useful functions to help us wrap. We suppose that \
the user has read the APIs docs of :class:`NeuronATGFBase <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase>`.

Firstly, we should determine the input. In SpikingJelly, the CUDA kernels will be used as input args, rather than created by the autograd Function (we did this before version 0.0.0.0.12).\
The forward function is defined as:

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def forward(ctx, x_seq: torch.Tensor, v_init: torch.Tensor, v_th: float, v_reset: float or None,
                    forward_kernel: IFNodeFPTTKernel, backward_kernel: IFNodeBPTTKernel):

Then, we will create ``py_dict`` and use :class:`NeuronATGFBase.pre_forward <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase.pre_forward>` to preprocess it:

.. code-block:: python

        py_dict = {
            'x_seq': x_seq,
            'v_init': v_init,
            'v_th': v_th,
            'v_reset': v_reset
        }
        requires_grad, blocks, threads, py_dict = NeuronATGFBase.pre_forward(py_dict)

And we can call the forward CUDA kernel directly:

.. code-block:: python

    forward_kernel((blocks,), (threads,), py_dict)

Do not forget to save the params for backward:

.. code-block:: python

    NeuronATGFBase.ctx_save(ctx, requires_grad, py_dict['h_seq'], blocks=blocks, threads=threads,
                           numel=py_dict['numel'], N=py_dict['N'], v_th=py_dict['v_th'], v_reset=py_dict['v_reset'],
                           backward_kernel=backward_kernel)
 
Finally, we return the spikes and membrane potential of ``T`` time-steps. Note that we should return ``v_v_seq[1:]`` because \
``v_v_seq[0]`` is ``v_init``:

.. code-block:: python

    return py_dict['spike_seq'], py_dict['v_v_seq'][1:, ]

The full codes of the python forward autograd function are:

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


Now we need to implement the backward autograd function. Note that the input args for backward are the gradients of output args of \
forward. Thus, the input args are:

.. code-block:: python

    class IFNodeATGF(torch.autograd.Function):
        @staticmethod
        def backward(ctx, grad_spike_seq: torch.Tensor, grad_v_seq: torch.Tensor):

We use :class:`NeuronATGFBase.pre_backward <spikingjelly.activation_based.auto_cuda.neuron_kernel.NeuronATGFBase.pre_backward>` to preprocess args to \
get the args for the CUDA kernel:

.. code-block:: python

    backward_kernel, blocks, threads, py_dict = NeuronATGFBase.pre_backward(ctx, grad_spike_seq, grad_v_seq)


And then we can call the backward kernel:

.. code-block:: python

    backward_kernel((blocks,), (threads,), py_dict)

Finally, we return the gradients. Note that the number of return args is identical to the number of input args for forward:

.. code-block:: python

    return py_dict['grad_x_seq'], py_dict['grad_v_init'], None, None, None, None

The full codes are:

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


Implement the CUPY backend
-------------------------------------
We have implemented  ``IFNodeFPTTKernel, IFNodeBPTTKernel, IFNodeATGF``. Now we can use them to implement the simplified IF neuron with CUPY backend. 


Here are the codes:


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

Let us check the output error compared with the python neuron:

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

The outputs are:

.. code-block:: bash

    max error of y_seq tensor(0., device='cuda:0')
    max error of x_seq.grad tensor(1.3113e-06, device='cuda:0')

We can find that the error is almost zero, indicating that our implementation is correct.

Then let us evaluate the speed. The following experiment is running on ``NVIDIA Quadro RTX 6000``: 

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

The outputs are:

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

We can find that when using ``T >= 4``, our neuron with CUPY kernel is much faster than the python neuron.

When ``T`` is small, due to the jit acceleration used in SpikingJelly, the python neuron is faster. It is caused by that the \
jit is faster when the operation is simple. For example, we can hardly write an element-wise CUDA kernel that is faster than jit.
