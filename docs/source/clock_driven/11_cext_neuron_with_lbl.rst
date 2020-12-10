使用CUDA增强的神经元与逐层传播进行加速
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

CUDA加速的神经元
-----------------------
``spikingjelly.cext.neuron`` 中的神经元与 ``spikingjelly.clock_driven.neuron`` 中的同名神经元，在前向传播和反向传播时的计算结果完全相同。但 ``spikingjelly.cext.neuron`` 将各种运算都封装到了一个CUDA内核；``spikingjelly.clock_driven.neuron`` 则是使用PyTorch来实现神经元，每一个Python函数都需要调用一次相应的CUDA后端，这种频繁的调用存在很大的开销。现在让我们通过一个简单的实验，来对比两个模块中LIF神经元的运行耗时：

.. code-block:: python

    from spikingjelly import cext
    from spikingjelly.cext import neuron as cext_neuron
    from spikingjelly.clock_driven import neuron, surrogate, layer
    import torch


    def cal_forward_t(multi_step_neuron, x, repeat_times):
        with torch.no_grad():
            used_t = cext.cal_fun_t(repeat_times, x.device, multi_step_neuron, x)
            multi_step_neuron.reset()
            return used_t * 1000


    def forward_backward(multi_step_neuron, x):
        multi_step_neuron(x).sum().backward()
        multi_step_neuron.reset()
        x.grad.zero_()


    def cal_forward_backward_t(multi_step_neuron, x, repeat_times):
        x.requires_grad_(True)
        used_t = cext.cal_fun_t(repeat_times, x.device, forward_backward, multi_step_neuron, x)
        return used_t * 1000


    device = 'cuda:0'
    lif = layer.MultiStepContainer(neuron.LIFNode(surrogate_function=surrogate.ATan(alpha=2.0)))
    lif_cuda = layer.MultiStepContainer(cext_neuron.LIFNode(surrogate_function='ATan', alpha=2.0))
    lif_cuda_tt = cext_neuron.MultiStepLIFNode(surrogate_function='ATan', alpha=2.0)
    lif.to(device)
    lif_cuda.to(device)
    lif_cuda_tt.to(device)
    N = 2 * 20
    print('forward')
    lif.eval()
    lif_cuda.eval()
    lif_cuda_tt.eval()
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_t(lif, x, 1024), cal_forward_t(lif_cuda, x, 1024), cal_forward_t(lif_cuda_tt, x, 1024))

    print('forward and backward')
    lif.train()
    lif_cuda.train()
    lif_cuda_tt.train()
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_backward_t(lif, x, 1024), cal_forward_backward_t(lif_cuda, x, 1024),
              cal_forward_backward_t(lif_cuda_tt, x, 1024))

实验机器使用 `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` 的CPU和 `GeForce RTX 2080` 的GPU。运行结果如下：

.. code-block:: bash

    forward
    8 1.6701502286196046 0.44249044822208816 0.05478178627527086
    16 3.237690732930787 0.8110604935609445 0.0550979889339942
    32 6.348427949433244 1.538134750262543 0.055016983878886094
    64 12.587608936428296 2.986507736295607 0.05504425234903465
    128 25.135914108886936 5.8784374023161945 0.06540481217598426
    forward and backward
    8 4.832853653624625 1.8915147916231945 0.33051693708330276
    16 9.511920674867724 3.5159952340109157 0.32920849980655476
    32 18.870338058150082 6.747562522832595 0.32978799936245196
    64 38.79206964529658 13.195514010476472 0.36697773248306476
    128 75.05335126097634 27.016243242997007 0.3330824065415072

将结果画成柱状图：

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.*
    :width: 100%

可以发现，使用CUDA封装操作的 ``spikingjelly.cext.neuron`` 速度明显快于原生PyTorch的神经元实现。