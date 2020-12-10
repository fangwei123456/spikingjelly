前向传播的方式
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_


CUDA加速的神经元
-----------------------

``spikingjelly.cext.neuron`` 中的神经元与 ``spikingjelly.clock_driven.neuron`` 中的同名神经元，在前向传播和反向传播时的计算结果完全相同。但 ``spikingjelly.cext.neuron`` 将各种运算都封装到了一个CUDA内核；``spikingjelly.clock_driven.neuron`` 则是使用PyTorch来实现神经元，每一个Python函数都需要调用一次相应的CUDA后端。现在让我们通过一个简单的实验，来对比两个模块中LIF神经元的运行耗时：

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
    N = 2*20
    print('forward')
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_t(lif, x, 1024), cal_forward_t(lif_cuda, x, 1024), cal_forward_t(lif_cuda_tt, x, 1024))

    print('forward and backward')
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_backward_t(lif, x, 1024), cal_forward_backward_t(lif_cuda, x, 1024), cal_forward_backward_t(lif_cuda_tt, x, 1024))

实验机器使用 `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` 的CPU和 `GeForce RTX 2080 Ti` 的GPU。运行结果如下：

.. code-block:: bash

    forward
    8 1.2689701984527346 0.5531465321837459 0.06358328437272576
    16 2.5922743875526066 1.0690318631532136 0.06530838709295494
    32 4.906598455818312 2.0490410443017026 0.06877212354083895
    64 9.582090764070017 4.050089067732188 0.08626037742942572
    128 19.352127595993807 7.874332742630941 0.11617418294918025
    forward and backward
    8 4.799259775609244 1.4362369111040607 0.2897263620980084
    16 7.427763028317713 3.084241311171354 0.2840051633938856
    32 15.380504060431122 5.489842319093441 0.4225145885357051
    64 32.96750279241678 10.161389542645338 0.28885948904644465
    128 63.52909050156086 20.467097838263726 0.2954222113658034

将结果画出柱状图：

.. image:: ../_static/tutorials/clock_driven/10_forward_pattern/exe_time_f.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/10_forward_pattern/exe_time_fb.*
    :width: 100%