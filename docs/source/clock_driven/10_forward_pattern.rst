前向传播的方式
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

.. code-block:: python

    from spikingjelly import cext
    from spikingjelly.cext import neuron as cext_neuron
    from spikingjelly.clock_driven import neuron, surrogate, layer
    import torch

    def cal_forward_t(multi_step_neuron, x, repeat_times):
        with torch.no_grad():
            used_t = cext.cal_fun_t(repeat_times, x.device, multi_step_neuron, x)
            multi_step_neuron.reset()
            return used_t

    def forward_backward(multi_step_neuron, x):
        multi_step_neuron(x).sum().backward()
        multi_step_neuron.reset()
        x.grad.zero_()

    def cal_forward_backward_t(multi_step_neuron, x, repeat_times):
        x.requires_grad_(True)
        used_t = cext.cal_fun_t(repeat_times, x.device, forward_backward, multi_step_neuron, x)
        return used_t

    device = 'cuda:0'
    lif = layer.MultiStepContainer(neuron.LIFNode(surrogate_function=surrogate.ATan(alpha=2.0)))
    lif_cuda = layer.MultiStepContainer(cext_neuron.LIFNode(surrogate_function='ATan', alpha=2.0))
    lif_cuda_tt = cext_neuron.MultiStepLIFNode(surrogate_function='ATan', alpha=2.0)
    lif.to(device)
    lif_cuda.to(device)
    lif_cuda_tt.to(device)
    N = 128 * 28 * 28
    print('forward')
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_t(lif, x, 1024), cal_forward_t(lif_cuda, x, 1024), cal_forward_t(lif_cuda_tt, x, 1024))

    print('forward and backward')
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        print(T, cal_forward_backward_t(lif, x, 1024), cal_forward_backward_t(lif_cuda, x, 1024), cal_forward_backward_t(lif_cuda_tt, x, 1024))


GeForce RTX 2080

.. code-block:: bash

    forward
    8 0.0017787316378417017 0.0007725339578428247 0.00015642789321645978
    16 0.0034476295772947196 0.0014541315672431665 0.0002429631640552543
    32 0.009004615353660483 0.003705305618041166 0.0004458090174921381
    64 0.013518321067294892 0.0056778287248562265 0.0008845650272633065
    128 0.03509577211070791 0.011248737328969582 0.0017893795111376676
    forward and backward
    8 0.00520956037007636 0.0019439102429714694 0.00034730490142464987
    16 0.009879665125481552 0.004046029102482862 0.00048325695388484746
    32 0.02146396137550255 0.008629620229385182 0.0008620421272098611
    64 0.04052260834896515 0.02441813969926443 0.0016582453868068114
    128 0.10904121043722625 0.08174578305579416 0.003309445638478792
