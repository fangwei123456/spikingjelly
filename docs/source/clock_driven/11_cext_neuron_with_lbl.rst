使用CUDA增强的神经元与逐层传播进行加速
======================================

本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

CUDA加速的神经元
-----------------------
在 :class:`spikingjelly.clock_driven.neuron` 提供了多步版本的神经元。与单步版本相比，多步神经元增加了cupy后端。cupy后端将各种运算都封装到
了一个CUDA内核，因此速度比默认pytorch后端更快。现在让我们通过一个简单的实验，来对比两个模块中LIF神经元的运行耗时：

.. code-block:: python

    from spikingjelly.clock_driven import neuron, surrogate, cu_kernel_opt
    import torch


    def cal_forward_t(multi_step_neuron, x, repeat_times):
        with torch.no_grad():
            used_t = cu_kernel_opt.cal_fun_t(repeat_times, x.device, multi_step_neuron, x)
            multi_step_neuron.reset()
            return used_t * 1000


    def forward_backward(multi_step_neuron, x):
        multi_step_neuron(x).sum().backward()
        multi_step_neuron.reset()
        x.grad.zero_()


    def cal_forward_backward_t(multi_step_neuron, x, repeat_times):
        x.requires_grad_(True)
        used_t = cu_kernel_opt.cal_fun_t(repeat_times, x.device, forward_backward, multi_step_neuron, x)
        return used_t * 1000


    device = 'cuda:0'
    repeat_times = 1024
    ms_lif = neuron.MultiStepLIFNode(surrogate_function=surrogate.ATan(alpha=2.0))


    ms_lif.to(device)
    N = 2 ** 20
    print('forward')
    ms_lif.eval()
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        ms_lif.backend = 'torch'
        print(T, cal_forward_t(ms_lif, x, repeat_times), end=', ')
        ms_lif.backend = 'cupy'
        print(cal_forward_t(ms_lif, x, repeat_times))

    print('forward and backward')
    ms_lif.train()
    for T in [8, 16, 32, 64, 128]:
        x = torch.rand(T, N, device=device)
        ms_lif.backend = 'torch'
        print(T, cal_forward_backward_t(ms_lif, x, repeat_times), end=', ')
        ms_lif.backend = 'cupy'
        print(cal_forward_backward_t(ms_lif, x, repeat_times))

实验机器使用 `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` 的CPU和 `GeForce RTX 2080 Ti` 的GPU。运行结果如下：

.. code-block:: bash

    forward
    8 1.495931436011233 0.5668559867899603 0.23965466994013696
    16 2.921243163427789 1.0935631946722424 0.42392046202621714
    32 5.7503134660237265 2.1567279295595654 0.800143975766332
    64 11.510705337741456 4.273202213653349 1.560730856454029
    128 22.884282833274483 8.508097553431071 3.1778080651747587
    forward and backward
    8 6.444244811291355 4.052604411526772 1.4819166492543445
    16 16.360167272978288 11.785220529191065 2.8625220465983148
    32 47.86415797116206 38.88952818761027 5.645714411912195
    64 157.53049964018828 139.59021832943108 11.367870506774125
    128 562.9168437742464 526.8922436650882 22.945806705592986

将结果画成柱状图：

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.*
    :width: 100%

可以发现，使用CUDA封装操作的 ``spikingjelly.cext.neuron`` 速度明显快于原生PyTorch的神经元实现。

加速深度脉冲神经网络
-----------------------
现在让我们用CUDA封装的多步LIF神经元，重新实现 :doc:`../clock_driven/4_conv_fashion_mnist` 中的网络，并进行速度对比。我们只需要更改一下网络结构，无需进行其他的改动：

.. code-block:: python

    class Net(nn.Module):
        def __init__(self, tau, T, v_threshold=1.0, v_reset=0.0):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                cext_neuron.MultiStepIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
                layer.SeqToANNContainer(
                        nn.MaxPool2d(2, 2),  # 14 * 14
                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                ),
                cext_neuron.MultiStepIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
            )
            self.fc = nn.Sequential(
                layer.SeqToANNContainer(
                        nn.MaxPool2d(2, 2),  # 7 * 7
                        nn.Flatten(),
                ),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False)),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(128 * 3 * 3, 128, bias=False)),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
                layer.SeqToANNContainer(nn.Linear(128, 10, bias=False)),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0)
            )


        def forward(self, x):
            x_seq = self.static_conv(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            # [N, C, H, W] -> [1, N, C, H, W] -> [T, N, C, H, W]

            out_spikes_counter = self.fc(self.conv(x_seq)).sum(0)
            return out_spikes_counter / self.T

完整的代码可见于 :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist_cuda_lbl`。我们按照与 :doc:`../clock_driven/4_conv_fashion_mnist` 中完全相同的输入参数和设备（`Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` 的CPU和 `GeForce RTX 2080 Ti` 的GPU）来运行，结果如下：

.. code-block:: bash

    saving net...
    saved
    epoch=0, t_train=26.745780434459448, t_test=1.4819979975000024, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.8705, train_times=468
    saving net...
    saved
    epoch=1, t_train=26.087690989486873, t_test=1.502928489819169, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.8913, train_times=936
    saving net...
    saved
    epoch=2, t_train=26.281963238492608, t_test=1.4901704853400588, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.8977, train_times=1404
    saving net...
    saved

    ...

    epoch=96, t_train=26.286096683703363, t_test=1.5033660298213363, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.9428, train_times=45396
    saving net...
    saved
    epoch=97, t_train=26.185854725539684, t_test=1.4934641849249601, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.943, train_times=45864
    saving net...
    saved
    epoch=98, t_train=26.256993867456913, t_test=1.5093903196975589, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.9437, train_times=46332
    epoch=99, t_train=26.200945735909045, t_test=1.4959839908406138, device=cuda:0, dataset_dir=./fmnist, batch_size=128, learning_rate=0.001, T=8, log_dir=./logs2, max_test_accuracy=0.9437, train_times=46800

最终的正确率是94.37%，与 :doc:`../clock_driven/11_cext_neuron_with_lbl` 中的94.4%相差无几，两者在训练过程中的训练batch正确率和测试集正确率曲线如下：

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.*
    :width: 100%

两个网络使用了完全相同的随机种子，最终的性能略有差异，可能是CUDA和PyTorch的计算数值误差导致的。在日志中记录了训练和测试所需要的时间，我们可以发现，训练耗时为原始网络的64%，推理耗时为原始网络的58%，速度有了明显提升。