Accelerate with CUDA-Enhanced Neuron and Layer-by-Layer Propagation
============================================================================================

Authors: `fangwei123456 <https://github.com/fangwei123456>`_

CUDA-Enhanced Neuron
-----------------------
:class:`spikingjelly.clock_driven.neuron` provides the multi-step version of neurons. Compared with the single-step neuron,
the multi-step neuron can use cupy backend. The cupy backend fuses operations in a single cuda kernel, which is much faster
than naive pytorch backend. Let us run a simple experiment to compare LIF neurons in both module:

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

The code is running at a Ubuntu server with `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` CPU and `GeForce RTX 2080 Ti` GPU. The outputs are:

.. code-block:: bash

    forward
    8 1.9180845527841939, 0.8166529733273364
    16 3.8143536958727964, 1.6002442711169351
    32 7.6071328955436, 3.2570467449772877
    64 15.181676714490777, 6.82808195671214
    128 30.344632044631226, 14.053565065751172
    forward and backward
    8 8.131792200288146, 1.6501817200662572
    16 21.89934094545265, 3.210343387223702
    32 66.34630815216269, 6.41730432241161
    64 226.20835550819152, 13.073845567419085
    128 827.6064751953811, 26.71502177403795

We plot the results in a bar chart:

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.*
    :width: 100%

It can be found that cupy backend is much faster than naive pytorch backend.

Accelerate Deep SNNs
-----------------------
Now let us use the CUDA-Enhanced Multi-Step neuron to re-implement the network in :doc:`../clock_driven_en/4_conv_fashion_mnist` and compare their speeds. There is no need to modify the training codes. We can only change the network's codes:

.. code-block:: python

    class CupyNet(nn.Module):
        def __init__(self, T):
            super().__init__()
            self.T = T

            self.static_conv = nn.Sequential(
                nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
                nn.BatchNorm2d(128),
            )

            self.conv = nn.Sequential(
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), backend='cupy'),
                layer.SeqToANNContainer(
                        nn.MaxPool2d(2, 2),  # 14 * 14
                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                ),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), backend='cupy'),
                layer.SeqToANNContainer(
                    nn.MaxPool2d(2, 2),  # 7 * 7
                    nn.Flatten(),
                ),
            )
            self.fc = nn.Sequential(
                layer.SeqToANNContainer(nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False)),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), backend='cupy'),
                layer.SeqToANNContainer(nn.Linear(128 * 4 * 4, 10, bias=False)),
                neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), backend='cupy'),
            )


        def forward(self, x):
            x_seq = self.static_conv(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            # [N, C, H, W] -> [1, N, C, H, W] -> [T, N, C, H, W]

            return self.fc(self.conv(x_seq)).mean(0)

The fully codes are available at :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist`. Run this example with the same arguments and devices as those in :doc:`../clock_driven_en/4_conv_fashion_mnist`. The outputs are:

.. code-block:: shell

    (pytorch-env) root@e8b6e4800dae4011eb0918702bd7ddedd51c-fangw1598-0:/# python -m spikingjelly.clock_driven.examples.conv_fashion_mnist -opt SGD -data_dir /userhome/datasets/FashionMNIST/ -amp -cupy

    Namespace(T=4, T_max=64, amp=True, b=128, cupy=True, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    CupyNet(
      (static_conv): Sequential(
        (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
      )
      (conv): Sequential(
        (0): MultiStepIFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (1): SeqToANNContainer(
          (module): Sequential(
            (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (1): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (2): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
          )
        )
        (2): MultiStepIFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (3): SeqToANNContainer(
          (module): Sequential(
            (0): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (1): Flatten(start_dim=1, end_dim=-1)
          )
        )
      )
      (fc): Sequential(
        (0): SeqToANNContainer(
          (module): Linear(in_features=6272, out_features=2048, bias=False)
        )
        (1): MultiStepIFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
        (2): SeqToANNContainer(
          (module): Linear(in_features=2048, out_features=10, bias=False)
        )
        (3): MultiStepIFNode(
          v_threshold=1.0, v_reset=0.0, detach_reset=False
          (surrogate_function): ATan(alpha=2.0, spiking=True)
        )
      )
    )
    Mkdir ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp_cupy.
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=True, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp_cupy
    epoch=0, train_loss=0.028574782584865507, train_acc=0.8175080128205128, test_loss=0.020883125430345536, test_acc=0.8725, max_test_acc=0.8725, total_time=13.037598133087158
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=True, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp_cupy

    ...

    epoch=62, train_loss=0.001055751721853287, train_acc=0.9977463942307693, test_loss=0.010815625159442425, test_acc=0.934, max_test_acc=0.9346, total_time=11.059867858886719
    Namespace(T=4, T_max=64, amp=True, b=128, cupy=True, data_dir='/userhome/datasets/FashionMNIST/', device='cuda:0', epochs=64, gamma=0.1, j=4, lr=0.1, lr_scheduler='CosALR', momentum=0.9, opt='SGD', out_dir='./logs', resume=None, step_size=32)
    ./logs/T_4_b_128_SGD_lr_0.1_CosALR_64_amp_cupy
    epoch=63, train_loss=0.0010632637413514631, train_acc=0.9980134882478633, test_loss=0.010720000202953816, test_acc=0.9324, max_test_acc=0.9346, total_time=11.128222703933716

We get 93.46% accuracy, which is very close to 93.3% in :doc:`../clock_driven/11_cext_neuron_with_lbl`. Here are training logs:

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.*
    :width: 100%

In fact, we set an identical seed in both examples, but get a different results, which maybe caused by the numerical errors between cupy and pytorch functions. It can be found that the training execution time with cupy backend is 69% of the naive PyTorch SNN.