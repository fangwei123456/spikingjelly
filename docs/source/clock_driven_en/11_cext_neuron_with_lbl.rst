Accelerate with CUDA-Enhanced Neuron and Layer-by-Layer Propagation
==================================================================

Authors: `fangwei123456 <https://github.com/fangwei123456>`_

CUDA-Enhanced Neuron
-----------------------
The neuron sharing the same name in ``spikingjelly.cext.neuron`` and  ``spikingjelly.clock_driven.neuron`` do the same calculation when forward and backward. However, ``spikingjelly.cext.neuron`` fuses operations in one CUDA kernel, while ``spikingjelly.clock_driven.neuron`` uses PyTorch functions to perform operations and each PyTorch function needs to call a CUDA function, which causes grate overhead for calling CUDA kernels. Let us run a simple code to compare LIF neurons in both module:

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

The code is running at a Ubuntu server with `Intel(R) Xeon(R) Gold 6148 CPU @ 2.40GHz` CPU and `GeForce RTX 2080 Ti` GPU. The outputs are:

.. code-block:: bash

    forward
    8 1.2027182438032469 0.31767665768711595 0.03739786916412413
    16 2.3161539784268825 0.5838696361024631 0.038602679524046835
    32 4.504981370700989 1.1035655998057337 0.03964870666095521
    64 8.957112172538473 2.1011443768657045 0.04390397953102365
    128 17.928721825228422 4.223306857966236 0.057996856412501074
    forward and backward
    8 4.733593518722046 1.3954817559351795 0.2895549105232931
    16 7.594537261866208 3.1417532363775535 0.5625875119221746
    32 15.621844995621359 5.426582512882305 0.5158364547241945
    64 32.04859962170303 11.180313862496405 0.28680579453066457
    128 60.52553526205884 20.54842408415425 0.2843772117557819

We plot the results in a bar chart:

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_f.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/exe_time_fb.*
    :width: 100%

It can be found that neurons in ``spikingjelly.cext.neuron`` are faster than naive PyTorch neuron.

Accelerate Deep SNNs
-----------------------
Now let us use the CUDA-Enhanced Multi-Step neuron to re-implement the network in :doc:`../clock_driven_en/4_conv_fashion_mnist` and compare their speeds. There is no need to modify the training codes. We can only change the network's codes:

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
                    nn.Sequential(
                        nn.MaxPool2d(2, 2),  # 14 * 14
                        nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
                        nn.BatchNorm2d(128),
                    )
                ),
                cext_neuron.MultiStepIFNode(v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
            )
            self.fc = nn.Sequential(
                layer.SeqToANNContainer(
                    nn.Sequential(
                        nn.MaxPool2d(2, 2),  # 7 * 7
                        nn.Flatten(),
                    )
                ),
                layer.MultiStepDropout(0.5),
                layer.SeqToANNContainer(nn.Linear(128 * 7 * 7, 128 * 3 * 3, bias=False)),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
                layer.MultiStepDropout(0.5),
                nn.Linear(128 * 3 * 3, 128, bias=False),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0),
                layer.SeqToANNContainer(nn.Linear(128, 10, bias=False)),
                cext_neuron.MultiStepLIFNode(tau=tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function='ATan', alpha=2.0)
            )


        def forward(self, x):
            x_seq = self.static_conv(x).unsqueeze(0).repeat(self.T, 1, 1, 1, 1)
            # [N, C, H, W] -> [1, N, C, H, W] -> [T, N, C, H, W]

            out_spikes_counter = self.fc(self.conv(x_seq)).sum(0)
            return out_spikes_counter / self.T

The fully codes are available at :class:`spikingjelly.clock_driven.examples.conv_fashion_mnist_cuda_lbl`. Run this example with the same arguments and devices as those in :doc:`../clock_driven_en/4_conv_fashion_mnist`. The outputs are:

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

The highest accuracy on test dataset is 94.37%, which is very close to 94.4% in :doc:`../clock_driven/11_cext_neuron_with_lbl`. The accuracy curves on training batch and test dataset during training are as followed:

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/train.*
    :width: 100%

.. image:: ../_static/tutorials/clock_driven/11_cext_neuron_with_lbl/test.*
    :width: 100%

In fact, we set an identical seed in both examples, but get a different results, which maybe caused by the numerical errors between CUDA and PyTorch functions. The logs also record the execution time of training and testing. It can be found that the training execution time of the SNN with CUDA-Enhanced neurons and Layer-by-Layer propagation is 64% of the naive PyTorch SNN, and the testing execution time is 58% of the naive PyTorch SNN.