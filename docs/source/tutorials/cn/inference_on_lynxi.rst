在灵汐芯片上推理
=======================================
本教程作者： `fangwei123456 <https://github.com/fangwei123456>`_

在GPU上训练float16模型
-------------------------------------------

我们使用的是 `灵汐科技 <https://www.lynxi.com/>`_ 的lynxi HP300芯片，完全支持float16，对于float32也可支持但会有一定的计算误差。从我们的使用经验来看，使用float32容易出现误差逐层累计的情况，\
因此最好使用float16。

将 :class:`spikingjelly.activation_based.examples.conv_fashion_mnist` 中的网络稍作更改，改为使用float16训练：

.. code-block:: python

    # ...
    net = CSNN(T=args.T, channels=args.channels, use_cupy=args.cupy).half()
    # ...
    for img, label in train_data_loader:
        optimizer.zero_grad()
        img = img.to(args.device).half()
        label = label.to(args.device)
        label_onehot = F.one_hot(label, 10).half()
        # ...
        train_acc += (out_fr.argmax(1) == label).half().sum().item()
        # ...
    # ...

    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device).half()
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).half()
            # ...
            test_acc += (out_fr.argmax(1) == label).half().sum().item()
            # ...

将修改后的文件保存为 `w1.py`，进行训练。需要注意，训练时不再使用AMP：

.. code-block:: shell

    python w1.py -T 4 -device cuda:0 -b 128 -epochs 64 -data-dir /datasets/FashionMNIST/ -cupy -opt sgd -lr 0.1 -j 8

训练完成后：

.. code-block:: shell

    Namespace(T=4, device='cuda:0', b=128, epochs=64, j=8, data_dir='/datasets/FashionMNIST/', out_dir='./logs', resume=None, amp=False, cupy=True, opt='sgd', momentum=0.9, lr=0.1, channels=128, save_es=None)
    ./logs/T4_b128_sgd_lr0.1_c128_cupy
    epoch = 63, train_loss = 0.0041, train_acc = 0.9836, test_loss = 0.0110, test_acc = 0.9312, max_test_acc = 0.9330
    train speed = 8056.0318 images/s, test speed = 11152.5812 images/s
    escape time = 2022-08-16 10:52:51

最高正确率为 `0.9330`，模型保存在 `./logs/T4_b128_sgd_lr0.1_c128_cupy` 中：

.. code-block:: shell

    cxhpc@lxnode01:~/fangwei/tempdir/fmnist_test/logs/T4_b128_sgd_lr0.1_c128_cupy$ ls
    args.txt  checkpoint_latest.pth  checkpoint_max.pth  events.out.tfevents.1660617801.mlg-ThinkStation-P920.3234566.0


模型编译
-------------------------------------------

并非所有SpikingJelly中的模块都支持灵汐的芯片。为了正确编译， :class:`spikingjelly.activation_based.lynxi_exchange` 提供了将SpikingJelly的部分网络层 \
转换到支持的网络层的函数。可以通过 :class:`spikingjelly.activation_based.lynxi_exchange.to_lynxi_supported_module` 将一个网络层或使用 :class:`spikingjelly.activation_based.lynxi_exchange.to_lynxi_supported_modules` \
将多个网络层进行转换。

需要注意的是，灵汐的芯片不支持5D的tensor，而 `shape = [T, N, C, H, W]` 经常出现在多步模式下的卷积层之间。对于使用多步模式的网络，使用 :class:`to_lynxi_supported_module <spikingjelly.activation_based.lynxi_exchange.to_lynxi_supported_module>` 或 \
:class:`to_lynxi_supported_modules <spikingjelly.activation_based.lynxi_exchange.to_lynxi_supported_modules>` 进行转换时，会将输入视作 `shape = [TN, *]`。

例如，查看转换神经元的源代码可以发现，在多步模式下，输入被当作 `shape = [TN, *]`，首先被reshape到 `shape = [T, N, *]` 然后才进行计算。由于灵汐不支持5D的tensor，神经元 \
内部直接reshape为3D的tensor：

.. code-block:: python

    # spikingjelly/activation_based/lynxi_exchange.py
    class BaseNode(nn.Module):
        # ...
        def forward(self, x: torch.Tensor, v: torch.Tensor = None):
            # ...
            elif self.step_mode == 'm':
                x = x.reshape(self.T, x.shape[0] // self.T, -1)
                # ...

接下来，我们将训练好的识别FashionMNIST的网络进行转换。原始网路的定义如下：

.. code-block:: python

    # spikingjelly/activation_based/examples/conv_fashion_mnist.py
    class CSNN(nn.Module):
        def __init__(self, T: int, channels: int, use_cupy=False):
            super().__init__()
            self.T = T

            self.conv_fc = nn.Sequential(
            layer.Conv2d(1, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 14 * 14

            layer.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False),
            layer.BatchNorm2d(channels),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            layer.MaxPool2d(2, 2),  # 7 * 7

            layer.Flatten(),
            layer.Linear(channels * 7 * 7, channels * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),

            layer.Linear(channels * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            )

            functional.set_step_mode(self, step_mode='m')

            if use_cupy:
                functional.set_backend(self, backend='cupy')

        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            x_seq = x.unsqueeze(0).repeat(self.T, 1, 1, 1, 1)  # [N, C, H, W] -> [T, N, C, H, W]
            x_seq = self.conv_fc(x_seq)
            fr = x_seq.mean(0)
            return fr
        

我们首先加载原始网络：

.. code-block:: python

    net_sj = conv_fashion_mnist.CSNN(T=args.T, channels=args.channels)
    net_sj.eval()
    ckp = torch.load(args.pt_path, map_location='cpu')
    print(f'max_test_acc={ckp["max_test_acc"]}')
    net_sj.load_state_dict(ckp['net'])

然后转换为支持的网络层：

.. code-block:: python

    module_list = lynxi_exchange.to_lynxi_supported_modules(net_sj.conv_fc, args.T)

需要注意的是，根据原始网络的定义，`net_sj.conv_fc` 的输出 `shape = [T, N, C]`；我们转换后，输出的 `shape = [TN, C]`。为了得到分类结果，我们需要求得发放率。

因此，新建一个网络：

.. code-block:: python

    class InferenceNet(nn.Module):
        def __init__(self, T: int, modules_list: list):
            super().__init__()
            self.T = T
            self.module_list = nn.Sequential(*modules_list)

        def forward(self, x: torch.Tensor):
            # x.shape = [N, C, H, W]
            x = x.repeat(self.T, 1, 1, 1)

            # [N, C, H, W] -> [T, N, C, H, W]
            x = self.module_list(x)

            # [TN, *] -> [T, N, *]
            x = x.reshape(self.T, x.shape[0] // self.T, -1)

            return x.mean(0)

    
        net = InferenceNet(args.T, module_list)
        net.eval()
        print(net)

输出为：

.. code-block:: shell

    InferenceNet(
        (module_list): Sequential(
            (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (2): IFNode()
            (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
            (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
            (6): IFNode()
            (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
            (8): Flatten(start_dim=1, end_dim=-1)
            (9): Linear(in_features=6272, out_features=2048, bias=False)
            (10): IFNode()
            (11): Linear(in_features=2048, out_features=10, bias=False)
            (12): IFNode()
        )
    )

接下来对模型进行编译：

.. code-block:: python

    output_path = lynxi_exchange.compile_lynxi_model(lynxi_model_path, net, in_data_type='float16',
                                                        out_data_type='float16',
                                                        input_shape_dict={'x': torch.Size([batch_size, 1, 28, 28])})

    


推理
-------------------------------------------

推理时，首先加载编译好的网络：

.. code-block:: python

    net_lynxi = lynxi_exchange.load_lynxi_model(device_id, output_path)

然后将pytorch的输入tensor转换为灵汐的tensor，送入网络；将输出的灵汐tensor转化为pytorch的tensor，便于计算正确率：

.. code-block:: python

    test_acc = 0
    test_samples = 0

    with torch.no_grad():
        for img, label in tqdm.tqdm(test_data_loader, disable=False):
            y = net_lynxi(lynxi_exchange.torch_tensor_to_lynxi(img, device_id))
            y = lynxi_exchange.lynxi_tensor_to_torch(y, shape=[label.shape[0], 10], dtype='float16')
            test_acc += (y.argmax(1) == label).half().sum().item()
            test_samples += img.shape[0]

    test_acc = test_acc / test_samples
    print(f'lynxi inference accuracy = {test_acc}')

最终正确率为：

.. code-block:: shell

    lynxi inference accuracy = 0.9316

完整代码和输入输出
-------------------------------------------

完整的代码位于 `spikingjelly/activation_based/examples/lynxi_fmnist_inference.py`，运行的命令行参数为：

.. code-block:: shell

    (fangwei) cxhpc@lxnode01:~/fangwei/spikingjelly$ python -m spikingjelly.activation_based.examples.lynxi_fmnist_inference -epochs
    
    lynxi_exchange.py[line:185]-CRITICAL: lyngor.version=1.1.0
    usage: test.py [-h] [-T T] [-j N] [-data-dir DATA_DIR] [-channels CHANNELS]
                [-b B] [-pt-path PT_PATH] [-out-model-path OUT_MODEL_PATH]
                [-lynxi-device LYNXI_DEVICE]

    Inference on Lynxi chips

    optional arguments:
    -h, --help            show this help message and exit
    -T T                  simulating time-steps
    -j N                  number of data loading workers (default: 4)
    -data-dir DATA_DIR    root dir of Fashion-MNIST dataset
    -channels CHANNELS    channels of CSNN
    -b B                  batch size
    -pt-path PT_PATH      checkpoint file path for conv_fashion_mnist.CSNN
    -out-model-path OUT_MODEL_PATH
                            path for saving the model compiled by lynxi
    -lynxi-device LYNXI_DEVICE
                            device id for lynxi


完整的输出日志为：

.. code-block:: shell

    CRITICAL:root:lyngor.version=1.1.0
    lynxi_exchange.py[line:185]-CRITICAL: lyngor.version=1.1.0
    Namespace(T=4, b=16, channels=128, data_dir=None, j=4, lynxi_device=0, out_model_path='/home/cxhpc/fangwei/tempdir/fmnist_test/lynxi_model', pt_path='/home/cxhpc/fangwei/tempdir/fmnist_test/logs/T4_b128_sgd_lr0.1_c128_cupy/checkpoint_max.pth')
    max_test_acc=0.933
    InferenceNet(
    (module_list): Sequential(
        (0): Conv2d(1, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (1): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (2): IFNode()
        (3): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (4): Conv2d(128, 128, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1), bias=False)
        (5): BatchNorm2d(128, eps=1e-05, momentum=0.1, affine=True, track_running_stats=True)
        (6): IFNode()
        (7): MaxPool2d(kernel_size=2, stride=2, padding=0, dilation=1, ceil_mode=False)
        (8): Flatten(start_dim=1, end_dim=-1)
        (9): Linear(in_features=6272, out_features=2048, bias=False)
        (10): IFNode()
        (11): Linear(in_features=2048, out_features=10, bias=False)
        (12): IFNode()
    )
    )
    util.py[line:268]-INFO: [optimize] Total time running optimize(2): 1.9529 seconds
    util.py[line:268]-INFO: [apu_build+optimize] Total time running apu_build(1): 4.8967 seconds
    Aborted (core dumped)
    builder.py[line:252]-ERROR: abc_map is error, error num is -2
    INFO: build abc map failed, try to build by auto mode
    util.py[line:268]-INFO: [optimize] Total time running optimize(1519): 1.2367 seconds
    util.py[line:268]-INFO: [apu_build+optimize] Total time running apu_build(1518): 4.1377 seconds
    lx_map compile option:
        git tag      : LX_APU_0626
        APU_LOG_LEVEL: 1
        isNewCmd     : true
        gen_golden   : false
        savePDF      : false
        sanityCheck  : false
        dynPattern   : false
        release      : true
        logFile      : "APU.log"
        batch        : 1
    MC conv info: 
        bHasMCConv   : true
        bFastModeConv: true

    test thread received convert primitive worker done message. 占用CPU时间 =     0.62s (累计用时     0.62s)
    ====================================
    test thread received resource assign worker done message.   占用CPU时间 =     0.71s (累计用时     1.33s)
    ====================================
    test thread received core slice worker done message.        占用CPU时间 =    65.14s (累计用时    66.47s)
    ====================================
    test thread received core map worker done message.          占用CPU时间 =   693.75s (累计用时   760.22s)
    ====================================
    test thread received core mem arrange worker done message.  占用CPU时间 =   129.37s (累计用时   889.59s)
    ====================================
    test thread received rc cfg worker done message.            占用CPU时间 =   176.15s (累计用时  1065.74s)
    ====================================
    test thread received route map worker done message.         占用CPU时间 =    17.04s (累计用时  1082.78s)
    ====================================
            支持多batch编译，最大支持数准确值请参考batchsize=2的信息说明,当前结果: 10
    test thread received print worker done message.             占用CPU时间 =    23.28s (累计用时  1106.06s)
    ====================================
    util.py[line:268]-INFO: [map] Total time running apu_map(3034): 1110.0334 seconds
    util.py[line:268]-INFO: [build+map] Total time running build(0): 1136.2683 seconds
    ['net_params.json', 'Net_0']
    100%|█████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 625/625 [02:36<00:00,  4.00it/s]
    lynxi inference accuracy = 0.9316