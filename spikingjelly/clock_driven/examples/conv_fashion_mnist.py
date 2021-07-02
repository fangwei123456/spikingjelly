import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
from spikingjelly.clock_driven import neuron, functional, surrogate, layer
from torch.utils.tensorboard import SummaryWriter
import os
import time
import argparse
import numpy as np
from torch.cuda import amp
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)

class PythonNet(nn.Module):
    def __init__(self, T):
        super().__init__()
        self.T = T

        self.static_conv = nn.Sequential(
            nn.Conv2d(1, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
        )

        self.conv = nn.Sequential(
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2),  # 14 * 14

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2)  # 7 * 7

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 7 * 7, 128 * 4 * 4, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
            nn.Linear(128 * 4 * 4, 10, bias=False),
            neuron.IFNode(surrogate_function=surrogate.ATan()),
        )


    def forward(self, x):
        x = self.static_conv(x)

        out_spikes_counter = self.fc(self.conv(x))
        for t in range(1, self.T):
            out_spikes_counter += self.fc(self.conv(x))

        return out_spikes_counter / self.T

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

def main():
    '''
    * :ref:`API in English <conv_fashion_mnist.main-en>`

    .. _conv_fashion_mnist.main-cn:

    Classify Fashion-MNIST

    optional arguments:
      -h, --help            show this help message and exit
      -T T                  simulating time-steps
      -device DEVICE        device
      -b B                  batch size
      -epochs N             number of total epochs to run
      -j N                  number of data loading workers (default: 4)
      -data_dir DATA_DIR    root dir of Fashion-MNIST dataset
      -out_dir OUT_DIR      root dir for saving logs and checkpoint
      -resume RESUME        resume from the checkpoint path
      -amp                  automatic mixed precision training
      -cupy                 use cupy neuron and multi-step forward mode
      -opt OPT              use which optimizer. SDG or Adam
      -lr LR                learning rate
      -momentum MOMENTUM    momentum for SGD
      -lr_scheduler LR_SCHEDULER
                            use which schedule. StepLR or CosALR
      -step_size STEP_SIZE  step_size for StepLR
      -gamma GAMMA          gamma for StepLR
      -T_max T_MAX          T_max for CosineAnnealingLR


    使用卷积-全连接的网络结构，进行Fashion MNIST识别。这个函数会初始化网络进行训练，并显示训练过程中在测试集的正确率。会将训练过
    程中测试集正确率最高的网络保存在 ``tensorboard`` 日志文件的同级目录下。这个目录的位置，是在运行 ``main()``
    函数时由用户输入的。

    训练100个epoch，训练batch和测试集上的正确率如下：

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%

    * :ref:`中文API <conv_fashion_mnist.main-cn>`

    .. _conv_fashion_mnist.main-en:

    The network with Conv-FC structure for classifying Fashion MNIST. This function initials the network, starts training
    and shows accuracy on test dataset. The net with the max accuracy on test dataset will be saved in
    the root directory for saving ``tensorboard`` logs, which is inputted by user when running the ``main()``  function.

    After 100 epochs, the accuracy on train batch and test dataset is as followed:

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/train.*
        :width: 100%

    .. image:: ./_static/tutorials/clock_driven/4_conv_fashion_mnist/test.*
        :width: 100%
    '''
    parser = argparse.ArgumentParser(description='Classify Fashion-MNIST')
    parser.add_argument('-T', default=4, type=int, help='simulating time-steps')
    parser.add_argument('-device', default='cuda:0', help='device')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-epochs', default=64, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 4)')
    parser.add_argument('-data_dir', type=str, help='root dir of Fashion-MNIST dataset')
    parser.add_argument('-out_dir', type=str, default='./logs', help='root dir for saving logs and checkpoint')

    parser.add_argument('-resume', type=str, help='resume from the checkpoint path')
    parser.add_argument('-amp', action='store_true', help='automatic mixed precision training')
    parser.add_argument('-cupy', action='store_true', help='use cupy neuron and multi-step forward mode')

    parser.add_argument('-opt', type=str, help='use which optimizer. SDG or Adam')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-momentum', default=0.9, type=float, help='momentum for SGD')
    parser.add_argument('-lr_scheduler', default='CosALR', type=str, help='use which schedule. StepLR or CosALR')
    parser.add_argument('-step_size', default=32, type=float, help='step_size for StepLR')
    parser.add_argument('-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('-T_max', default=64, type=int, help='T_max for CosineAnnealingLR')
    # python w1.py -opt SGD -data_dir /userhome/datasets/FashionMNIST/ -amp
    # python w1.py -opt SGD -data_dir /userhome/datasets/FashionMNIST/ -amp -cupy
    args = parser.parse_args()
    print(args)

    if args.cupy:
        net = CupyNet(T=args.T)
    else:
        net = PythonNet(T=args.T)
    print(net)
    net.to(args.device)

    optimizer = None
    if args.opt == 'SGD':
        optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=args.momentum)
    elif args.opt == 'Adam':
        optimizer = torch.optim.Adam(net.parameters(), lr=args.lr)
    else:
        raise NotImplementedError(args.opt)

    lr_scheduler = None
    if args.lr_scheduler == 'StepLR':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    elif args.lr_scheduler == 'CosALR':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.T_max)
    else:
        raise NotImplementedError(args.lr_scheduler)


    train_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=True,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    test_set = torchvision.datasets.FashionMNIST(
            root=args.data_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=args.j
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=False,
        num_workers=args.j
    )

    scaler = None
    if args.amp:
        scaler = amp.GradScaler()

    start_epoch = 0
    max_test_acc = 0

    if args.resume:
        checkpoint = torch.load(args.resume, map_location='cpu')
        net.load_state_dict(checkpoint['net'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_scheduler.load_state_dict(checkpoint['lr_scheduler'])
        start_epoch = checkpoint['epoch'] + 1
        max_test_acc = checkpoint['max_test_acc']

    out_dir = os.path.join(args.out_dir, f'T_{args.T}_b_{args.b}_{args.opt}_lr_{args.lr}_')
    if args.lr_scheduler == 'CosALR':
        out_dir += f'CosALR_{args.T_max}'
    elif args.lr_scheduler == 'StepLR':
        out_dir += f'StepLR_{args.step_size}_{args.gamma}'
    else:
        raise NotImplementedError(args.lr_scheduler)

    if args.amp:
        out_dir += '_amp'
    if args.cupy:
        out_dir += '_cupy'

    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
        print(f'Mkdir {out_dir}.')

    with open(os.path.join(out_dir, 'args.txt'), 'w', encoding='utf-8') as args_txt:
        args_txt.write(str(args))

    writer = SummaryWriter(os.path.join(out_dir, 'fmnist_logs'), purge_step=start_epoch)

    for epoch in range(start_epoch, args.epochs):
        start_time = time.time()
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for frame, label in train_data_loader:
            optimizer.zero_grad()
            frame = frame.float().to(args.device)
            label = label.to(args.device)
            label_onehot = F.one_hot(label, 10).float()
            if args.amp:
                with amp.autocast():
                    out_fr = net(frame)
                    loss = F.mse_loss(out_fr, label_onehot)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                out_fr = net(frame)
                loss = F.mse_loss(out_fr, label_onehot)
                loss.backward()
                optimizer.step()

            train_samples += label.numel()
            train_loss += loss.item() * label.numel()
            train_acc += (out_fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        train_loss /= train_samples
        train_acc /= train_samples

        writer.add_scalar('train_loss', train_loss, epoch)
        writer.add_scalar('train_acc', train_acc, epoch)
        lr_scheduler.step()

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for frame, label in test_data_loader:
                frame = frame.float().to(args.device)
                label = label.to(args.device)
                label_onehot = F.one_hot(label, 10).float()
                out_fr = net(frame)
                loss = F.mse_loss(out_fr, label_onehot)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (out_fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)

        test_loss /= test_samples
        test_acc /= test_samples
        writer.add_scalar('test_loss', test_loss, epoch)
        writer.add_scalar('test_acc', test_acc, epoch)

        save_max = False
        if test_acc > max_test_acc:
            max_test_acc = test_acc
            save_max = True

        checkpoint = {
            'net': net.state_dict(),
            'optimizer': optimizer.state_dict(),
            'lr_scheduler': lr_scheduler.state_dict(),
            'epoch': epoch,
            'max_test_acc': max_test_acc
        }

        if save_max:
            torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_max.pth'))

        torch.save(checkpoint, os.path.join(out_dir, 'checkpoint_latest.pth'))

        print(args)
        print(out_dir)
        print(
            f'epoch={epoch}, train_loss={train_loss}, train_acc={train_acc}, test_loss={test_loss}, test_acc={test_acc}, max_test_acc={max_test_acc}, total_time={time.time() - start_time}')


if __name__ == '__main__':
    main()