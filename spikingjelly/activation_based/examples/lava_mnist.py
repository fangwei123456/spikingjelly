import logging

logging.getLogger().setLevel(logging.INFO)
import torch
import torch.nn as nn
import torchvision
from torchvision import transforms
from spikingjelly.activation_based import functional, lava_exchange, surrogate, encoding, neuron
import torch.nn.functional as F
from torch.utils.data import DataLoader
import os
import argparse
import h5py


def export_hdf5(net, filename):
    # network export to hdf5 format
    h = h5py.File(filename, 'w')
    layer = h.create_group('layer')
    for i, b in enumerate(net):
        handle = layer.create_group(f'{i}')
        b.export_hdf5(handle)


class MNISTNet(nn.Module):
    def __init__(self, channels: int = 16):
        super().__init__()
        self.conv_fc = nn.Sequential(
            lava_exchange.BlockContainer(
                nn.Conv2d(1, channels, kernel_size=3, stride=1, padding=1, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),
            # 14 * 14

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=3, stride=1, padding=1, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            lava_exchange.BlockContainer(
                nn.Conv2d(channels, channels, kernel_size=2, stride=2, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            # 7 * 7

            lava_exchange.BlockContainer(
                nn.Flatten(),
                None
            ),
            lava_exchange.BlockContainer(
                nn.Linear(channels * 7 * 7, 128, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),

            lava_exchange.BlockContainer(
                nn.Linear(128, 10, bias=False),
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True)
            ),
        )

    def to_lava(self):
        ret = []

        for i in range(self.conv_fc.__len__()):
            m = self.conv_fc[i]
            if isinstance(m, lava_exchange.BlockContainer):
                ret.append(m.to_lava_block())

        return nn.Sequential(*ret)

    def forward(self, x):
        return self.conv_fc(x)


def main():
    # python -m spikingjelly.activation_based.examples.lava_mnist -T 32 -device cuda:0 -b 128 -epochs 16 -data-dir /datasets/MNIST/ -lr 0.1 -channels 16

    parser = argparse.ArgumentParser()
    parser.add_argument('-T', default=16, type=int, help='simulating time-steps')
    parser.add_argument('-b', default=128, type=int, help='batch size')
    parser.add_argument('-device', default='cuda:0', type=str, help='device')
    parser.add_argument('-data-dir', type=str, default='/datasets/MNIST/', help='root dir of the MNIST dataset')
    parser.add_argument('-channels', default=16, type=int, help='channels of CSNN')
    parser.add_argument('-epochs', default=16, type=int, help='training epochs')
    parser.add_argument('-lr', default=0.1, type=float, help='learning rate')
    parser.add_argument('-out-dir', default='./', type=str, help='path for saving weights')

    args = parser.parse_args()
    print(args)

    net = MNISTNet(channels=args.channels)
    net.to(args.device)
    functional.set_step_mode(net, step_mode='m')
    optimizer = torch.optim.SGD(net.parameters(), lr=args.lr, momentum=0.9)

    encoder = encoding.PoissonEncoder(step_mode='m')

    train_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=True,
        transform=transforms.ToTensor(),
        download=True)

    test_set = torchvision.datasets.MNIST(
        root=args.data_dir,
        train=False,
        transform=transforms.ToTensor(),
        download=True)

    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_set,
        batch_size=args.b,
        shuffle=True,
        drop_last=True,
        num_workers=4,
        pin_memory=True
    )

    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_set,
        batch_size=args.b,
        shuffle=False,
        drop_last=False,
        num_workers=4,
        pin_memory=True
    )
    max_test_acc = -1
    for epoch in range(args.epochs):
        net.train()
        train_loss = 0
        train_acc = 0
        train_samples = 0
        for img, label in train_data_loader:
            optimizer.zero_grad()
            img = img.to(args.device)
            label = label.to(args.device)
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

            fr = net(encoder(img)).mean(0)
            loss = F.cross_entropy(fr, label)
            loss.backward()
            optimizer.step()

            train_samples += label.shape[0]
            train_loss += loss.item() * label.shape[0]
            train_acc += (fr.argmax(1) == label).float().sum().item()

            functional.reset_net(net)
        train_loss /= train_samples
        train_acc /= train_samples

        net.eval()
        test_loss = 0
        test_acc = 0
        test_samples = 0
        with torch.no_grad():
            for img, label in test_data_loader:
                img = img.to(args.device)
                label = label.to(args.device)
                img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)

                fr = net(encoder(img)).mean(0)
                loss = F.cross_entropy(fr, label)

                test_samples += label.numel()
                test_loss += loss.item() * label.numel()
                test_acc += (fr.argmax(1) == label).float().sum().item()
                functional.reset_net(net)
        test_loss /= test_samples
        test_acc /= test_samples
        max_test_acc = max(max_test_acc, test_acc)
        print(args)
        print(
            f'epoch = {epoch}, train_loss ={train_loss: .4f}, train_acc ={train_acc: .4f}, test_loss ={test_loss: .4f}, test_acc ={test_acc: .4f}, max_test_acc ={max_test_acc: .4f}')

    print('finish training')
    print('test acc[sj] =', test_acc)

    net_ladl = net.to_lava().to(args.device)
    net_ladl.eval()
    test_loss = 0
    test_acc = 0
    test_samples = 0
    with torch.no_grad():
        for img, label in test_data_loader:
            img = img.to(args.device)
            label = label.to(args.device)
            img = img.unsqueeze(0).repeat(args.T, 1, 1, 1, 1)
            img = encoder(img)
            img = lava_exchange.TNX_to_NXT(img)
            fr = net_ladl(img).mean(-1)
            loss = F.cross_entropy(fr, label)

            test_samples += label.numel()
            test_loss += loss.item() * label.numel()
            test_acc += (fr.argmax(1) == label).float().sum().item()

    test_loss /= test_samples
    test_acc /= test_samples

    print('test acc[lava dl] =', test_acc)

    torch.save(net.state_dict(), os.path.join(args.out_dir, 'net.pt'))
    print(f"save net.state_dict() to {os.path.join(args.out_dir, 'net.pt')}")
    torch.save(net_ladl.state_dict(), os.path.join(args.out_dir, 'net_ladl.pt'))
    print(f"save net_ladl.state_dict() to {os.path.join(args.out_dir, 'net_ladl.pt')}")
    export_hdf5(net_ladl, os.path.join(args.out_dir, 'net_la.net'))
    print(f"export net_ladl to {os.path.join(args.out_dir, 'net_la.net')}")


if __name__ == '__main__':
    main()





