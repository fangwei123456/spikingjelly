import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.datasets
from spikingjelly.clock_driven.model import train_classify
from spikingjelly.clock_driven import neuron, surrogate, layer
from spikingjelly.clock_driven.functional import seq_to_ann_forward
from torchvision import transforms
import os, argparse
_seed_ = 2020
import random
random.seed(2020)

torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.cuda.manual_seed_all(_seed_)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

try:
    import cupy
    backend = 'cupy'
except ImportError:
    backend = 'torch'

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.sn1 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend)
        self.fc2 = nn.Linear(32, 10)
        self.sn2 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x.squeeze_(1)  # [N, H, W]
        x = x.permute(2, 0, 1)  # [W, N, H]
        x = seq_to_ann_forward(x, self.fc1)
        x = self.sn1(x)
        x = seq_to_ann_forward(x, self.fc2)
        x = self.sn2(x)
        return x.mean(0)

class StatefulSynapseNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.sn1 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend)
        self.sy1 = layer.MultiStepContainer(layer.SynapseFilter(tau=2., learnable=True))
        self.fc2 = nn.Linear(32, 10)
        self.sn2 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x.squeeze_(1)  # [N, H, W]
        x = x.permute(2, 0, 1)  # [W, N, H]
        x = self.fc1(x)
        x = self.sn1(x)
        x = self.sy1(x)
        x = self.fc2(x)
        x = self.sn2(x)
        return x.mean(0)

class FeedBackNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(28, 32)
        self.sn1 = layer.MultiStepContainer(
            layer.LinearRecurrentContainer(
                neuron.IFNode(surrogate_function=surrogate.ATan(), detach_reset=True),
                32, 32
            )
        )
        self.fc2 = nn.Linear(32, 10)
        self.sn2 = neuron.MultiStepIFNode(surrogate_function=surrogate.ATan(), detach_reset=True, backend=backend)

    def forward(self, x: torch.Tensor):
        # x.shape = [N, C, H, W]
        x.squeeze_(1)  # [N, H, W]
        x = x.permute(2, 0, 1)  # [W, N, H]
        x = seq_to_ann_forward(x, self.fc1)
        x = self.sn1(x)
        x = seq_to_ann_forward(x, self.fc2)
        x = self.sn2(x)
        return x.mean(0)

def main(args):
    dir_prefix = f'{args.model}_b{args.batch_size}_e{args.epochs}_{args.opt}_lr{args.lr}_wd{args.weight_decay}'

    if args.opt == 'sgd':
        dir_prefix += f'_m{args.momentum}'

    if args.lrs == 'step':
        dir_prefix += f'_steplrs{args.step_size}_{args.step_gamma}'
    elif args.lrs == 'cosa':
        if args.cosa_tmax is None:
            args.cosa_tmax = args.epochs
        dir_prefix += f'_cosa{args.cosa_tmax}'
    elif args.lrs == None:
        pass
    else:
        raise NotImplementedError

    if args.amp:
        dir_prefix += '_amp'

    tb_dir = None
    if args.tb:
        tb_dir = os.path.join(args.output_dir, dir_prefix + '_tb')

    pt_dir = os.path.join(args.output_dir, dir_prefix + '_pt')

    if args.model == 'plain':
        model = Net()
    elif args.model == 'feedback':
        model = FeedBackNet()
    elif args.model == 'stateful-synapse':
        model = StatefulSynapseNet()
    model.to(args.device)

    if args.opt == 'sgd':
        optimizer = torch.optim.SGD(params=model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.opt == 'adam':
        optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    else:
        raise NotImplementedError(f'Please implement the codes with args.opt')

    if args.lrs == 'step':
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.step_gamma)
    elif args.lrs == 'cosa':
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.cosa_tmax)
    elif args.lrs == None:
        lr_scheduler = None
    else:
        raise NotImplementedError

    def mse_loss(y, label):
        return F.mse_loss(y, F.one_hot(label, 10).to(y))

    transform_train = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2860, 0.3530),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(0.2860, 0.3530),
    ])
    train_set = torchvision.datasets.FashionMNIST(root=args.data_path, train=True, transform=transform_train)
    test_set = torchvision.datasets.FashionMNIST(root=args.data_path, train=False, transform=transform_test)

    train_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        shuffle=True, drop_last=True, num_workers=args.workers, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, num_workers=args.workers, pin_memory=True)


    train_classify.train_eval_loop(args=args, device=args.device, model=model, criterion=mse_loss, optimizer=optimizer, lr_scheduler=lr_scheduler, train_data_loader=train_data_loader, test_data_loader=test_data_loader, max_epoch=args.epochs, use_amp=args.amp, tb_log_dir=tb_dir, pt_dir=pt_dir, resume_pt=args.resume)

def parse_args():
    '''
    python -m spikingjelly.clock_driven.examples.rsnn_sequential_fmnist --data-path /raid/wfang/datasets/FashionMNIST --tb --device cuda:7 --amp --model plain
    '''
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')
    parser.add_argument('--data-path', default='/userhome/datasets/FashionMNIST', help='dataset')
    parser.add_argument('--device', default='cuda:0', help='device')
    parser.add_argument('-b', '--batch-size', default=128, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                        help='number of data loading workers (default: 16)')

    parser.add_argument('--lr', default=0.1, type=float, help='initial learning rate')
    parser.add_argument('--opt', default='sgd', type=str, help='optimizer (sgd or adam)')
    parser.add_argument('--lrs', default='cosa', help='lr schedule (cosa(CosineAnnealingLR), step(StepLR)) or None')
    parser.add_argument('--step-size', default=30, type=int, help='step_size for StepLR')
    parser.add_argument('--step-gamma', default=0.1, type=float, help='gamma for StepLR')
    parser.add_argument('--cosa-tmax', default=None, type=int, help='T_max for CosineAnnealingLR. If none, it will be set to epochs')

    parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                        help='Momentum for SGD')
    parser.add_argument('--wd', '--weight-decay', default=0, type=float,
                        metavar='W', help='weight decay (default: 0)',
                        dest='weight_decay')
    parser.add_argument('--output-dir', default='./logs', help='path where to save')
    parser.add_argument('--resume', default='', help='resume from checkpoint')
    parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                        help='start epoch')
    parser.add_argument(
        "--cache-dataset",
        dest="cache_dataset",
        help="Cache the datasets for quicker initialization. It also serializes the transforms",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')


    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--model', type=str, help='"plain", "feedback", or "stateful-synapse"')

    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()
    main(args)