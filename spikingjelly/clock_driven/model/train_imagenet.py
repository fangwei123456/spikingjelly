import torch
import torch.nn as nn
import torch.nn.functional as F
import argparse
import hashlib
from . import train_classify
import os
import torch.utils.data
from torchvision import transforms
import time
import torchvision
# reference: https://github.com/pytorch/vision/blob/main/references/classification/train.py
def parse_args():
    parser = argparse.ArgumentParser(description='PyTorch Classification Training')

    parser.add_argument('--data-path', default='/home/wfang/datasets/ImageNet', help='dataset')
    parser.add_argument('--device', default='cuda', help='device')
    parser.add_argument('-b', '--batch-size', default=32, type=int)
    parser.add_argument('--epochs', default=90, type=int, metavar='N',
                        help='number of total epochs to run')
    parser.add_argument('-j', '--workers', default=16, type=int, metavar='N',
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
    parser.add_argument(
        "--sync-bn",
        dest="sync_bn",
        help="Use sync batch norm",
        action="store_true",
    )

    # Mixed precision training parameters
    parser.add_argument('--amp', action='store_true',
                        help='Use AMP training')

    # distributed training parameters
    parser.add_argument('--world-size', default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='env://', help='url used to set up distributed training')

    parser.add_argument('--tb', action='store_true',
                        help='Use TensorBoard to record logs')
    parser.add_argument('--T', default=None, type=int, help='simulation steps')
    parser.add_argument('--local_rank', default=None, type=int)
    args = parser.parse_args()
    return args

def _get_cache_path(filepath):
    h = hashlib.sha1(filepath.encode()).hexdigest()
    cache_path = os.path.join("~", ".torch", "vision", "datasets", "imagefolder", h[:10] + ".pt")
    cache_path = os.path.expanduser(cache_path)
    return cache_path

def load_data(traindir, valdir, cache_dataset, distributed):
    # Data loading code
    print("Loading data")
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                     std=[0.229, 0.224, 0.225])

    print("Loading training data")
    st = time.time()
    cache_path = _get_cache_path(traindir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading train_set from {}".format(cache_path))
        train_set, _ = torch.load(cache_path)
    else:
        train_set = torchvision.datasets.ImageFolder(
            traindir,
            transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving train_set to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((train_set, traindir), cache_path)
    print("Took", time.time() - st)

    print("Loading validation data")
    cache_path = _get_cache_path(valdir)
    if cache_dataset and os.path.exists(cache_path):
        # Attention, as the transforms are also cached!
        print("Loading val_set from {}".format(cache_path))
        val_set, _ = torch.load(cache_path)
    else:
        val_set = torchvision.datasets.ImageFolder(
            valdir,
            transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                normalize,
            ]))
        if cache_dataset:
            print("Saving val_set to {}".format(cache_path))
            utils.mkdir(os.path.dirname(cache_path))
            utils.save_on_master((val_set, valdir), cache_path)

    print("Creating data loaders")
    if distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_set)
        val_sampler = torch.utils.data.distributed.DistributedSampler(val_set)
    else:
        train_sampler = torch.utils.data.RandomSampler(train_set)
        val_sampler = torch.utils.data.SequentialSampler(val_set)

    return train_set, val_set, train_sampler, val_sampler

def main(model: nn.Module, criterion, args, cal_acc1_acc5):
    model = train_classify.distributed_training_init(args, model)

    dir_prefix = f'b{args.batch_size}_e{args.epochs}_{args.opt}_lr{args.lr}_wd{args.weight_decay}'

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


    if args.T is not None:
        dir_prefix += f'_T{args.T}'

    if args.world_size != 1:
        dir_prefix += f'_ws{args.world_size}'

    if args.sync_bn:
        dir_prefix += '_sbn'
    if args.amp:
        dir_prefix += '_amp'

    tb_dir = None
    if args.tb:
        tb_dir = os.path.join(args.output_dir, dir_prefix + '_tb')

    pt_dir = os.path.join(args.output_dir, dir_prefix + '_pt')

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

    train_dir = os.path.join(args.data_path, 'train')
    val_dir = os.path.join(args.data_path, 'val')

    train_set, val_set, train_sampler, val_sampler = load_data(train_dir, val_dir, args.cache_dataset, args.distributed)
    train_data_loader = torch.utils.data.DataLoader(
        train_set, batch_size=args.batch_size,
        sampler=train_sampler, num_workers=args.workers, pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size,
        sampler=val_sampler, num_workers=args.workers, pin_memory=True)


    train_classify.train_eval_loop(args=args, device=args.device, model=model, criterion=criterion, optimizer=optimizer, lr_scheduler=lr_scheduler, train_data_loader=train_data_loader, test_data_loader=test_data_loader, max_epoch=args.epochs, use_amp=args.amp, tb_log_dir=tb_dir, pt_dir=pt_dir, resume_pt=args.resume, cal_acc1_acc5=cal_acc1_acc5)


'''
from spikingjelly.clock_driven.model import train_imagenet, spiking_resnet
from spikingjelly.clock_driven import neuron, surrogate

if __name__ == '__main__':
    # python -m torch.distributed.launch --nproc_per_node=2 w1.py --data-path /gdata/ImageNet2012 -j 8 --opt sgd
    net = spiking_resnet.multi_step_spiking_resnet18(T=4, multi_step_neuron=neuron.MultiStepIFNode, surrogate_function=surrogate.ATan(), detach_reset=True, backend='cupy')
    args = train_imagenet.parse_args()
    train_imagenet.main(net, args)
'''











