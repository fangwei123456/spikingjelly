import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
from torch.utils.tensorboard import SummaryWriter

import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from spikingjelly.clock_driven.ann2snn.examples.model_lib.cifar10 import resnet

def main(log_dir=None):
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    z_norm_mean = (0.4914, 0.4822, 0.4465)
    z_norm_std = (0.2023, 0.1994, 0.2010)

    # device = 'cuda:0'
    # dataset_dir = 'G:/Dataset/cifar10'
    # batch_size = 128
    # learning_rate = 1e-3
    # T = 200
    # train_epoch = 20
    # model_name = 'cifar10resnet18'

    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存CIFAR10数据集的位置，例如“./”\n input root directory for saving CIFAR10 dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“128”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“200”\n input simulating steps, e.g., "200": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“10”\n input training epochs, e.g., "10": '))
    model_name = input('输入模型名字，例如“cifar10resnet18”\n input model name, for log_dir generating , e.g., "cifar10resnet18": ')

    load = False
    if log_dir == None:
        log_dir = './log-' + model_name + str(time.time())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    print("All the temp files are saved to ", log_dir)

    ann_transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(z_norm_mean, z_norm_std),
    ])

    ann_transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(z_norm_mean, z_norm_std),
    ])

    snn_transform = transforms.Compose([
        transforms.ToTensor()
    ])

    ann_train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=ann_transform_train,
        download=True)
    snn_train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=snn_transform,
        download=True)
    ann_train_data_loader = torch.utils.data.DataLoader(
        ann_train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    ann_test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=ann_transform_test,
        download=True)
    snn_test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=snn_transform,
        download=True)
    ann_test_data_loader = torch.utils.data.DataLoader(
        dataset=ann_test_data_dataset,
        batch_size=100,
        shuffle=False,
        drop_last=False)
    snn_test_data_loader = torch.utils.data.DataLoader(
        dataset=snn_test_data_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False)

    config = utils.Config.default_config
    print('ann2snn config:\n\t', config)
    utils.Config.store_config(os.path.join(log_dir, 'default_config.json'), config)

    loss_function = nn.CrossEntropyLoss()

    ann = resnet.ResNet18().to(device)
    checkpoint_state_dict = torch.load('./model_lib/cifar10/checkpoint/ResNet18-state-dict.pth')
    ann.load_state_dict(checkpoint_state_dict)

    writer = SummaryWriter(log_dir)

    print('Directly load model', model_name + '.pth')


    # 加载用于归一化模型的数据
    # Load the data to normalize the model
    norm_set_len = int(snn_train_data_dataset.data.shape[0] / 500)
    print('Using %d pictures as norm set'%(norm_set_len))
    norm_set = snn_train_data_dataset.data[:norm_set_len, :, :].astype(np.float32) / 255
    norm_tensor = torch.from_numpy(norm_set.transpose(0,3,1,2).astype(np.float32))

    utils.onnx_ann2snn(model_name=model_name,
                       ann=ann,
                       norm_tensor=norm_tensor,
                       loss_function=loss_function,
                       test_data_loader=snn_test_data_loader,
                       device=device,
                       T=T,
                       log_dir=log_dir,
                       config=config,
                       z_score=(z_norm_mean,z_norm_std)
                       )

if __name__ == "__main__":
    main(log_dir="log-cifar10resnet181597814743.3173714")
