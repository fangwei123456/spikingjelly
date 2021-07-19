import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
from spikingjelly.clock_driven.ann2snn.examples.model_sample.cifar10 import resnet
import matplotlib.pyplot as plt

def main(log_dir=None):
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input training device, e.g., "cpu" or "cuda:0": ')
    parser_device = input('输入分析模型的设备，例如“cpu”或“cuda:0”\n input parsing device, e.g., "cpu" or "cuda:0": ')
    simulator_device = parser_device
    # simulator_device = input('输入SNN仿真的设备（支持多线程），例如“cpu,cuda:0”或“cuda:0,cuda:1”\n input SNN simulating device (support multithread), e.g., "cpu,cuda:0" or "cuda:0,cuda:1": ').split(',')
    dataset_dir = input('输入保存cifar10数据集的位置，例如“./”\n input root directory for saving cifar10 dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“128”\n input batch_size, e.g., "128": '))
    T = int(input('输入仿真时长，例如“400”\n input simulating steps, e.g., "400": '))
    model_name = input('输入模型名字，例如“resnet18_cifar10”\n input model name, for log_dir generating , e.g., "resnet18_cifar10": ')

    z_norm_mean = (0.4914, 0.4822, 0.4465)
    z_norm_std = (0.2023, 0.1994, 0.2010)

    load = False
    if log_dir == None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = model_name + '-' + current_time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)

    if not load:
        writer = SummaryWriter(log_dir)

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    train_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=True,
        transform=transform,
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        dataset=train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=False)
    test_data_dataset = torchvision.datasets.CIFAR10(
        root=dataset_dir,
        train=False,
        transform=transform,
        download=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=test_data_dataset,
        batch_size=50,
        shuffle=True,
        drop_last=False)

    ann = resnet.ResNet18().to(train_device)
    loss_function = nn.CrossEntropyLoss()
    checkpoint_state_dict = torch.load('./SJ-cifar10-resnet18_model-sample.pth')
    ann.load_state_dict(checkpoint_state_dict)

    # 加载用于归一化模型的数据
    # Load the data to normalize the model
    percentage = 0.004  # load 0.004 of the data
    norm_data_list = []
    for idx, (imgs, targets) in enumerate(train_data_loader):
        norm_data_list.append(imgs)
        if idx == int(len(train_data_loader) * percentage) - 1:
            break
    norm_data = torch.cat(norm_data_list)
    print('use %d imgs to parse' % (norm_data.size(0)))

    onnxparser = parser(name=model_name,
                        log_dir=log_dir + '/parser',
                        kernel='onnx',
                        z_norm=(z_norm_mean, z_norm_std))

    snn = onnxparser.parse(ann, norm_data.to(parser_device))
    ann_acc = utils.val_ann(torch.load(onnxparser.ann_filename).to(train_device),train_device,test_data_loader,loss_function)
    torch.save(snn, os.path.join(log_dir, 'snn-' + model_name + '.pkl'))
    fig = plt.figure('simulator')
    sim = classify_simulator(snn,
                             log_dir=log_dir + '/simulator',
                             device=simulator_device,
                             canvas=fig
                             )
    sim.simulate(test_data_loader,
                 T=T,
                 online_drawer=True,
                 ann_acc=ann_acc,
                 fig_name=model_name,
                 step_max=True
                 )

if __name__ == '__main__':
    utils.download_sample_pth("https://ndownloader.figshare.com/files/26676110",'./SJ-cifar10-resnet18_model-sample.pth')
    main('./resnet18_cifar10')