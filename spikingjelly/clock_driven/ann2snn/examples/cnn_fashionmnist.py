import torch
import torch.nn as nn
import torchvision
import os
from torch.utils.tensorboard import SummaryWriter
import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from spikingjelly.clock_driven.ann2snn import parser, classify_simulator
import matplotlib.pyplot as plt

class ANN(nn.Module):
    def __init__(self):
        super().__init__()
        # 网络结构：类似AlexNet的结构
        # Network structure: AlexNet-like structure
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(0.2),
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.Dropout2d(0.2),
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
            nn.Linear(256 * 3 * 3, 1024),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 10)
        )

    def forward(self,x):
        x = self.network(x)
        return x

def main(log_dir=None):
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    parser_device = input('输入分析模型的设备，例如“cpu”或“cuda:0”\n input parsing device, e.g., "cpu" or "cuda:0": ')
    simulator_device = parser_device
    # simulator_device = input(
    #     '输入SNN仿真的设备（支持多线程），例如“cpu,cuda:0”或“cuda:0,cuda:1”\n input SNN simulating device (support multithread), e.g., "cpu,cuda:0" or "cuda:0,cuda:1": ').split(
    #     ',')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving FashionMNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“128”\n input batch_size, e.g., "128": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“400”\n input simulating steps, e.g., "400": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“100”\n input training epochs, e.g., "100": '))
    model_name = input('输入模型名字，例如“cnn_fashionmnist”\n input model name, for log_dir generating , e.g., "cnn_fashionmnist": ')

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
        if not os.path.exists(os.path.join(log_dir, model_name + '.pkl')):
            print('%s has no model to load.' % (log_dir))
            load = False
        else:
            load = True

    if not load:
        writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    # initialize data loader
    train_data_dataset = torchvision.datasets.FashionMNIST(
        root=dataset_dir,
        train=True,
        transform=torchvision.transforms.ToTensor(),
        download=True)
    train_data_loader = torch.utils.data.DataLoader(
        train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=torchvision.datasets.FashionMNIST(
            root=dataset_dir,
            train=False,
            transform=torchvision.transforms.ToTensor(),
            download=True),
        batch_size=100,
        shuffle=True,
        drop_last=False)

    ann = ANN().to(train_device)
    loss_function = nn.CrossEntropyLoss()
    if not load:
        optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate, weight_decay=5e-4)
        best_acc = 0.0
        for epoch in range(train_epoch):
            # 使用utils中预先写好的训练程序训练网络
            # 训练程序的写法和经典ANN中的训练也是一样的
            # Train the network using a pre-prepared code in ''utils''
            utils.train_ann(net=ann,
                            device=train_device,
                            data_loader=train_data_loader,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            epoch=epoch
                            )
            # 使用utils中预先写好的验证程序验证网络输出
            # Validate the network using a pre-prepared code in ''utils''
            acc = utils.val_ann(net=ann,
                                device=train_device,
                                data_loader=test_data_loader,
                                loss_function=loss_function,
                                epoch=epoch
                                )
            if best_acc <= acc:
                utils.save_model(ann, log_dir, model_name + '.pkl')
            writer.add_scalar('val_accuracy', acc, epoch)
    ann = torch.load(os.path.join(log_dir, model_name + '.pkl'))
    print('validating best model...')
    ann_acc = utils.val_ann(net=ann,
                            device=train_device,
                            data_loader=test_data_loader,
                            loss_function=loss_function
                            )

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
                        kernel='onnx')
    snn = onnxparser.parse(ann, norm_data.to(parser_device))

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
    main('./cnn_fashionmnist')
