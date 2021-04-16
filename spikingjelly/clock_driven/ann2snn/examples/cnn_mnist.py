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
        # 网络结构：三层卷积块串联一个全连接层，每个卷积块由一个卷积层、一个批正则化、一个ReLU激活和一个平均池化层组成
        # Network structure: Three convolution blocks connected with a full-connection layer, each convolution
        # block consists of a convolution layer, a batch normalization, a ReLU activation and an average pool
        # layer.
        self.network = nn.Sequential(
            nn.Conv2d(1, 32, 3, 1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Conv2d(32, 32, 3, 1),
            nn.BatchNorm2d(32, eps=1e-3),
            nn.ReLU(),
            nn.AvgPool2d(2, 2),

            nn.Flatten(),
            nn.Linear(32, 10),
            nn.ReLU()
        )

    def forward(self,x):
        x = self.network(x)
        return x


def main(log_dir=None):
    '''
        :return: None

        使用Conv-ReLU-[Conv-ReLU]-全连接-ReLU的网络结构训练并转换为SNN，进行MNIST识别。运行示例：

        .. code-block:: python

            >>> import spikingjelly.clock_driven.ann2snn.examples.cnn_mnist as cnn_mnist
            >>> cnn_mnist.main()
            输入运行的设备，例如“cpu”或“cuda:0”
             input device, e.g., "cpu" or "cuda:0": cuda:15
            输入保存MNIST数据集的位置，例如“./”
             input root directory for saving MNIST dataset, e.g., "./": ./mnist
            输入batch_size，例如“64”
             input batch_size, e.g., "64": 128
            输入学习率，例如“1e-3”
             input learning rate, e.g., "1e-3": 1e-3
            输入仿真时长，例如“100”
             input simulating steps, e.g., "100": 100
            输入训练轮数，即遍历训练集的次数，例如“10”
             input training epochs, e.g., "10": 10
            输入模型名字，用于自动生成日志文档，例如“cnn_mnist”
             input model name, for log_dir generating , e.g., "cnn_mnist"

            Epoch 0 [1/937] ANN Training Loss:2.252 Accuracy:0.078
            Epoch 0 [101/937] ANN Training Loss:1.423 Accuracy:0.669
            Epoch 0 [201/937] ANN Training Loss:1.117 Accuracy:0.773
            Epoch 0 [301/937] ANN Training Loss:0.953 Accuracy:0.795
            Epoch 0 [401/937] ANN Training Loss:0.865 Accuracy:0.788
            Epoch 0 [501/937] ANN Training Loss:0.807 Accuracy:0.792
            Epoch 0 [601/937] ANN Training Loss:0.764 Accuracy:0.795
            Epoch 0 [701/937] ANN Training Loss:0.726 Accuracy:0.835
            Epoch 0 [801/937] ANN Training Loss:0.681 Accuracy:0.880
            Epoch 0 [901/937] ANN Training Loss:0.641 Accuracy:0.889
            100%|██████████| 100/100 [00:00<00:00, 116.12it/s]
            Epoch 0 [100/100] ANN Validating Loss:0.327 Accuracy:0.881
            Save model to: cnn_mnist-XXXXX\cnn_mnist.pkl
            ......
            --------------------simulator summary--------------------
            time elapsed: 46.55072790000008 (sec)
            ---------------------------------------------------------
    '''
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    train_device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    parser_device = input('输入分析模型的设备，例如“cpu”或“cuda:0”\n input parsing device, e.g., "cpu" or "cuda:0": ')
    simulator_device = parser_device
    # simulator_device = input(
    #     '输入SNN仿真的设备（支持多线程），例如“cpu,cuda:0”或“cuda:0,cuda:1”\n input SNN simulating device (support multithread), e.g., "cpu,cuda:0" or "cuda:0,cuda:1": ').split(
    #     ',')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“10”\n input training epochs, e.g., "10": '))
    model_name = input('输入模型名字，例如“cnn_mnist”\n input model name, for log_dir generating , e.g., "cnn_mnist": ')

    load = False
    if log_dir == None:
        from datetime import datetime
        current_time = datetime.now().strftime('%b%d_%H-%M-%S')
        log_dir = model_name+'-'+current_time
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir,model_name+'.pkl')):
            print('%s has no model to load.'%(log_dir))
            load = False
        else:
            load = True

    if not load:
        writer = SummaryWriter(log_dir)

    # 初始化数据加载器
    # initialize data loader
    train_data_dataset = torchvision.datasets.MNIST(
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
        dataset=torchvision.datasets.MNIST(
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
    percentage = 0.004 # load 0.004 of the data
    norm_data_list = []
    for idx, (imgs, targets) in enumerate(train_data_loader):
        norm_data_list.append(imgs)
        if idx == int(len(train_data_loader) * percentage) - 1:
            break
    norm_data = torch.cat(norm_data_list)
    print('use %d imgs to parse' % (norm_data.size(0)))

    # 调用parser，使用kernel为onnx
    # Call parser, use onnx kernel
    onnxparser = parser(name=model_name,
                        log_dir=log_dir + '/parser',
                        kernel='onnx')
    snn = onnxparser.parse(ann, norm_data.to(parser_device))

    # 保存转换好的SNN模型
    # Save SNN model
    torch.save(snn, os.path.join(log_dir,'snn-'+model_name+'.pkl'))
    fig = plt.figure('simulator')

    # 定义用于分类的SNN仿真器
    # define simulator for classification task
    sim = classify_simulator(snn,
                             log_dir=log_dir + '/simulator',
                             device=simulator_device,
                             canvas=fig
                             )
    # 仿真SNN
    # Simulate SNN
    sim.simulate(test_data_loader,
                T=T,
                online_drawer=True,
                ann_acc=ann_acc,
                fig_name=model_name,
                step_max=True
                )

if __name__ == '__main__':
    main('./cnn_mnist')
