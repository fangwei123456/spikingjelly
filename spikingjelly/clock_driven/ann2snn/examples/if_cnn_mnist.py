import torch
import torch.nn as nn
import torchvision
import os
import time
import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from torch.utils.tensorboard import SummaryWriter


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

            >>> import spikingjelly.clock_driven.ann2snn.examples.if_cnn_mnist as if_cnn_mnist
            >>> if_cnn_mnist.main()
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
            输入模型名字，用于自动生成日志文档，例如“mnist”
             input model name, for log_dir generating , e.g., "mnist"

            如果main函数的输入不是具有有效文件的文件夹，自动生成一个日志文件文件夹
            If the input of the main function is not a folder with valid files, an automatic log file folder is automatically generated.
            第一行输出为保存日志文件的位置，例如“./log-mnist1596804385.476601”
             Terminal outputs root directory for saving logs, e.g., "./": ./log-mnist1596804385.476601

            Epoch 0 [1/937] ANN Training Loss:2.252 Accuracy:0.078
            Epoch 0 [101/937] ANN Training Loss:1.424 Accuracy:0.669
            Epoch 0 [201/937] ANN Training Loss:1.117 Accuracy:0.773
            Epoch 0 [301/937] ANN Training Loss:0.953 Accuracy:0.795
            Epoch 0 [401/937] ANN Training Loss:0.865 Accuracy:0.788
            Epoch 0 [501/937] ANN Training Loss:0.807 Accuracy:0.792
            Epoch 0 [601/937] ANN Training Loss:0.764 Accuracy:0.795
            Epoch 0 [701/937] ANN Training Loss:0.726 Accuracy:0.834
            Epoch 0 [801/937] ANN Training Loss:0.681 Accuracy:0.880
            Epoch 0 [901/937] ANN Training Loss:0.641 Accuracy:0.888
            Epoch 0 [100/100] ANN Validating Loss:0.328 Accuracy:0.881
            Save model to: ./log-mnist1596804385.476601\mnist.pkl
            ...
            Epoch 9 [901/937] ANN Training Loss:0.036 Accuracy:0.990
            Epoch 9 [100/100] ANN Validating Loss:0.042 Accuracy:0.988
            Save model to: ./log-mnist1596804957.0179427\mnist.pkl
    '''
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    device = input('输入运行的设备，例如“cpu”或“cuda:0”\n input device, e.g., "cpu" or "cuda:0": ')
    dataset_dir = input('输入保存MNIST数据集的位置，例如“./”\n input root directory for saving MNIST dataset, e.g., "./": ')
    batch_size = int(input('输入batch_size，例如“64”\n input batch_size, e.g., "64": '))
    learning_rate = float(input('输入学习率，例如“1e-3”\n input learning rate, e.g., "1e-3": '))
    T = int(input('输入仿真时长，例如“100”\n input simulating steps, e.g., "100": '))
    train_epoch = int(input('输入训练轮数，即遍历训练集的次数，例如“10”\n input training epochs, e.g., "10": '))
    model_name = input('输入模型名字，例如“mnist”\n input model name, for log_dir generating , e.g., "mnist": ')

    load = False
    if log_dir == None:
        log_dir = './log-'+model_name+str(time.time())
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        print("All the temp files are saved to ", log_dir)
    else:
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        if not os.path.exists(os.path.join(log_dir,model_name+'.pkl')):
            print('Such log_dir has no model to load.')
            load = False
        else:
            load = True
        print("All the temp files are saved to ", log_dir)


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

    # 加载默认的配置并保存和输出
    # load default configuration, save and print
    config = utils.Config.default_config
    print('ann2snn config:\n\t', config)
    utils.Config.store_config(os.path.join(log_dir, 'default_config.json'), config)

    ann = ANN().to(device)
    loss_function = nn.CrossEntropyLoss()
    if not load:
        optimizer = torch.optim.Adam(ann.parameters(), lr=learning_rate, weight_decay=5e-4)
        best_acc = 0.0
        for epoch in range(train_epoch):
            # 使用utils中预先写好的训练程序训练网络
            # 训练程序的写法和经典ANN中的训练也是一样的
            # Train the network using a pre-prepared code in ''utils''
            utils.train_ann(net=ann,
                            device=device,
                            data_loader=train_data_loader,
                            optimizer=optimizer,
                            loss_function=loss_function,
                            epoch=epoch
                            )
            # 使用utils中预先写好的验证程序验证网络输出
            # Validate the network using a pre-prepared code in ''utils''
            acc = utils.val_ann(net=ann,
                                device=device,
                                data_loader=test_data_loader,
                                loss_function=loss_function,
                                epoch=epoch
                                )
            if best_acc <= acc:
                utils.save_model(ann, log_dir, model_name + '.pkl')
            writer.add_scalar('val_accuracy', acc, epoch)
    else:
        print('Directly load model',model_name+'.pkl')

    # 加载用于归一化模型的数据
    # Load the data to normalize the model
    norm_set_len = int(train_data_dataset.data.shape[0] / 500)
    print('Using %d pictures as norm set'%(norm_set_len))
    norm_set = train_data_dataset.data[:norm_set_len, :, :].float() / 255
    norm_tensor = torch.FloatTensor(norm_set).view(-1,1,28,28)

    # ANN2SNN标准转化，直接调用可以对模型进行转化并对SNN进行仿真测试
    # ANN2SNN standard conversion, direct calling of the function can transform the model and simulate the SNN.
    utils.pytorch_ann2snn(model_name=model_name,
                              norm_data=norm_tensor,
                              test_data_loader=test_data_loader,
                              device=device,
                              T=T,
                              log_dir=log_dir,
                              config=config
                              )

if __name__ == '__main__':
    main('./log-mnist1602839987.3347948')