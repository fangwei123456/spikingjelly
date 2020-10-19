import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import os
import time
import numpy as np
from PIL import Image
from torch.utils.tensorboard import SummaryWriter

import spikingjelly.clock_driven.ann2snn.examples.utils as utils
from spikingjelly.clock_driven.ann2snn.examples.model_lib.imagenet import resnet

def main(log_dir=None):
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)

    z_norm_mean = (0.485, 0.456, 0.406)
    z_norm_std = (0.229, 0.224, 0.225)

    # example setting
    device = 'cuda:0'
    dataset_dir = 'Dataset/ILSVRC2012'
    batch_size = 64
    learning_rate = 1e-4
    T = 2000
    train_epoch = 40
    model_name = 'imagenetresnet50'

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
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(z_norm_mean, z_norm_std),
    ])

    ann_transform_test = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(z_norm_mean, z_norm_std),
    ])

    snn_transform = transforms.Compose([
        transforms.Resize(int(224 / 0.875)),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    ann_train_data_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'train'),
        transform=ann_transform_train
    )
    snn_train_data_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'train'),
        transform=snn_transform
    )
    ann_train_data_loader = torch.utils.data.DataLoader(
        dataset=ann_train_data_dataset,
        batch_size=batch_size,
        shuffle=True,
        #num_workers=4,
        drop_last=True,
        pin_memory=True)

    ann_test_data_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'val'),
        transform=ann_transform_test
    )
    snn_test_data_dataset = torchvision.datasets.ImageFolder(
        root=os.path.join(dataset_dir, 'val'),
        transform=snn_transform
    )
    ann_test_data_loader = torch.utils.data.DataLoader(
        dataset=ann_test_data_dataset,
        batch_size=batch_size,
        shuffle=False,
        #num_workers=4,
        drop_last=False,
        pin_memory=True)
    snn_test_data_loader = torch.utils.data.DataLoader(
        dataset=snn_test_data_dataset,
        batch_size=16,
        shuffle=False,
        #num_workers=4,
        drop_last=False,
        pin_memory=True)

    config = utils.Config.default_config
    print('ann2snn config:\n\t', config)
    utils.Config.store_config(os.path.join(log_dir, 'default_config.json'), config)

    loss_function = nn.CrossEntropyLoss()

    ann = resnet.resnet50().to(device)
    checkpoint_state_dict = torch.load('./model_lib/imagenet/checkpoint/ResNet50-state-dict.pth')
    ann.load_state_dict(checkpoint_state_dict)

    # writer = SummaryWriter(log_dir)

    print('Directly load model', model_name + '.pth')

    # 加载用于归一化模型的数据
    # Load the data to normalize the model
    norm_set_len = int(len(snn_train_data_dataset.samples) / 500)
    print('Using %d pictures as norm set' % (norm_set_len))
    norm_set_list = []
    for idx,(datapath,target) in enumerate(snn_train_data_dataset.samples):
        norm_set_list.append(snn_transform(Image.open(datapath)))
        if idx==norm_set_len-1:
            break
    norm_tensor = torch.stack(norm_set_list)

    ann_acc = utils.val_ann(net=ann,
                      device=device,
                      data_loader=ann_test_data_loader,
                      loss_function=loss_function)

    # def hook(module,input,output):
    #     print(module.__class__.__name__)
    #     print(output.reshape(-1)[10:20])
    #
    # handle = []
    # for m in ann.modules():
    #     handle.append(m.register_forward_hook(hook))

    #print(norm_tensor[10,:,:,:].shape)

    # z_score_layer = nn.BatchNorm2d(num_features=len(z_norm_std))
    # norm_mean = torch.from_numpy(np.array(z_norm_mean).astype(np.float32))
    # norm_std = torch.from_numpy(np.array(z_norm_std).astype(np.float32))
    # z_score_layer.weight.data = torch.ones_like(z_score_layer.weight.data)
    # z_score_layer.bias.data = torch.zeros_like(z_score_layer.bias.data)
    # z_score_layer.running_var.data = torch.pow(norm_std, exponent=2) - z_score_layer.eps
    # z_score_layer.running_mean.data = norm_mean
    # z_score_layer.to('cuda:0')
    # z_score_layer.eval()
    # x = z_score_layer(torch.ones(1,3,224,224).to('cuda:0'))
    # print(x.reshape(-1)[10:20])
    # ann.eval()
    # ann(x)



    # for h in handle:
    #     h.remove()

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
    main("./log-imagenetresnet501602856657.8549194")
