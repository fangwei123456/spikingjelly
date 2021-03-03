import torch
import os
import numpy as np
from tqdm import tqdm
import requests

def train_ann(net, device, data_loader, optimizer, loss_function, epoch=None):
    '''
    * :ref:`API in English <train_ann-en>`

    .. _train_ann-cn:

    :param net: 训练的模型
    :param device: 运行的设备
    :param data_loader: 训练集
    :param optimizer: 神经网络优化器
    :param loss_function: 损失函数
    :param epoch: 当前训练期数
    :return: ``None``

    经典的神经网络训练程序预设，便于直接调用训练网络

    * :ref:`中文API <train_ann-cn>`

    .. _train_ann-en:

    :param net: network to train
    :param device: running device
    :param data_loader: training data loader
    :param optimizer: neural network optimizer
    :param loss_function: neural network loss function
    :param epoch: current training epoch
    :return: ``None``

    Preset classic neural network training program
    '''
    net.train()
    losses = []
    correct = 0.0
    total = 0.0
    for batch, (img, label) in enumerate(data_loader):
        img = img.to(device)
        optimizer.zero_grad()
        out = net(img)
        loss = loss_function(out, label.to(device))
        loss.backward()
        optimizer.step()
        losses.append(loss.item())
        correct += (out.max(dim=1)[1] == label.to(device)).float().sum().item()
        total += out.shape[0]
        if batch % 100 == 0:
            acc = correct / total
            print('Epoch %d [%d/%d] ANN Training Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                             batch + 1,
                                                                             len(data_loader),
                                                                             np.array(losses).mean(),
                                                                             acc))
            correct = 0.0
            total = 0.0


def val_ann(net, device, data_loader, loss_function, epoch=None):
    '''
    * :ref:`API in English <val_ann-en>`

    .. _val_ann-cn:

    :param net: 待验证的模型
    :param device: 运行的设备
    :param data_loader: 测试集
    :param epoch: 当前训练期数
    :return: 验证准确率

    经典的神经网络训练程序预设，便于直接调用训练网络

    * :ref:`中文API <val_ann-cn>`

    .. _val_ann-en:

    :param net: network to test
    :param device: running device
    :param data_loader: testing data loader
    :param epoch: current training epoch
    :return: testing accuracy

    Preset classic neural network training program
    '''
    net.eval()
    correct = 0.0
    total = 0.0
    losses = []
    with torch.no_grad():
        for batch, (img, label) in enumerate(tqdm(data_loader)):
            img = img.to(device)
            out = net(img)
            loss = loss_function(out, label.to(device))
            correct += (out.argmax(dim=1) == label.to(device)).float().sum().item()
            total += out.shape[0]
            losses.append(loss.item())
        acc = correct / total
        if epoch == None:
            print('ANN Validating Accuracy:%.3f' % (acc))
        else:
            print('Epoch %d [%d/%d] ANN Validating Loss:%.3f Accuracy:%.3f' % (epoch,
                                                                               batch + 1,
                                                                               len(data_loader),
                                                                               np.array(losses).mean(),
                                                                               acc))
    return acc


def save_model(net, log_dir, file_name):
    '''
    * :ref:`API in English <save_model-en>`

    .. _save_model-cn:

    :param net: 要保存的模型
    :param log_dir: 日志文件夹
    :param file_name: 文件名
    :return: ``None``

    保存模型的参数，以两种形式保存，分别为Pytorch保存的完整模型（适用于网络模型中只用了Pytorch预设模块的）
    以及模型参数（适用于网络模型中有自己定义的非参数模块无法保存完整模型）

    * :ref:`中文API <save_model-cn>`

    .. _save_model-en:

    :param net: network model to save
    :param log_dir: log file folder
    :param file_name: file name
    :return: ``None``

    Save the model, which is saved in two forms, the full model saved by Pytorch (for the network model only possessing
    the Pytorch preset module) and model parameters only (for network models that have their own defined nonparametric
    modules. In that case, Pytorch cannot save the full model)
    '''
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    torch.save(net, os.path.join(log_dir, file_name))
    torch.save(net.state_dict(), os.path.join(log_dir, 'param_' + file_name))
    print('Save model to:', os.path.join(log_dir, file_name))


def download_sample_pth(url, filename):
    '''
    * :ref:`API in English <download_sample_pth-en>`

    .. _download_sample_pth-cn:

    :param url: 链接
    :param filename: 文件名
    :return: ``None``

    下载例子的模型文件

    * :ref:`中文API <download_sample_pth-cn>`

    .. _download_sample_pth-en:

    :param url: links
    :param filename: file name
    :return: ``None``

    Download model state dict for examples
    '''
    print('Downloading %s from %s, please wait...'%(filename,url))
    r = requests.get(url, allow_redirects=True)
    open(filename, 'wb').write(r.content)