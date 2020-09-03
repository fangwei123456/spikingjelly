# /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs_cloudbrain.py
# 训练监控

import requests


def push_to_wechat(title, content, token='692f86a853d34106b0c2ab1470b35ba5'):
    url = 'http://pushplus.hxtrip.com/send?token=' + token + '&title=' + title + '&content=' + content
    print(url)
    print(requests.get(url))


import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from spikingjelly.clock_driven import functional, layer, surrogate
from spikingjelly.clock_driven.neuron import PLIFNode, LIFNode
from torch.utils.tensorboard import SummaryWriter
import os
import readline
from spikingjelly.datasets.cifar10_dvs import CIFAR10DVS
import numpy as np
import time
import argparse

torch.backends.cudnn.benchmark = True
_seed_ = 2020
torch.manual_seed(_seed_)  # use torch.manual_seed() to seed the RNG for all devices (both CPU and CUDA)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(_seed_)


class Net(nn.Module):
    def __init__(self, init_tau, use_plif, use_max_pool, v_threshold=1.0, v_reset=0.0):
        super().__init__()
        self.T = 20
        self.train_times = 0
        self.max_test_accuracy = 0
        self.epoch = 0

        self.conv = nn.Sequential(
            nn.Conv2d(2, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),  # 64 * 64

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),  # 32 * 32

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),  # 16 * 16

            nn.Conv2d(128, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.MaxPool2d(2, 2) if use_max_pool else nn.AvgPool2d(2, 2),  # 8 * 8

        )
        self.fc = nn.Sequential(
            nn.Flatten(),
            layer.Dropout(0.5, behind_spiking_layer=use_max_pool),
            nn.Linear(128 * 8 * 8, 128 * 4 * 4, bias=False),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            layer.Dropout(0.5, behind_spiking_layer=True),
            nn.Linear(128 * 4 * 4, 128, bias=False),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
            nn.Linear(128, 100, bias=False),
            PLIFNode(init_tau=init_tau, clamp=True,
                     clamp_function=PLIFNode.sigmoid,
                     inverse_clamp_function=PLIFNode.inverse_sigmoid,
                     v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()) if use_plif else
            LIFNode(tau=init_tau, v_threshold=v_threshold, v_reset=v_reset, surrogate_function=surrogate.ATan()),
        )
        self.boost = nn.AvgPool1d(10, 10)

    def forward(self, x):
        return self.boost(self.fc(self.conv(x)).unsqueeze(1)).squeeze()





if __name__ == '__main__':
    '''
    运行完成的：
    /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs.py -init_tau 16.0 -use_plif 1 -use_max_pool 1
    
    正在运行的：
    /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs.py -init_tau 2.0 -use_plif 1 -use_max_pool 1
    /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs.py -init_tau 2.0 -use_plif 0 -use_max_pool 1
    
    
    尚未运行的
    /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs.py -init_tau 16.0 -use_plif 0 -use_max_pool 1
    /userhome/anaconda3/envs/pytorch-env/bin/python /userhome/release5/cifar10_dvs.py -init_tau 2.0 -use_plif 1 -use_max_pool 0

    '''
    parser = argparse.ArgumentParser()
    # init_tau, batch_size, learning_rate, T_max, log_dir, use_plif
    parser.add_argument('-init_tau', type=float)
    parser.add_argument('-batch_size', type=int, default=16)
    parser.add_argument('-learning_rate', type=float, default=1e-3)
    parser.add_argument('-T_max', type=int, default=64)
    parser.add_argument('-use_plif', action='store_true', default=False)
    parser.add_argument('-use_max_pool', action='store_true', default=False)
    args = parser.parse_args()
    print(args)

    init_tau = args.init_tau
    batch_size = args.batch_size
    learning_rate = args.learning_rate
    T_max = args.T_max
    use_plif = args.use_plif
    use_max_pool = args.use_max_pool
    log_dir = f'/userhome/logs/cifar10_dvs_atan_sigmoid_init_tau_{init_tau}_use_plif_{use_plif}_T_max_{T_max}_use_max_pool_{use_max_pool}'
    print(log_dir)
    exit()
    device = 'cuda:0'
    dataset_dir = '/userhome/datasets/dvs_cifar10_frames_20_split_by_number_normalization_max'
    # dataset_dir = '/ssd_datasets/wfang/dvs_cifar10/dvs_cifar10_frames_20_split_by_number_normalization_max'
    # log_dir = input('输入保存tensorboard日志文件的位置，例如“./”  ')
    # 初始化数据加载器

    train_data_loader = torch.utils.data.DataLoader(
        dataset=CIFAR10DVS(dataset_dir, train=True, split_ratio=0.9),
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        drop_last=True,
        pin_memory=True)
    test_data_loader = torch.utils.data.DataLoader(
        dataset=CIFAR10DVS(dataset_dir, train=False, split_ratio=0.9),
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        drop_last=False,
        pin_memory=True)

    writer = SummaryWriter(log_dir)

    # 初始化网络
    if os.path.exists(log_dir + '/net.pt'):
        net = torch.load(log_dir + '/net.pt', map_location=device)
        print(net.train_times, net.max_test_accuracy)
    else:
        net = Net(init_tau=init_tau, use_plif=use_plif, use_max_pool=use_max_pool).to(device)
    print(net)
    # 使用Adam优化器
    optimizer = torch.optim.Adam(net.parameters(), lr=learning_rate)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max)

    if os.path.exists(log_dir + '/optimizer.pt'):
        optimizer.load_state_dict(torch.load(log_dir + '/optimizer.pt', map_location=device))
    if os.path.exists(log_dir + '/scheduler.pt'):
        scheduler.load_state_dict(torch.load(log_dir + '/scheduler.pt', map_location=device))
    train_log_data_list = []
    for net.epoch in range(net.epoch, 999999999):
        start_time = time.time()
        net.eval()
        with torch.no_grad():
            # 每遍历一次全部数据集，就在测试集上测试一次
            test_sum = 0
            correct_sum = 0
            for img, label in test_data_loader:
                img = img.to(device).permute(1, 0, 2, 3, 4)  # [10, N, 2, 34, 34]
                label = label.to(device)
                out_spikes_counter = net(img[0])
                for t in range(1, net.T):
                    out_spikes_counter += net(img[t])

                correct_sum += (out_spikes_counter.max(1)[1] == label).float().sum().item()
                test_sum += label.numel()
                functional.reset_net(net)
            test_accuracy = correct_sum / test_sum
            # 写入日志到tensorboard
            print('Writing....')

            writer.add_scalar('test_accuracy', test_accuracy, net.epoch)
            if train_log_data_list.__len__() != 0:
                for item in train_log_data_list:
                    writer.add_scalar(item[0], item[1], item[2])
                train_log_data_list.clear()
            print('test_accuracy', test_accuracy)
            if net.max_test_accuracy < test_accuracy:
                print('save model with test_accuracy = ', test_accuracy)
                net.max_test_accuracy = test_accuracy
                torch.save(net, log_dir + '/net_max.pt')
                torch.save(optimizer.state_dict(), log_dir + '/optimizer_max.pt')
                torch.save(scheduler.state_dict(), log_dir + '/scheduler_max.pt')
                push_to_wechat(log_dir, str(test_accuracy))

            torch.save(net, log_dir + '/net.pt')
            torch.save(optimizer.state_dict(), log_dir + '/optimizer.pt')
            torch.save(scheduler.state_dict(), log_dir + '/scheduler.pt')

            print('Written.')

        print(
            'log_dir={}, max_test_accuracy={}, train_times={}, epoch={}'.format(
                log_dir, net.max_test_accuracy, net.train_times, net.epoch
            ))
        print(args)

        net.train()
        for img, label in train_data_loader:
            img = img.to(device).permute(1, 0, 2, 3, 4)  # [10, N, 2, 34, 34]
            label = label.to(device)
            optimizer.zero_grad()

            # 运行T个时长，out_spikes_counter是shape=[batch_size, 10]的tensor
            # 记录整个仿真时长内，输出层的10个神经元的脉冲发放次数
            out_spikes_counter = net(img[0])
            for t in range(1, net.T):
                out_spikes_counter += net(img[t])

            # out_spikes_counter / T 得到输出层10个神经元在仿真时长内的脉冲发放频率
            out_spikes_counter_frequency = out_spikes_counter / net.T
            loss = functional.spike_mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).bool())
            # loss = F.mse_loss(out_spikes_counter_frequency, F.one_hot(label, 10).float())
            loss.backward()
            optimizer.step()
            functional.reset_net(net)

            # 正确率的计算方法如下。认为输出层中脉冲发放频率最大的神经元的下标i是分类结果
            if net.train_times % 256 == 0:
                accuracy = (out_spikes_counter_frequency.argmax(dim=1) == label).float().mean().item()
                train_log_data_list.append(('train_accuracy', accuracy, net.train_times))
                train_log_data_list.append(('train_loss', loss.item(), net.train_times))
                if use_plif:
                    with torch.no_grad():
                        plif_idx = 0
                        for m in net.modules():
                            if isinstance(m, PLIFNode):
                                train_log_data_list.append(('tau_' + str(plif_idx), m.tau(), net.train_times))
                                plif_idx += 1
            net.train_times += 1

        scheduler.step()
        speed_per_epoch = time.time() - start_time
        print('speed per epoch', speed_per_epoch)




