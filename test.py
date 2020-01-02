import torch
from encoding.gaussian_tuning_curve import GaussianEncoder
import numpy as np
from module.tempotron import Tempotron
from module.spikeprop import SpikePropLayer
import module.postsynaptic_potential_kernel_function as kernel_function

if __name__ == "__main__":

    batch_size = 4
    T = 100
    tau = 15.0
    tau_s = 15.0 / 4
    csv_data = np.loadtxt('./dataset/iris.csv', skiprows=1, delimiter=',')  # 最后一列为分类
    data_num = csv_data.shape[0]
    feature_num = csv_data.shape[1] - 1  # 最后一列是标签，前面是特征
    per_class_neuron_num = 8
    enc_neuron_num = 12
    class_num = 3
    print('数据数', data_num, '特征数', feature_num, '类别数', class_num)
    x_data = torch.from_numpy(csv_data[:, 0: feature_num]).float()
    y_data = torch.from_numpy(csv_data[:, feature_num]).int()
    x_train_min = x_data.min(0)[0]
    x_train_max = x_data.max(0)[0]
    encoder = GaussianEncoder(x_train_min, x_train_max, enc_neuron_num)
    t_spike = encoder.encode(x_data).view(data_num, -1)

    sp1 = SpikePropLayer(feature_num * enc_neuron_num, 32, T, 0.1, kernel_function.exp_decay_kernel,
                         kernel_function.grad_exp_decay_kernel, tau, tau_s).cuda().train()
    sp2 = SpikePropLayer(32, 16, T, 0.1, kernel_function.exp_decay_kernel, kernel_function.grad_exp_decay_kernel, tau,
                         tau_s).cuda().train()

    tp = Tempotron(16, class_num, T, kernel_function.exp_decay_kernel, tau, tau_s).cuda().train()
    optimizer = torch.optim.SGD(
        [{'params': sp1.parameters()}, {'params': sp2.parameters()}, {'params': tp.parameters()}], lr=1e-3)
    while 1:
        optimizer.zero_grad()
        batch_i = np.random.randint(low=0, high=data_num, size=[batch_size])
        v_max = tp(sp2(sp1(t_spike[batch_i].cuda())))
        y_r = F.one_hot(y_data[batch_i].long(), num_classes=class_num).float().cuda()
        wrong_id = ((v_max >= 1.0).float() != y_r).float()
        loss = torch.sum(torch.pow((v_max - 1) * wrong_id, 2))
        print(loss.item())
        loss.backward()
        optimizer.step()
