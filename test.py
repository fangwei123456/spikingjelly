import torch
import torch.nn as nn
import torch.nn.functional as F
from module.postsynaptic_potential_kernel_function import exp_decay_kernel
batch_size = 4
T = 100
tau = 15.0
tau_s = 15.0 / 4
in_num = 4
out_num = 2
in_spike = torch.randint(low=0, high=T, size=[batch_size, in_num]).float().unsqueeze(2)
t_sequence = torch.arange(start=0, end=T).repeat([in_num, 1]).float()  # shape=[in_num, T]
time_sequence = (t_sequence - in_spike).float()  # [batch_size, in_num, T]
weight = torch.rand(size=[out_num, in_num])
v_threshold = 0.1

v_input = exp_decay_kernel(time_sequence, tau, tau_s)  # [batch_size, in_num, T]
membrane_voltage = v_input.permute(0, 2, 1).matmul(weight.t()).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]
out_spike = torch.ones(size=[batch_size, out_num]) * T
for i in range(membrane_voltage.shape[0]):
    for j in range(membrane_voltage.shape[1]):
        for k in range(membrane_voltage.shape[2]):
            if membrane_voltage[i][j][k] >= v_threshold:
                out_spike[i, j] = k
                break
print('out_spike', out_spike)

for i in range(out_num):
    t_spike = time_sequence[:, :, out_spike[:, i].long()].detach()  # batch_size个输出层中第i个神经元的脉冲发放时间 [batch_size, in_num, batch_size]
    t_spike.requires_grad_(True)
    v_input = exp_decay_kernel(t_spike, tau, tau_s)  # [batch_size, in_num, batch_size]
    membrane_voltage = v_input.permute(0, 2, 1).matmul(weight.t()).permute(0, 2, 1)  # [batch_size, out_num, batch_size]
