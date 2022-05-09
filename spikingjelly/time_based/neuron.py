import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class Tempotron(nn.Module):
    def __init__(self, in_features, out_features, T, tau=15.0, tau_s=15.0 / 4, v_threshold=1.0):
        '''
        :param in_features: 输入数量，含义与nn.Linear的in_features参数相同
        :param out_features: 输出数量，含义与nn.Linear的out_features参数相同
        :param T: 仿真周期
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :param v_threshold: 阈值电压

        Gutig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature \
        Neuroscience, 2006, 9(3): 420-428. 中提出的Tempotron模型

        '''
        super().__init__()
        self.tau = tau
        self.tau_s = tau_s
        self.T = T
        self.v_threshold = v_threshold
        self.fc = nn.Linear(in_features, out_features, bias=False)

        # v0要求使psp_kernel的最大值为v_threshold，通过求极值点，计算出最值
        t_max = (tau * tau_s * math.log(tau / tau_s)) / (tau - tau_s)
        self.v0 = self.v_threshold / (math.exp(-t_max / tau) - math.exp(-t_max / tau_s))

    @staticmethod
    def psp_kernel(t: torch.Tensor, tau, tau_s):
        '''
        :param t: 表示时刻的tensor
        :param tau: LIF神经元的积分时间常数
        :param tau_s: 突触上的电流的衰减时间常数
        :return: t时刻突触后的LIF神经元的电压值
        '''
        # 指数衰减的脉冲输入
        return (torch.exp(-t / tau) - torch.exp(-t / tau_s)) * (t >= 0).float()

    @staticmethod
    def mse_loss(v_max, v_threshold, label, num_classes):
        '''
        :param v_max: Tempotron神经元在仿真周期内输出的最大电压值，与forward函数在ret_type == 'v_max'时的返回值相\
        同。shape=[batch_size, out_features]的tensor
        :param v_threshold: Tempotron的阈值电压，float或shape=[batch_size, out_features]的tensor
        :param label: 样本的真实标签，shape=[batch_size]的tensor
        :param num_classes: 样本的类别总数，int
        :return: 分类错误的神经元的电压，与阈值电压之差的均方误差
        '''
        wrong_mask = ((v_max >= v_threshold).float() != F.one_hot(label, num_classes)).float()
        return torch.sum(torch.pow((v_max - v_threshold) * wrong_mask, 2)) / label.shape[0]

    def forward(self, in_spikes: torch.Tensor, ret_type):
        '''
        :param in_spikes: shape=[batch_size, in_features]

        in_spikes[:, i]表示第i个输入脉冲的脉冲发放时刻，介于0到T之间，T是仿真时长

        in_spikes[:, i] < 0则表示无脉冲发放
        :param ret_type: 返回值的类项，可以为'v','v_max','spikes'
        :return:

        ret_type == 'v': 返回一个shape=[batch_size, out_features, T]的tensor，表示out_features个Tempotron神经元在仿真时长T\
        内的电压值

        ret_type == 'v_max': 返回一个shape=[batch_size, out_features]的tensor，表示out_features个Tempotron神经元在仿真时长T\
        内的峰值电压

        ret_type == 'spikes': 返回一个out_spikes，shape=[batch_size, out_features]的tensor，表示out_features个Tempotron神\
        经元的脉冲发放时刻，out_spikes[:, i]表示第i个输出脉冲的脉冲发放时刻，介于0到T之间，T是仿真时长。out_spikes[:, i] < 0\
        表示无脉冲发放
        '''
        t = torch.arange(0, self.T).to(in_spikes.device)  # t = [0, 1, 2, ..., T-1] shape=[T]
        t = t.view(1, 1, t.shape[0]).repeat(in_spikes.shape[0], in_spikes.shape[1], 1)  # shape=[batch_size, in_features, T]
        in_spikes = in_spikes.unsqueeze(-1).repeat(1, 1, self.T)  # shape=[batch_size, in_features, T]
        v_in = self.v0 * self.psp_kernel(t - in_spikes, self.tau, self.tau_s) * (in_spikes >= 0).float()  # in_spikes[:, i] < 0的位置，输入电压也为0
        v_out = self.fc(v_in.permute(0, 2, 1)).permute(0, 2, 1)  # shape=[batch_size, out_features, T]

        if ret_type == 'v':
            return v_out
        elif ret_type == 'v_max':
            return F.max_pool1d(v_out, kernel_size=self.T).squeeze()
        elif ret_type == 'spikes':
            max_index = v_out.argmax(dim=2)  # shape=[batch_size, out_features]

            # 用soft arg max的方法，将tempotron扩展到多层
            t = torch.arange(0, self.T).to(in_spikes.device)  # t = [0, 1, 2, ..., T-1] shape=[T]
            t = t.view(1, 1, t.shape[0]).repeat(in_spikes.shape[0], v_out.shape[1], 1)  # shape=[batch_size, out_features, T]
            max_index_soft = (F.softmax(v_out * self.T, dim=2) * t).sum(dim=2)  # shape=[batch_size, out_features]
            v_max = F.max_pool1d(v_out, kernel_size=self.T).squeeze()
            mask = (v_max >= self.v_threshold).float() * 2 - 1
            # mask_soft = torch.tanh(v_max - self.v_threshold)
            # mask中的元素均为±1，表示峰值电压是否过阈值
            max_index = max_index * mask
            # max_index_soft = max_index_soft * mask_soft
            max_index_soft = max_index_soft * mask
            # print('max_index\n', max_index, '\nmax_index_soft\n', max_index_soft)
            return max_index_soft + (max_index - max_index_soft).detach()
        else:
            raise ValueError


