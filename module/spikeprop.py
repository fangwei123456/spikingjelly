import torch
import torch.nn as nn
import torch.nn.functional as F
import math

'''
Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.
Yang J, Yang W, Wu W. A remark on the error-backpropagation learning algorithm for spiking neural networks[J]. Applied Mathematics Letters, 2012, 25(8): 1118-1120.
'''


class SpikePropLayer(nn.Module):
    '''
    一层SpikePropLayer
    输入[batch_size, in_num]，是in_num个神经元的脉冲发放时间
    输出[batch_size, out_num]，是out_num个神经元的脉冲发放时间
    '''

    def __init__(self, in_num, out_num, T, v_threshold, kernel_function, grad_kernel_function, *args):
        super(SpikePropLayer, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.fc = nn.Linear(in_num, out_num, bias=False)
        self.T = T
        self.v_threshold = v_threshold
        self.t_sequence = torch.arange(start=0, end=T).repeat([in_num, 1]).float()  # shape=[in_num, T]
        self.kernel_function = kernel_function
        self.grad_kernel_function = grad_kernel_function
        self.args = args

    def forward(self, in_spike):
        '''
        :param in_spike:输入[batch_size, in_num]，是in_num个神经元的脉冲发放时间

        :return:
        '''
        device = in_spike.device
        if self.t_sequence.device != device:
            self.t_sequence = self.t_sequence.to(device)
        '''
                create time_sequence with shape=[batch_size, in_num, T]
                t_sequence = [
                [0, 1, 2, ..., T-1],
                [0, 1, 2, ..., T-1],
                ...................,
                [0, 1, 2, ..., T-1]
                ]
                time_sequence[?][i] = [
                0 - t_spike[?][i], 1 - t_spike[?][i], ..., T - 1 - t_spike[?][i]
                ]


        '''
        time_sequence = (self.t_sequence - in_spike.unsqueeze(2)).float()  # [batch_size, in_num, T]

        v_input = self.kernel_function(time_sequence, *self.args)  # [batch_size, in_num, T]
        membrane_voltage = self.fc(v_input.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]

        return VoltageToSpike.apply(membrane_voltage, self.fc.weight.detach(), time_sequence.detach(), self.v_threshold,
                                    self.kernel_function, self.grad_kernel_function, *self.args)


class VoltageToSpike(torch.autograd.Function):
    '''
    输入膜电位电压，输出脉冲发放时间
    脉冲定义为膜电位电压首次到达阈值的时刻，只允许一次脉冲发放
    '''

    @staticmethod
    def forward(ctx, membrane_voltage, weight, time_sequence, v_threshold, kernel_function, grad_kernel_function, *args):
        '''
        输入膜电位membrane_voltage shape=[batch_size, out_num, T]
        输出脉冲发放时刻out_spike shape=[batch_size, out_num]
        '''
        batch_size = membrane_voltage.shape[0]
        in_num = weight.shape[1]
        out_num = weight.shape[0]
        T = membrane_voltage.shape[2]
        device = membrane_voltage.device
        out_spike = torch.ones(size=[batch_size, out_num], device=device).float() * T
        # 对于不发放脉冲的神经元，把脉冲发放时间设置R，因此初始化时都是T
        # todo cuda backend
        # find the first time that membrane_voltage >= v_threshold
        for i in range(membrane_voltage.shape[0]):
            for j in range(membrane_voltage.shape[1]):
                for k in range(membrane_voltage.shape[2]):
                    '''
                    找到电压首次过阈值的时刻
                    无法用argmin之类的操作完成，因为当存在多个最小值时，argmin返回的最小位置是不确定的
                    '''
                    if membrane_voltage[i][j][k] >= v_threshold:
                        out_spike[i][j] = k
                        break

        '''
        下面的计算是为了反向传播做准备
        使用以下符号
        输入脉冲S1, [batch_size, in_num]
        经过核函数作用，得到电压X1, [batch_size, in_num, T]
        经过权重W1, [out_num, in_num], 作用，得到电压Y1, [batch_size, out_num, T]
        由电压Y1产生脉冲S2, [batch_size, out_num]
        '''
        dX1_dt = torch.zeros(size=[batch_size, in_num, out_num], device=device)
        # dX1_dt[:, :, i]是out_spike[:, i]时刻输入层神经元电压对时间的导数

        for b in range(batch_size):
            for i in range(out_num):
                if out_spike[b, i].int() != T:
                    dX1_dt[b, :, i] = grad_kernel_function(time_sequence[b, :, out_spike[b, i].int()], *args)

        ctx.save_for_backward(out_spike, dX1_dt, weight, torch.tensor(data=[args.__len__(), T], dtype=int))
        return out_spike  # shape=[batch_size, out_num]

    @staticmethod
    def backward(ctx, grad_output):
        '''
        :param ctx:
        :param grad_output:
        :return:

        grad_output.shape=[batch_size, out_num], same with out_spike

        使用以下符号
        输入脉冲S1, [batch_size, in_num]
        经过核函数作用，得到电压X1, [batch_size, in_num, T]
        经过权重weight=W1, [out_num, in_num], 作用，得到电压Y1, [batch_size, out_num, T]
        由电压Y1产生脉冲S2, [batch_size, out_num]

        '''
        out_spike, dX1_dt, weight, shape_tensor = ctx.saved_tensors

        batch_size, out_num = grad_output.shape
        in_num = weight.shape[1]
        args_len = shape_tensor[0].item()
        T = shape_tensor[1].item()
        '''
        使用以下符号
        输入脉冲S1, [batch_size, in_num]
        经过核函数作用，得到电压X1, [batch_size, in_num, T]
        经过权重weight=W1, [out_num, in_num], 作用，得到电压Y1, [batch_size, out_num, T]
        由电压Y1产生脉冲S2, [batch_size, out_num]
        首先求解Y1, [batch_size, out_num, T]在脉冲发放时刻对时间的导数，尺寸应该为[batch_size, out_num]
        d(Y1) / dt = d(X1.matmul(W1.t())) / dt = (d(X1) / dt).matmul(W1.t())
        由于out_num个输出神经元的脉冲发放时间并不相同，因此输出层神经元i在脉冲发放时刻t_i的电压Y1[:, i, t_i]对时间的导数
        应该为X1[:, :, t_i]对时间的导数，再矩阵乘W1[i].t()
        X1[:, :, t_i]对时间的导数为t_i时刻核函数对时间的导数

        dX1_dt shape=[batch_size, in_num, out_num]
        dX1_dt[:, :, i]是out_spike[:, i]时刻输入层神经元电压对时间的导数
        '''
        dY1_dt = torch.zeros_like(out_spike)

        for i in range(out_num):
            '''
            需要注意，不发放脉冲的情况，脉冲发放时间被设置成T，在kernel_cuntion中会得到0
            而在grad_kernel_cuntion中也会计算得到0
            '''
            dY1_dt[:, i] = dX1_dt[:, :, i].matmul(weight[i].t())

        '''
        接下来计算d(S2) / d(Y1)，它应该就是-1 / dY1_dt
        通过乘(dY1_dt != 0).float()，防止-1/0出现nan的情况
        grad_output shape=[batch_size, out_num]
        grad_output = d(Loss) / d(S2)
        '''
        dS2_dY1 = -1 / dY1_dt * ((dY1_dt != 0).float())  # [batch_size, out_num]
        dL_dY1 = dS2_dY1 * grad_output  # [batch_size, out_num]
        grad_membrane_voltage = torch.zeros(size=[batch_size, out_num, T], device=grad_output.device)
        for b in range(batch_size):
            for i in range(out_num):
                if out_spike[b, i].int() != T:
                    grad_membrane_voltage[b, i, out_spike[b, i].int()] = dL_dY1[b, i]
                    # 仅在脉冲发放时刻有导数，其他时刻都0
        ret = []
        ret.append(grad_membrane_voltage)
        for i in range(5 + args_len):
            ret.append(None)
        return tuple(ret)



















