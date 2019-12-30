import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
Bohte S M, Kok J N, La Poutre H. Error-backpropagation in temporally encoded networks of spiking neurons[J]. Neurocomputing, 2002, 48(1-4): 17-37.
Yang J, Yang W, Wu W. A remark on the error-backpropagation learning algorithm for spiking neural networks[J]. Applied Mathematics Letters, 2012, 25(8): 1118-1120.
'''
def psp_kernel(t, tau=15.0, tau_s=15.0 / 4):
    '''
    postsynaptic potentials
    :param t:
    :param tau:
    :param tau_s:
    :return:
    '''
    t_ = F.relu(t)
    return torch.exp(-t_ / tau) - torch.exp(-t_ / tau_s)

def grad_psp_kernel(t, tau=15.0, tau_s=15.0 / 4):
    '''
    postsynaptic potentials
    :param t:
    :param tau:
    :param tau_s:
    :return:
    '''
    t_ = F.relu(t)
    return -torch.exp(-t_ / tau) / tau + torch.exp(-t_ / tau_s) / tau_s

class SpikePropLayer(nn.Module):
    '''
    一层SpikePropLayer
    输入[batch_size, in_num]，是in_num个神经元的脉冲发放时间
    输出[batch_size, out_num]，是out_num个神经元的脉冲发放时间
    '''
    def __init__(self, in_num, out_num, T):
        super(SpikePropLayer, self).__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.fc = nn.Linear(in_num, out_num, bias=False)
        self.T = T
        self.t_sequence = torch.arange(start=0, end=T).repeat([in_spike.shape[1], 1])  # shape=[in_num, T]

    def forward(self, in_spike, kernel_function, grad_kernel_function, *args):
        '''
        :param in_spike:输入[batch_size, in_num]，是in_num个神经元的脉冲发放时间

        :return:
        '''
        batch_size = in_spike.shape[0]
        device = in_spike.device
        if self.t_sequence.device != device:
            self.t_sequence = self.t_sequence.to(device)
        time_sequence = (self.t_sequence - in_spike.unsqueeze(2)).float()  # [batch_size, in_num, T]

        v_input = kernel_function(time_sequence, *args)  # [batch_size, in_num, T]
        membrane_voltage = self.fc(v_input.permute(0, 2, 1)).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]


class VoltageToSpike(torch.autograd.Function):
    '''
    脉冲定义为膜电位电压首次到达阈值的时刻，只允许一次脉冲发放
    '''
    @staticmethod
    def forward(ctx, weight, in_spike, v_threshold, T, kernel_function, grad_kernel_cuntion, *args):
        '''
        :param ctx:
        :param weight: shape=[out_num, in_num]
        :param in_spike: shape=[batch_size, in_num]
        :param v_threshold: a float num
        :param T: stimulation time
        :param kernel_function: kernel function for calculating membrane voltage
        kernel_function is defined as
        def f(t, *args)
        :param grad_kernel_function: gradient of kernel_function
        :param args: agrs for kernel_function
        :return: out_spike, shape=[batch_size, out_num]
        '''
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
        batch_size = in_spike.shape[0]
        in_num = in_spike.shape[1]
        out_num = weight.shape[0]

        device = weight.device
        t_sequence = torch.arange(start=0, end=T).repeat([in_spike.shape[1], 1]).to(device)  # shape=[in_num, T]
        time_sequence = (t_sequence - in_spike.unsqueeze(2)).float()  # [batch_size, in_num, T]

        v_input = kernel_function(time_sequence, *args)  # [batch_size, in_num, T]
        membrane_voltage = v_input.permute(0, 2, 1).matmul(weight.t()).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]

        out_spike = -torch.ones(size=membrane_voltage.shape[0:2], device=device)
        # todo cuda backend
        # find the first time that membrane_voltage >= v_threshold
        for i in range(membrane_voltage.shape[0]):
            for j in range(membrane_voltage.shape[1]):
                for k in range(membrane_voltage.shape[2]):
                    if membrane_voltage[i][j][k] >= v_threshold:
                        out_spike[i][j] = k
                        break
        out_spike[out_spike < 0] = T
        # spike time is T means that no spike because t_sequence - T < 0 holds forever

        t_sequence_spkie = torch.zeros(size=[batch_size, in_num, out_num], device=device)  # 脉冲发放时刻的输入
        X1_spike = torch.zeros(size=[batch_size, in_num, out_num], device=device)  # 脉冲发放时刻的输入层电压
        dX1_dt = torch.zeros(size=[batch_size, in_num, out_num], device=device)  # dX1_dt[:, :, i]是out_spike[:, i]时刻输入层神经元电压对时间的导数


        for b in range(batch_size):
            for i in range(out_num):
                t_sequence_spkie[b, :, i] = time_sequence[b, :, out_spike[b, i].int()]
                X1_spike[b, :, i] = v_input[b, :, out_spike[b, i].int()]
                dX1_dt[b, :, i] = grad_kernel_cuntion(t_sequence_spkie[b, :, i], *args)

        dY1_dt_0 = (out_spike != T).float()

        ctx.save_for_backward(t_sequence_spkie, X1_spike, dX1_dt, dY1_dt_0, weight)


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


        t_sequence_spkie, X1_spike, dX1_dt, dY1_dt_0, weight = ctx.saved_tensors
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
        batch_size = t_sequence_spkie.shape[0]
        in_num = t_sequence_spkie.shape[1]
        out_num = t_sequence_spkie.shape[2]
        dY1_dt = torch.zeros(size=[batch_size, out_num], device=grad_output.device)

        for i in range(out_num):
            '''
            需要注意，不发放脉冲的情况，脉冲发放时间被设置成T，在kernel_cuntion中会得到0
            而在grad_kernel_cuntion中也会计算得到0
            '''
            dY1_dt[:, i] = dX1_dt[:, :, i].matmul(weight[i].t())

        '''
        接下来计算d(S2) / d(Y1)，它应该就是-1 / dY1_dt
        通过乘dY1_dt_0，防止-1/0出现nan的情况
        grad_output shape=[batch_size, out_num]
        grad_output = d(Loss) / d(S2)
        '''
        dS2_dY1 = -1 / dY1_dt * dY1_dt_0  # [batch_size, out_num]
        dL_W1 = torch.zeros_like(weight)  # [out_num, in_num]
        dL_dY1 = dS2_dY1 * grad_output  # [batch_size, out_num]
        dL_dS1 = torch.zeros(size=[batch_size, in_num], device=grad_output.device)
        '''
        Y1 = X1.matmul(W1.t())  
        任意时刻t
        [batch_size, t, out_num] = [batch_size, t, in_num] [in_num, out_num]
        grad_output = d(Loss) / d(S2)  shape=[batch_size, out_num]
        dS2_dY1.shape=[batch_size, out_num]
        dL_dY1.shape=[batch_size, out_num]
        X1_spike.shape=[batch_size, in_num, out_num]
        '''
        for i in range(out_num):
            '''
            dL_W1[i] shape=[in_num]
            X1_spike[:, :, i] shape=[batch_size, in_num]
            dL_dY1[:, i] shape=[batch_size]
            Y1[:, i] shape=[batch_size]
            Y1[:, i] = X1_spike[:, :, i].matmul(W1[i])
            '''
            dL_W1[i] = X1_spike[:, :, i].t().matmul(dL_dY1[:, i])

        '''
        dL_dS1 shape=[batch_size, in_num]
        先求解dL_dX1 shape=[batch_size, in_num]
            
        dX1_dt shape=[batch_size, in_num, out_num]
        dX1_dt[:, :, i]是out_spike[:, i]时刻输入层神经元电压对时间的导数
        
        X1_spike[:, :, i] shape=[batch_size, in_num]
        dL_dY1[:, i] shape=[batch_size]
        Y1[:, i] shape=[batch_size]
        Y1[:, i] = X1_spike[:, :, i].matmul(W1[i])
        
        dL_dX1 shape=[batch_size, in_num]
        '''
        dL_dX1 = dS2_dY1.matmul(weight.t())




        return dL_W1, grad_S1, None, None, None, None

if __name__ == "__main__":

    batch_size = 4
    in_num = 10
    out_num = 2
    T = 100
    tau = 15.0
    tau_s = 15.0 / 4
    in_spike = torch.randint(low=0, high=T, size=[batch_size, in_num])
    W = torch.rand(size=[out_num, in_num])
    W.requires_grad_(True)

    y = SpikeToSpike.apply(W, in_spike, 0.2, T, psp_kernel, grad_psp_kernel, tau, tau_s)
    print(y)
    y.sum().backward()

