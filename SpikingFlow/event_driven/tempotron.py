import torch
import torch.nn as nn
import torch.nn.functional as F


class Tempotron(nn.Module):
    def __init__(self, in_num, out_num, T, kernel_function, *args):
        '''
        :param in_num: 输入脉冲的数量，也可以看作是输入神经元的数量
        :param out_num: 输出电压的数量，也可以看作是Tempotron神经元的数量
        :param T: 仿真时长
        :param kernel_function: :math:`v(t)` 函数
        :param args: :math:`v(t)` 函数的参数

        Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature neuroscience, 2006, 9(3): 420.

        一个Tempotron神经元，接受in_num个输入。输入是脉冲发放时间，只允许发放一次脉冲，例如

        in_spike = [3, 5, 2]

        表示有3个输入，3个输入的脉冲发放时间分别是3，5，2

        kernel_function一般定义为指数衰减的形式，而且当t<0时输出0，例如 :math:`K(t)=exp(-t), t>=0`

        则输入的不带权重的核电压分别为 :math:`K(t-3)`, :math:`K(t-5)`, :math:`K(t-2)`

        经过突触（连接权值）的作用后，Tempotron的电压为 :math:`w_1K(t-3) + w_2K(t-5) + w_3K(t-2)`

        Tempotron神经元的阈值为1.0，只要在整个仿真时长内有一个时刻，电压超过了阈值，则
        认为Temporton输出1，否则输出0

        因此单个Tempotron神经元可以用来做二分类，而多个则可以做多分类

        使用Tempotron神经元的神经网络，损失函数定义为分类错误的神经元的峰值电压与阈值1.0的差值，这种
        损失函数可以直接使用PyTorch的自动微分机制进行训练

        Tempotron的劣势在于，输入是脉冲，而输出是电压，只能做一层，而且限定了神经元只能发放一次脉冲（因为
        Tempotron神经元只考虑它的峰值电压是否过阈值）
        '''
        super().__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.fc = nn.Linear(in_num, out_num, bias=False)
        self.T = T
        self.t_sequence = torch.arange(start=0, end=T).repeat([in_num, 1]).float()  # shape=[in_num, T]
        self.kernel_function = kernel_function
        self.args = args
        self.pool = nn.MaxPool1d(kernel_size=T)




    def forward(self, t_spike):
        '''
        :param t_spike: 脉冲发放时刻
        :return: out_num个神经元在仿真时长内的最大电压
        '''


        # create time_sequence with shape=[batch_size, in_num, T]
        # self.t_sequence = [
        # [0, 1, 2, ..., T-1],
        # [0, 1, 2, ..., T-1],
        # ...................,
        # [0, 1, 2, ..., T-1]
        # ]
        # time_sequence[?][i] = [
        # 0 - t_spike[?][i], 1 - t_spike[?][i], ..., T - 1 - t_spike[?][i]
        # ]
        
        
        if self.t_sequence.device != t_spike.device:
            self.t_sequence = self.t_sequence.to(t_spike.device)
        time_sequence = (self.t_sequence - t_spike.unsqueeze(2))  # [batch_size, in_num, T]
        v_input = self.kernel_function(time_sequence, *self.args).permute(0, 2, 1)  # [batch_size, in_num, T] -> [batch_size, T, in_num]

        membrane_voltage = self.fc(v_input).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]

        max_membrane_voltage = self.pool(membrane_voltage)

        # [batch_size, out_num, 1] -> [batch_size, out_num]
        max_membrane_voltage = max_membrane_voltage.squeeze(2)


        return max_membrane_voltage













