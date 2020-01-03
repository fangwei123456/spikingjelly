import torch
import torch.nn as nn
import torch.nn.functional as F

class HardForwardSoftBackwardSigmoid(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, k=0.1):
        '''
        输入x，对x进行逐元素操作
        前向传播时
        x[i]>0 y[i]=1
        x[i]<=0 y[i]=0
        反向传播时，则是按照前向传播为sigmoid(k * x)来计算导数
        :param x: tensor，任意形状
        :param k: float，控制反向传播的导数的平滑程度，k越小则导数越平滑
        前向传播可以看作是k无穷大
        :return: y，与x相同形状
        '''
        ctx.save_for_backward(x, torch.tensor(data=k, device=x.device))
        return torch.gt(x, 0).float()

    @staticmethod
    def backward(ctx, grad_output):
        ret = None
        if ctx.needs_input_grad[0]:
            x, k = ctx.saved_tensors
            sigmoid_kx = torch.sigmoid(k * x)
            ret = grad_output * k * sigmoid_kx * (1 - sigmoid_kx)
        return ret, None

hard_forward_soft_backward_sigmoid = HardForwardSoftBackwardSigmoid.apply

class VoltageToSpike(nn.Module):
    def __init__(self, spike_mode, v_threshold):
        '''
        将电压转换为脉冲
        :param spike_mode:
        'threshold': 过阈值则认为脉冲发放，即t时刻发放脉冲，当且仅当v(t-1)<v_threshold，而v(t+1)>v_threshold
        'delta': 电压增量大于阈值则认为脉冲发放，v(t)-v(t-1)>v_threshold
        这两种模式都允许多脉冲的形式
        '''
        super().__init__()
        assert spike_mode == 'threshold' or spike_mode == 'delta', 'not implemented mode'
        self.spike_mode = spike_mode
        self.v_threshold = v_threshold

        if self.spike_mode == 'threshold':
            exit(0)
        elif self.spike_mode == 'delta':
            self.conv_kernel = torch.tensor(data=[-1, 1]).view(1, 1, 2)



    def forward(self, x):
        '''
        :param x: 输入电压，shape=[batch_size, *, T]
        out_num个神经元在t=0, 1, ..., T-1时刻的电压值
        :return:脉冲，shape=[batch_size, *, T]
        out_num个神经元在t=0, 1, ..., T-1时刻是否有脉冲，有则为1，没有则为0
        '''
        x_shape = x.shape

        if self.spike_mode == 'threshold':
            exit(0)
        elif self.spike_mode == 'delta':
            if self.conv_kernel.device != x.device:
                self.conv_kernel = self.conv_kernel.to(x.device)
            diff = F.conv1d(x.view(-1, 1, x.shape[2]), self.conv_kernel, padding=1)
            # diff中保存的是v(t)-t(t-1)，也就是前向一阶差分
            spkie = hard_forward_soft_backward_sigmoid(diff - self.v_threshold, 0.5)
            # spkie shape=[*, T] 为1的位置表示该时刻有脉冲











