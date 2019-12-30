import torch
import torch.nn as nn
import torch.nn.functional as F
import math
'''
Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature neuroscience, 2006, 9(3): 420.
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


class Tempotron(nn.Module):
    def __init__(self, in_num, out_num, T=500, tau=15.0, tau_s=15.0 / 4):
        super().__init__()
        self.in_num = in_num
        self.out_num = out_num
        self.W = nn.Linear(in_num, out_num, bias=False)
        self.T = T
        self.t_sequence = torch.arange(start=0, end=T).repeat([in_num, 1])  # shape=[in_num, T]
        self.tau = tau
        self.tau_s = tau_s
        t_max = (tau * tau_s * math.log(tau / tau_s)) / (tau - tau_s)
        v_max = math.exp(-t_max / tau) - math.exp(-t_max / tau_s)
        self.v0 = 1 / v_max
        self.pool = nn.MaxPool1d(kernel_size=T, return_indices=True)


    def forward(self, t_spike, v_threshold):
        '''

        :param t_spike: shape=[batch_size, in_num]
        t_spike[?][i] is the spike time of input afferent[i]
        :param v_threshold: threshold membrane voltage of tempotrons

        :return:
        membrane_voltage, t_max_membrane_voltage, t_spike_tempotron

        membrane_voltage, shape=[batch_size, out_num, T]
        membrane_voltage[?][i][j] is the tempotoron[i]'s membrane voltage at t=j

        max_membrane_voltage, shape=[batch_size, out_num]

        t_max_membrane_voltage, shape=[batch_size, out_num]
        t_max_membrane_voltage[?][i] is the time when tempotoron[i]'s membrane voltage gets its max value

        '''

        '''
        create time_sequence with shape=[batch_size, in_num, T]
        self.t_sequence = [
        [0, 1, 2, ..., T-1],
        [0, 1, 2, ..., T-1],
        ...................,
        [0, 1, 2, ..., T-1]
        ]
        time_sequence[?][i] = [
        0 - t_spike[?][i], 1 - t_spike[?][i], ..., T - 1 - t_spike[?][i]
        ]
        
        
        '''

        time_sequence = (self.t_sequence - t_spike.unsqueeze(2)).float()   # [batch_size, in_num, T]
        v_input = psp_kernel(time_sequence, self.tau, self.tau_s).permute(0, 2, 1)  # [batch_size, in_num, T] -> [batch_size, T, in_num]
        membrane_voltage = self.v0 * self.W(v_input).permute(0, 2, 1)  # permute(0, 2, 1) [batch_size, T, out_num] -> [batch_size, out_num, T]
        max_membrane_voltage, t_max_membrane_voltage = self.pool(membrane_voltage)

        # [batch_size, out_num, 1] -> [batch_size, out_num]
        max_membrane_voltage = max_membrane_voltage.squeeze(2)
        t_max_membrane_voltage = t_max_membrane_voltage.squeeze(2)


        return membrane_voltage, max_membrane_voltage, t_max_membrane_voltage

if __name__ == "__main__":
    batch_size = 4
    in_num = 10
    out_num = 2
    T = 100
    tau = 15.0
    tau_s = 15.0 / 4
    tm = Tempotron(in_num, out_num, T, tau, tau_s)

    in_spike = torch.randint(low=0, high=T, size=[batch_size, in_num])
    membrane_voltage, max_membrane_voltage, t_max_membrane_voltage, t_spike_tempotron = tm(in_spike, 0.2)
    print(membrane_voltage)
    print(max_membrane_voltage)
    print(t_max_membrane_voltage)













