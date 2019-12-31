import torch
import torch.nn as nn
import torch.nn.functional as F
'''
Gütig R, Sompolinsky H. The tempotron: a neuron that learns spike timing–based decisions[J]. Nature neuroscience, 2006, 9(3): 420.
'''



class Tempotron(nn.Module):
    def __init__(self, in_num, out_num, T, kernel_function, *args):
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
        if self.t_sequence.device != t_spike.device:
            self.t_sequence = self.t_sequence.to(t_spike.device)
        time_sequence = (self.t_sequence - t_spike.unsqueeze(2))  # [batch_size, in_num, T]
        v_input = self.kernel_function(time_sequence, *self.args).permute(0, 2, 1)  # [batch_size, in_num, T] -> [batch_size, T, in_num]

        membrane_voltage = self.fc(v_input).permute(0, 2, 1)  # [batch_size, T, out_num] -> [batch_size, out_num, T]

        max_membrane_voltage = self.pool(membrane_voltage)

        # [batch_size, out_num, 1] -> [batch_size, out_num]
        max_membrane_voltage = max_membrane_voltage.squeeze(2)


        return max_membrane_voltage

if __name__ == "__main__":
    exit(0)













