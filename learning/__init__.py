import torch
import torch.nn as nn
import torch.nn.functional as F
import simulating
class STDPLearning(simulating.Simulator):
    def __init__(self, A, tau_a, B, tau_b):
        super().__init__()
        '''
        使用STDP学习规则来更新module参数的仿真器
        记dt = post_spike_time - pre_spike_time
        dt >= 0 则 dw = A * exp( - |dt| / tau_a)
        dt < 0 则 dw = B * exp( - |dt| / tau_b)
        :param A: 系数，一般为正数
        :param tau_a: 系数
        :param B: 系数，一般为负数
        :param tau_b: 系数
        '''
        self.A = A
        self.tau_a = tau_a
        self.B = B
        self.tau_b = tau_b
        self.is_learning_list = []  # 记录某个module是否参与学习（参数更新）


    def append(self, new_module, is_learning=False):
        '''
        :param new_module: 新添加的模型
        :param is_learning: 该模型是否参与学习
        :return: 无返回值
        '''
        super().append(new_module)
        self.is_learning_list.append(is_learning)

    def step(self, input_data):
        '''
        :param input_data: 输入数据
        :return: 输出值
        input_data -> module[0] -> x[0] -> module[1] -> ... -> x[n-2] -> module[n-1] -> x[n-1]
        对于module[i]，若is_learning_list[i]==True，则使用STDP规则更新它的参数
        在pipeline中搜索位于module[i]前后的、数据类型为bool的tensor，这2个tensor就分别是输入脉冲和输出脉冲
        '''