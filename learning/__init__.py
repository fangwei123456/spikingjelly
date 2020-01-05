import torch
import torch.nn as nn
import torch.nn.functional as F
import simulating
class STDPLearning(simulating.Simulator):
    def __init__(self, A, tau_a, B, tau_b, learning_rate):
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
        :param learning_rate: 学习率
        '''
        self.A = A
        self.tau_a = tau_a
        self.B = B
        self.tau_b = tau_b
        self.learning_rate = learning_rate
        self.is_learning_list = []  # 记录某个module是否参与学习（参数更新）
        self.pre_post_spike_index = []  # 记录参与学习的module的前后脉冲数据在pipeline中的index
        self.pre_post_spike = []  # 记录参与学习的module前后脉冲发放时间

    def append(self, new_module, is_learning=False):
        '''
        :param new_module: 新添加的模型
        :param is_learning: 该模型是否参与学习
        :return: 无返回值
        '''
        super().append(new_module)
        self.is_learning_list.append(is_learning)
        if is_learning:
            self.pre_post_spike.append([])
            self.pre_post_spike_index.append([])
        else:
            self.pre_post_spike.append(None)
            self.pre_post_spike_index.append(None)

    def step(self, input_data):
        '''
        :param input_data: 输入数据
        :return: 输出值
        x[0] -> module[0] -> x[1] -> module[1] -> ... -> x[n-1] -> module[n-1] -> x[n]
        对于module[i]，若is_learning_list[i]==True，则使用STDP规则更新它的参数
        在input_data和pipeline中搜索位于module[i]前后的、数据类型为bool的tensor
        这2个tensor就分别是输入脉冲和输出脉冲
        '''
        self.pipeline[0] = input_data

        if self.simulated_steps < self.module_list.__len__():
            '''
            x[simulated_steps+1] = module[simulated_steps](x[simulated_steps])
            x[simulated_steps] = module[simulated_steps-1](x[simulated_steps-1])
            ...
            x[1] = module[0](x[0])
            '''
            for i in range(self.simulated_steps + 1, 0, -1):
                self.pipeline[i] = self.module_list[i - 1](self.pipeline[i - 1])
            self.simulated_steps += 1


            if self.simulated_steps == self.module_list.__len__():

                for i in range(self.is_learning_list.__len__()):
                    if self.is_learning_list[i]:

                        # 前向查找pre脉冲在pipeline中的位置
                        pre_index = i
                        while 1:
                            assert pre_index >= 0, 'can not find pre spike of module[' + str(i) + ']'
                            if isinstance(self.pipeline[pre_index], torch.Tensor) and \
                                    self.pipeline[pre_index].dtype == torch.bool:
                                self.pre_post_spike_index[i].append(pre_index)
                                break
                            else:
                                pre_index = pre_index - 1
                        # 后向查找post脉冲在pipeline中的位置
                        post_index = i + 1
                        while 1:
                            assert post_index <= self.module_list.__len__(),\
                                'can not find post spike of module[' + str(i) + ']'
                            if isinstance(self.pipeline[post_index], torch.Tensor) and \
                                    self.pipeline[post_index].dtype == torch.bool:
                                self.pre_post_spike_index[i].append(post_index)
                                break
                            else:
                                post_index = post_index + 1
                print('module_list', self.module_list)
                print('is_learning_list', self.is_learning_list)
                print('pipeline', self.pipeline)
                print('pre_post_spike_index', self.pre_post_spike_index)

                # 记录当前时刻各个需要学习的module的脉冲
                for i in range(self.is_learning_list.__len__()):
                    if self.is_learning_list[i]:
                        pre_spike = self.pipeline[self.pre_post_spike_index[i][0]]
                        post_spike = self.pipeline[self.pre_post_spike_index[i][1]]
                        self.pre_post_spike[i].append(pre_index.float() * self.simulated_steps)
                        self.pre_post_spike[i].append(post_index.float() * self.simulated_steps)

                return self.pipeline[-1]
            else:
                return None
        else:
            for i in range(self.module_list.__len__(), 0, -1):
                '''
                x[n] = module[n-1](x[n-1])
                x[n-1] = module[n-2](x[n-2])
                ...
                x[1] = module[0](x[0])
                '''
                self.pipeline[i] = self.module_list[i - 1](self.pipeline[i - 1])
            self.simulated_steps += 1
            return self.pipeline[-1]