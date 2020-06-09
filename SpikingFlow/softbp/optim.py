import torch
from torch import nn
from torch.optim.optimizer import Optimizer

class AdamRewiring(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=0, amsgrad=False, T=1e-5, l1=1e-5):
        '''
        .. attention::
            该算法的收敛性尚未得到任何证明，以及在基于softbp的SNN上的剪枝可靠性也未知。

        :param params: （原始Adam）网络参数的迭代器，或者由字典定义的参数组
        :param lr: （原始Adam）学习率
        :param betas: （原始Adam）用于计算运行时梯度平均值的以及平均值平方的两个参数
        :param eps: （原始Adam）除法计算时，加入到分母中的小常数，用于提高数值稳定性
        :param weight_decay: （原始Adam）L2范数惩罚因子
        :param amsgrad: （原始Adam）是否使用AMSGrad算法
        :param T: Deep R算法中的温度参数
        :param l1: Deep R算法中的L1惩罚参数

        G. Bellec et al, "Deep Rewiring: Training very sparse deep networks," ICLR 2018.

        该实现将论文中的基于SGD优化算法的 `Deep R`_ 算法移植到 `Adam: A Method for Stochastic Optimization`_ 优化算法上，是基于Adam算法在Pytorch中的 `官方实现`_ 修改而来。

        .. _Adam\: A Method for Stochastic Optimization:
            https://arxiv.org/abs/1412.6980
        .. _官方实现:
            https://github.com/pytorch/pytorch/blob/6e2bb1c05442010aff90b413e21fce99f0393727/torch/optim/adam.py
        .. _Deep R:
            https://openreview.net/pdf?id=BJ_wN01C-
        '''
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay, amsgrad=amsgrad, T=T, l1=l1)
        super(AdamRewiring, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(AdamRewiring, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('amsgrad', False)

    @torch.no_grad()
    def step(self, closure=None):
        '''
        :param closure: （原始Adam）传入的闭包，可用于评估模型并返回损失

        执行单步参数更新
        '''
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad
                if grad.is_sparse:
                    raise RuntimeError('Adam does not support sparse gradients, please consider SparseAdam instead')
                amsgrad = group['amsgrad']

                state = self.state[p]

                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if amsgrad:
                        # Maintains max of all exp. moving avg. of sq. grad. values
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                    # 记录各参数初始符号
                    state['sign'] = torch.sign(p)
                    # 记录被置零（休眠状态）的参数mask
                    state['dormant'] = (p != 0.0).float()

                dormant = state['dormant']
                sgn = state['sign']

                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if amsgrad:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                beta1, beta2 = group['betas']

                state['step'] += 1
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']

                if group['weight_decay'] != 0:
                    grad = grad.add(p, alpha=group['weight_decay'])

                # Decay the first and second moment running average coefficient
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                if amsgrad:
                    # Maintains the maximum of all 2nd moment running avg. till now
                    torch.max(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    # Use the max. for normalizing running avg. of gradient
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(group['eps'])

                step_size = group['lr'] / bias_correction1          

                p.addcdiv_(exp_avg, denom, value=-step_size)

                # l1
                p.add_(-group['l1'] * step_size * sgn)

                # 扰动项
                rand_normal = torch.randn_like(p)
                p.add_(rand_normal * group['T'] * step_size)

                # 裁剪越过0的参数：保证各参数符号与sgn中对应的初始符号始终一致，否则变为0
                p.mul_(dormant).mul_(sgn).clamp_(min=0.0).mul_(sgn)

                state['dormant'] = (p != 0.0).float()

        return loss