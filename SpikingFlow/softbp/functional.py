import torch
import torch.nn as nn
import torch.nn.functional as F

def spike_cluster(v: torch.Tensor, v_threshold, T_in: int):
    '''
    :param v: shape=[T, N]，N个神经元在t=[0, 1, ..., T-1]时刻的电压值
    :param v_threshold: 神经元的阈值电压，float或者是shape=[N]的tensor
    :param T_in: 脉冲聚类的距离阈值。一个脉冲聚类满足，内部任意2个相邻脉冲的距离不大于T_in，而其内部任一脉冲与外部的脉冲距离大于T_in
    :return:
    N_o: shape=[N]，N个神经元的输出脉冲的脉冲聚类的数量

    k_positive: shape=[N]，bool类型的tensor，索引。需要注意的是，k_positive可能是一个全False的tensor

    k_negative: shape=[N]，bool类型的tensor，索引。需要注意的是，k_negative可能是一个全False的tensor

    Gu P, Xiao R, Pan G, et al. STCA: Spatio-Temporal Credit Assignment with Delayed Feedback in Deep Spiking Neural Networks[C]. international joint conference on artificial intelligence, 2019: 1366-1372.\
    一文提出的脉冲聚类方法。如果想使用该文中定义的损失，可以参考如下代码：

    .. code-block:: python

        v_k_negative = out_v * k_negative.float().sum(dim=0)
        v_k_positive = out_v * k_positive.float().sum(dim=0)
        loss0 = ((N_o > N_d).float() * (v_k_negative - 1.0)).sum()
        loss1 = ((N_o < N_d).float() * (1.0 - v_k_positive)).sum()
        loss = loss0 + loss1
    '''
    with torch.no_grad():

        spike = (v >= v_threshold).float()
        T = v.shape[0]

        N_o = torch.zeros_like(v[1])
        spikes_num = torch.ones_like(v[1]) * T * 2
        min_spikes_num = torch.ones_like(v[1]) * T * 2
        min_spikes_num_t = torch.ones_like(v[1]) * T * 2
        last_spike_t = - torch.ones_like(v[1]) * T_in * 2
        # 初始时，认为上一次的脉冲发放时刻是- T_in * 2，这样即便在0时刻发放脉冲，其与上一个脉冲发放时刻的间隔也大于T_in

        for t in range(T):
            delta_t = (t - last_spike_t) * spike[t]
            # delta_t[i] == 0的神经元i，当前时刻无脉冲发放
            # delta_t[i] > 0的神经元i，在t时刻释放脉冲，距离上次释放脉冲的时间差为delta_t[i]

            mask0 = (delta_t > T_in)  # 在t时刻释放脉冲，且距离上次释放脉冲的时间高于T_in的神经元
            mask1 = torch.logical_and(delta_t <= T_in, spike[t].bool())  # t时刻释放脉冲，但距离上次释放脉冲的时间不超过T_in的神经元



            temp_mask = torch.logical_and(mask0, min_spikes_num > spikes_num)
            min_spikes_num_t[temp_mask] = last_spike_t[temp_mask]
            min_spikes_num[temp_mask] = spikes_num[temp_mask]

            spikes_num[mask0] = 1
            N_o[mask0] += 1
            spikes_num[mask1] += 1
            last_spike_t[spike[t].bool()] = t




        mask = (spikes_num < min_spikes_num)
        min_spikes_num[mask] = spikes_num[mask]
        min_spikes_num_t[mask] = last_spike_t[mask]

        # 开始求解k_positive
        v_ = v.clone()
        v_min = v_.min().item()
        v_[spike.bool()] = v_min
        last_spike_t = - torch.ones_like(v[1]) * T_in * 2
        # 初始时，认为上一次的脉冲发放时刻是- T_in * 2，这样即便在0时刻发放脉冲，其与上一个脉冲发放时刻的间隔也大于T_in

        # 遍历t，若t距离上次脉冲发放时刻的时间不超过T_in则将v_设置成v_min
        for t in range(T):
            delta_t = (t - last_spike_t)

            mask = torch.logical_and(delta_t <= T_in, (1 - spike[t]).bool())
            # 表示与上次脉冲发放时刻距离不超过T_in且当前时刻没有释放脉冲（这些位置如果之后释放了脉冲，也会被归类到上次脉冲
            # 所在的脉冲聚类里）
            v_[t][mask] = v_min

            last_spike_t[spike[t].bool()] = t

        # 反着遍历t，若t距离下次脉冲发放时刻的时间不超过T_in则将v_设置成v_min
        next_spike_t = torch.ones_like(v[1]) * T_in * 2 + T
        for t in range(T - 1, -1, -1):
            delta_t = (next_spike_t - t)

            mask = torch.logical_and(delta_t <= T_in, (1 - spike[t]).bool())
            # 表示与下次脉冲发放时刻距离不超过T_in且当前时刻没有释放脉冲（这些位置如果之后释放了脉冲，也会被归类到下次脉冲
            # 所在的脉冲聚类里）
            v_[t][mask] = v_min

            next_spike_t[spike[t].bool()] = t


        k_positive = v_.argmax(dim=0)
        k_negative = min_spikes_num_t.long()
        arrange = torch.arange(0, T, device=v.device).unsqueeze(1).repeat(1, v.shape[1])
        k_positive = (arrange == k_positive)
        k_negative = (arrange == k_negative)

        # 需要注意的是，如果脉冲聚类太密集，导致找不到符合要求的k_positive，例如脉冲为[1 0 1 1]，T_in=1，此时得到的v_在0到T均为v_min，k_positive
        # 是1，但实际上1的位置不符合k_positive的定义，因为这个位置发放脉冲后，会与已有的脉冲聚类合并，不能生成新的脉冲聚类
        # 这种情况下，v_中的所有元素均为v_min
        # 用k_positive_mask来记录，k_positive_mask==False的神经元满足这种情况，用k_positive与k_positive_mask做and操作，可以去掉这些
        # 错误的位置
        # 但是v_.max(dim=0)[0] == v_min，也就是k_positive_mask==False的神经元，在0到T时刻的v_均为v_min，只有两种情况：
        #   1.v在0到T全部过阈值，一直在发放脉冲，因此才会出现v_在0到T均为v_min，这种情况下k_positive_mask==False
        #   2.v本身在0到T均为v_min，且从来没有发放脉冲，这是一种非常极端的情况，
        #     这种情况下k_positive_mask应该为True但却被设置成False，应该修正
        k_positive_mask = (v_.max(dim=0)[0] != v_min)

        # 修正情况2
        k_positive_mask[v.max(dim=0)[0] == v_min] = True
        # 在没有这行修正代码的情况下，如果v是全0的tensor，会错误的出现k_positive为空tensor

        k_positive = torch.logical_and(k_positive, k_positive_mask)

        return N_o, k_positive, k_negative
