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

def spike_similar_loss(spikes:torch.Tensor, labels:torch.Tensor, sim_type='strict', loss_type='mse'):
    '''
    :param spikes: shape=[N, M, T]，N个数据生成的脉冲
    :param labels: shape=[N, C]，N个数据的标签，labels[i][k] == 1表示数据i属于第k类，labels[i][k] == 0则表示数据i不属于第k类，允许多标签
    :param sim_type: 如何定义脉冲的“相似”，可以为'strict'或'relaxed'
    :param loss_type: 返回哪种损失，可以为'mse', 'l1', 'bce'
    :return:
    shape=[1]的tensor，相似损失

    将N个数据输入到输出层有M个神经元的SNN，运行T步，得到shape=[N, M, T]的脉冲。这N个数据的标签为shape=[N, C]的labels。

    用shape=[N, N]的矩阵sim表示相似矩阵，sim[i][j] == 1表示数据i与数据j相似，sim[i][j] == 0表示数据i与数据j不相似。若\\
    labels[i]与labels[j]共享至少同一个标签，则认为他们相似，否则不相似。

    用shape=[N, N]的矩阵sim_p表示脉冲相似矩阵，sim_p[i][j]的取值为0到1，值越大表示数据i与数据j的脉冲越相似。

    当sim_type == 'strict'时，数据i的脉冲 :math:`s_{i}` 和数据j的脉冲 :math:`s_{j}` 的相似度计算按照

    .. math::
        sim_{p_{ij}} = \\frac{1}{2}(\\frac{\\sum_{m=0}^{M-1} \\sum_{t=0}^{T-1}(2s_{i,m,t} - 1)(2s_{j,m,t} - 1)}{MT} + 1)

    当sim_type == 'relaxed'时，数据i的脉冲 :math:`s_{i}` 和数据j的脉冲 :math:`s_{j}` 的相似度计算按照

    .. math::
        sim_{p_{ij}} = \\frac{\\sum_{m=0}^{M-1} \\sum_{t=0}^{T-1}s_{i,m,t}s_{j,m,t}}{\\sqrt{|s_{i}||s_{j}|} + \\epsilon}

    其中 :math:`\\epsilon` 是一个很小的正数，可以为1e-6，防止出现除以0导致的数值不稳定。

    .. tip::
        将脉冲看作是一维的向量。

        'strict'其实是将脉冲从0,1线性变换到-1,1，然后求解两个脉冲向量的夹角余弦值。再将取值范围为[-1,1]的余弦值线性变换到[0,1]。

        'relaxed'是直接求解两个脉冲向量的夹角余弦值，由于脉冲向量的元素都为0或1，因此不会出现负余弦值，求解出的余弦值范围均为[0,1]。

    对于相似的数据，根据输入的loss_type，返回度量sim与sim_p差异的损失。

    loss_type == 'mse'时，返回sim与sim_p的均方误差（也就是l2误差）。

    loss_type == 'l1'时，返回sim与sim_p的l1误差。

    loss_type == 'bce'时，返回sim与sim_p的二值交叉熵误差。

    '''

    spikes = spikes.flatten(start_dim=1)
    if sim_type == 'strict':
        spikes = spikes * 2 - 1  # 0 1变换到-1 1
        sim_p = spikes.mm(spikes.t()) / spikes.shape[1]
        # shape=[N, N] sim[i][j]表示i与j的输出脉冲乘积，可以表示相似度。1表示完全相同，-1表示完全不相同
        # / spikes.shape[1]是为了归一化
        sim_p = (sim_p + 1) / 2  # -1 1变换到0 1
    elif sim_type == 'relaxed':
        spikes_norm2 = spikes.norm(dim=1, keepdim=True)  # shape=[N, 1]，表示N个脉冲的2范数（向量的长度）
        sim_p = spikes.mm(spikes.t()) / (spikes_norm2.mm(spikes_norm2.t()) + 1e-6)  # + 1e-6防止出现除以0
    else:
        raise NotImplementedError

    labels = labels.float()
    sim = labels.mm(labels.t()).clamp_max(1)  # labels.mm(labels.t())[i][j]位置的元素表现输入数据i和数据数据j有多少个相同的标签
    # 将大于1的元素设置为1，因为共享至少同一个标签，就认为他们相似

    if loss_type == 'mse':
        return F.mse_loss(sim_p, sim)
    elif loss_type == 'l1':
        return F.l1_loss(sim_p, sim)
    elif loss_type == 'bce':
        return F.binary_cross_entropy(sim_p, sim)
    else:
        raise NotImplementedError