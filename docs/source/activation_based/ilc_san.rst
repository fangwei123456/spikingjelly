使用层内连接增强的脉冲行动器网络进行连续动作空间下的强化学习
====================================
本教程作者：\ `Ding Chen <https://github.com/lucifer2859>`__

本节教程将介绍如何使用替代梯度方法训练一个层内连接增强的脉冲行动器网络。

从头搭建一个 `层内连接增强的脉冲行动器网络 <https://ieeexplore.ieee.org/document/10423179>`_
-------------------------

我们使用TD3算法将层内连接增强的脉冲行动器网络（ILC-SAN）与深度行动器网络进行协调训练。ILC-SAN首先采用群体编码器将状态编码为对应的脉冲序列，然后经由骨干SNN处理后输入到群体解码器中，得到最终的连续动作，其具体代码如下：

.. code-block:: python

    class PopSpikeActor(nn.Module):
        def __init__(self, obs_dim, act_dim, enc_pop_dim, dec_pop_dim, hidden_sizes,
                     mean_range, std, spike_ts, encode, decode, act_limit):
            super().__init__()
            self.act_limit = act_limit
            self.obs_dim = obs_dim
            self.act_dim = act_dim
            self.spike_ts = spike_ts

            if encode == 'pop-det':
                self.encoder = encoding.PopSpikeEncoderDeterministic(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
            elif encode == 'pop-ran':
                self.encoder = encoding.PopSpikeEncoderRandom(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
            else: # 'pop-raw'
                self.encoder = encoding.PopEncoder(obs_dim, enc_pop_dim, spike_ts, mean_range, std)
            self.snn = SpikeMLP(obs_dim * enc_pop_dim, act_dim, dec_pop_dim, hidden_sizes)
            self.decoder = PopDecoder(act_dim, dec_pop_dim, spike_ts, decode)

        def forward(self, obs):
            in_pop_vals = self.encoder(obs)
            out_pop_spikes = self.snn(in_pop_vals)
            return self.act_limit * torch.tanh(self.decoder(out_pop_spikes))

群体编码器采用可学习的高斯感受野，根据后续的处理方式不同可以分为三类，其中采用确定性编码的群体编码器如下：

.. code-block:: python

    class PopSpikeEncoderDeterministic(nn.Module):
        """ Learnable Population Coding Spike Encoder with Deterministic Spike Trains"""

        def __init__(self, obs_dim, pop_dim, spike_ts, mean_range, std):
            super().__init__()
            self.obs_dim = obs_dim
            self.pop_dim = pop_dim
            self.encoder_neuron_num = obs_dim * pop_dim
            self.spike_ts = spike_ts

            # Compute evenly distributed mean and variance
            tmp_mean = torch.zeros(1, obs_dim, pop_dim)
            delta_mean = (mean_range[1] - mean_range[0]) / (pop_dim - 1)
            for num in range(pop_dim):
                tmp_mean[0, :, num] = mean_range[0] + delta_mean * num
            tmp_std = torch.zeros(1, obs_dim, pop_dim) + std

            self.mean = nn.Parameter(tmp_mean)
            self.std = nn.Parameter(tmp_std)

            self.neurons = neuron.IFNode(v_threshold=0.999, v_reset=None, surrogate_function=surrogate.DeterministicPass(), detach_reset=True)

            functional.set_step_mode(self, step_mode='m')
            functional.set_backend(self, backend='torch')

        def forward(self, obs):
            obs = obs.view(-1, self.obs_dim, 1)

            # Receptive Field of encoder population has Gaussian Shape
            pop_act = torch.exp(-(1. / 2.) * (obs - self.mean).pow(2) / self.std.pow(2)).view(-1, self.encoder_neuron_num)
            pop_act = pop_act.unsqueeze(0).repeat(self.spike_ts, 1, 1)

            return self.neurons(pop_act)

其中脉冲神经元选择软重置的IF神经元，替代函数这里选择\ ``surrogate.DeterministicPass``\。

骨干SNN采用基于电流的LIF神经元，其具体代码如下：

.. code-block:: python

    class SpikeMLP(nn.Module):
        def __init__(self, in_pop_dim, act_dim, dec_pop_dim, hidden_sizes):
            super().__init__()
            self.in_pop_dim = in_pop_dim
            self.out_pop_dim = act_dim * dec_pop_dim
            self.act_dim = act_dim
            self.hidden_sizes = hidden_sizes
            self.hidden_num = len(hidden_sizes)
            
            # Define Layers
            hidden_layers = OrderedDict([
                ('Linear0', layer.Linear(in_pop_dim, hidden_sizes[0])),
                (neuron_type + '0', neuron.CLIFNode(surrogate_function=surrogate.Rect()))
            ])
            if self.hidden_num > 1:
                for hidden_layer in range(1, self.hidden_num):
                    hidden_layers['Linear' + str(hidden_layer)] = layer.Linear(hidden_sizes[hidden_layer-1], hidden_sizes[hidden_layer])
                    hidden_layers[neuron_type + str(hidden_layer)] = neuron.CLIFNode(surrogate_function=surrogate.Rect())

            hidden_layers['Linear' + str(self.hidden_num)] = layer.Linear(hidden_sizes[-1], self.out_pop_dim)
            hidden_layers[neuron_type + str(self.hidden_num)] = neuron.ILCCLIFNode(act_dim, dec_pop_dim, surrogate_function=surrogate.Rect())

            self.hidden_layers = nn.Sequential(hidden_layers)

            functional.set_step_mode(self, step_mode='m')
            functional.set_backend(self, backend='torch')

        def forward(self, in_pop_spikes):
            return self.hidden_layers(in_pop_spikes)

群体解码器采用可学习层与非脉冲神经元层，其具体代码如下：

.. code-block:: python

    class PopDecoder(nn.Module):
        """ Learnable Population Coding Decoder """
        def __init__(self, act_dim, pop_dim, spike_ts, decode='last-mem'):
            super().__init__()
            self.act_dim = act_dim
            self.pop_dim = pop_dim
            self.spike_ts = spike_ts
            self.decode = decode

            if decode == 'fr-mlp':
                self.decoder = nn.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim)
            else:
                self.decoder = nn.Sequential(
                    layer.Conv1d(act_dim, act_dim, pop_dim, groups=act_dim),
                    neuron.NonSpikingIFNode(decode=decode)
                )

                functional.set_step_mode(self, step_mode='m')
                functional.set_backend(self, backend='torch')

        def forward(self, out_pop_spikes):
            if self.decode == 'fr-mlp':
                out_pop_fr = out_pop_spikes.mean(dim=0).view(-1, self.act_dim, self.pop_dim)
                return self.decoder(out_pop_fr).view(-1, self.act_dim)

            out_pop_spikes = out_pop_spikes.view(self.spike_ts, -1, self.act_dim, self.pop_dim)
            return self.decoder(out_pop_spikes).view(-1, self.act_dim)

其中非脉冲神经元的膜电压编码方法需要通过参数\ ``decode``\ 设置。

SpikingJelly中提供了4种膜电压编码方法，用作非脉冲神经元中膜电压序列的统计量，其中\ ``last-mem``\代表最终膜电压，\ ``max-mem``\代表最大膜电压，\ ``max-abs-mem``\代表最大绝对值的膜电压，而\ ``mean-mem``\代表平均膜电压。通过这种方式，SNN可以输出任意大小的浮点值，适用于强化学习中的连续动作值。

训练ILC-SAN
-----------

首先指定好训练参数如学习率等以及若干其他配置

优化器默认使用Adam

训练代码的编写需要遵循以下三个要点：

1. 脉冲神经元的输出是二值的。因此网络需要运行一段时间，即使用\ ``T``\ 个时刻后非脉冲神经元的膜电压统计量作为决策依据。

2. ILC-SAN的损失函数与TD3算法相同。

3. 每次网络仿真结束后，需要\ **重置**\ 网络状态

ILC-SAN的完整代码位于\ ``activation_based.examples.ILC-SAN``\。

.. code-block:: shell

    usage: hybrid_td3_cuda_norm.py [--env ENV] [--encoder_pop_dim ENC_POP_DIM] [--decoder_pop_dim DEC_POP_DIM] [--encoder_var ENC_VAR] [--start_model_idx IDX] [--num_model N] [--epochs E] [--device_id DEVICE_ID] [--root_dir ROOT_DIR] [--encode ENC] [--decode DEC]

    Solve the continuous control tasks from OpenAI Gym

    options:
      --env ENV                         the continuous control tasks from OpenAI Gym
      --encoder_pop_dim ENC_POP_DIM     the input population sizes per state dimension
      --decoder_pop_dim DEC_POP_DIM     the size of output populations corresponding to each action dimension
      --encoder_var ENC_VAR             the initial standard deviation of Gaussian receptive fields for the population encoder
      --start_model_idx IDX             the start index of the model for training
      --num_model N                     the number of models for training
      --epochs E                        the number of training epochs per model
      --device_id DEVICE_ID             the cuda ID of training device, e.g., "0" or "1"
      --root_dir ROOT_DIR               the directory for storing files
      --encode ENC                      the type of population encoder, e.g., "pop-det", "pop-ran" or "pop-raw"
      --decode DEC                      the type of population decoder, e.g., "fr-mlp", "last-mem", "max-mem", "max-abs-mem" or "mean-mem"

需要注意的是，训练这样的SNN，所需显存数量与仿真时长 ``T`` 线性相关，更长的 ``T`` 相当于使用更小的仿真步长，训练更为“精细”，但训练效果不一定更好。\ ``T``
太大时，SNN在时间上展开后会变成一个非常深的网络，这将导致BPTT计算梯度时容易衰减或爆炸。

训练结果
--------

详细的训练结果与分析可以参见 `相关论文 <https://ieeexplore.ieee.org/document/10423179>`_。
