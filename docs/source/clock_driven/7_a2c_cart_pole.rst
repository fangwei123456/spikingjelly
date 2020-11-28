强化学习A2C
=======================================
本教程作者：`lucifer2859 <https://github.com/lucifer2859>`_

本节教程使用SNN重新实现 `actor-critic.py <https://github.com/lucifer2859/Policy-Gradients/blob/master/actor-critic.py>`_。
请确保你已经阅读了原版代码以及相关论文，因为本教程是对原代码的扩展。

状态输入
同DQN一样我们使用另一种常用的使SNN输出浮点值的方法：将神经元的阈值设置成无穷大，使其不发放脉冲，用神经元最后时刻的电压作为输出值。神经元实现这
种神经元非常简单，只需要继承已有神经元，重写 ``forward`` 函数即可。LIF神经元的电压不像IF神经元那样是简单的积分，因此我们使用LIF
神经元来改写：

.. code-block:: python

    class NonSpikingLIFNode(neuron.LIFNode):
        def forward(self, dv: torch.Tensor):
            self.neuronal_charge(dv)
            # self.neuronal_fire()
            # self.neuronal_reset()
            return self.v

接下来，搭建我们的Spiking Actor-Critic Network，网络的结构非常简单，全连接-IF神经元-全连接-NonSpikingLIF神经元，全连接-IF神经元起到
编码器的作用，而全连接-NonSpikingLIF神经元则可以看作一个决策器：

.. code-block:: python

    class ActorCritic(nn.Module):
        def __init__(self, num_inputs, num_outputs, hidden_size, T=16):
            super(ActorCritic, self).__init__()
            
            self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                neuron.IFNode(),
                nn.Linear(hidden_size, 1),
                NonSpikingLIFNode(tau=2.0)
            )
            
            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                neuron.IFNode(),
                nn.Linear(hidden_size, num_outputs),
                NonSpikingLIFNode(tau=2.0)
            )

            self.T = T
            
        def forward(self, x):
            for t in range(self.T):
                self.critic(x)
                self.actor(x)
            value = self.critic[-1].v
            probs = F.softmax(self.actor[-1].v, dim=1)
            dist  = Categorical(probs)

            return dist, value


训练网络
--------------------
训练部分的代码，与ANN版本几乎相同，使用env返回的Observation作为输入。

SNN的训练代码如下，我们会保存训练过程中使得奖励最大的模型参数：

.. code-block:: python

    while step_idx < max_steps:

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            functional.reset_net(model)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(action.cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            state = next_state
            step_idx += 1
            
            if step_idx % 1000 == 0:
                test_reward = test_env()
                print('Step: %d, Reward: %.2f' % (step_idx, test_reward))
                writer.add_scalar('Spiking-A2C-multi_env-' + env_name + '/Reward', test_reward, step_idx)
                
        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        functional.reset_net(model)
        returns = compute_returns(next_value, rewards, masks)
        
        log_probs = torch.cat(log_probs)
        returns   = torch.cat(returns).detach()
        values    = torch.cat(values)

        advantage = returns - values

        actor_loss  = - (log_probs * advantage.detach()).mean()
        critic_loss = advantage.pow(2).mean()

        loss = actor_loss + 0.5 * critic_loss - 0.001 * entropy

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

另外一个需要注意的地方是，SNN是有状态的，因此每次前向传播后，不要忘了将网络 ``reset``。

完整的代码可见于 `clock_driven/examples/Spiking_A2C.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/Spiking_A2C.py>`_。可以从命令行直接启动训练：

.. code-block:: python

    >>> python Spiking_A2C.py

ANN与SNN的性能对比
---------------------------
训练1e5个步骤的性能曲线：

.. image:: ../_static/tutorials/clock_driven/\7_a2c_cart_pole/Spiking-A2C-CartPole-v0.*
    :width: 100%

用相同处理方式的ANN训练1e5个步骤的性能曲线(完整的代码可见于 `clock_driven/examples/A2C.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/A2C.py>`_)：

.. image:: ../_static/tutorials/clock_driven/\7_a2c_cart_pole/A2C-CartPole-v0.*
    :width: 100%