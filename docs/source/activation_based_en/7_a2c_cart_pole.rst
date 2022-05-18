Reinforcement Learning: Advantage Actor Critic (A2C)
=============================================================
Author: `lucifer2859 <https://github.com/lucifer2859>`_

Translator: `LiutaoYu <https://github.com/LiutaoYu>`_

This tutorial applies a spiking neural network to reproduce `actor-critic.py <https://github.com/lucifer2859/Policy-Gradients/blob/master/actor-critic.py>`_.
Please make sure that you have read the original tutorial and corresponding codes before proceeding.

Here, we apply the same method as the previous DQN tutorial to make SNN output floating numbers.
We set the firing threshold of a neuron to be infinity, which won't fire at all, and we adopt the final membrane potential to represent Q function. It is convenient to implement such neurons in the ``SpikingJelly`` framework: just inherit everything from LIF neuron ``neuron.LIFNode`` and rewrite its ``forward`` function.

.. code-block:: python

    class NonSpikingLIFNode(neuron.LIFNode):
        def forward(self, dv: torch.Tensor):
            self.neuronal_charge(dv)
            # self.neuronal_fire()
            # self.neuronal_reset()
            return self.v


The basic structure of the Spiking Actor-Critic Network is very simple: input layer, IF neuron layer, and NonSpikingLIF neuron layer,
between which are fully linear connections.
The IF neuron layer is an encoder to convert the CartPole's state variables to spikes,
and the NonSpikingLIF neuron layer can be regraded as the decision making unit.

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


Training the network
---------------------------
The code of this part is almost the same with the ANN version.
But note that the SNN version here adopts ``Observation`` returned by ``env`` as the network input.

Following is the training code of the SNN version.
During the training process, we will save the model parameters responsible for the largest reward.

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

It should be emphasized here that, we need to ``reset`` the network after each forward process,
because SNN is retentive while each trial should be started with a clean network state.

The integrated script can be found here `clock_driven/examples/Spiking_A2C.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/Spiking_A2C.py>`_.
And we can start the training process in a Python Console as follows.

.. code-block:: python

    >>> python Spiking_A2C.py

Performance comparison between ANN and SNN
------------------------------------------------------
Here is the reward curve during the training process of 1e5 episodes:

.. image:: ../_static/tutorials/clock_driven/\7_a2c_cart_pole/Spiking-A2C-CartPole-v0.*
    :width: 100%

And here is the result of the ANN version with the same settings.
The integrated code can be found here `clock_driven/examples/A2C.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/A2C.py>`_.

.. image:: ../_static/tutorials/clock_driven/\7_a2c_cart_pole/A2C-CartPole-v0.*
    :width: 100%
