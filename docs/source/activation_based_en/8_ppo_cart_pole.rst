Reinforcement Learning: Proximal Policy Optimization (PPO)
===============================================================
Author: `lucifer2859 <https://github.com/lucifer2859>`_

Translator: `LiutaoYu <https://github.com/LiutaoYu>`_

This tutorial applies a spiking neural network to reproduce `ppo.py <https://github.com/lucifer2859/Policy-Gradients/blob/master/ppo.py>`_.
Please make sure that you have read the original tutorial and corresponding codes before proceeding.

Here, we apply the same method as the previous DQN tutorial to make SNN output floating numbers.
We set the firing threshold of a neuron to be infinity, which won't fire at all, and we adopt the final membrane potential to represent Q function.
It is convenient to implement such neurons in the ``SpikingJelly`` framework: just inherit everything from LIF neuron ``neuron.LIFNode`` and rewrite the ``forward`` function.

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
        def __init__(self, num_inputs, num_outputs, hidden_size, T=16, std=0.0):
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

            self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

            self.T = T
            
        def forward(self, x):
            for t in range(self.T):
                self.critic(x)
                self.actor(x)
            value = self.critic[-1].v
            mu    = self.actor[-1].v
            std   = self.log_std.exp().expand_as(mu)
            dist  = Normal(mu, std)
            return dist, value


Training the network
---------------------------
The code of this part is almost the same with the ANN version.
But note that the SNN version here adopts ``Observation`` returned by ``env`` as the network input.

Following is the training code of the SNN version.
During the training process, we will save the model parameters responsible for the largest reward.

.. code-block:: python

    # GAE
    def compute_gae(next_value, rewards, masks, values, gamma=0.99, tau=0.95):
        values = values + [next_value]
        gae = 0
        returns = []
        for step in reversed(range(len(rewards))):
            delta = rewards[step] + gamma * values[step + 1] * masks[step] - values[step]
            gae = delta + gamma * tau * masks[step] * gae
            returns.insert(0, gae + values[step])
        return returns

    # Proximal Policy Optimization Algorithm
    # Arxiv: "https://arxiv.org/abs/1707.06347"
    def ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantage):
        batch_size = states.size(0)
        ids = np.random.permutation(batch_size)
        ids = np.split(ids[:batch_size // mini_batch_size * mini_batch_size], batch_size // mini_batch_size)
        for i in range(len(ids)):
            yield states[ids[i], :], actions[ids[i], :], log_probs[ids[i], :], returns[ids[i], :], advantage[ids[i], :]

    def ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantages, clip_param=0.2):
        for _ in range(ppo_epochs):
            for state, action, old_log_probs, return_, advantage in ppo_iter(mini_batch_size, states, actions, log_probs, returns, advantages):
                dist, value = model(state)
                functional.reset_net(model)
                entropy = dist.entropy().mean()
                new_log_probs = dist.log_prob(action)

                ratio = (new_log_probs - old_log_probs).exp()
                surr1 = ratio * advantage
                surr2 = torch.clamp(ratio, 1.0 - clip_param, 1.0 + clip_param) * advantage

                actor_loss  = - torch.min(surr1, surr2).mean()
                critic_loss = (return_ - value).pow(2).mean()

                loss = 0.5 * critic_loss + actor_loss - 0.001 * entropy

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

    while step_idx < max_steps:

        log_probs = []
        values    = []
        states    = []
        actions   = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)
            functional.reset_net(model)

            action = dist.sample()
            next_state, reward, done, _ = envs.step(torch.max(action, 1)[1].cpu().numpy())

            log_prob = dist.log_prob(action)
            entropy += dist.entropy().mean()
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(torch.FloatTensor(reward).unsqueeze(1).to(device))
            masks.append(torch.FloatTensor(1 - done).unsqueeze(1).to(device))
            
            states.append(state)
            actions.append(action)
            
            state = next_state
            step_idx += 1
            
            if step_idx % 100 == 0:
                test_reward = test_env()
                print('Step: %d, Reward: %.2f' % (step_idx, test_reward))
                writer.add_scalar('Spiking-PPO-' + env_name + '/Reward', test_reward, step_idx)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        functional.reset_net(model)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values
        
        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)


It should be emphasized here that, we need to ``reset`` the network after each forward process,
because SNN is retentive while each trial should be started with a clean network state.

The integrated script can be found here `clock_driven/examples/Spiking_PPO.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/Spiking_PPO.py>`_.
And we can start the training process in a Python Console as follows.

.. code-block:: python

    >>> python Spiking_PPO.py

Performance comparison between ANN and SNN
------------------------------------------------------
Here is the reward curve during the training process of 1e5 episodes:

.. image:: ../_static/tutorials/clock_driven/\8_ppo_cart_pole/Spiking-PPO-CartPole-v0.*
    :width: 100%

And here is the result of the ANN version with the same settings.
The integrated code can be found here `clock_driven/examples/PPO.py <https://github.com/fangwei123456/spikingjelly/blob/master/spikingjelly/clock_driven/examples/PPO.py>`_.

.. image:: ../_static/tutorials/clock_driven/\8_ppo_cart_pole/PPO-CartPole-v0.*
    :width: 100%