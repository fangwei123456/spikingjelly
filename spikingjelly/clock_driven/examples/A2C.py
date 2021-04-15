import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Categorical

from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven.examples.common.multiprocessing_env import SubprocVecEnv

if __name__ == '__main__':
    # Use CUDA
    use_cuda = torch.cuda.is_available()
    device = torch.device('cuda' if use_cuda else 'cpu')

    # Set Seed
    seed = 1

    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


    # Create Environments
    num_envs = 4
    env_name = 'CartPole-v0'

    def make_env():
        def _thunk():
            env = gym.make(env_name)
            env.seed(seed)
            return env

        return _thunk

    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    env = gym.make(env_name)
    env.seed(seed)

    # Neural Network
    class ActorCritic(nn.Module):
        def __init__(self, num_inputs, num_outputs, hidden_size):
            super(ActorCritic, self).__init__()

            self.critic = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, 1)
            )

            self.actor = nn.Sequential(
                nn.Linear(num_inputs, hidden_size),
                nn.ReLU(),
                nn.Linear(hidden_size, num_outputs),
                nn.Softmax(dim=1),
            )

        def forward(self, x):
            value = self.critic(x)
            probs = self.actor(x)
            dist  = Categorical(probs)
            return dist, value


    def test_env(vis=False):
        state = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            next_state, reward, done, _ = env.step(dist.sample().cpu().numpy()[0])
            state = next_state
            if vis: env.render()
            total_reward += reward
        return total_reward


    # A2C: Synchronous Advantage Actor Critic
    def compute_returns(next_value, rewards, masks, gamma=0.99):
        R = next_value
        returns = []
        for step in reversed(range(len(rewards))):
            R = rewards[step] + gamma * R * masks[step]
            returns.insert(0, R)
        return returns

    num_inputs  = envs.observation_space.shape[0]
    num_outputs = envs.action_space.n

    print('State Num: %d, Action Num: %d' % (num_inputs, num_outputs))

    # Hyper params:
    hidden_size = 256
    lr = 3e-4
    num_steps = 5

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_steps = 100000
    step_idx = 0

    state = envs.reset()

    writer = SummaryWriter(logdir='./log')

    while step_idx < max_steps:

        log_probs = []
        values    = []
        rewards   = []
        masks     = []
        entropy = 0

        for _ in range(num_steps):
            state = torch.FloatTensor(state).to(device)
            dist, value = model(state)

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
                writer.add_scalar('A2C-multi_env-' + env_name + '/Reward', test_reward, step_idx)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
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

    # print(test_env(True))
    print('----------------------------')
    print('Complete')

    writer.close()