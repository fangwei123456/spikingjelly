import gym
import math
import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.distributions import Normal

from torch.utils.tensorboard import SummaryWriter
from spikingjelly.clock_driven.examples.common.multiprocessing_env import SubprocVecEnv


# Use CUDA
use_cuda = torch.cuda.is_available()
device   = torch.device('cuda' if use_cuda else 'cpu')

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


# Neural Network
class ActorCritic(nn.Module):
    def __init__(self, num_inputs, num_outputs, hidden_size, std=0.0):
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
        )

        self.log_std = nn.Parameter(torch.ones(1, num_outputs) * std)

    def forward(self, x):
        value = self.critic(x)
        mu = self.actor(x)
        std = self.log_std.exp().expand_as(mu)
        dist = Normal(mu, std)
        return dist, value

if __name__ == '__main__':
    envs = [make_env() for i in range(num_envs)]
    envs = SubprocVecEnv(envs)

    env = gym.make(env_name)
    env.seed(seed)


    
    def test_env(vis=False):
        state = env.reset()
        if vis: env.render()
        done = False
        total_reward = 0
        while not done:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            dist, _ = model(state)
            next_state, reward, done, _ = env.step(torch.max(dist.sample(), 1)[1].cpu().numpy()[0])
            state = next_state
            if vis: env.render()
            total_reward += reward
        return total_reward

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

    num_inputs  = envs.observation_space.shape[0]
    num_outputs = env.action_space.n

    print('State Num: %d, Action Num: %d' % (num_inputs, num_outputs))

    # Hyper params:
    hidden_size      = 32
    lr               = 1e-3
    num_steps        = 128
    mini_batch_size  = 256
    ppo_epochs       = 30

    model = ActorCritic(num_inputs, num_outputs, hidden_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    max_steps = 10000
    step_idx  = 0

    state = envs.reset()

    writer = SummaryWriter(logdir='./log')

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
                writer.add_scalar('PPO-' + env_name + '/Reward', test_reward, step_idx)

        next_state = torch.FloatTensor(next_state).to(device)
        _, next_value = model(next_state)
        returns = compute_gae(next_value, rewards, masks, values)

        returns   = torch.cat(returns).detach()
        log_probs = torch.cat(log_probs).detach()
        values    = torch.cat(values).detach()
        states    = torch.cat(states)
        actions   = torch.cat(actions)
        advantage = returns - values

        ppo_update(ppo_epochs, mini_batch_size, states, actions, log_probs, returns, advantage)

    print('----------------------------')
    print('Complete')

    writer.close()