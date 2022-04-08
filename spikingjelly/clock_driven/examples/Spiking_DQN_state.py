import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count
import matplotlib
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from spikingjelly.clock_driven import neuron, functional
import os

from torch.utils.tensorboard import SummaryWriter

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))

class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class NonSpikingLIFNode(neuron.LIFNode):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, dv: torch.Tensor):
        self.neuronal_charge(dv)
        # self.neuronal_fire()
        # self.neuronal_reset()
        return self.v


# Spiking DQN algorithm
class DQSN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, T=16):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            neuron.IFNode(),
            nn.Linear(hidden_size, output_size),
            NonSpikingLIFNode(tau=2.0)
        )

        self.T = T

    def forward(self, x):
        for t in range(self.T):
            self.fc(x)
            
        return self.fc[-1].v


def train(use_cuda, model_dir, log_dir, env_name, hidden_size, num_episodes, seed):
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    T = 16

    random.seed(seed)
    np.random.seed(seed)

    torch.cuda.manual_seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if use_cuda else "cpu")

    steps_done = 0

    writer = SummaryWriter(logdir=log_dir)

    env = gym.make(env_name).unwrapped
    env.seed(seed)

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQSN(n_states, hidden_size, n_actions, T).to(device)
    target_net = DQSN(n_states, hidden_size, n_actions, T).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    def select_action(state, steps_done):
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
                        math.exp(-1. * steps_done / EPS_DECAY)
        if sample > eps_threshold:
            with torch.no_grad():
                ac = policy_net(state).max(1)[1].view(1, 1)
                functional.reset_net(policy_net)
                return ac
        else:
            return torch.tensor([[random.randrange(env.action_space.n)]], device=device, dtype=torch.long)

    def optimize_model():
        if len(memory) < BATCH_SIZE:
            return

        transitions = memory.sample(BATCH_SIZE)

        batch = Transition(*zip(*transitions))

        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                                batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        state_action_values = policy_net(state_batch).gather(1, action_batch)

        next_state_values = torch.zeros(BATCH_SIZE, device=device)
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1)[0].detach()
        functional.reset_net(target_net)
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            if param.grad is not None:
                param.grad.data.clamp_(-1, 1)
        optimizer.step()
        functional.reset_net(policy_net)

    max_reward = 0
    max_pt_path = os.path.join(model_dir, f'policy_net_{hidden_size}_max.pt')
    pt_path = os.path.join(model_dir, f'policy_net_{hidden_size}.pt')

    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = torch.zeros([1, n_states], dtype=torch.float, device=device)

        total_reward = 0

        for t in count():
            action = select_action(state, steps_done)
            steps_done += 1
            next_state, reward, done, _ = env.step(action.item())
            total_reward += reward
            next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state
            if done and total_reward > max_reward:
                max_reward = total_reward
                torch.save(policy_net.state_dict(), max_pt_path)
                print(f'max_reward={max_reward}, save models')

            optimize_model()

            if done:
                print(f'Episode: {i_episode}, Reward: {total_reward}')
                writer.add_scalar('Spiking-DQN-state-' + env_name + '/Reward', total_reward, i_episode)
                break

        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('complete')
    torch.save(policy_net.state_dict(), pt_path)
    print('state_dict path is', pt_path)

    writer.close()


def play(use_cuda, pt_path, env_name, hidden_size, played_frames=60, save_fig_num=0, fig_dir=None, figsize=(12, 6), firing_rates_plot_type='bar', heatmap_shape=None):
    
    T = 16

    plt.rcParams['figure.figsize'] = figsize
    plt.ion()

    env = gym.make(env_name).unwrapped

    device = torch.device("cuda" if use_cuda else "cpu")

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    policy_net = DQSN(n_states, hidden_size, n_actions, T).to(device)
    policy_net.load_state_dict(torch.load(pt_path, map_location=device))

    env.reset()
    state = torch.zeros([1, n_states], dtype=torch.float, device=device)

    with torch.no_grad():
        functional.set_monitor(policy_net, True)
        delta_lim = 0
        over_score = 1e9

        for i in count():
            LIF_v = policy_net(state)  # shape=[1, 2]
            action = LIF_v.max(1)[1].view(1, 1).item()

            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (1, 0), colspan=3)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (1, 0))

            plt.xticks(np.arange(2), ('Left', 'Right'))
            plt.ylabel('Voltage')
            plt.title('Voltage of LIF neurons at last time step')
            delta_lim = (LIF_v.max() - LIF_v.min()) * 0.5
            plt.ylim(LIF_v.min() - delta_lim, LIF_v.max() + delta_lim)
            plt.yticks([])
            plt.text(0, LIF_v[0][0], str(round(LIF_v[0][0].item(), 2)), ha='center')
            plt.text(1, LIF_v[0][1], str(round(LIF_v[0][1].item(), 2)), ha='center')

            plt.bar(np.arange(2), LIF_v.squeeze(), color=['r', 'gray'] if action == 0 else ['gray', 'r'], width=0.5)
            
            if LIF_v.min() - delta_lim < 0:
                plt.axhline(0, color='black', linewidth=0.1)

            IF_spikes = np.asarray(policy_net.fc[1].monitor['s'])  # shape=[16, 1, 256]
            firing_rates = IF_spikes.mean(axis=0).squeeze()
            
            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (0, 4), rowspan=2, colspan=5)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (0, 1), rowspan=2, colspan=2)
            
            plt.title('Firing rates of IF neurons')

            if firing_rates_plot_type == 'bar':
                # 绘制柱状图
                plt.xlabel('Neuron index')
                plt.ylabel('Firing rate')
                plt.xlim(0, firing_rates.size)
                plt.ylim(0, 1.01)
                plt.bar(np.arange(firing_rates.size), firing_rates, width=0.5)

            elif firing_rates_plot_type == 'heatmap':
                # 绘制热力图
                heatmap = plt.imshow(firing_rates.reshape(heatmap_shape), vmin=0, vmax=1, cmap='ocean')
                plt.gca().invert_yaxis()
                cbar = heatmap.figure.colorbar(heatmap)
                cbar.ax.set_ylabel('Magnitude', rotation=90, va='top')
            
            functional.reset_net(policy_net)
            subtitle = f'Position={state[0][0].item(): .2f}, Velocity={state[0][1].item(): .2f}, Pole Angle={state[0][2].item(): .2f}, Pole Velocity At Tip={state[0][3].item(): .2f}, Score={i}'

            state, reward, done, _ = env.step(action)
            
            if done:
                over_score = min(over_score, i)
                subtitle = f'Game over, Score={over_score}'
            plt.suptitle(subtitle)

            state = torch.from_numpy(state).float().to(device).unsqueeze(0)
            screen = env.render(mode='rgb_array').copy()
            screen[300, :, :] = 0  # 画出黑线
            
            if firing_rates_plot_type == 'bar':
                plt.subplot2grid((2, 9), (0, 0), colspan=3)
            elif firing_rates_plot_type == 'heatmap':
                plt.subplot2grid((2, 3), (0, 0))
            
            plt.xticks([])
            plt.yticks([])
            plt.title('Game screen')
            plt.imshow(screen, interpolation='bicubic')
            plt.pause(0.001)
            
            if i < save_fig_num:
                plt.savefig(os.path.join(fig_dir, f'{i}.png'))
            
            if done and i >= played_frames:
                env.close()
                plt.close()
                break

    '''
    train(use_cuda=False, model_dir='./model/CartPole-v0/state', log_dir='./log', env_name='CartPole-v0', \
            hidden_size=256, num_episodes=500, seed=1)
    '''

    play(use_cuda=False, pt_path='./model/CartPole-v0/policy_net_256_max.pt', env_name='CartPole-v0', \
            hidden_size=256, played_frames=300)