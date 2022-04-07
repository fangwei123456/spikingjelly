import gym
import math
import random
import numpy as np
from collections import namedtuple
from itertools import count

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T

from torch.utils.tensorboard import SummaryWriter
import argparse


class ReplayMemory(object):
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


# DQN algorithm
class DQN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DQN, self).__init__()

        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        return self.fc2(x)

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='DQN State')
    parser.add_argument('--seed', type=int, default=1,
                        help='random seed (default: 1)')
    parser.add_argument('--use-cuda', default=False,
                        help='use cuda or not (default: False)')

    args = parser.parse_args()

    env_name = 'CartPole-v0'

    random.seed(args.seed)
    np.random.seed(args.seed)

    torch.cuda.manual_seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device("cuda" if args.use_cuda else "cpu")

    writer = SummaryWriter(log_dir='./log')

    env = gym.make(env_name).unwrapped
    env.seed(args.seed)

    # Replay Memory
    Transition = namedtuple('Transition',
                            ('state', 'action', 'next_state', 'reward'))



    # Hyperparameters and utilitie
    BATCH_SIZE = 128
    GAMMA = 0.999
    EPS_START = 0.9
    EPS_END = 0.05
    EPS_DECAY = 200
    TARGET_UPDATE = 10

    num_episodes = 500
    hidden_size = 256

    n_states = env.observation_space.shape[0]
    n_actions = env.action_space.n

    print('State Num: %d, Action Num: %d' % (n_states, n_actions))

    policy_net = DQN(n_states, hidden_size, n_actions).to(device)
    target_net = DQN(n_states, hidden_size, n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters())
    memory = ReplayMemory(10000)

    steps_done = 0

    def select_action(state):
        global steps_done
        sample = random.random()
        eps_threshold = EPS_END + (EPS_START - EPS_END) * \
            math.exp(-1. * steps_done / EPS_DECAY)
        steps_done += 1
        if sample > eps_threshold:
            with torch.no_grad():
                return policy_net(state).max(1)[1].view(1, 1)
        else:
            return torch.tensor([[random.randrange(n_actions)]], device=device, dtype=torch.long)


    # Training loop
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
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * GAMMA) + reward_batch

        # Compute Huber loss
        loss = F.smooth_l1_loss(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        optimizer.zero_grad()
        loss.backward()
        for param in policy_net.parameters():
            param.grad.data.clamp_(-1, 1)
        optimizer.step()


    # Train
    for i_episode in range(num_episodes):
        # Initialize the environment and state
        env.reset()
        state = torch.zeros([1, n_states], dtype=torch.float, device=device)

        total_reward = 0

        for t in count():
            # Select and perform an action
            action = select_action(state)
            next_state, reward, done, _ = env.step(action.item())
            next_state = torch.from_numpy(next_state).float().to(device).unsqueeze(0)
            total_reward += reward
            reward = torch.tensor([reward], device=device)

            if done:
                next_state = None

            memory.push(state, action, next_state, reward)

            state = next_state

            optimize_model()

            if done:
                print(f'Episode: {i_episode}, Reward: {total_reward}')
                writer.add_scalar('DQN-state-' + env_name + '/Reward', total_reward, i_episode)
                break

        # Update the target network, copying all weights and biases in DQN
        if i_episode % TARGET_UPDATE == 0:
            target_net.load_state_dict(policy_net.state_dict())

    print('Complete')

    writer.close()