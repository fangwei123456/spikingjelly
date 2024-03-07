from copy import deepcopy
import itertools
import numpy as np
import random
import gym
import pickle
import os

import torch
import torch.nn as nn
from torch.optim import Adam

from spikingjelly.activation_based import functional

from replay_buffer_norm import ReplayBuffer
from noisysan import NoisyPopSpikeActor
from core_cuda import MLPQFunction
from torch.utils.tensorboard import SummaryWriter


# Online noise reduction for noisy networks
K_FINAL = 1.0
CUR_R = 0

SUP_R = {
    'Ant-v3': 6000,
    'HalfCheetah-v3': 12000,
    'Hopper-v3': 4000,
    'Walker2d-v3': 6000,
    'Humanoid-v3': 6000,
    'HumanoidStandup-v2': 180000,
    'InvertedDoublePendulum-v2': 10000,
    'BipedalWalker-v3': 320,
}

INF_R = {
    'Ant-v3': 0,
    'HalfCheetah-v3': 0,
    'Hopper-v3': 0,
    'Walker2d-v3': 0,
    'Humanoid-v3': 0,
    'HumanoidStandup-v2': 0,
    'InvertedDoublePendulum-v2': 0,
    'BipedalWalker-v3': 0,
}


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


class SpikeActorDeepCritic(nn.Module):
    def __init__(self, observation_space, action_space, encoder_pop_dim, decoder_pop_dim, 
                 mean_range, std, spike_ts, beta, sigma_init, hidden_sizes=(256, 256), activation=nn.ReLU):
        super().__init__()
        obs_dim = observation_space.shape[0]
        act_dim = action_space.shape[0]
        act_limit = action_space.high[0]
        # build policy and value functions
        self.san = NoisyPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes, 
                                      mean_range, std, spike_ts, act_limit, beta, sigma_init)
            
        self.q1 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)
        self.q2 = MLPQFunction(obs_dim, act_dim, hidden_sizes, activation)

    def act(self, obs, use_noise=True):
        with torch.no_grad():
            action = self.san.act(obs).cpu().numpy() if not use_noise else self.san(obs).cpu().numpy()
            functional.reset_net(self.san)
            return action


def hybrid_td3(env_name, actor_critic=SpikeActorDeepCritic, ac_kwargs=dict(), seed=0,
               steps_per_epoch=10000, epochs=100, replay_size=int(1e6), gamma=0.99,
               polyak=0.995, san_lr=1e-4, q_lr=1e-3, batch_size=100, start_steps=10000,
               update_after=1000, update_every=50, act_noise=0.1, target_noise=0.2,
               noise_clip=0.5, policy_delay=2, num_test_episodes=10, max_ep_len=1000,
               save_freq=5, norm_clip_limit=3, norm_update=50, tb_comment='', model_idx=0, 
               root_dir='.'):
    global CUR_R
    
    # Set device
    device = torch.device('cuda')

    # Set random seed
    setup_seed(seed)

    env, test_env = gym.make(env_name), gym.make(env_name)
    obs_dim = env.observation_space.shape
    act_dim = env.action_space.shape[0]

    # Action limit for clamping: critically, assumes all dimensions share the same bound!
    act_limit = env.action_space.high[0]

    # Create actor-critic module and target networks
    ac = actor_critic(env.observation_space, env.action_space, **ac_kwargs)
    ac_targ = deepcopy(ac)
    ac.to(device)
    ac_targ.to(device)

    # Freeze target networks with respect to optimizers (only update via polyak averaging)
    for p in ac_targ.parameters():
        p.requires_grad = False
        
    # List of parameters for both Q-networks (save this for convenience)
    q_params = itertools.chain(ac.q1.parameters(), ac.q2.parameters())

    cn_length = ac.san.get_colored_noise_length()

    # Experience buffer
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=replay_size,
                                 clip_limit=norm_clip_limit, norm_update_every=norm_update,
                                 spike_ts=ac_kwargs['spike_ts'], cn_length=cn_length)

    # Set up function for computing TD3 Q-losses
    def compute_loss_q(data):
        o, a, r, o2, d = data['obs'], data['act'], data['rew'], data['obs2'], data['done']

        q1 = ac.q1(o, a)
        q2 = ac.q2(o, a)

        # Bellman backup for Q functions
        with torch.no_grad():
            san_targ = ac_targ.san.act(o2)
            functional.reset_net(ac_targ.san)

            # Target policy smoothing
            epsilon = torch.randn_like(san_targ) * target_noise
            epsilon = torch.clamp(epsilon, -noise_clip, noise_clip)
            a2 = san_targ + epsilon
            a2 = torch.clamp(a2, -act_limit, act_limit)

            # Target Q-values
            q1_san_targ = ac_targ.q1(o2, a2)
            q2_san_targ = ac_targ.q2(o2, a2)
            q_san_targ = torch.min(q1_san_targ, q2_san_targ)
            backup = r + gamma * (1 - d) * q_san_targ

        # MSE loss against Bellman backup
        loss_q1 = ((q1 - backup)**2).mean()
        loss_q2 = ((q2 - backup)**2).mean()
        loss_q = loss_q1 + loss_q2

        # Useful info for logging
        loss_info = dict(Q1Vals=q1.cpu().detach().numpy(),
                         Q2Vals=q2.cpu().detach().numpy())

        return loss_q, loss_info

    # Set up function for computing TD3 san loss
    def compute_loss_san(data):
        o = data['obs']
        ac.san.load_colored_noise(data['cn'])
        q1_san = ac.q1(o, ac.san(o))
        functional.reset_net(ac.san)

        noise_sigma = ac.san.get_noise_sigma()
        k = K_FINAL * max(0, CUR_R - INF_R[env_name]) / (SUP_R[env_name] - INF_R[env_name])

        return -q1_san.mean() + k * torch.pow(noise_sigma, 2)

    # Set up optimizers for policy and q-function
    san_optimizer = Adam(ac.san.parameters(), lr=san_lr)
    q_optimizer = Adam(q_params, lr=q_lr)

    def update(data, timer):
        # First run one gradient descent step for Q1 and Q2
        q_optimizer.zero_grad()
        loss_q, loss_info = compute_loss_q(data)
        loss_q.backward()
        q_optimizer.step()

        # Possibly update pi and target networks
        if timer % policy_delay == 0:
            # Freeze Q-networks so you don't waste computational effort computing gradients for them during the policy learning step.
            for p in q_params:
                p.requires_grad = False

            # Next run one gradient descent step for san.
            san_optimizer.zero_grad()
            loss_san = compute_loss_san(data)
            loss_san.backward()
            san_optimizer.step()

            # Unfreeze Q-networks so you can optimize it at next DDPG step.
            for p in q_params:
                p.requires_grad = True

            # Finally, update target networks by polyak averaging.
            with torch.no_grad():
                for p, p_targ in zip(ac.parameters(), ac_targ.parameters()):
                    # We use an in-place operations "mul_", "add_" to update target params, 
                    # as opposed to "mul" and "add", which would make new tensors.
                    p_targ.data.mul_(polyak)
                    p_targ.data.add_((1 - polyak) * p.data)

    def get_action(o, use_noise=True):
        return ac.act(torch.as_tensor(o, dtype=torch.float32, device=device), use_noise=use_noise)

    def test_agent():
        # Compuate the return mean test reward
        test_reward_sum = 0
        for j in range(num_test_episodes):
            o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
            while not(d or (ep_len == max_ep_len)):
                # Take deterministic actions at test time (noise_scale=0)
                a = get_action(replay_buffer.normalize_obs(o), use_noise=False)
                o, r, d, _ = test_env.step(a[0]) if len(a.shape) > 1 else test_env.step(a)
                ep_ret += r
                ep_len += 1
            test_reward_sum += ep_ret
        return test_reward_sum / num_test_episodes

    # Save rewards also create dir for saving parameters
    save_test_reward = []
    save_test_reward_steps = []
    try:
        os.mkdir(root_dir + '/params')
        print('Directory params Created')
    except FileExistsError:
        print('Directory params already exists')
    model_dir = root_dir + '/params/hybrid-td3_' + tb_comment
    try:
        os.mkdir(model_dir)
        print('Directory ', model_dir, ' Created')
    except FileExistsError:
        print('Directory ', model_dir, ' already exists')

    tb_dir = model_dir + '/tb/' + str(seed)
    writer = SummaryWriter(log_dir=tb_dir)

    # Prepare for interaction with environment
    total_steps = steps_per_epoch * epochs
    o, ep_ret, ep_len = env.reset(), 0, 0
    max_test_reward = 0
    
    ac.san.reset_noise(max_ep_len + 1)

    # Main loop: collect experience in env and update/log each epoch
    for t in range(total_steps):
        if t <= start_steps:
            a = env.action_space.sample()
        else:
            a = get_action(replay_buffer.normalize_obs(o), use_noise=True)

        # Step the env
        o2, r, d, _ = env.step(a[0]) if len(a.shape) > 1 else env.step(a)
        ep_ret += r
        ep_len += 1

        # Ignore the "done" signal if it comes from hitting the time horizon (that is,
        # when it's an artificial terminal signal that isn't based on the agent's state)
        d = False if ep_len == max_ep_len else d

        # Store experience to replay buffer
        replay_buffer.store(o, a, r, o2, d, ac.san.get_colored_noise())

        # Super critical, easy to overlook step: make sure to update most recent observation!
        o = o2

        # End of trajectory handling
        if d or (ep_len == max_ep_len):
            o, ep_ret, ep_len = env.reset(), 0, 0
            ac.san.reset_noise(max_ep_len + 1)

        # Update handling
        if t >= update_after and t % update_every == 0:
            for j in range(update_every):
                batch = replay_buffer.sample_batch(device, batch_size)
                update(data=batch, timer=j)

            ac.san.cancel_load()

        # End of epoch handling
        if (t+1) % steps_per_epoch == 0:
            epoch = (t+1) // steps_per_epoch

            # Test the performance of the deterministic version of the agent
            test_mean_reward = test_agent()
            save_test_reward.append(test_mean_reward)
            save_test_reward_steps.append(t + 1)
            print('Model: ', model_idx, ' Steps: ', t + 1, ' Mean Reward: ', test_mean_reward)

            CUR_R = test_mean_reward

            # Save model
            if epoch % save_freq == 0:
                if test_mean_reward > max_test_reward:
                    ac.san.to('cpu')
                    torch.save(ac.san.state_dict(), model_dir + '/model' + str(model_idx) + '_best.pt')
                    ac.san.to(device)
                    max_test_reward = test_mean_reward
                    pickle.dump([replay_buffer.mean, replay_buffer.var], open(model_dir + '/model' + str(model_idx) + '_best_mean_var.p', 'wb+'))
                    print('Weights saved in ', model_dir + '/model' + str(model_idx) + '_best.pt')

            if epoch == epochs:
                ac.san.to('cpu')
                torch.save(ac.san.state_dict(), model_dir + '/model' + str(model_idx) + '_last.pt')
                ac.san.to(device)
                pickle.dump([replay_buffer.mean, replay_buffer.var], open(model_dir + '/model' + str(model_idx) + '_last_mean_var.p', 'wb+'))
                print('Weights saved in ', model_dir + '/model' + str(model_idx) + '_last.pt')

            noise_sigma = ac.san.get_noise_sigma()
            writer.add_scalar(tag='nsn', scalar_value=noise_sigma.item(), global_step=epoch)

    # Save Test Reward List
    pickle.dump([save_test_reward, save_test_reward_steps], open(model_dir + '/model' + str(model_idx) + '_test_rewards.p', 'wb+'))
    writer.close()


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--start_model_idx', type=int, default=0)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--beta', type=float, default=0.0, help='the exponent for the power-spectrum of the generated noise')
    parser.add_argument('--sigma_init', type=float, default=0.5, help='the sigma for the generated noise')
    args = parser.parse_args()

    START_MODEL = args.start_model_idx
    NUM_MODEL = args.num_model
    AC_KWARGS = dict(hidden_sizes=[256, 256],
                     encoder_pop_dim=args.encoder_pop_dim,
                     decoder_pop_dim=args.decoder_pop_dim,
                     mean_range=(-3, 3),
                     std=math.sqrt(args.encoder_var),
                     spike_ts=5,
                     beta=args.beta,
                     sigma_init=args.sigma_init)

    COMMENT = args.env + '_NoisySAN_B' + str(args.beta) + '_S' + str(args.sigma_init)
    
    for num in range(START_MODEL, START_MODEL + NUM_MODEL):
        seed = num * 10
        hybrid_td3(env_name=args.env, actor_critic=SpikeActorDeepCritic, ac_kwargs=AC_KWARGS,
                   san_lr=1e-4, gamma=0.99, seed=seed, epochs=args.epochs, norm_clip_limit=3.0, 
                   tb_comment=COMMENT, model_idx=num, root_dir=args.root_dir)