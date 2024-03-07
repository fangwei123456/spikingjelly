import numpy as np
import torch
import gym
import pickle
import os

from spikingjelly.activation_based import functional

from replay_buffer_norm import ReplayBuffer
from noisysan import NoisyPopSpikeActor


def test_mujoco_render(san_model_file, mean_var_file, env_fn, encoder_pop_dim, decoder_pop_dim, 
                       mean_range, std, spike_ts, beta, sigma_init, hidden_sizes=(256, 256), norm_clip_limit=3):
    # Set device
    device = torch.device('cuda')
    # Set environment
    test_env = env_fn()
    obs_dim = test_env.observation_space.shape[0]
    act_dim = test_env.action_space.shape[0]
    act_limit = test_env.action_space.high[0]

    # Replay buffer for running z-score norm
    b_mean_var = pickle.load(open(mean_var_file, 'rb'))
    replay_buffer = ReplayBuffer(obs_dim=obs_dim, act_dim=act_dim, size=1, clip_limit=norm_clip_limit, norm_update_every=1)
    replay_buffer.mean = b_mean_var[0]
    replay_buffer.var = b_mean_var[1]

    # SAN
    san = NoisyPopSpikeActor(obs_dim, act_dim, encoder_pop_dim, decoder_pop_dim, hidden_sizes, mean_range, std, 
                             spike_ts, act_limit, beta, sigma_init)
    
    san.load_state_dict(torch.load(san_model_file))
    san.to(device)

    def get_action(o):
        a = san.act(torch.as_tensor(o, dtype=torch.float32, device=device)).cpu().numpy()
        functional.reset_net(san)
        return a

    # Start testing
    o, d, ep_ret, ep_len = test_env.reset(), False, 0, 0
    while not (d or (ep_len == 1000)):
        # test_env.render()
        with torch.no_grad():
            a = get_action(replay_buffer.normalize_obs(o))
        o, r, d, _ = test_env.step(a[0]) if len(a.shape) > 1 else test_env.step(a)
        ep_ret += r
        ep_len += 1

    return ep_ret


if __name__ == '__main__':
    import math
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument('--env', type=str, default='HalfCheetah-v3')
    parser.add_argument('--encoder_pop_dim', type=int, default=10)
    parser.add_argument('--decoder_pop_dim', type=int, default=10)
    parser.add_argument('--encoder_var', type=float, default=0.15)
    parser.add_argument('--num_model', type=int, default=10)
    parser.add_argument('--root_dir', type=str, default='.')
    parser.add_argument('--beta', type=float, default=0.0, help='the exponent for the power-spectrum of the generated noise')
    parser.add_argument('--sigma_init', type=float, default=0.5, help='the sigma for the generated noise')
    parser.set_defaults(noisy_nsn=True, noise_reduction=False)
    args = parser.parse_args()
    
    mean_range = (-3, 3)
    std = math.sqrt(args.encoder_var)
    spike_ts = 5

    file_dir = args.root_dir + '/params/hybrid-td3_' + args.env + '_NoisySAN_B' + str(args.beta) + '_S' + str(args.sigma_init)

    reward_list = []

    for model_idx in range(args.num_model):
        # Best epoch reward during training
        test_reward, _ = pickle.load(open(file_dir + '/model' + str(model_idx) + '_test_rewards.p', 'rb'))
        best_epoch_reward = 0
        for idx in range(20):
            if test_reward[(idx + 1) * 5 - 1] > best_epoch_reward:
                best_epoch_reward = test_reward[(idx + 1) * 5 - 1]

        best_rewards = [max(test_reward)]

        # Test Reward (last epoch idx)
        model_file_dir = file_dir + '/model' + str(model_idx) + '_last.pt'
        buffer_file_dir = file_dir + '/model' + str(model_idx) + '_last_mean_var.p'
        
        reward = test_mujoco_render(model_file_dir, 
                                    buffer_file_dir, 
                                    lambda : gym.make(args.env), 
                                    args.encoder_pop_dim, 
                                    args.decoder_pop_dim, 
                                    mean_range, std, 
                                    spike_ts, args.beta,
                                    args.sigma_init)

        best_rewards.append(reward)

        # Test Reward (best epoch idx)
        model_file_dir = file_dir + '/model' + str(model_idx) + '_best.pt'
        buffer_file_dir = file_dir + '/model' + str(model_idx) + '_best_mean_var.p'
        
        reward = test_mujoco_render(model_file_dir, 
                                    buffer_file_dir, 
                                    lambda : gym.make(args.env), 
                                    args.encoder_pop_dim, 
                                    args.decoder_pop_dim, 
                                    mean_range, std, 
                                    spike_ts, args.beta,
                                    args.sigma_init)

        best_rewards.append(reward)

        print('Model', model_idx, ', Reward:', best_rewards)

        reward_list.append(max(best_rewards))

    print('------------------------------------------------')
    print('mean: ', np.mean(reward_list))
    print('std: ', np.std(reward_list, ddof=1))