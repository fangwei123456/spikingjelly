import ptan
import argparse, os
from itertools import count

import numpy as np

import torch
import torch.optim as optim

from spikingjelly.activation_based import functional

from tensorboardX import SummaryWriter

from utils import model, common, atari_wrappers

PLAY_STEPS = 4
MODEL_SAVE_FREQUENCY = 10 ** 5

# Full evaluation phases of length T (Double DQN)
EVAL_LEN = 125000


def make_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, frame_stack=True, pytorch_img=True)
    return env


def make_test_env(params):
    env = atari_wrappers.make_atari(params['env_name'])
    env = atari_wrappers.wrap_deepmind(env, episode_life=False, clip_rewards=False, frame_stack=True, pytorch_img=True)
    return env


def eval_Q(env, net, cuda):
    device = torch.device("cuda" if cuda else "cpu")

    epsilon = 0.05

    value_estimate = 0.0
    step_cnt = 0

    with torch.no_grad():
        while step_cnt < EVAL_LEN:
            obs = env.reset()
            for _ in count():
                state = torch.tensor(np.expand_dims(obs, 0)).to(device).float() / 256
                q_v = net(state)
                q = q_v.data.cpu().numpy()
                n_actions = q.shape[1]
                actions = np.argmax(q, axis=1)
                mask = np.random.random(size=1) < epsilon
                rand_actions = np.random.choice(n_actions, sum(mask))
                actions[mask] = rand_actions

                obs, reward, done, _ = env.step(actions)

                value_estimate += q.max()
                step_cnt += 1

                functional.reset_net(net)

                if done:
                    break

    return value_estimate / EVAL_LEN


def test(env, net, n_episodes, cuda):
    device = torch.device("cuda" if cuda else "cpu")

    total_reward = 0.0
    epsilon = 0.05

    with torch.no_grad():
        for episode in range(n_episodes):
            obs = env.reset()
            ep_reward = 0.0
            for _ in count():
                state = torch.tensor(np.expand_dims(obs, 0)).to(device).float() / 256
                q_v = net(state)
                q = q_v.data.cpu().numpy()
                n_actions = q.shape[1]
                actions = np.argmax(q, axis=1)
                mask = np.random.random(size=1) < epsilon
                rand_actions = np.random.choice(n_actions, sum(mask))
                actions[mask] = rand_actions

                obs, reward, done, _ = env.step(actions)

                ep_reward += reward

                functional.reset_net(net)

                if done:
                    # print("Finished Episode {} with reward {}".format(episode, ep_reward))
                    total_reward += ep_reward
                    break

    return total_reward / n_episodes


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--cuda", default=False, action="store_true", help="Enable cuda")
    parser.add_argument("--game", type=str, default="breakout", help="ATARI game (gym)")
    parser.add_argument("--T", type=int, default=8, help="The simulation time")
    parser.add_argument("--dec_type", type=str, default="max-mem", help="The type of SNN decoder, e.g. max-mem, mean-mem, max-abs-mem, last-mem, fr-mlp")
    parser.add_argument("--early_stop", default=False, action="store_true", help="Use stop reward to stop early")
    parser.add_argument("--eval_q", default=False, action="store_true", help="Record the Q-value (eval)")
    parser.add_argument("--sticky_actions", default=False, action="store_true", help="Use sticky actions")
    parser.add_argument("--frame_num", type=int, default=20*10 ** 6, help="The number of frames")
    parser.add_argument("--seed", type=int, default=12, help="Random seed to use")
    args = parser.parse_args()

    device = torch.device("cuda" if args.cuda else "cpu")

    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    game = args.game + '-sa' if args.sticky_actions else args.game
    results_dir = 'results/' + game + '/snn/seed=' + str(args.seed) + '/T=' + str(args.T) + '/' + args.dec_type

    test_writer = SummaryWriter(logdir=results_dir)

    params = common.HYPERPARAMS[game]
    params['batch_size'] *= PLAY_STEPS

    env = make_env(params)
    test_env = make_test_env(params)

    if args.seed is not None:
        env.seed(args.seed)

    suffix = "" if args.seed is None else "_seed=%s" % args.seed
    writer = SummaryWriter(comment="-" + params['run_name'] + "-dsqn%s" % (suffix))
    net = model.DSQN(env.observation_space.shape, env.action_space.n, args.T, args.dec_type, args.cuda).to(device)
    net_ = model.DSQN(env.observation_space.shape, env.action_space.n, args.T, args.dec_type, args.cuda).to(device)
    tgt_net = ptan.agent.TargetNet(net, net_)
    selector = ptan.actions.EpsilonGreedyActionSelector(epsilon=params['epsilon_start'])
    epsilon_tracker = common.EpsilonTracker(selector, params)
    agent = ptan.agent.DQNAgent(net, selector, device=device)

    exp_source = ptan.experience.ExperienceSourceFirstLast(env, agent, gamma=params['gamma'], steps_count=1)
    buffer = ptan.experience.ExperienceReplayBuffer(exp_source, buffer_size=params['replay_size'])
    optimizer = optim.Adam(net.parameters(), lr=params['learning_rate'])

    frame_idx = 0
    max_frame_idx = 0
    max_reward = -float('Inf')

    backup_path = os.path.join(results_dir, 'dsqn_' + args.game + '_backup.pth')
    reload_flag = os.path.exists(backup_path)

    if reload_flag:
        print('Reloading!')
        checkpoint = torch.load(backup_path)
        net.load_state_dict(checkpoint['net'])
        tgt_net.sync()
        optimizer.load_state_dict(checkpoint['optimizer'])   
        frame_idx = checkpoint['frame_idx']
        max_frame_idx = checkpoint['max_frame_idx']
        max_reward = checkpoint['max_reward']

    print('------------------------------------------------')
    print('Frame: %d / %d' % (frame_idx, args.frame_num))
    print('Max Frame Idx: ', max_frame_idx , ', Max Reward: ', max_reward)
    print('------------------------------------------------')

    with common.RewardTracker(writer, args.frame_num) as reward_tracker:
        while True:
            frame_idx += PLAY_STEPS
            buffer.populate(PLAY_STEPS)
            epsilon_tracker.frame(frame_idx)

            new_rewards = exp_source.pop_total_rewards()
            if new_rewards:
                if reward_tracker.reward(new_rewards[0], frame_idx, selector.epsilon):
                    break

            if len(buffer) < params['replay_initial']:
                if reload_flag:
                    frame_idx -= PLAY_STEPS
                continue

            optimizer.zero_grad()
            batch = buffer.sample(params['batch_size'])
            loss_v = common.calc_loss_dqn(batch, net, tgt_net.target_model, gamma=params['gamma'], cuda=args.cuda, cuda_async=True)
            loss_v.backward()
            optimizer.step()

            functional.reset_net(net)
            functional.reset_net(tgt_net.target_model)

            if frame_idx % params['target_net_sync'] < PLAY_STEPS:
                tgt_net.sync()

            if frame_idx % MODEL_SAVE_FREQUENCY == 0:
                # Save the best-performance model
                test_reward = test(test_env, net, n_episodes=30, cuda=args.cuda)
                if test_reward > max_reward:
                    max_reward = test_reward
                    max_frame_idx = frame_idx
                    torch.save(net.state_dict(), os.path.join(results_dir, 'dsqn_' + args.game + '.pth'))
                
                print('------------------------------------------------')
                test_writer.add_scalar(game + '/reward', test_reward, frame_idx)
                print('F: %d, Max-F: %d, R: %.1f, Max-R: %.1f' % (frame_idx, max_frame_idx, test_reward, max_reward))
                
                if args.eval_q:
                    eval_q = eval_Q(env, net, cuda=args.cuda)
                    print('Eval-Q: %.1f' % (eval_q))
                    test_writer.add_scalar(game + '/eval_q', eval_q, frame_idx)

                print('------------------------------------------------')

                # Backup
                backup_state = {'net': net.state_dict(), 
                                'optimizer': optimizer.state_dict(),
                                'frame_idx': frame_idx,
                                'max_frame_idx': max_frame_idx,
                                'max_reward': max_reward}

                backup_path = os.path.join(results_dir, 'dsqn_' + args.game + '_backup.pth')
                torch.save(backup_state, backup_path)

                if args.early_stop and test_reward > params['stop_reward']:
                    print("Solved in %d frames!" % frame_idx)
                    break

    # Backup
    backup_state = {'net': net.state_dict(), 
                    'optimizer': optimizer.state_dict(),
                    'frame_idx': frame_idx,
                    'max_frame_idx': max_frame_idx,
                    'max_reward': max_reward}

    backup_path = os.path.join(results_dir, 'dsqn_' + args.game + '_backup.pth')
    torch.save(backup_state, backup_path)

    print('Max Frame Idx: ', max_frame_idx , ', Max Reward: ', max_reward)

    test_writer.close()