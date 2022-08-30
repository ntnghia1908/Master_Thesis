## In version 14 of environment, we implement pretrain and real trace

from env.greedy_env_v17_04 import Env
from old.DQN import DQN
from helper import write2csv
import matplotlib.pyplot as plt

import numpy as np
np.warnings.filterwarnings('ignore', category=np.VisibleDeprecationWarning)

reward_trace = [
    ['ep', 'utility', 'switch_penalty', 'rebuffering_penalty', 'no_switch', 'rebuffering_time', 'total_reward']]


def play_video(env, TrainNet, TargetNet, epsilon, copy_step, test=False):
    total_reward = 0
    iter = 0
    end_video = False
    observations = env.reset()
    losses = list()
    history = ""
    quality_reward = 0.0
    smooth_reward = 0.0
    while not end_video:
        # action = TrainNet.get_action(observations, epsilon)
        down_id = int(env.next_segment())
        predict_quality_index = TrainNet.get_action(observations, epsilon)
        action = (down_id - env.play_id - 1) * env.QUALITY_SPACE + predict_quality_index

        prev_observations = observations

        observations, reward, end_video, line, _, _, _, _, _ = env.step(action)

        history += line
        # total_reward += reward
        if end_video:
            quality_reward = env.quality_reward_cummulate
            no_switch = env.no_switch
            buffering_time = env.rebufer_time
            smooth_reward = env.smooth_penalty_cummulate
            rebuffering_reward = env.rebuffer_penalty_cummulate
            total_reward = quality_reward - smooth_reward - rebuffering_reward

        if not test:
            exp = {'s': prev_observations,
                   'a': action,
                   'r': reward,
                   's2': observations,
                   'done': end_video}
            TrainNet.add_experience(exp)
            loss = TrainNet.train(TargetNet)
            if isinstance(loss, int):
                losses.append(loss)
            else:
                losses.append(loss.numpy())
            iter += 1
            if iter % copy_step == 0:
                TargetNet.copy_weights(TrainNet)

    details=False
    if details:
        dirname = 'evaluateRL/lr5e-05_decay0.999_bwmarkovian_markovian_batch200_env18/details_fix100-50/'
        case='realtrace_'
        write2csv(dirname + '{}play_event'.format(case), env.play_event)
        write2csv(dirname + '{}down_event'.format(case), env.down_event)
        write2csv(dirname + '{}buffer_traces'.format(case), env.buffer_size_trace)
        write2csv(dirname + '{}bw1'.format(case), env.bw1_trace)
        write2csv(dirname + '{}bw2'.format(case), env.bw2_trace)

    env.reset()
    return total_reward, losses, quality_reward, no_switch, buffering_time, smooth_reward, rebuffering_reward


def main(env):
    env_version = 'v17_04_greedy'
    gamma = 0.9
    copy_step = 25
    num_states = env.state_space()
    num_actions = env.action_space()
    hidden_units = [256, 128, 128]
    max_experiences = 10000
    min_experiences = 100
    batch_size = 200
    lr = 1e-5

    N = 20000
    total_rewards = np.array([])
    epsilon = 1
    decay = 0.9999
    min_epsilon = 0.1
    epoch = []
    avg_rewards = []
    dirname = 'v17_04_greedy/'
    file_name = f'lr{lr}_decay{decay}_{len(hidden_units)}layer_mixbw_relu'

    TrainNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)
    TargetNet = DQN(num_states, num_actions, hidden_units, gamma, max_experiences, min_experiences, batch_size, lr)

    print("ep, utility, switch_penalty, rebuffering_penalty, no_switch, rebuffering_time, total_reward")
    print(file_name)

    for n in range(N):
        epoch.append(n)
        epsilon = max(min_epsilon, epsilon * decay)
        total_reward, losses, utility, no_switch, rebuffering_time, switch_penalty, rebuffering_penalty = play_video(env, TrainNet, TargetNet, epsilon, copy_step)
        total_rewards = np.append(total_rewards, [total_reward])
        avg_reward = total_rewards[max(0, n - 100):(n + 1)].mean()
        avg_rewards.append(avg_reward)
        reward_trace.append([n, utility, switch_penalty, rebuffering_penalty, no_switch, rebuffering_time, total_reward])

        print(n, utility, switch_penalty, rebuffering_penalty, no_switch, rebuffering_time, total_reward)

        if n != 0 and n % 100 == 0:
            plt.figure(figsize=(15, 3))
            plt.title(file_name)
            plt.plot(epoch, total_rewards)
            plt.plot(epoch, avg_rewards)

            plt.savefig(dirname + 'figs/{}.png'.format(file_name))
            plt.clf()
            f2 = dirname + 'history/history_{}'.format(file_name)
            np.save(f2, [total_rewards, avg_rewards])
            TrainNet.save_model(dirname + "model/DQNmodel_{}.h5".format(file_name))


if __name__ == '__main__':
    env = Env(test=True)
    main(env)
