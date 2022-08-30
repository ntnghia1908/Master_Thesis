from single_path import rl_agent as rl, SmoothThroughput as smooth
from old_envs import bola_basic as bola
import numpy as np
from old_envs.single_path.trainning_env import Env
import matplotlib.pyplot as plt

EVAL_EPS = 100
DEFAULT_QUALITY = 1

net_env = Env()
state = net_env.reset()

# evaluation for policy gradient agent require model/SGD.h5 to be generated
agent = rl.policy_gradient_agent()
agent.get_dist(np.array([state]))
agent.load_model()

ep = 0

agent_rewards = []
total_reward = 0

while True:
    if ep == EVAL_EPS:
        break

    dist = agent.get_dist(np.array([state]))
    action = agent.pick_action(dist)
    state, reward, done, sep_reward, _ = net_env.step(action)
    total_reward += reward
    if done:
        ep += 1
        agent_rewards.append(total_reward)
        total_reward = 0
        state = net_env.reset()

agent_rewards = np.array(agent_rewards)
agent_rewards = np.sort(agent_rewards)
y_axis = np.arange(1,len(agent_rewards)+1) / len(agent_rewards)
plt.plot(agent_rewards,y_axis,label = 'policy gradient',color="blue")

# evaluation for actor critic agent require model/A2C.h5 to be generated
# agent = rl.actor_critic_agent()
# agent.get_dist(np.array([state]))
# agent.load_model()

# ep = 0

# agent_rewards = []
# total_reward = 0

# while True:
#     if ep == EVAL_EPS:
#         break

#     dist = agent.get_dist(np.array([state]))
#     action = agent.pick_action(dist)
#     state, reward, done, sep_reward, _ = net_env.step(action)
#     total_reward += reward
#     if done:
#         ep += 1
#         agent_rewards.append(total_reward)
#         total_reward = 0
#         state = net_env.reset()

# # convert to cdf
# agent_rewards = np.array(agent_rewards)
# agent_rewards = np.sort(agent_rewards)
# y_axis = np.arange(1,len(agent_rewards)+1) / len(agent_rewards)
# plt.plot(agent_rewards,y_axis,label = 'actor critic',color="purple")

# evaluation for smooth_evaluate throughput
state = net_env.reset()
ep = 0
throughput_rewards = []
throughput = smooth.SmoothThroughput()

while True:
    if ep == EVAL_EPS:
        break

    action = throughput.predict(state[9:-1]*1e6) # next segment list
    state, reward, done, sep_reward, _ = net_env.step(action)
    throughput.add(state[0] * 1e6) # add network speed
    total_reward += reward
    if done:
        ep += 1
        throughput_rewards.append(total_reward)
        total_reward = 0
        state = net_env.reset()

# convert to cdf
throughput_rewards = np.array(throughput_rewards)
throughput_rewards = np.sort(throughput_rewards)
y_axis = np.arange(1,len(throughput_rewards)+1) / len(throughput_rewards)
plt.plot(throughput_rewards,y_axis,label = 'throughput',color="red")

# evaluation for bola basic
state = net_env.reset()
ep = 0
bola_rewards = []
bola_agent = bola.Bola()
bola_agent.add_data(state[7:9]*10,state[9:-1]*1e6,0)

while True:
    if ep == EVAL_EPS:
        break

    action = bola_agent.predict2()
    state, reward, done, sep_reward, delay = net_env.step(action)
    bola_agent.add_data(state[7:9]*10,state[9:-1]*1e6,delay)
    total_reward += reward
    if done:
        ep += 1
        bola_rewards.append(total_reward)
        total_reward = 0
        state = net_env.reset()

# convert to cdf
bola_rewards = np.array(bola_rewards)
bola_rewards = np.sort(bola_rewards)
y_axis = np.arange(1,len(bola_rewards)+1) / len(bola_rewards)
plt.plot(bola_rewards,y_axis,label = 'bola',color = "green")
plt.xlabel("reward")
plt.ylabel("CDF")

plt.title("Performance CDF")

plt.legend()
plt.show()
