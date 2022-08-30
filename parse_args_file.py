import argparse
import time


def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


def parse():
    parser = argparse.ArgumentParser(
        description="DASH singlepath evaluation. Hyperparameters are set to the bestmodel.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--eval_eps", type=int, default=30000, help="Episodes to evaluate the RL algorithms")
    parser.add_argument("--schedule_type", type=str, default="NO_DUP", choices=["GREEDY", "NO_DUP", "DUPLICATE"])
    parser.add_argument("--multi-discrete", type=str2bool, default=True, help="whether to use multi-discrete")
    parser.add_argument("--manual_test", type=str2bool, default=False, help="test with a model model")
    parser.add_argument("--log_qoe", type=str2bool, default=True, help="whether to use log QoE")
    parser.add_argument("--dataset", type=str, default="LTE", choices=["LTE", "FCC"],
                        help="dataset to evaluate the algorithm")
    parser.add_argument("--seed", type=int, default=int(1), help="global random seed")
    parser.add_argument("--algorithms", type=str, choices=["DQN", "PPO", "A2C" "None"], default=["PPO"], nargs="+",
                        help="Algorithms to run. Must be DQN, A2C and PPO.")
    parser.add_argument("--other_algorithms", type=str, choices=["Random", "Constant", "Bola", "Smooth", ""], default=[""], nargs="+",
                        help="Other algorithms to run, including Random, Constant and Smooth")
    parser.add_argument("--features_dim", type=int, default=256,
                        help="final embedding vector's shape "
                             "to feed into the policy and value network. "
                             "The observation will be extracted to have shape "
                             "(1, features_dim) before feeding into the networks")

    parser.add_argument("--policy_net_arch_units", type=int, default=64,
                        help="policy network unit. Default is 64 units")
    parser.add_argument("--policy_net_arch_layers", type=int, default=2,
                        help="policy network layer. Default is 2 layers")
    parser.add_argument("--value_net_arch_units", type=int, default=128,
                        help="value network unit. Default is 128. Note that this is also the Q-Network of DQN.")
    parser.add_argument("--value_net_arch_layers", type=int, default=2,
                        help="value network layer. Default is 2 layers")

    parser.add_argument("--act_func", type=str, default="tanh", choices=["tanh", "relu"],
                        help="activation function of policy and value networks")
    parser.add_argument("--n_envs", type=int, default=1,
                        help="number of parallel envs, use in A2C and PPO")
    parser.add_argument("--use_wandb", type=str2bool, default=True,
                        help="whether to use wandb online")
    parser.add_argument("--wandb_key", type=str, default="08b340c93b3947482fd16cbd380e08e234ef5fc0")
    parser.add_argument("--wandb_entity", type=str, default="ntnghia_iu")
    parser.add_argument("--wandb_project", type=str, default="multipath")
    parser.add_argument("--wandb_runname", type=str, default="")

    # DQN params
    dqn = parser.add_argument_group("DQN parameters")
    dqn.add_argument("--dqn_lr", type=float, default=0.000519797694783514, help="DQN learning rate")
    dqn.add_argument("--dqn_buffer_size", type=int, default=7880, help="DQN max buffer size")
    dqn.add_argument("--dqn_learning_starts", type=int, default=2323, help="DQN min buffer size to start learning")
    dqn.add_argument("--dqn_batch_size", type=int, default=295, help="DQN batch size")
    dqn.add_argument("--dqn_tau", type=float, default=1.0, help="DQN Polyak update coefficient")
    dqn.add_argument("--dqn_gamma", type=float, default=0.99, help="DQN discount factor")
    dqn.add_argument("--dqn_train_freq", type=int, default=42, help="DQN update the model every dqn_train_freq steps")
    dqn.add_argument("--dqn_grad_steps", type=int, default=2, help="DQN gradient step each rollout. "
                                                                   "-1 means to do as many grad steps as "
                                                                   "steps done in the rollout.")
    dqn.add_argument("--dqn_target_update_interval", type=int, default=92,
                     help="update the target network every this step.")
    dqn.add_argument("--dqn_exploration_fraction", type=float, default=0.4513628403209479 ,
                     help="fraction of the training period over which the exploration rate is reduced")
    dqn.add_argument("--dqn_exploration_initial_eps", type=float, default=1.0, help="initial exploration rate")
    dqn.add_argument("--dqn_exploration_final_eps", type=float, default=0.05, help="final exploration rate")
    dqn.add_argument("--dqn_max_grad_norm", type=float, default=10.0, help="max value for gradient clipping")

    a2c = parser.add_argument_group("A2C parameters")
    a2c.add_argument("--a2c_lr", type=float, default=3.308224595892156e-05, help="A2C learning rate")
    a2c.add_argument("--a2c_n_steps", type=int, default=192, help="A2C num steps to run for each env each update")
    a2c.add_argument("--a2c_gamma", type=float, default=0.99, help="A2C discount factor")
    a2c.add_argument("--a2c_gae_lambda", type=float, default=1.0, help="A2C GAE lambda")
    a2c.add_argument("--a2c_ent_coef", type=float, default=0.000000001,
                     help="A2C entropy coef. Increase to encourage exploration")
    a2c.add_argument("--a2c_vf_coef", type=float, default=0.32790564221824087, help="A2C value function coef")
    a2c.add_argument("--a2c_max_grad_norm", type=float, default=0.5, help="A2C max value for grad clip")
    a2c.add_argument("--a2c_rmsprop", type=str2bool, default=True, help="whether to use RMSProp or Adam as optimizer")
    a2c.add_argument("--a2c_norm_adv", type=str2bool, default=True, help="whether to use normalized advantage")

    # PPO parameters
    ppo = parser.add_argument_group("PPO parameters")
    ppo.add_argument("--ppo_lr", type=float, default=1.0676995458551102e-05, help="PPO learning rate")
    ppo.add_argument("--ppo_n_steps_coef", type=int, default=5,
                     help="PPO n_steps coef to run for each env per update: n_steps = ppo_n_steps * ppo_batch_size")
    ppo.add_argument("--ppo_batch_size", type=int, default=575, help="PPO batch size")
    ppo.add_argument("--ppo_n_epochs", type=int, default=10, help="PPO num epochs to run each update")
    ppo.add_argument("--ppo_gamma", type=float, default=1, help="PPO reward discount factor")
    ppo.add_argument("--ppo_gae_lambda", type=float, default=0.95, help="PPO GAE lambda coef")
    ppo.add_argument("--ppo_clip_range", type=float, default=0.3, help="PPO clip range")
    ppo.add_argument("--ppo_clip_range_vf", type=float, default=None, help="PPO Value function clipping parameter")
    ppo.add_argument("--ppo_ent_coef", type=float, default=0.00001, help="PPO entropy coef")
    ppo.add_argument("--ppo_vf_coef", type=float, default=0.30567628766494326, help="PPO value function coef")
    ppo.add_argument("--ppo_max_grad_norm", type=float, default=0.5, help="PPO Gradient clipping value")

    return parser