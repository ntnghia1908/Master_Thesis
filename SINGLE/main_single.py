import argparse
import shutil
import time

from evaluator_contrib_single import *
# from evaluator import *
from utility.helper import *
from utils_schedule_newstate import *
import parse_args_file
from feature_extractor_single import PensieveFeatureExtractor
import pickle


def main(seed):
    if not os.path.isdir("../results"):
        os.mkdir("../results")
    else:
        shutil.rmtree("../results")
        os.mkdir("../results")

    parser = parse_args_file.parse()
    args = parser.parse_args()
    if args.use_wandb:
        os.environ["WANDB_MODE"] = "online"
        wandb.login(key=args.wandb_key)
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, sync_tensorboard=True)
        wandb.run.name = args.wandb_runname
        wandb.config.update(args)
    else:
        wandb.init(project=args.wandb_project, entity=args.wandb_entity, sync_tensorboard=True)
        os.environ["WANDB_MODE"] = "offline"

    set_global_seed(seed)

    if args.act_func == "tanh":
        activation_fn = torch.nn.modules.activation.Tanh
    elif args.act_func == "relu":
        activation_fn = torch.nn.modules.activation.ReLU
    else:
        raise NotImplementedError("Agent network activation function is not supported.")

    if args.schedule_type == "GREEDY":
        schedule_type=Schedule.GREEDY
    elif args.schedule_type == "NO_DUP":
        schedule_type=Schedule.NO_DUPLICATE
    elif args.schedule_type == "DUPLICATE":
        schedule_type=Schedule.DUPLICATE
    else:
        raise NotImplementedError("Env not support schedule_type")

    on_policy_kwargs = dict(
        features_extractor_class=PensieveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=[dict(pi=[args.policy_net_arch_units] * args.policy_net_arch_layers,
                       vf=[args.value_net_arch_units] * args.value_net_arch_layers)],
        activation_fn=activation_fn
    )

    off_policy_kwargs = dict(
        features_extractor_class=PensieveFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=args.features_dim),
        net_arch=[args.value_net_arch_units] * args.value_net_arch_layers,
        activation_fn=activation_fn
    )

    A2C_params = dict(
        policy_kwargs=on_policy_kwargs,
        learning_rate=args.a2c_lr,
        n_steps=args.a2c_n_steps,
        gamma=args.a2c_gamma,
        gae_lambda=args.a2c_gae_lambda,
        ent_coef=args.a2c_ent_coef,
        vf_coef=args.a2c_vf_coef,
        max_grad_norm=args.a2c_max_grad_norm,
        use_rms_prop=args.a2c_rmsprop,
        normalize_advantage=args.a2c_norm_adv,
    )

    DQN_params = dict(
        policy_kwargs=off_policy_kwargs,
        learning_rate=args.dqn_lr,
        buffer_size=args.dqn_buffer_size,
        learning_starts=args.dqn_learning_starts,
        batch_size=args.dqn_batch_size,
        tau=args.dqn_tau,
        gamma=args.dqn_gamma,
        train_freq=args.dqn_train_freq,
        gradient_steps=args.dqn_grad_steps,
        target_update_interval=args.dqn_target_update_interval,
        exploration_fraction=args.dqn_exploration_fraction,
        exploration_initial_eps=args.dqn_exploration_initial_eps,
        exploration_final_eps=args.dqn_exploration_final_eps,
        max_grad_norm=args.dqn_max_grad_norm
    )

    PPO_params = dict(
        policy_kwargs=on_policy_kwargs,
        learning_rate=args.ppo_lr,
        n_steps=args.ppo_n_steps_coef * args.ppo_batch_size,
        batch_size=args.ppo_batch_size,
        n_epochs=args.ppo_n_epochs,
        gamma=args.ppo_gamma,
        gae_lambda=args.ppo_gae_lambda,
        clip_range=args.ppo_clip_range,
        clip_range_vf=args.ppo_clip_range_vf,
        ent_coef=args.ppo_ent_coef,
        vf_coef=args.ppo_vf_coef,
        max_grad_norm=args.ppo_max_grad_norm,
    )

    bwlist = {
        "fccup": pickle.load(open("../bw/FCCUp10Train.pickle", 'rb')),
        "fcclow":pickle.load(open("../bw/FCCLow10Train.pickle", 'rb')),
        "lteup": pickle.load(open("../bw/LTEUp10Train.pickle", 'rb')),
        "ltelow":pickle.load(open("../bw/LTELow10Train.pickle", 'rb'))
    }

    if "PPO" in args.algorithms:
        print("\n\n***EVALUATING: PPO***\n\n")

        ppo_eval = PPOEvaluator(EVAL_EPS=args.eval_eps, log_qoe=args.log_qoe, schedule_type=schedule_type,
                                multi_discrete = args.multi_discrete, bitrate_lists=bwlist, n_envs=args.n_envs)
        if not args.manual_test:
            ppo_eval.evaluate(str(seed), params=PPO_params)
            # ppo_eval.test()
        else:
            ppo_eval.test(str(seed), manual_test=args.manual_test)


if __name__ == "__main__":
    main(1)
