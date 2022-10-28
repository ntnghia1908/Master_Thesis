from menv_baseline import Env
from utility.helper import *
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
from utils_schedule_newstate import *
from sb3_contrib.common.maskable.evaluation import get_action_masks
from sb3_contrib.ppo_mask import MaskablePPO
from wandb_callback_bestModel import WandbCallback_bestModel

import parse_args_file

import datetime
import pickle
import torch
import time

NUM_STEP_PER_EP = Env.CHUNK_TIL_VIDEO_END_CAP - 1
device = "cuda" if torch.cuda.is_available() else "cpu"

parser = parse_args_file.parse()
args = parser.parse_args()

class BaseEvaluator:
    def __init__(self, schedule_type, EVAL_EPS, log_qoe=True, bitrate_lists=None, replace=True, multi_discrete=False, n_envs=1):
        self.env = Env(schedule_type=schedule_type, bitrate_lists=bitrate_lists, replace=replace,
                       multi_discrete=multi_discrete, train=True, debug_mode=False)
        self.start_time = time.time()
        self.log_dir= "tmp/PPOEvaluator"+str(self.start_time)
        os.makedirs(self.log_dir, exist_ok=True)
        self.env = Monitor(env=self.env, filename=self.log_dir)
        self.env.reset()
        self.env_kwargs = dict(
            log_qoe=log_qoe, bitrate_lists=bitrate_lists, replace=replace, schedule_type=schedule_type, train=True,
            multi_discrete=multi_discrete
        )
        self.n_envs = n_envs
        self.schedule_type = schedule_type
        self.multi_discrete = multi_discrete
        self.M_IN_K = self.env.M_IN_K
        self.HISTORY_SIZE = self.env.HISTORY_SIZE
        self.SEGMENT_SPACE = self.env.SEGMENT_SPACE
        self.QUALITY_SPACE = self.env.QUALITY_SPACE
        self.CHUNK_TIL_VIDEO_END = self.env.CHUNK_TIL_VIDEO_END_CAP
        self.VIDEO_CHUNK_LEN = self.env.VIDEO_CHUNK_LEN
        self.SCALE = 0.9
        self.EVAL_EPS = EVAL_EPS
        self.n_envs= n_envs
        self.print_template = "{0:^20}|{1:^20}|{2:^20}|{3:^20}|{4:^20}|{5:^20}|{6:^20}\n"
        self.write_template = "{},{},{},{},{},{},{}\n"

        def predict(self, *args):
            pass

        def evaluate(self, *args):
            pass

class PPOEvaluator(BaseEvaluator):
    name = "PPO"
    def evaluate(self, seed, params):
        self.logger = configure("./loggerPPO/", ["stdout", "csv", "tensorboard"])
        self.model = MaskablePPO("MultiInputPolicy", self.env, verbose=1, **params,
                             tensorboard_log=f"runs/{time.time()}PPO")
        self.model.set_logger(logger=self.logger)
        self.model.learn(total_timesteps=NUM_STEP_PER_EP*self.EVAL_EPS,
                         callback=WandbCallback_bestModel(verbose=1,
                                                model_save_path=f"./models/PPOEvaluator_{str(self.start_time)}_{args.schedule_type}_seed{args.seed}",
                                                log_dir=self.log_dir, model_save_freq=1000))
        end_time = time.time()
        seconds_elapsed = end_time - self.start_time
        print(f"Time elapsed: {str(datetime.timedelta(seconds=seconds_elapsed))}")


def test(model_path, file_ouput, bwTest1, bwTest2):
    env = Env(schedule_type=Env.SCHEDULE.NO_DUPLICATE, bitrate_lists=[bwTest1, bwTest2], replace=None,
                   multi_discrete=True, train=False, debug_mode=False)
    info_keywords = ("quality_reward_cummulate", "smooth_penalty_cummulate", "rebuffer_penalty_cummulate", "sum_reward")
    env_test = Monitor(env=env, info_keywords=info_keywords)
    test_agent = MaskablePPO.load(path=model_path, env=env)
    # test_agent.set_env(env=env_test)

    test_result = []
    for eps in range(min(len(bwTest1), len(bwTest2))):
        observations = env_test.reset()
        done = False
        while not done:
            action_masks = get_action_masks(env_test)
            predicted_action, states = test_agent.predict(observations, action_masks=action_masks, deterministic=True)
            observations, reward, done, info = env_test.step(predicted_action)

        epi_utility = info["quality_reward_cummulate"]
        epi_switch_penalty = info["smooth_penalty_cummulate"]
        epi_rebuffering_penalty = info["rebuffer_penalty_cummulate"]
        epi_reward = info["sum_reward"]

        # print(eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty)
        test_result.append([eps, epi_reward, epi_utility, epi_switch_penalty, epi_rebuffering_penalty])

    twodlist2csv(file_ouput, test_result)
    result = np.array(test_result).T
    REWARD_COL = 1
    return result[REWARD_COL ,:].mean(), result[REWARD_COL ,:].std()


if __name__ == '__main__':
    model_path = "./models/PPOEvaluator_seed18_1656295100.3684726/model.zip"

    result_path_dir= "results/PPOEvaluator_seed18_1656295100.3684726/bestmodel"
    if not os.path.exists(result_path_dir):
        os.makedirs(result_path_dir)

    print("Evaluate 20-20")
    bwTest1 = pickle.load(open("../bw/bw20Test.pickle", 'rb'))
    bwTest2 = pickle.load(open("../bw/bw20Test.pickle", 'rb'))
    random.shuffle(bwTest2)
    avgTest2020, stdTest2020 = test(model_path = model_path,
         file_ouput=os.path.join(result_path_dir, '20_20test'),
         bwTest1=bwTest1, bwTest2=bwTest2)
    print("TEST 2020:", avgTest2020, stdTest2020)

    print("Evaluate 20-15")
    bwTest1= pickle.load(open("../bw/bw20Test.pickle", 'rb'))
    bwTest2= pickle.load(open("../bw/bw15Test.pickle", 'rb'))
    avgTest2015, stdTest2015 = test(model_path= model_path,
         file_ouput= os.path.join(result_path_dir, '20_15test'),
         bwTest1=bwTest1, bwTest2=bwTest2)
    print("TEST 2015:", avgTest2015, stdTest2015)

    print("Evaluate 20-10")
    bwTest1 = pickle.load(open("../bw/bw20Test.pickle", 'rb'))
    bwTest2 = pickle.load(open("../bw/bw10Test.pickle", 'rb'))
    avgTest2010, stdTest2010 = test(model_path=model_path,
         file_ouput=os.path.join(result_path_dir, '20_10test'),
         bwTest1=bwTest1, bwTest2=bwTest2)
    print("TEST 2010:", avgTest2010, stdTest2010)

    print("Evaluate 20-05")
    bwTest1 = pickle.load(open("../bw/bw20Test.pickle", 'rb'))
    bwTest2 = pickle.load(open("../bw/bw05Test.pickle", 'rb'))
    bwTest2 = np.concatenate([bwTest2, bwTest2, bwTest2, bwTest2])
    avgTest2005, stdTest2005 = test(model_path=model_path,
         file_ouput=os.path.join(result_path_dir, '20_05test')
         , bwTest1=bwTest1, bwTest2=bwTest2)
    print("TEST 2005:", avgTest2005, stdTest2005)

    avg_all_test = (avgTest2005 + avgTest2010 + avgTest2015 + avgTest2020) /4
    print("avg_all_test:", avg_all_test)
