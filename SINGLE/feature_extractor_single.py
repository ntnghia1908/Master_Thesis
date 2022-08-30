import torch
import torch.nn as nn
import torch.nn.functional as F
import gym
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor


class PensieveFeatureExtractor(BaseFeaturesExtractor):
    def __init__(self, observation_space: gym.spaces.Dict, features_dim: int = 256, use_feature_extractor=True):
        super(PensieveFeatureExtractor, self).__init__(observation_space, features_dim)
        if use_feature_extractor:
            self.network_speed1 = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())
            # self.network_speed2 = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())

            self.next_chunk_size = nn.Sequential(nn.Conv1d(1, 128, 7, 7), nn.ReLU(), nn.Flatten())
            # self.quality_in_buffer = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())
            self.quality_in_buffer = nn.Sequential(nn.Flatten())
            self.buffer_size = nn.Sequential(nn.Linear(1, 128), nn.ReLU(), nn.Flatten())
            self.percentage_remain_video_chunks = nn.Sequential(nn.Linear(1, 128), nn.ReLU())
            self.last_play_quality = nn.Sequential(nn.Linear(7, 128), nn.ReLU(), nn.Flatten())

            self.delay_net1 = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())
            # self.delay_net2 = nn.Sequential(nn.Conv1d(1, 128, 4, 1), nn.ReLU(), nn.Flatten())

            # self.last_path_state1=nn.Sequential(nn.Flatten())
            # self.last_path_state2=nn.Sequential(nn.Flatten())

            self.last_layer = nn.Sequential(nn.Linear(2104, self.features_dim * 2), nn.Tanh(),
                                            nn.Linear(self.features_dim * 2, self.features_dim), nn.Tanh())
        # else:
        #     self.network_speed1 = nn.Sequential(nn.Flatten())
        #     self.network_speed2 = nn.Sequential(nn.Flatten())
        #
        #     self.next_chunk_size = nn.Sequential(nn.Flatten())
        #     self.quality_in_buffer = nn.Sequential(nn.Flatten())
        #     self.buffer_size = nn.Sequential(nn.Flatten())
        #     self.percentage_remain_video_chunks = nn.Sequential()
        #     self.last_play_quality = nn.Sequential(nn.Flatten())
        #
        #     self.delay_net1 = nn.Sequential(nn.Flatten())
        #     self.delay_net2 = nn.Sequential(nn.Flatten())
        #
        #
        #     self.last_path_state1 = nn.Sequential(nn.Flatten())
        #     self.last_path_state2 = nn.Sequential(nn.Flatten())
        #
        #     self.last_layer = nn.Sequential(nn.Linear(91, self.features_dim * 2), nn.Tanh(),
        #                         nn.Linear(self.features_dim * 2, self.features_dim), nn.Tanh())


    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        network_speed1 = self.network_speed1(observations["est_throughput1"].unsqueeze(-2))
        # network_speed2 = self.network_speed1(observations["est_throughput2"].unsqueeze(-2))
        next_chunk_size = self.next_chunk_size(observations["next_chunk_size"].unsqueeze(-2))
        # quality_in_buffer = self.quality_in_buffer(observations["quality_in_buffer"].unsqueeze(-2))
        chunk_state = F.one_hot(observations["quality_in_buffer"].to(torch.int64), num_classes=8)
        quality_in_buffer = self.quality_in_buffer(chunk_state)
        buffer_size = self.buffer_size(observations["buffer_size"])
        percentage_remain_video_chunks = self.percentage_remain_video_chunks(
            observations["percentage_remain_video_chunks"])
        last_play_quality = self.last_play_quality(observations["last_play_quality"])
        delay_net1 = self.delay_net1(observations["delay_path1"].unsqueeze(-2))
        # delay_net2=self.delay_net2(observations["delay_path2"].unsqueeze(-2))
        # last_path_state1=self.last_path_state1(observations["last_path_state1"].unsqueeze(-2))
        # last_path_state2=self.last_path_state1(observations["last_path_state1"].unsqueeze(-2))

        cat = torch.cat((network_speed1, next_chunk_size, quality_in_buffer, buffer_size,
                         percentage_remain_video_chunks, last_play_quality, delay_net1), dim=1)

        out = self.last_layer(cat)
        return out
