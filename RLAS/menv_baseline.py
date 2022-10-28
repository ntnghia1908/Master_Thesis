import math
import pickle
import random

import gym
import numpy as np

from utility.get_down_size import video_list_collector
from collections import Counter
# from stable_baselines3.common.env_checker import check_env
from utils_schedule_newstate import Event, Path, Schedule, TimeLine, DownEvent, DownFinishEvent, \
                    PlayEvent, PlayFinishEvent, SleepFinishEvent, FreezeFinishEvent, InactiveFinishEvent


class Env(gym.Env):
    EVENT = Event
    SCHEDULE = Schedule
    PATH = Path

    SAMPLE = 0.05  # sec
    IS_NOT_DOWNLOADED = -1
    MIC_SEC = 0.0001

    NETWORK_SEGMENT1 = 1  # sec
    NETWORK_SEGMENT2 = 1  # sec

    BUFFER_NORM_FACTOR = 10.0
    SMOOTH_PENALTY_COEFFICIENT = 1.0
    LOW_BUFFER_PENALTY_COEFFICIENT = 0.0
    DEFAULT_QUALITY = 0  # default video quality without agent

    MIN_BUFFER_THRESH = 6.0  # sec, min buffer threshold
    M_IN_K = 1000.0

    QOE_LIN = np.array([300.0, 700.0, 1200.0, 1500.0, 3000.0, 6000.0, 8000.0])  # for 04-second segment
    VIDEO_BIT_RATE = [300, 700, 1200, 1500, 3000, 6000, 8000]  # for 04-second segment

    SEGMENT_SPACE = 7
    QUALITY_SPACE = 7
    HISTORY_SIZE = 6
    CHUNK_TIL_VIDEO_END_CAP = 60
    BUFFER_THRESH = 30.0  # sec, max buffer limit
    VIDEO_CHUNK_LEN = int(4)  # sec, every time add this amount to buffer
    VIDEO_LEN = VIDEO_CHUNK_LEN * CHUNK_TIL_VIDEO_END_CAP  # in sec

    print_template = "{0:^10}|{1:^20}|{2:^20}|{3:^20}|{4:^20}|{5:^20}|{6:^20}"

    def __init__(self, schedule_type, bitrate_lists, video_trace='../utility/video_list_bunny8k',
                 log_qoe=True, debug_mode=False, multi_discrete=True,  replace=True, train=True,
                 bw_test1=None, bw_test2=None):
        assert len(self.QOE_LIN) == len(self.VIDEO_BIT_RATE)
        if log_qoe:
            self.UTILITY_SCORE = list(np.log(self.QOE_LIN / self.QOE_LIN[0]))
            self.REBUF_PENALTY_COEFFICIENT = 3.3  # 1 sec rebuffering -> 3 Mbps
        else:
            self.REBUF_PENALTY_COEFFICIENT = 4.3

        self.train = train
        if not self.train:
            self.test_round = 0

        self.debug_mode = debug_mode
        if self.debug_mode:
            print("--DEBUG--")

        self.schedule_type = schedule_type
        self.multi_discrete = multi_discrete

        # get video list
        self.video_list = video_list_collector(save_dir=video_trace).get_trace_matrix(self.VIDEO_BIT_RATE)

        # get bandwidth
        self.bws_list = bitrate_lists

        self.state = dict(
            est_throughput1=gym.spaces.Box(low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32),
            est_throughput2=gym.spaces.Box(low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32),
            next_chunk_size=gym.spaces.Box(low=0, high=float("inf"),
                                           shape=(self.QUALITY_SPACE * self.SEGMENT_SPACE,), dtype=np.float32),
            quality_in_buffer=gym.spaces.Box(low=self.IS_NOT_DOWNLOADED, high=self.QUALITY_SPACE-1,
                                             shape=(self.SEGMENT_SPACE,), dtype=np.int32),
            buffer_size=gym.spaces.Box(low=0, high=float("inf"), shape=(1,), dtype=np.float32),
            percentage_remain_video_chunks=gym.spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            last_play_quality=gym.spaces.Discrete(self.QUALITY_SPACE),  # actually this is included in quality_in_buffer
            delay_path1=gym.spaces.Box(low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32),
            delay_path2=gym.spaces.Box(low=0, high=float("inf"), shape=(self.HISTORY_SIZE,), dtype=np.float32),
        )

        self.action_space = gym.spaces.MultiDiscrete([self.SEGMENT_SPACE, self.QUALITY_SPACE])
        self.observation_space = gym.spaces.Dict(self.state)

    def choice_bw(self):
        if self.train:
            rand_group_bw1 = np.random.choice(['fcclow', 'fccup', 'ltelow', 'lteup'])
            rand_group_bw2 = np.random.choice(['fcclow', 'fccup', 'ltelow', 'lteup'])

            group1 = self.bws_list[rand_group_bw1]
            group2 = self.bws_list[rand_group_bw2]

            idx1= np.random.choice(len(group1))
            idx2 = np.random.choice(len(group2))

            self.bw1=group1[idx1]
            self.bw2=group2[idx2]

            self.init_net_seg1 = np.random.randint(0, len(self.bw1) - 1)
            self.init_net_seg2 = np.random.randint(0, len(self.bw2) - 1)
        else: # TEST
            bws1, bws2 =self.bws_list
            self.bw1 = bws1[self.test_round]
            self.bw2 = bws2[self.test_round]

            self.init_net_seg1 = 0
            self.init_net_seg2 = 0
            self.test_round += 1

    def reset(self):
        self.RTT1 = np.random.uniform(0.05, 0.1)  # Round Trip Time
        self.RTT2 = np.random.uniform(0.05, 0.1)  # Round Trip Time

        if not self.train:
            self.RTT1 = 0.1  # Round Trip Time
            self.RTT2 = 0.05  # Round Trip Time
        self.step_no = 1

        if self.debug_mode:
            self.RTT1 = 0.1  # Round Trip Time
            self.RTT2 = 0.05  # Round Trip Time

            print(self.print_template.format('step', 'step reward', 'smooth penalty', 'reward quality', 'rebufer_time',
                                         'low buffer', 'buffer_size'))
        self.choice_bw()
        self.last_quality = self.DEFAULT_QUALITY

        self.network_speed1 = np.zeros(self.HISTORY_SIZE)
        self.network_speed2 = np.zeros(self.HISTORY_SIZE)

        self.down_delay1 = np.ones(self.HISTORY_SIZE) # * 10.0
        self.down_delay2 = np.ones(self.HISTORY_SIZE) #* 10.0

        self.play_id = 0
        self.down_segment = np.array([self.IS_NOT_DOWNLOADED] * self.CHUNK_TIL_VIDEO_END_CAP, dtype=int)
        self.down_segment_f = np.array([self.IS_NOT_DOWNLOADED] * self.CHUNK_TIL_VIDEO_END_CAP, dtype=int)

        self.reward_qua = 0.0
        self.reward_smooth = 0.0
        self.rebufer_time = 0.0
        self.low_buffer = 0.0
        self.last_sum_reward_cummulate = 0.0
        self.est_throughput1 = 0.0
        self.est_throughput2 = 0.0
        self.delay1 = 1.0
        self.delay2 = 1.0
        self.end_of_video = False
        self.buffer_size = 0
        self.cur_time = 0.0

        # self.sleep_time = 0.0

        self.time_line = TimeLine()
        self.time_line.add(DownEvent(cur_time=0.0, down_id=-1, cur_path=Path.PATH1))

        self.down_segment[0] = self.DEFAULT_QUALITY  # in version 3, down_segment is updated when finish download

        segment_size = float(self.video_list[0][0]) * 8.0  # download video segment in bit
        delay = self.down_time(segment_size, self.cur_time, Path.PATH1)

        self.time_line.add(DownFinishEvent(cur_time=delay, downf_id=0, quality=self.DEFAULT_QUALITY, cur_path=Path.PATH1, est_throughput=self.est_throughput1, delay=self.delay1))

        # initialize some types of events
        self.time_line.add(SleepFinishEvent(self.SAMPLE, cur_path=Path.PATH2))
        self.time_line.add(PlayEvent(delay+self.MIC_SEC, play_id=0, play_quality=self.DEFAULT_QUALITY))

        # remove the current considering event from event and sort
        self.time_line.remove_event()
        self.time_line.sort()

        self.quality_reward_cummulate = 0.0
        self.smooth_penalty_cummulate = 0.0
        self.rebuffer_penalty_cummulate = 0.0
        self.low_buffer_penalty_cummulate= 0.0
        self.no_switch = 0

        self.state = dict(
            est_throughput1=np.zeros(self.HISTORY_SIZE),
            est_throughput2=np.zeros(self.HISTORY_SIZE),
            next_chunk_size=np.zeros(self.QUALITY_SPACE * self.SEGMENT_SPACE),
            quality_in_buffer=np.zeros(self.SEGMENT_SPACE),
            buffer_size=np.array([0]),
            percentage_remain_video_chunks=np.array([1]),
            last_play_quality=self.DEFAULT_QUALITY,
            delay_path1=self.down_delay1,
            delay_path2=self.down_delay2,
        )

        return self.state

    def state_space(self):
        return self.state.keys()

    def action_space(self):
        return [self.SEGMENT_SPACE , self.QUALITY_SPACE]


    def get_video_list(self):
        return self.video_list

    def current_buffer_size(self):
        return sum(self.down_segment == self.IS_NOT_DOWNLOADED)

    def feasible_action(self):
        if self.schedule_type == self.SCHEDULE.NO_DUPLICATE:
            for i in range(self.play_id + 1, min(self.CHUNK_TIL_VIDEO_END_CAP, self.play_id + 1 + self.SEGMENT_SPACE)):
                if self.down_segment[i] == self.IS_NOT_DOWNLOADED:
                    for j in range(self.QUALITY_SPACE):
                        f_action = np.append(f_action, (i - self.play_id - 1) * self.QUALITY_SPACE + j)

        if self.schedule_type == self.SCHEDULE.GREEDY:
            for i in range(self.play_id + 1,
                           min(self.CHUNK_TIL_VIDEO_END_CAP, self.play_id + 1 + self.SEGMENT_SPACE)):
                if self.down_segment[i] == self.IS_NOT_DOWNLOADED:
                    for j in range(self.QUALITY_SPACE):
                        f_action = np.append(f_action, (i - self.play_id - 1) * self.QUALITY_SPACE + j)
                    break
        return f_action

    # This function for multi-discrete
    def pick_next_segment(self):
        if self.schedule_type == self.SCHEDULE.NO_DUPLICATE:
            feasible_segment = np.array([], dtype=int)
            for i in range(self.play_id + 1,
                           min(self.CHUNK_TIL_VIDEO_END_CAP, self.play_id + 1 + self.SEGMENT_SPACE)):
                if self.down_segment[i] == self.IS_NOT_DOWNLOADED:
                    feasible_segment = np.append(feasible_segment, i - (self.play_id + 1))
            return feasible_segment

        if self.schedule_type == self.SCHEDULE.GREEDY:
            feasible_segment = np.array([], dtype=int)
            for i in range(self.play_id + 1,
                           min(self.CHUNK_TIL_VIDEO_END_CAP, self.play_id + 1 + self.SEGMENT_SPACE)):
                if self.down_segment[i] == self.IS_NOT_DOWNLOADED:
                    feasible_segment = np.append(feasible_segment, i - (self.play_id + 1))
                    break
            return feasible_segment

    # multi-discrete
    def action_masks(self):
        chunk_mask = np.zeros(self.SEGMENT_SPACE, dtype=bool)
        quality_mask = np.ones(self.QUALITY_SPACE, dtype=bool)
        chunk_mask[self.pick_next_segment()] = True
        return np.concatenate((chunk_mask, quality_mask),axis=None)

    def down_time(self, segment_size, cur_time, path):
        # calculate net_seg_id, seg_time_stamp from cur_time. Remember seg_time_stamp plus rtt
        # set network segment ID to position after sleeping and download lastmodel segment
        if path == Path.PATH1:
            delay = self.RTT1
            pass_seg = math.floor(cur_time / self.NETWORK_SEGMENT1)
            net_seg_id1 = self.init_net_seg1 + pass_seg
            seg_time_stamp = cur_time - pass_seg * self.NETWORK_SEGMENT1

            while True:  # download segment process finish after a full video segment is downloaded
                net_seg_id1 = net_seg_id1 % len(self.bw1)  # loop back to begin if finished
                network = self.bw1[net_seg_id1]  # network DL_bitrate in bps
                # self.bw1_trace.append([cur_time, network])
                # maximum possible throughput in bytes
                max_throughput = network * (self.NETWORK_SEGMENT1 - seg_time_stamp)

                if max_throughput > segment_size:  # finish download in network segment
                    seg_time_stamp += segment_size / network  # used time in network segment in second
                    delay += segment_size / network  # delay from begin in second
                    break
                else:
                    delay += self.NETWORK_SEGMENT1 - seg_time_stamp  # delay from begin in second
                    seg_time_stamp = 0  # used time of next network segment is 0s
                    segment_size -= max_throughput  # remain undownloaded part of video segment
                    net_seg_id1 += 1

        if path == Path.PATH2:

            delay = self.RTT2
            pass_seg = math.floor(cur_time / self.NETWORK_SEGMENT2)
            net_seg_id2 = self.init_net_seg2 + pass_seg
            seg_time_stamp = cur_time - pass_seg

            while True:
                net_seg_id2 = net_seg_id2 % len(self.bw2)  # loop back to begin if finished
                network = self.bw2[net_seg_id2]  # network DL_bitrate in bps
                # self.bw2_trace.append([cur_time, network])
                max_throughput = network * (
                        self.NETWORK_SEGMENT2 - seg_time_stamp)  # maximum possible throughput in bytes

                if max_throughput > segment_size:  # finish download in network segment
                    seg_time_stamp += segment_size / network  # used time in network segment in second
                    delay += segment_size / network  # delay from begin in second
                    break
                else:
                    delay += self.NETWORK_SEGMENT2 - seg_time_stamp  # delay from begin in second
                    seg_time_stamp = 0  # used time of next network segment is 0s
                    segment_size -= max_throughput
                    net_seg_id2 += 1
        return delay

    def pick_action(self, action):
        down_id = self.play_id + math.floor(
            action / self.QUALITY_SPACE) + 1  # self.play_id is playing or just finish playing
        down_quality = action % self.QUALITY_SPACE
        return down_id, down_quality

    def step(self, action):
        if self.multi_discrete:
            next_down_chunk, down_quality = action
            down_id = self.play_id + next_down_chunk + 1

        # NEW STEP
        cur_event = self.time_line.get_next_event()
        self.cur_down_path = cur_event.cur_path

        self.down_segment[down_id] = down_quality
        self.buffer_size = sum((self.down_segment[self.play_id:] != self.IS_NOT_DOWNLOADED)) * self.VIDEO_CHUNK_LEN

        segment_size = float(self.video_list[down_quality][down_id]) * 8.0  # download video segment in bits
        delay = self.down_time(segment_size, cur_event.cur_time, cur_event.cur_path)

        self.time_line.add(DownFinishEvent(cur_time=cur_event.cur_time + delay, downf_id=down_id,
                                           quality=down_quality, cur_path=cur_event.cur_path,
                                           est_throughput=segment_size / delay, delay=delay))

        # remove the current considering event from event
        self.time_line.remove_event()
        self.time_line.sort()

        # One step happen between 2 DOWN EVENT
        while True:
            cur_event = self.time_line.get_next_event()
            cur_time = cur_event.cur_time
            cur_path = cur_event.cur_path

            if cur_event.event_type == Event.DOWN:
                if len(self.pick_next_segment()) == 0:
                    self.time_line.add(SleepFinishEvent(cur_time=cur_time, cur_path=cur_path))
                else:
                    break

            elif cur_event.event_type == Event.DOWNF:
                self.down_segment_f[int(cur_event.downf_id)] = cur_event.quality

                if cur_event.cur_path == Path.PATH1:
                     self.est_throughput1 = cur_event.est_throughput  # in bits per second
                     self.delay1 = cur_event.delay
                else:
                    self.est_throughput2 = cur_event.est_throughput  # in bits per second
                    self.delay2 = cur_event.delay

                if (self.buffer_size > self.BUFFER_THRESH) or (self.IS_NOT_DOWNLOADED not in self.down_segment):
                    self.time_line.add(SleepFinishEvent(cur_time=cur_time+self.SAMPLE, cur_path=cur_path))
                else:
                    self.time_line.add(DownEvent(cur_time=cur_time+self.MIC_SEC, cur_path=cur_path))

            elif cur_event.event_type == Event.SLEEPF:  # this make an infinite loop
                if (self.buffer_size > self.BUFFER_THRESH) or (self.IS_NOT_DOWNLOADED not in self.down_segment):
                    self.time_line.add(SleepFinishEvent(cur_time=cur_time+self.SAMPLE, cur_path=cur_path))
                else:
                    self.time_line.add(DownEvent(cur_time=cur_time+self.MIC_SEC, cur_path=cur_path))

            elif cur_event.event_type == Event.PLAY:
                self.play_id = cur_event.play_id

                self.buffer_size = sum((self.down_segment_f[(self.play_id + 1):] != self.IS_NOT_DOWNLOADED)) * self.VIDEO_CHUNK_LEN

                play_quality = self.down_segment_f[self.play_id]
                last_play_quality = self.down_segment_f[self.play_id - 1]
                self.reward_qua += self.UTILITY_SCORE[play_quality]
                self.reward_smooth += np.abs(self.UTILITY_SCORE[play_quality] - \
                                             self.UTILITY_SCORE[last_play_quality])
                self.low_buffer += (max(0, self.MIN_BUFFER_THRESH - self.buffer_size)) ** 2
                if play_quality != last_play_quality:
                    self.no_switch += 1

                self.time_line.add(PlayFinishEvent(cur_time=cur_time+self.VIDEO_CHUNK_LEN, playf_id=self.play_id,
                                                   playf_quality=play_quality))

            elif cur_event.event_type == Event.PLAYF:
                self.play_id = cur_event.playf_id  # finish play_id

                if self.play_id == self.CHUNK_TIL_VIDEO_END_CAP - 1:
                    self.time_line.remove_event()
                    break

                if self.down_segment_f[self.play_id + 1] == self.IS_NOT_DOWNLOADED:
                    self.time_line.add(FreezeFinishEvent(cur_time+self.SAMPLE, next_play_id=self.play_id+1))
                else:
                    self.time_line.add(PlayEvent(cur_time=cur_time+self.MIC_SEC, play_id=self.play_id+1,
                                                 play_quality=self.down_segment_f[self.play_id+1]))

            elif cur_event.event_type == Event.REBUFFER:
                self.rebufer_time += self.SAMPLE # calculate rebuffering time
                # next chunk has not downloaded yet
                if self.down_segment_f[int(cur_event.next_play_id)] == self.IS_NOT_DOWNLOADED:
                    self.time_line.add(FreezeFinishEvent(cur_time=cur_time+self.SAMPLE,
                                                         next_play_id=cur_event.next_play_id))
                else:
                    self.time_line.add(PlayEvent(cur_time=cur_time+self.MIC_SEC, play_id=cur_event.next_play_id,
                                                 play_quality=cur_event.next_play_id))

            self.time_line.remove_event()
            self.time_line.sort()

        self.quality_reward_cummulate = self.reward_qua
        self.smooth_penalty_cummulate = self.SMOOTH_PENALTY_COEFFICIENT * self.reward_smooth
        self.rebuffer_penalty_cummulate = self.rebufer_time * self.REBUF_PENALTY_COEFFICIENT
        self.low_buffer_penalty_cummulate = self.LOW_BUFFER_PENALTY_COEFFICIENT * self.low_buffer
        sum_reward_cummulate = self.quality_reward_cummulate - self.smooth_penalty_cummulate - \
                     self.rebuffer_penalty_cummulate - self.low_buffer_penalty_cummulate

        step_reward = sum_reward_cummulate - self.last_sum_reward_cummulate
        step_reward_norm = step_reward / 100
        self.last_sum_reward_cummulate = sum_reward_cummulate

        if self.debug_mode:
            print(self.print_template.format(self.step_no, step_reward_norm, self.quality_reward_cummulate,
                                             self.smooth_penalty_cummulate, self.rebuffer_penalty_cummulate,
                                             self.low_buffer_penalty_cummulate, self.buffer_size))
            self.step_no += 1

        # CALCULATE NEW STATE
        next_chunk_size = np.array([])
        chunk_state = np.array([])
        for i in range(self.play_id + 1, self.play_id + 1 + self.SEGMENT_SPACE):
            if i < self.CHUNK_TIL_VIDEO_END_CAP:
                chunk_state = np.append(chunk_state, self.down_segment[i]+1)
                next_chunk_size = np.append(next_chunk_size, self.video_list[:, i])
            else:
                next_chunk_size = np.append(next_chunk_size, np.array([0] * self.QUALITY_SPACE))
                chunk_state = np.append(chunk_state, 0)

        self.network_speed1 = np.roll(self.network_speed1, axis=-1, shift=1)
        self.network_speed1[0] = self.est_throughput1 / 1e6  # in Mbps

        self.down_delay1 = np.roll(self.down_delay1, axis=-1, shift=1)
        self.down_delay1[0] = self.delay1  # in Mbps

        self.network_speed2 = np.roll(self.network_speed2, axis=-1, shift=1)
        self.network_speed2[0] = self.est_throughput2 / 1e6

        self.down_delay2 = np.roll(self.down_delay2, axis=-1, shift=1)
        self.down_delay2[0] = self.delay2  # in Mbps

        remain = self.CHUNK_TIL_VIDEO_END_CAP - self.play_id

        self.state = dict(
            est_throughput1=self.network_speed1,
            est_throughput2=self.network_speed2,
            next_chunk_size=next_chunk_size / 1e6,
            quality_in_buffer=chunk_state,
            buffer_size=np.array([self.buffer_size / self.BUFFER_NORM_FACTOR]),
            percentage_remain_video_chunks=np.array([remain / self.CHUNK_TIL_VIDEO_END_CAP]),
            last_play_quality=int(self.down_segment[self.play_id] / self.QUALITY_SPACE),
            delay_path1=self.down_delay1,
            delay_path2=self.down_delay2,
        )

        info = dict(
            step_no=self.step_no,
            step_reward_norm=step_reward_norm,
            quality_reward_cummulate=self.quality_reward_cummulate,
            smooth_penalty_cummulate=self.smooth_penalty_cummulate,
            rebuffer_penalty_cummulate=self.rebuffer_penalty_cummulate,
            low_buffer_penalty_cummulate=self.low_buffer_penalty_cummulate,
            sum_reward=self.last_sum_reward_cummulate,
            buffer_size=self.buffer_size,
            rebuffer_time=self.rebufer_time
        )

        # if terminate reset
        if self.play_id == self.CHUNK_TIL_VIDEO_END_CAP - 1:
            self.end_of_video = True

        return self.state, step_reward_norm, self.end_of_video, info
