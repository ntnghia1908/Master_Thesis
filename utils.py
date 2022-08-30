from enum import Enum
import os
import random
import numpy as np
import torch


class Path(Enum):
    PATH1 = 1
    PATH2 = 2


class Schedule(Enum):
    DUPLICATE = 1
    NO_DUPLICATE = 2
    GREEDY = 3


class Event:
    DOWN = 1 # [cur_time, DOWN, down_id, cur_path]
    DOWNF = 2 # [cur_time, DOWNF, down_id, cur_path]
    PLAY = 3  # [cur_time, PLAY, play_id, -1]
    PLAYF = 4  # [cur_time, PLAYF, play_id, -1]
    SLEEPF = 5  # [cur_time, SLEEPF, play_id, cur_path]
    FREEZEF = 6 # [cur_time, FREEZEF, next_play_id, cur_path]

    def __init__(self, cur_time, event_type):
        self.cur_time = cur_time
        self.event_type = event_type

    def __str__(self):
        pass


class DownEvent(Event):
    def __init__(self, cur_time, cur_path,  down_id=None):
        super().__init__(cur_time, event_type=Event.DOWN)
        self.down_id = down_id
        self.cur_path = cur_path

    def __str__(self):
        return "{}, DOWN, downID: {}, path: {}".format(self.cur_time, self.down_id, self.cur_path)


class DownFinishEvent(Event):
    def __init__(self, cur_time, downf_id, quality, cur_path):
        super().__init__(cur_time, event_type=Event.DOWNF)
        self.quality = quality
        self.downf_id = downf_id
        self.cur_path = cur_path

    def __str__(self):
        return "{}, DOWNF, downfID: {}, quality: {}, path: {}".\
            format(self.cur_time, self.downf_id, self.quality, self.cur_path)


class PlayEvent(Event):
    def __init__(self, cur_time, play_id, play_quality, cur_path=None):
        super(PlayEvent, self).__init__(cur_time=cur_time, event_type=Event.PLAY)
        self.play_id = play_id
        self.play_quality = play_quality
        self.cur_path = cur_path

    def __str__(self):
        return "{}, PLAY, playID: {}, quality: {}".format(self.cur_time, self.play_id, self.play_quality)


class PlayFinishEvent(Event):
    def __init__(self, cur_time, playf_id, playf_quality, cur_path=None):
        super(PlayFinishEvent, self).__init__(cur_time=cur_time, event_type=Event.PLAYF)
        self.playf_id = playf_id
        self.playf_quality = playf_quality
        self.cur_path = cur_path

    def __str__(self):
        return "{}, PLAYF, playFID: {}, playf_quality: {}".format(self.cur_time, self.playf_id, self.playf_quality)


class SleepFinishEvent(Event):
    def __init__(self, cur_time, cur_path,  playing_id=None):
        super().__init__(cur_time=cur_time, event_type=Event.SLEEPF)
        self.playing_id = playing_id
        self.cur_path = cur_path

    def __str__(self):
        return "{}, SLEEPF, playingID: {}, cur_path: {}".format(self.cur_time, self.playing_id, self.cur_path)


class FreezeFinishEvent(Event):
    def __init__(self, cur_time, next_play_id, cur_path=None):
        super().__init__(cur_time=cur_time, event_type=Event.FREEZEF)
        self.next_play_id = next_play_id
        self.cur_path = cur_path

    def __str__(self):
        return "{}, FreezeF, next_play_id: {}, cur_path: {}".format(self.cur_time, self.next_play_id, self.cur_path)


class TimeLine:
    def __init__(self):
        self.time_line = list()

    def add(self, event):
        self.time_line.append(event)

    def sort(self):
        self.time_line.sort(key=lambda e: e.cur_time)

    def get_next_event(self):
        return self.time_line[0]

    def remove_event(self):
        return self.time_line.pop(0)

    def show(self):
        for event in self.time_line:
            print(event)

    def clear(self):
        self.time_line = list()

def set_global_seed(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)


if __name__ == '__main__':
    down_event = DownEvent(1.2, 5, 1)
    print(down_event.__str__())

