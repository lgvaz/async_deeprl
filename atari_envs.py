import gym
from gym.wrappers import Monitor
import numpy as np
from cv2 import resize, cvtColor, COLOR_RGB2GRAY


def preprocess(img):
    ''' Converts image to grayscale and resizes to 84x84 '''
    return resize(cvtColor(img, COLOR_RGB2GRAY), (84, 84))


class AtariWrapper:
    '''
    Wraps atari gym env, the new env returns the last N frames.
    '''
    def __init__(self, env_name, num_stacked_frames, videodir=None):
        self.env = gym.make(env_name)
        self.num_stacked_frames = num_stacked_frames
        if videodir is not None:
            self.env = Monitor(env=self.env, directory=videodir, resume=True)
        # Do workaround for pong and breakout actions
        if env_name == 'Pong-v0' or env_name == 'Breakout-v0':
            print('Changing pong or breakout actions to [1, 2, 3]')
            self.valid_actions = [1, 2, 3]
        else:
            num_actions = self.env.action_space.n
            self.valid_actions = np.arange(num_actions)

    def reset(self):
        '''
        Reset the env and returns the first frame repeated N times
        '''
        state = self.env.reset()
        state = preprocess(state)
        self.states_hist = np.stack([state] * self.num_stacked_frames, axis=2)
        return self.states_hist

    def step(self, action):
        '''
        Execute the action on the gym env and returns the last N frames
        '''
        next_state, reward, done, info = self.env.step(self.valid_actions[action])
        next_state = preprocess(next_state)
        self.states_hist = np.append(self.states_hist[:, :, 1:], next_state[:, :, np.newaxis], axis=2)
        return self.states_hist, reward, done, info

    def close(self):
        self.env.close()
