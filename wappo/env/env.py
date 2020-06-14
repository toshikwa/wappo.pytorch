import gym
import numpy as np
from .vec_monitor import VecMonitor
from .vec_normalize import VecNormalize


def make_cartpole(num_envs=4, num_levels=0):
    venv = gym.vector.make(
        'cartpole-visual-v1', num_envs=num_envs, num_levels=num_levels)
    venv.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    venv.action_space = gym.spaces.Discrete(2)
    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    return venv
