import gym
import numpy as np
from procgen import ProcgenEnv

from .vec_env import VecExtractDictObs, TransposeImage
from .vec_monitor import VecMonitor
from .vec_normalize import VecNormalize


def make_venv(env_id, num_envs=4, num_levels=0, start_level=0,
              distribution_mode='easy'):
    if env_id == 'cartpole-visual-v1':
        venv = gym.vector.make(
            'cartpole-visual-v1', num_envs=num_envs,
            num_levels=num_levels, start_level=start_level)
        venv.observation_space = gym.spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        venv.action_space = gym.spaces.Discrete(2)
    else:
        venv = ProcgenEnv(
            env_name=env_id, num_envs=num_envs, num_levels=num_levels,
            start_level=start_level, distribution_mode=distribution_mode)
        venv = VecExtractDictObs(venv, "rgb")
        venv = TransposeImage(venv)

    venv = VecMonitor(venv=venv, filename=None, keep_buf=100)
    venv = VecNormalize(venv=venv, ob=False)
    return venv
