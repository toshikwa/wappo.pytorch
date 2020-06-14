# This file is mainly derived from https://github.com/openai/baselines.

from abc import ABC, abstractmethod
import numpy as np


def tile_images(nhwc):
    nhwc = np.asarray(nhwc)
    N, h, w, c = nhwc.shape
    H = int(np.ceil(np.sqrt(N)))
    W = int(np.ceil(float(N)/H))
    nhwc = np.array(list(nhwc) + [nhwc[0]*0 for _ in range(N, H*W)])
    HWhwc = np.reshape(nhwc, (H, W, h, w, c))
    HhWwc = HWhwc.transpose(0, 2, 1, 3, 4)
    Hh_Ww_c = np.reshape(HhWwc, (H*h, W*w, c))
    return Hh_Ww_c


class VecEnv(ABC):
    closed = False
    viewer = None
    metadata = {'render.modes': ['human', 'rgb_array']}

    def __init__(self, num_envs, observation_space, action_space):
        self.num_envs = num_envs
        self.observation_space = observation_space
        self.action_space = action_space

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_async(self, actions):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close_extras(self):
        pass

    def close(self):
        if self.closed:
            return
        if self.viewer is not None:
            self.viewer.close()
        self.close_extras()
        self.closed = True

    def step(self, actions):
        self.step_async(actions)
        return self.step_wait()

    def render(self, mode='human'):
        imgs = self.get_images()
        bigimg = tile_images(imgs)
        if mode == 'human':
            self.get_viewer().imshow(bigimg)
            return self.get_viewer().isopen
        elif mode == 'rgb_array':
            return bigimg
        else:
            raise NotImplementedError

    def get_images(self):
        raise NotImplementedError

    @property
    def unwrapped(self):
        if isinstance(self, VecEnvWrapper):
            return self.venv.unwrapped
        else:
            return self

    def get_viewer(self):
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.SimpleImageViewer()
        return self.viewer


class VecEnvWrapper(VecEnv):

    def __init__(self, venv, observation_space=None, action_space=None):
        self.venv = venv
        super().__init__(
            num_envs=venv.num_envs,
            observation_space=observation_space or venv.observation_space,
            action_space=action_space or venv.action_space)

    def step_async(self, actions):
        self.venv.step_async(actions)

    @abstractmethod
    def reset(self):
        pass

    @abstractmethod
    def step_wait(self):
        pass

    def close(self):
        return self.venv.close()

    def render(self, mode='human'):
        return self.venv.render(mode=mode)

    def get_images(self):
        return self.venv.get_images()

    def __getattr__(self, name):
        return getattr(self.venv, name)
