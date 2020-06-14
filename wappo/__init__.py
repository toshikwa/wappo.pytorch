from gym.envs.registration import register
from .env import CartPoleVisualEnv


register(
    id='cartpole-visual-v1',
    entry_point=CartPoleVisualEnv,
)
