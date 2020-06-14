from gym.envs.registration import register
from .env.cartpole import CartPoleVisualEnv


register(
    id='cartpole-visual-v1',
    entry_point=CartPoleVisualEnv,
)
