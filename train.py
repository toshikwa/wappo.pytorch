from datetime import datetime
import os
import argparse
import yaml
import gym
import numpy as np
import torch

from wappo.env import VecMonitor, VecNormalize
from wappo.agent import PPOAgent


def main(args):
    with open(args.config) as f:
        config = yaml.load(f, Loader=yaml.SafeLoader)

    device = torch.device(
        'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

    # Make environments.
    source_venv = gym.vector.make(
        'cartpole-visual-v1', num_envs=config['env']['source_num_envs'],
        num_levels=config['env']['source_num_levels'])
    source_venv.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    source_venv.action_space = gym.spaces.Discrete(2)
    source_venv = VecMonitor(venv=source_venv, filename=None, keep_buf=100)
    source_venv = VecNormalize(venv=source_venv, ob=False)

    target_venv = gym.vector.make(
        'cartpole-visual-v1', num_envs=config['env']['target_num_envs'],
        num_levels=config['env']['target_num_levels'])
    target_venv.observation_space = gym.spaces.Box(
        low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
    target_venv.action_space = gym.spaces.Discrete(2)
    target_venv = VecMonitor(venv=target_venv, filename=None, keep_buf=100)
    target_venv = VecNormalize(venv=target_venv, ob=False)

    # Specify the directory to log.
    time = datetime.now().strftime("%Y%m%d-%H%M")
    log_dir = os.path.join(
        args.log_dir, f'PPO-{args.seed}-{time}')

    # PPO agent.
    agent = PPOAgent(
        source_venv=source_venv, target_venv=target_venv, device=device,
        log_dir=log_dir, **config['ppo'])
    agent.run()
    agent.save_models(filename='final_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ppo.yaml')
    parser.add_argument('--log_dir', default='logs/ppo')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true', default=0)
    main(parser.parse_args())
