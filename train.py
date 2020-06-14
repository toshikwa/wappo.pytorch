from datetime import datetime
import os
import argparse
import yaml
import torch
from pyvirtualdisplay import Display

from wappo.agent import PPOAgent, WAPPOAgent
from wappo.env import make_cartpole


def main(args):

    with Display(visible=0, size=(100, 100), backend="xvfb") as disp:
        with open(args.config) as f:
            config = yaml.load(f, Loader=yaml.SafeLoader)

        device = torch.device(
            'cuda' if args.cuda and torch.cuda.is_available() else 'cpu')

        # Make environments.
        source_venv = make_cartpole(
            num_envs=config['env']['source_num_envs'],
            num_levels=config['env']['source_num_levels'])
        target_venv = make_cartpole(
            num_envs=config['env']['target_num_envs'],
            num_levels=config['env']['target_num_levels'])

        # Specify the directory to log.
        name = 'wappo' if args.wappo else 'ppo'
        time = datetime.now().strftime("%Y%m%d-%H%M")
        log_dir = os.path.join(
            args.log_dir, 'cartpole-visual-v1', f'{name}-{args.seed}-{time}')

        # Agent.
        if args.wappo:
            agent = WAPPOAgent(
                source_venv=source_venv, target_venv=target_venv,
                device=device, log_dir=log_dir,
                **config['ppo'], **config['wappo'])
        else:
            agent = PPOAgent(
                source_venv=source_venv, target_venv=target_venv,
                device=device, log_dir=log_dir, **config['ppo'])

        agent.run()
        agent.save_models(filename='final_model.pth')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/ppo.yaml')
    parser.add_argument('--log_dir', default='logs')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--cuda', action='store_true')
    parser.add_argument('--wappo', action='store_true')
    main(parser.parse_args())
