from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from wappo.storage import SourceStorage, TargetStorage


class BaseAgent(ABC):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, gamma=0.999, rollout_length=128,
                 num_minibatches=8, epochs_ppo=3, clip_range_ppo=0.2,
                 coef_value=0.5, coef_ent=0.01, lambd=0.95,
                 max_grad_norm=0.5):
        super().__init__()

        self.source_venv = source_venv
        self.target_venv = target_venv
        self.device = device

        # Storage.
        self.source_storage = SourceStorage(
            source_venv.num_envs, rollout_length, epochs_ppo,
            source_venv.observation_space.shape, gamma, lambd, device)
        self.target_storage = TargetStorage(
            target_venv.num_envs, rollout_length,
            target_venv.observation_space.shape, device)

        # Reset envs and store initial states.
        self.states = torch.tensor(
            self.source_venv.reset(), dtype=torch.uint8, device=self.device)
        self.target_states = torch.tensor(
            self.target_venv.reset(), dtype=torch.uint8, device=self.device)
        self.source_storage.init_states(self.states)
        self.target_storage.init_states(self.target_states)

        # For logging.
        self.model_dir = os.path.join(log_dir, 'model')
        summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.steps = 0
        self.update_steps = 0
        self.writer = SummaryWriter(log_dir=summary_dir)
        self.source_return = deque([0.0], maxlen=100)
        self.target_return = deque([0.0], maxlen=100)

        # Batch size.
        total_batches = rollout_length * source_venv.num_envs
        self.batch_size = total_batches // num_minibatches
        # Unroll length.
        self.rollout_length = rollout_length
        # Number of staps to update.
        self.num_updates = num_steps // total_batches

        # Hyperparameters.
        self.num_envs = source_venv.num_envs
        self.gamma = gamma
        self.num_minibatches = num_minibatches
        self.epochs_ppo = epochs_ppo
        self.clip_range_ppo = clip_range_ppo
        self.coef_value = coef_value
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def run(self):
        self.writer.add_image(
            'image/source_env', self.states[0], self.steps)
        self.writer.add_image(
            'image/target_env', self.target_states[0], self.steps)

        for step in range(self.num_updates):
            self.run_source()
            self.run_target()
            self.update()

            print(f"Steps: {self.steps}   "
                  f"Source Return: {np.mean(self.source_return):5.3f}  "
                  f"Target Return: {np.mean(self.target_return):5.3f}")

            self.writer.add_scalar(
                'return/source', np.mean(self.source_return), self.steps)
            self.writer.add_scalar(
                'return/target', np.mean(self.target_return), self.steps)
            self.save_models(os.path.join(self.model_dir, f'step{self.steps}'))

        self.save_models(os.path.join(self.model_dir, 'final'))

    def run_source(self):
        self.steps += self.rollout_length * self.num_envs

        for _ in range(self.rollout_length):
            with torch.no_grad():
                values, actions, log_probs = self.ppo_network(self.states)
            next_states, rewards, dones, infos = \
                self.source_venv.step(actions.cpu().numpy().flatten())

            for info in infos:
                if 'episode' in info.keys():
                    self.source_return.append(info['episode']['r'])

            self.states = torch.tensor(
                next_states, dtype=torch.uint8, device=self.device)

            self.source_storage.insert(
                self.states, actions, rewards, dones, log_probs, values)

        with torch.no_grad():
            next_values = self.ppo_network.calculate_values(self.states)

        self.source_storage.end_rollout(next_values)

    def run_target(self):
        for _ in range(self.rollout_length):
            with torch.no_grad():
                _, actions, _ = self.ppo_network(self.target_states)

            next_states, _, _, infos = \
                self.target_venv.step(actions.cpu().numpy().flatten())

            for info in infos:
                if 'episode' in info.keys():
                    self.target_return.append(info['episode']['r'])

            self.target_states = torch.tensor(
                next_states, dtype=torch.uint8, device=self.device)

            self.target_storage.insert(self.target_states)

    @abstractmethod
    def update(self):
        pass

    @abstractmethod
    def save_models(self, save_dir):
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        pass

    @abstractmethod
    def load_models(self, save_dir):
        pass
