from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from wappo.storage import SourceStorage, TargetStorage
from wappo.utils import ResultLogger


class BaseAgent(ABC):

    def __init__(self, venv_source, venv_target, log_dir, device,
                 num_steps=10**6, gamma=0.999, rollout_length=128,
                 num_minibatches=8, epochs_ppo=3, clip_range_ppo=0.2,
                 coef_value=0.5, coef_ent=0.01, lambd=0.95,
                 max_grad_norm=0.5):
        super().__init__()

        self.venv_source = venv_source
        self.venv_target = venv_target
        self.device = device

        # Storage.
        self.storage_source = SourceStorage(
            venv_source.num_envs, rollout_length, epochs_ppo,
            venv_source.observation_space.shape, gamma, lambd, device)
        self.storage_target = TargetStorage(
            venv_target.num_envs, rollout_length,
            venv_target.observation_space.shape, device)

        # Reset envs and store initial states.
        self.states_source = torch.tensor(
            self.venv_source.reset(), dtype=torch.uint8, device=self.device)
        self.states_target = torch.tensor(
            self.venv_target.reset(), dtype=torch.uint8, device=self.device)
        self.storage_source.init_states(self.states_source)
        self.storage_target.init_states(self.states_target)

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
        self.result_logger = ResultLogger(log_dir)

        # Batch size.
        total_batches = rollout_length * venv_source.num_envs
        self.batch_size = total_batches // num_minibatches
        # Unroll length.
        self.rollout_length = rollout_length
        # Number of staps to update.
        self.num_updates = num_steps // total_batches

        # Hyperparameters.
        self.num_envs = venv_source.num_envs
        self.gamma = gamma
        self.num_minibatches = num_minibatches
        self.epochs_ppo = epochs_ppo
        self.clip_range_ppo = clip_range_ppo
        self.coef_value = coef_value
        self.coef_ent = coef_ent
        self.max_grad_norm = max_grad_norm

    def run(self):
        self.writer.add_image(
            'image/source_env', self.states_source[0], self.steps)
        self.writer.add_image(
            'image/target_env', self.states_target[0], self.steps)
        for step in range(self.num_updates):
            self.run_source()
            self.run_target()
            self.update()

            source_return = np.mean(self.source_return)
            target_return = np.mean(self.target_return)

            print(f"Steps: {self.steps}   "
                  f"Source Return: {source_return:5.3f}  "
                  f"Target Return: {target_return:5.3f}")

            self.writer.add_scalar(
                'return/source', source_return, self.steps)
            self.writer.add_scalar(
                'return/target', target_return, self.steps)

            self.result_logger.add(self.steps, source_return, target_return)
            self.save_models(os.path.join(self.model_dir, f'step{self.steps}'))

        self.save_models(os.path.join(self.model_dir, 'final'))

    def run_source(self):
        self.steps += self.rollout_length * self.num_envs

        for _ in range(self.rollout_length):
            with torch.no_grad():
                values, actions, log_probs = \
                    self.network_ppo(self.states_source)
            next_states, rewards, dones, infos = \
                self.venv_source.step(actions.cpu().numpy().flatten())

            for info in infos:
                if 'episode' in info.keys():
                    self.source_return.append(info['episode']['r'])

            self.states_source = torch.tensor(
                next_states, dtype=torch.uint8, device=self.device)

            self.storage_source.insert(
                self.states_source, actions, rewards, dones, log_probs, values)

        with torch.no_grad():
            next_values = self.network_ppo.calculate_values(self.states_source)

        self.storage_source.end_rollout(next_values)

    def run_target(self):
        for _ in range(self.rollout_length):
            with torch.no_grad():
                _, actions, _ = self.network_ppo(self.states_target)

            next_states, _, _, infos = \
                self.venv_target.step(actions.cpu().numpy().flatten())

            for info in infos:
                if 'episode' in info.keys():
                    self.target_return.append(info['episode']['r'])

            self.states_target = torch.tensor(
                next_states, dtype=torch.uint8, device=self.device)

            self.storage_target.insert(self.states_target)

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
