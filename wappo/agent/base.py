from abc import ABC, abstractmethod
from collections import deque
import numpy as np
import os
import torch
from torch.utils.tensorboard import SummaryWriter

from wappo.storage import SourceStorage, TargetStorage


class BaseAgent(ABC):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, memory_size=10000, batch_size=256,
                 unroll_length=128, gamma=0.999, clip_param=0.2,
                 num_gradient_steps=4, value_loss_coef=0.5,
                 entropy_coef=0.01, lambd=0.95, max_grad_norm=0.5):

        self.source_venv = source_venv
        self.target_venv = target_venv
        self.device = device

        # Storage.
        self.source_storage = SourceStorage(
            unroll_length, source_venv.num_envs, batch_size,
            source_venv.observation_space.shape, gamma, lambd,
            num_gradient_steps, device)

        self.target_storage = TargetStorage(
            memory_size, target_venv.num_envs, batch_size,
            target_venv.observation_space.shape, torch.device('cpu'))

        # For logging.
        self.model_dir = os.path.join(log_dir, 'model')
        summary_dir = os.path.join(log_dir, 'summary')
        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)
        if not os.path.exists(summary_dir):
            os.makedirs(summary_dir)

        self.steps = 0
        self.writer = SummaryWriter(log_dir=summary_dir)
        self.source_return = deque([0.0], maxlen=10)
        self.target_return = deque([0.0], maxlen=10)

        # Batch size.
        self.batch_size = batch_size
        # Unroll length.
        self.unroll_length = unroll_length
        # Number of staps to update.
        self.num_updates = num_steps // (unroll_length * source_venv.num_envs)

        # Hyperparameters.
        self.num_envs = source_venv.num_envs
        self.gamma = gamma
        self.clip_param = clip_param
        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef
        self.max_grad_norm = max_grad_norm

    def run(self):
        self.states = torch.tensor(
            self.source_venv.reset(), dtype=torch.uint8, device=self.device)
        self.source_storage.init_states(self.states)
        self.target_states = torch.tensor(
            self.target_venv.reset(), dtype=torch.uint8, device=self.device)
        self.target_storage.insert(self.target_states)

        for step in range(self.num_updates):
            self.run_target()
            self.run_source()
            self.update()

            print(f"\rSteps: {self.steps}   "
                  f"Source Return: {np.mean(self.source_return):5.3f}  "
                  f"Target Return: {np.mean(self.target_return):5.3f}")

            self.writer.add_scalar(
                'return/source', np.mean(self.source_return), self.steps)
            self.writer.add_scalar(
                'return/target', np.mean(self.target_return), self.steps)

    def run_source(self):
        self.steps += self.unroll_length * self.num_envs

        for _ in range(self.unroll_length):
            with torch.no_grad():
                values, actions, action_log_probs = \
                    self.ppo_network(self.states)
            next_states, rewards, dones, infos = \
                self.source_venv.step(actions.cpu().numpy().flatten())

            for info in infos:
                if 'episode' in info.keys():
                    self.source_return.append(info['episode']['r'])

            self.states = torch.tensor(
                next_states, dtype=torch.uint8, device=self.device)

            self.source_storage.insert(
                self.states, actions, rewards, dones, action_log_probs,
                values)

        with torch.no_grad():
            next_values = self.ppo_network.calculate_values(self.states)

        self.source_storage.end_rollout(next_values)

    def run_target(self):
        for _ in range(self.unroll_length):
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
