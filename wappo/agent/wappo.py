import os
import numpy as np
import torch
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from .ppo import PPOAgent
from wappo.network import CriticNetwork


class WAPPOAgent(PPOAgent):

    def __init__(self, venv_source, venv_target, log_dir, device,
                 num_steps=10**6, lr_ppo=5e-4, gamma=0.999,
                 rollout_length=16, num_minibatches=8, epochs_ppo=3,
                 clip_range_ppo=0.2, coef_value=0.5, coef_ent=0.01,
                 lambd=0.95, max_grad_norm=0.5, num_initial_blocks=1,
                 lr_critic=5.e-4, epochs_critic=5, coef_conf=10.0,
                 clip_range_critic=0.01):
        super().__init__(
            venv_source, venv_target, log_dir, device, num_steps, lr_ppo,
            gamma, rollout_length, num_minibatches, epochs_ppo, clip_range_ppo,
            coef_value, coef_ent, lambd, max_grad_norm, num_initial_blocks)

        # Adversarial network.
        self.network_critic = CriticNetwork(
            feature_dim=self.network_ppo.feature_dim).to(device)

        # Optimizers.
        self.optim_critic = RMSprop(
            self.network_critic.parameters(), lr=lr_critic)

        self.epochs_critic = epochs_critic
        self.coef_conf = coef_conf
        self.clip_range_critic = clip_range_critic

    def update(self):
        loss_policies = []
        loss_values = []
        loss_critics = []
        loss_confs = []

        for samples in self.storage_source.iter(self.batch_size):
            states_target = self.storage_target.sample(self.batch_size)

            loss_policy, loss_value, loss_conf = \
                self.update_ppo(*samples, states_target)
            loss_critic = self.update_critic(samples[0], states_target)

            self.update_steps += 1
            loss_policies.append(loss_policy)
            loss_values.append(loss_value)
            loss_confs.append(loss_conf)
            loss_critics.append(loss_critic)

        self.writer.add_scalar(
            'loss/policy', np.mean(loss_policies), self.steps)
        self.writer.add_scalar(
            'loss/value', np.mean(loss_values), self.steps)
        self.writer.add_scalar(
            'loss/critic', np.mean(loss_critics), self.steps)
        self.writer.add_scalar(
            'loss/conf', np.mean(loss_confs), self.steps)

    def update_ppo(self, states_source, actions, values_old, value_targets,
                   log_probs_old, advantages, states_target):
        # Include confusion loss every 5 updates.
        if_update_conf = float(self.update_steps % self.epochs_critic == 0)

        values, log_probs, mean_entropy, source_features = \
            self.network_ppo.evaluate_actions(states_source, actions)

        # >>> Value >>> #
        values_clipped = values_old + (values - values_old).clamp(
            -self.clip_range_ppo, self.clip_range_ppo)
        loss_value1 = (values - value_targets).pow(2)
        loss_value2 = (values_clipped - value_targets).pow(2)
        loss_value = torch.max(loss_value1, loss_value2).mean()
        # >>> Value >>> #

        # >>> Policy >>> #
        ratio = torch.exp(log_probs - log_probs_old)
        loss_policy1 = -ratio * advantages
        loss_policy2 = -torch.clamp(
            ratio,
            1.0 - self.clip_range_ppo,
            1.0 + self.clip_range_ppo
        ) * advantages
        loss_policy = torch.max(loss_policy1, loss_policy2).mean()
        # >>> Policy >>> #

        # >>> Confusion >>> #
        target_features = \
            self.network_ppo.calculate_features(states_target)
        source_preds = self.network_critic(source_features)
        target_preds = self.network_critic(target_features)
        loss_conf = torch.mean(source_preds) - torch.mean(target_preds)
        # >>> Confusion >>> #

        # >>> Total >>> #
        loss = \
            loss_policy \
            - self.coef_ent * mean_entropy \
            + self.coef_value * loss_value \
            + if_update_conf * self.coef_conf * loss_conf
        # >>> Total >>> #

        self.optim_ppo.zero_grad()
        loss.backward()
        clip_grad_norm_(self.network_ppo.parameters(), self.max_grad_norm)
        self.optim_ppo.step()

        return loss_policy.detach().item(), loss_value.detach().item(), \
            loss_conf.detach().item()

    def update_critic(self, states_source, states_target):
        with torch.no_grad():
            source_features = \
                self.network_ppo.calculate_features(states_source)
            target_features = \
                self.network_ppo.calculate_features(states_target)

        source_preds = self.network_critic(source_features)
        target_preds = self.network_critic(target_features)

        loss_critic = - torch.mean(source_preds) + torch.mean(target_preds)

        self.optim_critic.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.network_critic.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        for p in self.network_critic.parameters():
            p.data.clamp_(-self.clip_range_critic, self.clip_range_critic)

        return loss_critic.detach().item()

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.network_critic.state_dict(),
            os.path.join(save_dir, 'network_critic.pth'))

    def load_models(self, save_dir):
        super().load_models(save_dir)
        self.network_critic.load_state_dict(
            torch.load(os.path.join(save_dir, 'network_critic.pth')))
