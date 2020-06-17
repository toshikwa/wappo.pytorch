import os
import numpy as np
import torch
from torch.optim import RMSprop
from torch.nn.utils import clip_grad_norm_

from .ppo import PPOAgent
from wappo.network import AdversarialNetwork


class WAPPOAgent(PPOAgent):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, lr_ppo=5e-4, gamma=0.999,
                 rollout_length=16, num_minibatches=8, epochs_ppo=3,
                 clip_range_ppo=0.2, coef_value=0.5, coef_ent=0.01,
                 lambd=0.95, max_grad_norm=0.5, use_impala=True,
                 lr_critic=1e-4, epochs_critic=5, coef_conf=10.0,
                 clip_range_adv=0.01):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, lr_ppo,
            gamma, rollout_length, num_minibatches, epochs_ppo, clip_range_ppo,
            coef_value, coef_ent, lambd, max_grad_norm, use_impala)

        # Adversarial network.
        self.adv_network = AdversarialNetwork(
            feature_dim=self.ppo_network.feature_dim).to(device)

        # Optimizers.
        self.optim_critic = RMSprop(
            self.adv_network.parameters(), lr=lr_critic)

        self.epochs_critic = epochs_critic
        self.coef_conf = coef_conf
        self.clip_range_adv = clip_range_adv

    def update(self):
        loss_policies = []
        loss_values = []
        loss_critics = []
        loss_confs = []

        for samples in self.source_storage.iter(self.batch_size):
            self.update_steps += 1
            target_states = self.target_storage.sample(self.batch_size)

            # If update using confusion loss every 5 updates.
            update_conf = self.update_steps % self.epochs_critic == 0

            # Update PPO network.
            loss_policy, loss_value, loss_conf = \
                self.update_ppo(*samples, target_states, update_conf)
            loss_policies.append(loss_policy)
            loss_values.append(loss_value)
            if update_conf:
                loss_confs.append(loss_conf)

            # Update Critic network.
            loss_critics.append(self.update_critic(samples[0], target_states))

        self.writer.add_scalar(
            'loss/policy', np.mean(loss_policies), self.steps)
        self.writer.add_scalar(
            'loss/value', np.mean(loss_values), self.steps)
        self.writer.add_scalar(
            'loss/critic', np.mean(loss_critics), self.steps)
        self.writer.add_scalar(
            'loss/conf', np.mean(loss_confs), self.steps)

    def update_ppo(self, states, actions, values_old, value_targets,
                   log_probs_old, advantages, target_states,
                   update_conf=False):
        values, log_probs, mean_entropy, source_features = \
            self.ppo_network.evaluate_actions(states, actions)

        # >>> Value >>> #
        values_clipped = values_old + (values - values_old).clamp(
            -self.clip_range_ppo, self.clip_range_ppo)
        loss_value1 = (values - value_targets).pow(2)
        loss_value2 = (values_clipped - value_targets).pow(2)
        loss_value = 0.5 * torch.max(loss_value1, loss_value2).mean()
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
        if update_conf:
            target_features = self.ppo_network.body_net(target_states)
            source_preds = self.adv_network(source_features)
            target_preds = self.adv_network(target_features)
            loss_conf = torch.mean(source_preds) - torch.mean(target_preds)
        else:
            loss_conf = torch.tensor(0.0)
        # >>> Confusion >>> #

        # >>> Total >>> #
        loss = loss_policy - self.coef_ent * mean_entropy + \
            self.coef_value * loss_value + self.coef_conf * loss_conf
        # >>> Total >>> #

        self.optim_ppo.zero_grad()
        loss.backward()
        clip_grad_norm_(self.ppo_network.parameters(), self.max_grad_norm)
        self.optim_ppo.step()

        return loss_policy.detach().item(), loss_value.detach().item(), \
            loss_conf.detach().item()

    def update_critic(self, source_states, target_states):
        with torch.no_grad():
            source_features = self.ppo_network.body_net(source_states)
            target_features = self.ppo_network.body_net(target_states)

        source_preds = self.adv_network(source_features)
        target_preds = self.adv_network(target_features)

        loss_critic = - torch.mean(source_preds) + torch.mean(target_preds)

        self.optim_critic.zero_grad()
        loss_critic.backward()
        clip_grad_norm_(self.adv_network.parameters(), self.max_grad_norm)
        self.optim_critic.step()

        for p in self.adv_network.parameters():
            p.data.clamp_(-self.clip_range_adv, self.clip_range_adv)

        return loss_critic.detach().item()

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.adv_network.state_dict(),
            os.path.join(save_dir, 'adv_network.pth'))

    def load_models(self, save_dir):
        super().load_models(save_dir)
        self.adv_network.load_state_dict(
            torch.load(os.path.join(save_dir, 'adv_network.pth')))
