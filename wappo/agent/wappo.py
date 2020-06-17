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
                 clip_range_ppo=0.2, value_coef=0.5, ent_coef=0.01,
                 lambd=0.95, max_grad_norm=0.5, use_impala=True,
                 lr_critic=1e-4, lr_conf=1e-4, batch_size_adv=512,
                 epochs_critic=5, clip_range_adv=0.01):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, lr_ppo,
            gamma, rollout_length, num_minibatches, epochs_ppo, clip_range_ppo,
            value_coef, ent_coef, lambd, max_grad_norm, use_impala)

        # Adversarial network.
        self.adv_network = AdversarialNetwork(
            feature_dim=self.ppo_network.feature_dim).to(device)

        # Optimizers.
        self.optim_critic = RMSprop(
            self.adv_network.parameters(), lr=lr_critic)
        self.optim_conf = RMSprop(
            self.ppo_network.body_net.parameters(), lr=lr_conf)

        self.batch_size_adv = batch_size_adv
        self.epochs_critic = epochs_critic
        self.clip_range_adv = clip_range_adv

    def update(self):
        loss_policies = []
        loss_values = []
        loss_critics = []
        loss_confs = []

        for samples in self.source_storage.iter(self.batch_size_ppo):
            loss_critics.append(self.update_critic())
            loss_confs.append(self.update_conf())

            loss_policy, loss_value = self.update_ppo(samples)
            loss_policies.append(loss_policy)
            loss_values.append(loss_value)

        self.writer.add_scalar(
            'loss/policy', np.mean(loss_policies), self.steps)
        self.writer.add_scalar(
            'loss/value', np.mean(loss_values), self.steps)
        self.writer.add_scalar(
            'loss/critic', np.mean(loss_critics), self.steps)
        self.writer.add_scalar(
            'loss/conf', np.mean(loss_confs), self.steps)

    def update_conf(self):
        source_states = self.source_storage.sample(self.batch_size_adv)
        target_states = self.target_storage.sample(self.batch_size_adv)

        source_features = self.ppo_network.body_net(source_states)
        target_features = self.ppo_network.body_net(target_states)

        source_preds = self.adv_network(source_features)
        target_preds = self.adv_network(target_features)

        loss_conf = -torch.mean(source_preds) + torch.mean(target_preds)

        self.optim_conf.zero_grad()
        loss_conf.backward()
        clip_grad_norm_(self.ppo_network.parameters(), self.max_grad_norm)
        self.optim_conf.step()

        return loss_conf.detach().item()

    def update_critic(self):
        loss_critics = []

        for _ in range(self.epochs_critic):
            source_states = self.source_storage.sample(self.batch_size_adv)
            target_states = self.target_storage.sample(self.batch_size_adv)

            with torch.no_grad():
                source_features = self.ppo_network.body_net(source_states)
                target_features = self.ppo_network.body_net(target_states)

            source_preds = self.adv_network(source_features)
            target_preds = self.adv_network(target_features)

            loss_critic = torch.mean(source_preds) - torch.mean(target_preds)

            self.optim_critic.zero_grad()
            loss_critic.backward()
            clip_grad_norm_(self.adv_network.parameters(), self.max_grad_norm)
            self.optim_critic.step()

            for p in self.adv_network.parameters():
                p.data.clamp_(-self.clip_range_adv, self.clip_range_adv)

            loss_critics.append(loss_critic.detach().item())

        return np.mean(loss_critics)

    def calculate_gradient_penalty(self, source_features, target_features):
        # Random weight term for interpolation between real and fake samples.
        alpha = torch.rand(
            source_features.size(0), 1, dtype=torch.float, device=self.device)

        # Get random interpolation between real and fake samples.
        interpolates = alpha.mul(source_features).add_(
            (1 - alpha).mul(target_features)).requires_grad_(True)
        preds = self.adv_network(interpolates)

        # Calculate gradients using autograd.grad for second order derivatives.
        gradients = torch.autograd.grad(
            outputs=preds.sum(), inputs=interpolates, create_graph=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.adv_network.state_dict(),
            os.path.join(save_dir, 'adv_network.pth'))

    def load_models(self, save_dir):
        super().load_models(save_dir)
        self.adv_network.load_state_dict(
            torch.load(os.path.join(save_dir, 'adv_network.pth')))
