import os
import torch
from torch import nn
from torch.optim import Adam

from .base import BaseAgent
from wappo.network import PPONetwork


class PPOAgent(BaseAgent):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, memory_size=10000, batch_size=256,
                 unroll_length=128, lr=5e-4, adam_eps=1e-5, gamma=0.999,
                 ppo_clip_param=0.2, num_gradient_steps=4, value_loss_coef=0.5,
                 entropy_coef=0.01, lambd=0.95, max_grad_norm=0.5):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, memory_size,
            batch_size, unroll_length, gamma, ppo_clip_param,
            num_gradient_steps, value_loss_coef, entropy_coef, lambd,
            max_grad_norm)

        # PPO network.
        self.ppo_network = PPONetwork(
            self.source_venv.observation_space.shape,
            self.source_venv.action_space.n).to(device)

        # Optimizer.
        self.ppo_optim = Adam(
            self.ppo_network.parameters(), lr=lr, eps=adam_eps)

    def update(self):
        self.update_ppo()

    def update_ppo(self):
        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        num_iters = 0

        for sample in self.source_storage.iter():
            states, actions, pred_values, \
                target_values, log_probs_old, advs = sample

            # Reshape to do in a single forward pass for all steps.
            values, action_log_probs, dist_entropy = \
                self.ppo_network.evaluate_actions(states, actions)

            ratio = torch.exp(action_log_probs - log_probs_old)

            policy_loss = -torch.min(ratio * advs, torch.clamp(
                ratio, 1.0 - self.ppo_clip_param,
                1.0 + self.ppo_clip_param) * advs).mean()

            value_pred_clipped = pred_values + (
                values - pred_values
            ).clamp(-self.ppo_clip_param, self.ppo_clip_param)

            value_loss = 0.5 * torch.max(
                (values - target_values).pow(2),
                (value_pred_clipped - target_values).pow(2)
            ).mean()

            self.ppo_optim.zero_grad()
            loss = policy_loss + self.value_loss_coef * value_loss \
                - self.entropy_coef * dist_entropy
            loss.backward()
            nn.utils.clip_grad_norm_(
                self.ppo_network.parameters(), self.max_grad_norm)
            self.ppo_optim.step()

            num_iters += 1
            mean_policy_loss += policy_loss.detach()
            mean_value_loss += value_loss.detach()

        mean_policy_loss /= num_iters
        mean_value_loss /= num_iters

        self.writer.add_scalar('loss/policy', mean_policy_loss, self.steps)
        self.writer.add_scalar('loss/value', mean_value_loss, self.steps)

    def save_models(self, filename):
        torch.save(
            self.ppo_network.state_dict(),
            os.path.join(self.model_dir, filename))

    def load_models(self, filename):
        self.ppo_network.load_state_dict(
            torch.load(os.path.join(self.model_dir, filename)))
