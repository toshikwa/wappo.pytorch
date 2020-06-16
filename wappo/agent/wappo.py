import os
import torch
from torch.optim import RMSprop

from .ppo import PPOAgent
from wappo.network import AdversarialNetwork


class WAPPOAgent(PPOAgent):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, memory_size=10**4, lr_ppo=5e-4, gamma=0.999,
                 rollout_length=16, num_minibatches=8, epochs_ppo=3,
                 clip_range_ppo=0.2, value_coef=0.5, ent_coef=0.01,
                 lambd=0.95, max_grad_norm=0.5, use_impala=True, lr_adv=5e-4,
                 epochs_critic=5, clip_range_adv=0.01):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, memory_size,
            lr_ppo, gamma, rollout_length, num_minibatches, epochs_ppo,
            clip_range_ppo, value_coef, ent_coef, lambd, max_grad_norm,
            use_impala)

        # Adversarial network.
        self.adv_network = AdversarialNetwork(
            feature_dim=self.ppo_network.feature_dim).to(device)

        # Optimizers.
        self.optim_critic = RMSprop(self.adv_network.parameters(), lr=lr_adv)
        self.optim_conf = RMSprop(
            self.ppo_network.body_net.parameters(), lr=lr_adv)

        self.epochs_critic = epochs_critic
        self.clip_range_adv = clip_range_adv

    def update(self):
        mean_loss_policy = 0.0
        mean_loss_value = 0.0
        mean_loss_conf = 0.0
        mean_loss_critic = 0.0
        num_iters = 0

        for samples in self.source_storage.iter(self.batch_size):
            loss_policy, loss_value = self.update_ppo(samples)
            mean_loss_conf += self.update_conf()
            mean_loss_critic += self.update_critic()

            mean_loss_policy += loss_policy
            mean_loss_value = loss_value
            num_iters += 1

        mean_loss_policy /= num_iters
        mean_loss_value /= num_iters

        self.writer.add_scalar('loss/policy', mean_loss_policy, self.steps)
        self.writer.add_scalar('loss/value', mean_loss_value, self.steps)
        self.writer.add_scalar('loss/conf', mean_loss_conf, self.steps)
        self.writer.add_scalar('loss/critic', mean_loss_critic, self.steps)

    def update_conf(self):
        source_states = self.source_storage.sample(self.batch_size)
        target_states = self.target_storage.sample(
            self.batch_size, self.device)

        source_features = self.ppo_network.body_net(source_states)
        target_features = self.ppo_network.body_net(target_states)

        source_preds = self.adv_network(source_features)
        target_preds = self.adv_network(target_features)

        self.optim_conf.zero_grad()
        loss_conf = -torch.mean(source_preds) + torch.mean(target_preds)
        loss_conf.backward()
        self.optim_conf.step()

        return loss_conf.detach().item()

    def update_critic(self):
        mean_loss_critic = 0.0

        for _ in range(self.epochs_critic):
            source_states = self.source_storage.sample(self.batch_size)
            target_states = self.target_storage.sample(
                self.batch_size, self.device)

            with torch.no_grad():
                source_features = self.ppo_network.body_net(source_states)
                target_features = self.ppo_network.body_net(target_states)

            source_preds = self.adv_network(source_features)
            target_preds = self.adv_network(target_features)

            loss_critic = torch.mean(source_preds) - torch.mean(target_preds)

            self.optim_critic.zero_grad()
            loss_critic.backward()
            self.optim_critic.step()

            for p in self.adv_network.parameters():
                p.data.clamp_(
                    -self.clip_range_adv, self.clip_range_adv)

            mean_loss_critic += loss_critic.detach().item()

        mean_loss_critic /= self.epochs_critic
        return mean_loss_critic

    def save_models(self, save_dir):
        super().save_models(save_dir)
        torch.save(
            self.adv_network.state_dict(),
            os.path.join(save_dir, 'adv_network.pth'))

    def load_models(self, save_dir):
        super().load_models(save_dir)
        self.adv_network.load_state_dict(
            torch.load(os.path.join(save_dir, 'adv_network.pth')))
