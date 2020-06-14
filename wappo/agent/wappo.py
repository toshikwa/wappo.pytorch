import torch
from torch.optim import Adam

from .ppo import PPOAgent
from wappo.network import CriticNetwork


class WAPPOAgent(PPOAgent):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, memory_size=10000, batch_size=256,
                 unroll_length=128, lr=5e-4, adam_eps=1e-5, gamma=0.999,
                 ppo_clip_param=0.2, num_gradient_steps=4, value_loss_coef=0.5,
                 entropy_coef=0.01, lambd=0.95, max_grad_norm=0.5,
                 num_critic=5, weight_clip_param=0.01):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, memory_size,
            batch_size, unroll_length, lr, adam_eps, gamma, ppo_clip_param,
            num_gradient_steps, value_loss_coef, entropy_coef, lambd,
            max_grad_norm)

        # Critic network.
        self.critic_network = CriticNetwork().to(device)

        # Optimizer.
        self.critic_optim = Adam(
            self.critic_network.parameters(), lr=lr, eps=adam_eps)

        self.num_critic = num_critic
        self.weight_clip_param = weight_clip_param

    def update(self):
        self.update_ppo()
        self.update_adversarial()

    def update_adversarial(self):
        mean_critic_loss = 0.0

        for i in range(self.num_critic):
            source_states = self.source_storage.sample()
            target_states = self.target_storage.sample(self.device)
            with torch.no_grad():
                source_features = self.ppo_network.body_net(source_states)
                target_features = self.ppo_network.body_net(target_states)

            # Critic's predictions.
            source_preds = self.critic_network(source_features)
            target_preds = self.critic_network(target_features)

            # Calculate critic's loss.
            critic_loss = -torch.mean(source_preds) + torch.mean(target_preds)

            self.critic_optim.zero_grad()
            critic_loss.backward()
            self.critic_optim.step()

            # Clip weights of the critic.
            for p in self.critic_network.parameters():
                p.data.clamp_(-self.weight_clip_param, self.weight_clip_param)

            mean_critic_loss += critic_loss.detach() / self.num_critic

        source_states = self.source_storage.sample()
        target_states = self.target_storage.sample(self.device)
        source_features = self.ppo_network.body_net(source_states)
        target_features = self.ppo_network.body_net(target_states)

        # Critic's predictions.
        source_preds = self.critic_network(source_features)
        target_preds = self.critic_network(target_features)

        # Calculate encoder's loss.
        encoder_loss = torch.mean(source_preds) - torch.mean(target_preds)

        self.ppo_optim.zero_grad()
        encoder_loss.backward()
        self.ppo_optim.step()

        self.writer.add_scalar(
            'loss/critic', mean_critic_loss, self.steps)
        self.writer.add_scalar(
            'loss/encoder', encoder_loss.item(), self.steps)

    def calculate_gradient_penalty(self, source_features, target_features):
        # Random weight term for interpolation between real and fake samples.
        alpha = torch.rand(
            source_features.size(0), 1, dtype=torch.float, device=self.device)

        # Get random interpolation between real and fake samples.
        interpolates = alpha.mul(source_features).add_(
            (1 - alpha).mul(target_features)).requires_grad_(True)
        preds = self.critic_network(interpolates)

        # Calculate gradients using autograd.grad for second order derivatives.
        gradients = torch.autograd.grad(
            outputs=preds.sum(), inputs=interpolates, create_graph=True)[0]

        gradients = gradients.view(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty
