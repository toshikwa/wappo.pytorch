import torch
from torch.optim import RMSprop

from .ppo import PPOAgent
from wappo.network import CriticNetwork


class WAPPOAgent(PPOAgent):

    def __init__(self, source_venv, target_venv, log_dir, device,
                 num_steps=10**6, memory_size=10000, batch_size=256,
                 unroll_length=128, ppo_lr=5e-4, gamma=0.99,
                 ppo_clip_param=0.2, num_gradient_steps=3, value_loss_coef=0.5,
                 entropy_coef=0.01, lambd=0.95, max_grad_norm=0.5,
                 use_deep=True, critic_lr=5e-4, num_critic=5, use_gp=True,
                 lambda_gp=10.0, weight_clip_param=0.01):
        super().__init__(
            source_venv, target_venv, log_dir, device, num_steps, memory_size,
            batch_size, unroll_length, ppo_lr, gamma, ppo_clip_param,
            num_gradient_steps, value_loss_coef, entropy_coef, lambd,
            max_grad_norm, use_deep)

        # Critic network.
        self.critic_network = CriticNetwork(
            feature_dim=self.ppo_network.feature_dim).to(device)

        # Optimizer.
        self.critic_optim = RMSprop(
            self.critic_network.parameters(), lr=critic_lr)

        self.num_critic = num_critic
        self.use_gp = use_gp
        self.lambda_gp = lambda_gp
        self.weight_clip_param = weight_clip_param

    def update(self):
        mean_policy_loss = 0.0
        mean_value_loss = 0.0
        mean_feature_loss = 0.0
        mean_critic_loss = 0.0
        num_iters = 0

        for samples in self.source_storage.iter():
            for _ in range(self.num_critic):
                mean_critic_loss += self.update_critic() / self.num_critic

            policy_loss, value_loss, feature_loss = self.update_ppo(samples)
            mean_policy_loss += policy_loss
            mean_value_loss += value_loss
            mean_feature_loss += feature_loss
            num_iters += 1

        mean_policy_loss /= num_iters
        mean_value_loss /= num_iters
        mean_feature_loss /= num_iters
        mean_critic_loss /= num_iters

        self.writer.add_scalar('loss/policy', mean_policy_loss, self.steps)
        self.writer.add_scalar('loss/value', mean_value_loss, self.steps)
        self.writer.add_scalar('loss/feature', mean_feature_loss, self.steps)
        self.writer.add_scalar('loss/critic', mean_critic_loss, self.steps)

    def update_ppo(self, samples):
        states, actions, pred_values, \
            target_values, log_probs_old, advs = samples

        values, action_log_probs, dist_entropy, source_features = \
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

        target_states = self.target_storage.sample(self.device)
        target_features = self.ppo_network.body_net(target_states)

        source_preds = self.critic_network(source_features)
        target_preds = self.critic_network(target_features)

        feature_loss = -torch.mean(source_preds) + torch.mean(target_preds)

        self.ppo_optim.zero_grad()
        loss = policy_loss + self.value_loss_coef * value_loss \
            - self.entropy_coef * dist_entropy + feature_loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.ppo_network.parameters(), self.max_grad_norm)
        self.ppo_optim.step()

        return policy_loss.detach().item(), value_loss.detach().item(), \
            feature_loss.detach().item()

    def update_critic(self):
        source_states = self.source_storage.sample()
        target_states = self.target_storage.sample(self.device)

        with torch.no_grad():
            source_features = self.ppo_network.body_net(source_states)
            target_features = self.ppo_network.body_net(target_states)

        # Critic's predictions.
        source_preds = self.critic_network(source_features)
        target_preds = self.critic_network(target_features)

        if self.use_gp:
            penalty = -self.lambda_gp * self.calculate_gradient_penalty(
                source_features, target_features)
        else:
            penalty = 0.0

        # Calculate critic's loss.
        critic_loss = \
            torch.mean(source_preds) - torch.mean(target_preds) + penalty

        self.critic_optim.zero_grad()
        critic_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.critic_network.parameters(), self.max_grad_norm)
        self.critic_optim.step()

        # Clip weights of the critic.
        if not self.use_gp:
            for p in self.critic_network.parameters():
                p.data.clamp_(
                    -self.weight_clip_param, self.weight_clip_param)

        return critic_loss.detach().item()

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
