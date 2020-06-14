from functools import partial
import torch
from torch import nn


def init_fn(m, gain=1):
    if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
        nn.init.orthogonal_(m.weight.data, gain=gain)
        nn.init.constant_(m.bias.data, 0)


class Flatten(nn.Module):

    def forward(self, x):
        return x.view(x.size(0), -1)


class PPONetwork(nn.Module):

    def __init__(self, img_shape, action_dim, feature_dim=512):
        super().__init__()
        self.body_net = CNNBody(img_shape[0], feature_dim)
        self.value_net = nn.Linear(feature_dim, 1).apply(init_fn)
        self.policy_net = Categorical(feature_dim, action_dim)

    def forward(self, states, deterministic=False):
        features = self.body_net(states)
        values = self.value_net(features)
        action_dists = self.policy_net(features)

        if deterministic:
            actions = action_dists.mode()
        else:
            actions = action_dists.sample()
        log_probs = action_dists.log_probs(actions)
        return values, actions, log_probs

    def calculate_values(self, states):
        features = self.body_net(states)
        values = self.value_net(features)
        return values

    def evaluate_actions(self, states, actions):
        features = self.body_net(states)
        values = self.value_net(features)
        action_dists = self.policy_net(features)

        log_probs = action_dists.log_probs(actions)
        mean_entropy = action_dists.entropy().mean()
        return values, log_probs, mean_entropy


class WAPPONetwork(PPONetwork):

    def __init__(self, img_shape, action_dim, feature_dim=512):
        super().__init__(img_shape, action_dim, feature_dim)

        self.critic_net = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1)).apply(partial(
                init_fn, gain=nn.init.calculate_gain('leaky_relu', 0.2)))

    def predict(self, features):
        return self.critic_net(features)


class CNNBody(nn.Module):

    def __init__(self, num_channels, feature_dim=512):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(num_channels, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, stride=1),
            nn.ReLU(),
            Flatten(),
            nn.Linear(32 * 4 * 4, feature_dim),
            nn.ReLU()
        ).apply(partial(init_fn, gain=nn.init.calculate_gain('relu')))

    def forward(self, states):
        if states.dtype == torch.uint8:
            states = states.float() / 255.0
        return self.net(states)


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(
            actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):

    def __init__(self, feature_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(feature_dim, action_dim).apply(
            partial(init_fn, gain=0.01))

    def forward(self, features):
        return FixedCategorical(logits=self.linear(features))
