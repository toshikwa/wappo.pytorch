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

    def __init__(self, img_shape, action_dim):
        super().__init__()
        self.body_net = ImpalaCNNBody(img_shape[0])
        self.value_net = nn.Linear(256, 1)
        self.policy_net = Categorical(256, action_dim)

    def forward(self, states, deterministic=False):
        x, _ = self.body_net(states)
        values = self.value_net(x)
        action_dists = self.policy_net(x)

        if deterministic:
            actions = action_dists.mode()
        else:
            actions = action_dists.sample()
        log_probs = action_dists.log_probs(actions)
        return values, actions, log_probs

    def calculate_values(self, states):
        x, _ = self.body_net(states)
        values = self.value_net(x)
        return values

    def calculate_features(self, states):
        return self.body_net.calculate_features(states)

    def evaluate_actions(self, states, actions):
        x, features = self.body_net(states)
        values = self.value_net(x)
        action_dists = self.policy_net(x)
        log_probs = action_dists.log_probs(actions)
        mean_entropy = action_dists.entropy().mean()
        return values, log_probs, mean_entropy, features

    @property
    def feature_dim(self):
        return self.body_net.feature_dim


class ResidualBlock(nn.Module):

    def __init__(self, num_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(num_channels, num_channels, 3, 1, padding=1),
        )

    def forward(self, x):
        return self.net(x) + x


class ConvSequence(nn.Module):

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, padding=1),
            nn.MaxPool2d(3, 2, padding=1),
            ResidualBlock(out_channels),
            ResidualBlock(out_channels)
        )

    def forward(self, x):
        return self.net(x)


class ImpalaCNNBody(nn.Module):

    def __init__(self, num_channels, depths=(16, 32, 32),
                 num_initial_blocks=1):
        super().__init__()
        assert 1 <= num_initial_blocks <= 3

        self.feature_dim = depths[num_initial_blocks-1] * \
            (64 // 2 ** num_initial_blocks) ** 2

        in_channels = num_channels
        nets = []
        for out_channels in depths:
            nets.append(ConvSequence(in_channels, out_channels))
            in_channels = out_channels

        nets.append(
            nn.Sequential(
                Flatten(),
                nn.LeakyReLU(0.2),
                nn.Linear(32 * 8 * 8, 256),
                nn.LeakyReLU(0.2),
            )
        )

        self.initial_net = nn.Sequential(
            *[nets.pop(0) for _ in range(num_initial_blocks)])
        self.net = nn.Sequential(*nets)

    def forward(self, states):
        assert states.dtype == torch.uint8
        states = states.float().div_(255.0)
        features = self.initial_net(states)
        return self.net(features), features.view(-1, self.feature_dim)

    def calculate_features(self, states):
        assert states.dtype == torch.uint8
        states = states.float().div_(255.0)
        return self.initial_net(states).view(-1, self.feature_dim)


class FixedCategorical(torch.distributions.Categorical):

    def sample(self):
        return super().sample().unsqueeze(-1)

    def log_probs(self, actions):
        return super().log_prob(actions.squeeze(-1)).view(
            actions.size(0), -1).sum(-1).unsqueeze(-1)

    def mode(self):
        return self.probs.argmax(dim=-1, keepdim=True)


class Categorical(nn.Module):

    def __init__(self, input_dim, action_dim):
        super().__init__()
        self.linear = nn.Linear(input_dim, action_dim).apply(
            partial(init_fn, gain=0.01))

    def forward(self, x):
        return FixedCategorical(logits=self.linear(x))


class CriticNetwork(nn.Module):

    def __init__(self, feature_dim=16*32*32):
        super().__init__()

        self.net = nn.Sequential(
            nn.Tanh(),
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
            nn.Linear(512, 512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(512, 1))

    def forward(self, features):
        return self.net(features)
