import numpy as np
import torch

from .utils import normalize


class SourceStorage:

    def __init__(self, num_envs, rollout_length, epochs_ppo, img_shape,
                 gamma, lambd, device):

        # Transitions.
        self.states = torch.empty(
            rollout_length + 1, num_envs, *img_shape, device=device,
            dtype=torch.uint8)
        self.rewards = torch.empty(
            rollout_length, num_envs, 1, device=device)
        self.actions = torch.empty(
            rollout_length, num_envs, 1, device=device, dtype=torch.long)
        self.dones = torch.empty(
            rollout_length + 1, num_envs, 1, device=device)

        # Log of action probabilities based on the current policy.
        self.log_probs = torch.empty(
            rollout_length, num_envs, 1, device=device)
        # Predictions of V(s_t) based on the current value function.
        self.values = torch.empty(
            rollout_length + 1, num_envs, 1, device=device)
        # Target estimate of V(s_t) based on rollouts.
        self.value_targets = torch.empty(
            rollout_length, num_envs, 1, device=device)

        self.advantages = torch.empty(
            rollout_length, num_envs, 1, device=device)

        self.step = 0
        self.total_batches = num_envs * rollout_length

        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.epochs_ppo = epochs_ppo
        self.img_shape = img_shape
        self.gamma = gamma
        self.lambd = lambd
        self._is_ready = False

    def init_states(self, states):
        self.states[0].copy_(states)
        self.dones[0].fill_(0)

    def insert(self, next_states, actions, rewards, dones, log_probs,
               values):
        self.states[self.step + 1].copy_(next_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(torch.from_numpy(rewards[..., None]))
        self.dones[self.step + 1].copy_(torch.from_numpy(dones[..., None]))

        self.log_probs[self.step].copy_(log_probs)
        self.values[self.step].copy_(values)
        self.step = (self.step + 1) % self.rollout_length

    def end_rollout(self, next_value):
        assert not self._is_ready
        self._is_ready = True

        self.values[-1].copy_(next_value)
        advantage = 0
        for step in reversed(range(self.rollout_length)):
            td_error = self.rewards[step] + \
                self.gamma * self.values[step+1] * (1 - self.dones[step])\
                - self.values[step]
            advantage = td_error + \
                self.gamma * self.lambd * (1 - self.dones[step]) * advantage
            self.value_targets[step].copy_(advantage + self.values[step])

        self.advantages.copy_(normalize(self.value_targets - self.values[:-1]))

    def iter(self, batch_size):
        assert self._is_ready

        for _ in range(self.epochs_ppo):
            indices = np.arange(self.total_batches)
            np.random.shuffle(indices)

            for start in range(0, self.total_batches, batch_size):
                idxes = indices[start:start+batch_size]
                states = self.states[:-1].view(-1, *self.img_shape)[idxes]
                actions = self.actions.view(-1, self.actions.size(-1))[idxes]
                values = self.values[:-1].view(-1, 1)[idxes]
                value_targets = self.value_targets.view(-1, 1)[idxes]
                log_probs = self.log_probs.view(-1, 1)[idxes]
                advantages = self.advantages.view(-1, 1)[idxes]

                yield states, actions, values, value_targets, \
                    log_probs, advantages

        self.states[0].copy_(self.states[-1])
        self.dones[0].copy_(self.dones[-1])
        self._is_ready = False

    def sample(self, batch_size):
        indices = np.random.randint(
            low=0, high=self.total_batches, size=batch_size)
        states = self.states.view(-1, *self.img_shape)[indices]
        return states


class TargetStorage:

    def __init__(self, num_envs, rollout_length, img_shape, device):

        self.states = torch.empty(
            rollout_length + 1, num_envs, *img_shape, device=device,
            dtype=torch.uint8)

        self.step = 0
        self.total_batches = num_envs * (rollout_length + 1)

        self.num_envs = num_envs
        self.rollout_length = rollout_length
        self.img_shape = img_shape

    def init_states(self, states):
        self.states[0].copy_(states)

    def insert(self, next_states):
        self.states[self.step + 1].copy_(next_states)
        self.step = (self.step + 1) % self.rollout_length

    def sample(self, batch_size):
        indices = np.random.randint(
            low=0, high=self.total_batches, size=batch_size)
        states = self.states.view(-1, *self.img_shape)[indices]
        return states
