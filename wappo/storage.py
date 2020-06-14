import numpy as np
import torch
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler


class SourceStorage:

    def __init__(self, unroll_length, num_envs, batch_size, img_shape,
                 gamma, lambd, num_gradient_steps, device):

        # Transitions.
        self.states = torch.zeros(
            unroll_length + 1, num_envs, *img_shape, device=device)
        self.rewards = torch.zeros(
            unroll_length, num_envs, 1, device=device)
        self.actions = torch.zeros(
            unroll_length, num_envs, 1, device=device, dtype=torch.long)
        self.dones = torch.ones(
            unroll_length + 1, num_envs, 1, device=device)

        # Log of action probabilities based on the current policy.
        self.action_log_probs = torch.zeros(
            unroll_length, num_envs, 1, device=device)
        # Predictions of V(s_t) based on the current value function.
        self.pred_values = torch.zeros(
            unroll_length + 1, num_envs, 1, device=device)
        # Target estimate of V(s_t) based on rollouts.
        self.target_values = torch.zeros(
            unroll_length, num_envs, 1, device=device)

        self.step = 0
        self.unroll_length = unroll_length
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.img_shape = img_shape
        self.gamma = gamma
        self.lambd = lambd
        self.num_gradient_steps = num_gradient_steps

        self._is_ready = False

    def init_states(self, states):
        self.states[0].copy_(states)

    def insert(self, next_states, actions, rewards, dones, action_log_probs,
               pred_values):
        self.states[self.step + 1].copy_(next_states)
        self.actions[self.step].copy_(actions)
        self.rewards[self.step].copy_(torch.from_numpy(rewards[..., None]))
        self.dones[self.step + 1].copy_(torch.from_numpy(dones[..., None]))

        self.action_log_probs[self.step].copy_(action_log_probs)
        self.pred_values[self.step].copy_(pred_values)
        self.step = (self.step + 1) % self.unroll_length

    def end_rollout(self, next_value):
        assert not self._is_ready
        self._is_ready = True

        self.pred_values[-1].copy_(next_value)
        adv = 0
        for step in reversed(range(self.unroll_length)):
            td_error = self.rewards[step] + \
                self.gamma * self.pred_values[step+1] * (1 - self.dones[step])\
                - self.pred_values[step]
            adv = td_error + \
                self.gamma * self.lambd * (1 - self.dones[step]) * adv
            self.target_values[step] = adv + self.pred_values[step]

    def iter(self):
        assert self._is_ready

        # Calculate advantages.
        all_advs = self.target_values - self.pred_values[:-1]
        all_advs = (all_advs - all_advs.mean()) / (all_advs.std() + 1e-5)

        for _ in range(self.num_gradient_steps):
            # Sampler for indices.
            sampler = BatchSampler(
                SubsetRandomSampler(
                    range(self.num_envs * self.unroll_length)),
                self.batch_size, drop_last=True)

            for indices in sampler:
                states = self.states[:-1].view(-1, *self.img_shape)[indices]
                actions = self.actions.view(-1, self.actions.size(-1))[indices]
                pred_values = self.pred_values[:-1].view(-1, 1)[indices]
                target_values = self.target_values.view(-1, 1)[indices]
                action_log_probs = self.action_log_probs.view(-1, 1)[indices]
                advs = all_advs.view(-1, 1)[indices]

                yield states, actions, pred_values, \
                    target_values, action_log_probs, advs

        self.states[0].copy_(self.states[-1])
        self.dones[0].copy_(self.dones[-1])
        self._is_ready = False


class TargetStorage:

    def __init__(self, memory_size, num_envs, batch_size, img_shape, device):

        self.states = torch.zeros(
            memory_size, num_envs, *img_shape, device=device)

        self._p = 0
        self._n = 0
        self.memory_size = memory_size
        self.num_envs = num_envs
        self.batch_size = batch_size
        self.img_shape = img_shape

    def insert(self, states):
        self.states[self._p].copy_(states)
        self._p = (self._p + 1) % self.memory_size
        self._n = min(self._n + 1, self.memory_size)

    def sample(self):
        indices = np.random.randint(low=0, high=self._n, size=self.batch_size)
        states = self.states[:self._n].view(-1, *self.img_shape)[indices]
        return states
