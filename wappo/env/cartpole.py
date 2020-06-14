# -*- coding: utf-8 -*-
"""
Classic cart-pole system implemented by Rich Sutton et al.
Copied from http://incompleteideas.net/sutton/book/code/pole.c
permalink: https://perma.cc/C9ZM-652R
"""

import math
import gym
from gym import spaces
from gym.utils import seeding
import numpy as np


class CartPoleVisualEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self, num_levels=0, start_level=0):
        self.gravity = 9.8
        self.masscart = 1.0
        self.masspole = 0.1
        self.total_mass = (self.masspole + self.masscart)
        self.polelength = 5
        self.polewidth = 5
        self.cartwidth = 20
        self.cartheight = 10
        self.polemass_length = (self.masspole * self.polelength)
        self.force_mag = 10.
        self.tau = 0.02
        self.kinematics_integrator = 'euler'
        self.polecolor = np.array([0., 0., 1.])
        self.cartcolor = np.array([1., 1., 0.])
        self.axlecolor = np.array([1., 0., 1.])
        self.trackcolor = np.array([0., 1., 1.])
        self.backgroundcolor = np.array([1., 1., 1.])

        # Angle at which to fail the episode.
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4

        self.action_space = spaces.Discrete(2)
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)

        self.start_level = start_level
        self.num_levels = num_levels
        self.seed_set()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed_set(self):
        if self.num_levels == 0:
            seed = np.random.randint(low=0, high=2**32)
        else:
            seed = self.num_levels
        self.np_random, self.seed = seeding.np_random(seed)

    def step(self, action):
        assert self.action_space.contains(action)
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)
        temp = (force + self.polemass_length * theta_dot *
                theta_dot * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / \
            (self.polelength * (4.0/3.0 - self.masspole * costheta *
             costheta / self.total_mass))
        xacc = temp - \
            self.polemass_length * thetaacc * costheta / self.total_mass

        if self.kinematics_integrator == 'euler':
            x = x + self.tau * x_dot
            x_dot = x_dot + self.tau * xacc
            theta = theta + self.tau * theta_dot
            theta_dot = theta_dot + self.tau * thetaacc
        else:
            x_dot = x_dot + self.tau * xacc
            x = x + self.tau * x_dot
            theta_dot = theta_dot + self.tau * thetaacc
            theta = theta + self.tau * theta_dot

        self.state = (x, x_dot, theta, theta_dot)
        done = x < -self.x_threshold or x > self.x_threshold \
            or theta < -self.theta_threshold_radians \
            or theta > self.theta_threshold_radians
        done = bool(done)

        if not done:
            reward = 1.0
        elif self.steps_beyond_done is None:
            self.steps_beyond_done = 0
            reward = 1.0
        else:
            self.steps_beyond_done += 1
            reward = 0.0

        img = self.render()
        done = np.int64(done).astype(np.int32)
        dic = {"level_seed": np.int32(self.seed)}
        return self._process_obs(img), reward, done, dic

    def reset(self):
        self.seed_set()
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        img = self.render()
        self.change_color()
        img = self.render()
        return self._process_obs(img)

    def _process_obs(self, img):
        return np.transpose(img, axes=(2, 0, 1)).astype(np.uint8)

    def change_color(self):
        self.polecolor = np.clip(self.np_random.normal(0.5, 0.5, 3), 0., 1.)
        self.cartcolor = np.clip(self.np_random.normal(0.5, 0.5, 3), 0., 1.)
        self.axlecolor = np.clip(self.np_random.normal(0.5, 0.5, 3), 0., 1.)
        self.trackcolor = np.clip(self.np_random.normal(0.5, 0.5, 3), 0., 1.)
        self.backgroundcolor = np.clip(
            self.np_random.normal(0.5, 0.5, 3), 0., 1.)

        self.pole.set_color(
            self.polecolor[0], self.polecolor[1], self.polecolor[2])
        self.axle.set_color(
            self.axlecolor[0], self.axlecolor[1], self.axlecolor[2])
        self.cart.set_color(
            self.cartcolor[0], self.cartcolor[1], self.cartcolor[2])
        self.track.set_color(
            self.trackcolor[0], self.trackcolor[1], self.trackcolor[2])
        self.background.set_color(
            self.backgroundcolor[0], self.backgroundcolor[1],
            self.backgroundcolor[2])

    def render(self, mode='rgb'):
        screen_width = 64
        screen_height = 64

        world_width = self.x_threshold*2
        scale = screen_width/world_width
        carty = 10
        polewidth = self.polewidth
        polelen = scale * 2 * self.polelength
        cartwidth = self.cartwidth
        cartheight = self.cartheight

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_width, screen_height)
            self.background = rendering.FilledPolygon(
                [(0, 0), (0, 64), (64, 64), (64, 0)])
            self.background.set_color(
                self.backgroundcolor[0], self.backgroundcolor[1],
                self.backgroundcolor[2])
            self.viewer.add_geom(self.background)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight / 4.0
            self.cart = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.carttrans = rendering.Transform()
            self.cart.add_attr(self.carttrans)
            self.viewer.add_geom(self.cart)
            l, r, t, b = \
                -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            self.pole = rendering.FilledPolygon(
                [(l, b), (l, t), (r, t), (r, b)])
            self.pole.set_color(
                self.polecolor[0], self.polecolor[1], self.polecolor[2])
            self.poletrans = rendering.Transform(translation=(0, axleoffset))
            self.pole.add_attr(self.poletrans)
            self.pole.add_attr(self.carttrans)
            self.viewer.add_geom(self.pole)
            self.axle = rendering.make_circle(polewidth/2)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(
                self.axlecolor[0], self.axlecolor[1], self.axlecolor[2])
            self.cart.set_color(
                self.cartcolor[0], self.cartcolor[1], self.cartcolor[2])
            self.viewer.add_geom(self.axle)
            self.track = rendering.Line((0, carty), (screen_width, carty))
            self.track.set_color(
                self.trackcolor[0], self.trackcolor[1], self.trackcolor[2])
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0]*scale+screen_width / 2.0
        self.carttrans.set_translation(cartx, carty)
        self.poletrans.set_rotation(-x[2])

        return self.viewer.render(return_rgb_array=True)

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
