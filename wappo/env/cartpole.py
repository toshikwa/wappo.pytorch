# It's mainly derived from https://github.com/joshnroy/gym-cartpole-visual.

import math
import numpy as np
from random import randrange
import gym
from gym import spaces
from gym.utils import seeding
from gym.envs.classic_control import rendering


class CartPoleVisualEnv(gym.Env):

    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }
    screen_width = 64
    screen_height = 64

    def __init__(self, num_levels=0, start_level=0, rendering_scale=1):
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
        self.rendering_scale = rendering_scale

        # Angle at which to fail the episode.
        self.theta_threshold_radians = 12 * 2 * math.pi / 360
        self.x_threshold = 2.4
        self.offset = start_level

        # Observations are images.
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(3, 64, 64), dtype=np.uint8)
        self.action_space = spaces.Discrete(2)

        self.num_levels = num_levels
        if self.num_levels == 0:
            self.seed = self.seed_set(randrange(2**32))
        else:
            to_set = self.offset
            self.seed = self.seed_set(to_set)

        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

        if self.num_levels == 0:
            self.seed = self.seed_set(randrange(2**32))
        else:
            self.seed = self.seed_set(self.offset)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.render()
        self.change_color()
        self.render()

    def seed_set(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return seed

    def step(self, action):
        state = self.state
        x, x_dot, theta, theta_dot = state
        force = self.force_mag if action == 1 else -self.force_mag
        costheta = math.cos(theta)
        sintheta = math.sin(theta)

        temp = (
            force + self.polemass_length * theta_dot * theta_dot * sintheta
        ) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.polelength *
            (4 / 3.0 - self.masspole * costheta * costheta / self.total_mass))
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
        done = x < -self.x_threshold \
            or x > self.x_threshold \
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
        dic = {'level_seed': np.int32(self.seed)}
        return self._process_obs(img), reward, done, dic

    def reset(self):
        if self.num_levels == 0:
            self.seed = self.seed_set(randrange(2**32))
        else:
            self.seed = self.seed_set(self.offset)
        self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,))
        self.steps_beyond_done = None
        self.change_color()
        return self._process_obs(self.render())

    def _process_obs(self, img):
        return np.transpose(img, axes=(2, 0, 1))

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
        world_width = self.x_threshold * 2
        scale = self.screen_width / world_width
        carty = 10
        polewidth = self.polewidth
        polelen = scale * 2 * self.polelength
        cartwidth = self.cartwidth
        cartheight = self.cartheight

        if self.viewer is None:
            self.viewer = rendering.Viewer(
                self.screen_width * self.rendering_scale,
                self.screen_height * self.rendering_scale)
            self.background = rendering.FilledPolygon([
                (0, 0),
                (0, 64 * self.rendering_scale),
                (64 * self.rendering_scale, 64 * self.rendering_scale),
                (64 * self.rendering_scale, 0)])
            self.background.set_color(
                self.backgroundcolor[0], self.backgroundcolor[1],
                self.backgroundcolor[2])
            self.viewer.add_geom(self.background)
            l, r, t, b = -cartwidth/2, cartwidth/2, cartheight/2, -cartheight/2
            axleoffset = cartheight / 4.0
            self.cart = rendering.FilledPolygon([
                (l * self.rendering_scale, b * self.rendering_scale),
                (l * self.rendering_scale, t * self.rendering_scale),
                (r * self.rendering_scale, t * self.rendering_scale),
                (r * self.rendering_scale, b * self.rendering_scale)])
            self.carttrans = rendering.Transform()
            self.cart.add_attr(self.carttrans)
            self.viewer.add_geom(self.cart)
            l, r, t, b = \
                -polewidth/2, polewidth/2, polelen-polewidth/2, -polewidth/2
            self.pole = rendering.FilledPolygon([
                (l * self.rendering_scale, b * self.rendering_scale),
                (l * self.rendering_scale, t * self.rendering_scale),
                (r * self.rendering_scale, t * self.rendering_scale),
                (r * self.rendering_scale, b * self.rendering_scale)])
            self.pole.set_color(
                self.polecolor[0], self.polecolor[1], self.polecolor[2])
            self.poletrans = rendering.Transform(
                translation=(0, axleoffset * self.rendering_scale))
            self.pole.add_attr(self.poletrans)
            self.pole.add_attr(self.carttrans)
            self.viewer.add_geom(self.pole)
            self.axle = rendering.make_circle(
                polewidth / 2 * self.rendering_scale)
            self.axle.add_attr(self.poletrans)
            self.axle.add_attr(self.carttrans)
            self.axle.set_color(
                self.axlecolor[0], self.axlecolor[1], self.axlecolor[2])
            self.cart.set_color(
                self.cartcolor[0], self.cartcolor[1], self.cartcolor[2])
            self.viewer.add_geom(self.axle)

            top = (carty + 0.5) * self.rendering_scale
            bottom = (carty - 0.5) * self.rendering_scale
            self.track = rendering.FilledPolygon([
                (0, bottom),
                (0, top),
                (self.screen_width * self.rendering_scale, top),
                (self.screen_width * self.rendering_scale, bottom)
            ])
            self.track.set_color(
                self.trackcolor[0], self.trackcolor[1], self.trackcolor[2])
            self.viewer.add_geom(self.track)

        if self.state is None:
            return None

        x = self.state
        cartx = x[0] * scale + self.screen_width / 2.0
        self.carttrans.set_translation(
            cartx * self.rendering_scale, carty * self.rendering_scale)
        self.poletrans.set_rotation(-x[2] * self.rendering_scale)

        return self.viewer.render(return_rgb_array=True)[
            ::self.rendering_scale, ::self.rendering_scale]

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
