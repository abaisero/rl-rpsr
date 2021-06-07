from __future__ import annotations

import logging

import gym
import numpy as np
from gym.utils import seeding

from .model import BSR_Model


class BSR(gym.Env):  # pylint: disable=abstract-method
    def __init__(self, model: BSR_Model, seed=None):
        super().__init__()
        self.seed(seed)

        self.model = model
        self.discount = model.discount
        self.action_space = gym.spaces.Discrete(len(model.actions))
        self.observation_space = gym.spaces.Discrete(len(model.observations))
        # TODO actually compute this,  it's not just min/max anymore
        self.reward_range = model.R.min(), model.R.max()

        self.state = None

    def seed(self, seed):  # pylint: disable=signature-differs
        self.np_random, seed_ = seeding.np_random(seed)
        return [seed_]

    def reset(self):  # pylint: disable=arguments-differ
        self.state = self.model.start.copy()
        return self.state

    def step(self, action):
        reward = self.model.expected_reward(self.state, action)

        p = self.model.observation_probs(self.state, action)
        try:
            observation = self.np_random.multinomial(1, p).argmax()
        except ValueError:
            logger = logging.getLogger(__name__)
            logger.exception(
                'Exception on sample, p=%s.  Clipping between 0.0 and 1.0', p
            )
            # TODO find a better way to handle these kinds of numerical errors
            p = np.clip(p, 0.0, 1.0)
            observation = self.np_random.multinomial(1, p).argmax()

        self.state = self.model.dynamics(self.state, action, observation)

        done = False
        info = {'observation': observation}

        return self.state, reward, done, info
