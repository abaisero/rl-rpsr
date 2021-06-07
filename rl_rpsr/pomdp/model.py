from __future__ import annotations

import logging

import gym
from gym_pomdps import POMDP
from rl_rpsr import matrices

__all__ = ['POMDP_Model']


class POMDP_Model:
    def __init__(self, env: POMDP):
        self.env = env
        self.T = env.T
        self.O = env.O
        self.R = matrices.R(env)
        self.G = matrices.G(env)
        self.D = matrices.D(env)

        self.discount = env.model.discount
        self.states = env.model.states
        self.actions = env.model.actions
        self.observations = env.model.observations
        self.start = env.start

        self.state_space = env.state_space
        self.action_space = env.action_space
        self.observation_space = env.observation_space
        self.reward_range = env.reward_range

    @staticmethod
    def make(name) -> POMDP_Model:
        logger = logging.getLogger(__name__)
        logger.info('making %s', name)

        try:
            env = gym.make(name)
        except gym.error.Error:
            logger.info('could not gym.make %s. Loading from filename', name)
            try:
                with open(name) as f:
                    env = POMDP(f.read(), episodic=False)
            except FileNotFoundError:
                logger.exception('could not open filename %s', name)
                raise

        return POMDP_Model(env)
