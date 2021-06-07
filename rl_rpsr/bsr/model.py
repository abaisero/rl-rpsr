from __future__ import annotations

from gym_pomdps.belief import belief_step, expected_obs, expected_reward
from rl_rpsr.pomdp import POMDP_Model

__all__ = ['BSR_Model']


class BSR_Model:
    def __init__(self, pomdp_model: POMDP_Model):
        self.pomdp_model = pomdp_model

        self.R = pomdp_model.R
        self.G = pomdp_model.G

        self.discount = pomdp_model.discount
        self.actions = pomdp_model.actions
        self.observations = pomdp_model.observations
        self.start = pomdp_model.start

        self.state_space = pomdp_model.state_space
        self.action_space = pomdp_model.action_space
        self.observation_space = pomdp_model.observation_space
        self.reward_range = pomdp_model.reward_range

        self.rank = self.state_space.n

    def dynamics(self, state, action, observation):
        return belief_step(self.pomdp_model.env, state, action, observation)

    def observation_probs(self, state, action):
        return expected_obs(self.pomdp_model.env, state, action)

    def expected_reward(self, state, action):
        return expected_reward(self.pomdp_model.env, state, action)
