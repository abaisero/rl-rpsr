from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet

import numpy as np
import numpy.linalg as la

from rl_psr.core import Interaction, Test
from rl_psr.pomdp import POMDP_Model

from .search import outcome, outcome_matrix

__all__ = ['PSR_Model']


class PSR_Model:
    def __init__(self, pomdp_model: POMDP_Model, Q: FrozenSet[Test]):
        self.pomdp_model = pomdp_model

        self.U = self.outcome_matrix(Q)
        self.U_PI = la.pinv(self.U)

        # (|A|, |O|, |Q|, |Q|) array, M_{ao} \in \mathbb{R}^{|Q|\times|Q|}
        self.M_ao = np.einsum(
            'ij,aojk,kl->aoil', self.U.T, pomdp_model.G, self.U_PI.T
        )

        # (|A|, |O|, |Q|, |Q|) array, M_{aoQ} \in \mathbb{R}^{|Q|\times|Q|}
        self.M_aoQ = np.array(
            [
                [
                    np.column_stack(
                        [self._m(q.prepend(Interaction(a, o))) for q in Q]
                    )
                    for o in range(pomdp_model.observation_space.n)
                ]
                for a in range(pomdp_model.action_space.n)
            ]
        )

        # (|A|, |O|, |Q|) array, m_{ao} \in \mathbb{R}^{|Q|}
        self.m_ao = np.array(
            [
                [
                    self._m(Interaction(a, o).as_test())
                    for o in range(pomdp_model.observation_space.n)
                ]
                for a in range(pomdp_model.action_space.n)
            ]
        )

        # (|Q|, |A|) array
        self.R = self.U_PI @ pomdp_model.R

        self.discount = pomdp_model.discount
        self.actions = pomdp_model.actions
        self.observations = pomdp_model.observations
        self.start = self.psr(pomdp_model.start)

        self.action_space = pomdp_model.action_space
        self.observation_space = pomdp_model.observation_space
        self.reward_set = set((self.U @ self.R).flatten())
        self.reward_range = min(self.reward_set), max(self.reward_set)

        self.rank = self.U.shape[1]

    def outcome(self, test: Test):
        return outcome(self.pomdp_model, test)

    def outcome_matrix(self, tests: FrozenSet[Test]):
        return outcome_matrix(self.pomdp_model, tests)

    def psr(self, belief):
        return belief @ self.U

    @lru_cache(maxsize=None)
    def _m(self, test: Test):
        # TODO decompose outcome(intent) into individual matrices stuff
        # return self.U_PI @ self.M(intent.test) @ self...
        return self.U_PI @ self.outcome(test)

    def dynamics(self, state, action, observation):
        M = self.M_aoQ[action, observation]
        m = self.m_ao[action, observation]
        return (state @ M) / (state @ m)

    def observation_probs(self, state, action):
        return state @ self.m_ao[action, :, :].T

    def expected_reward(self, state, action):
        return state @ self.R[:, action]

    def R_as_pomdp(self):
        return self.U @ self.R
