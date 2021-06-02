from __future__ import annotations

from functools import lru_cache
from typing import FrozenSet

import numpy as np
import numpy.linalg as la
from rl_psr.core import Intent, Interaction
from rl_psr.pomdp import POMDP_Model

from .search import outcome, outcome_matrix

__all__ = ['RPSR_Model']


class RPSR_Model:
    def __init__(self, pomdp_model: POMDP_Model, I: FrozenSet[Intent]):
        self.pomdp_model = pomdp_model

        self.V = self.outcome_matrix(I)
        self.V_PI = la.pinv(self.V)

        # (|A|, |O|, |I|, |I|) array, M_{ao} \in \mathbb{R}^{|I|\times|I|}
        self.M_ao = np.einsum(
            'ij,aojk,kl->aoil', self.V.T, pomdp_model.G, self.V_PI.T
        )

        # (|A|, |O|, |I|, |I|) array, M_{ao} \in \mathbb{R}^{|I|\times|I|}
        self.M_aoI = np.array(
            [
                [
                    np.column_stack(
                        [self._m(i.prepend(Interaction(a, o))) for i in I]
                    )
                    for o in range(pomdp_model.observation_space.n)
                ]
                for a in range(pomdp_model.action_space.n)
            ]
        )

        # (|A|, |O|, |I|) array, m_{ao} \in \mathbb{R}^{|I|}
        self.m_ao = np.array(
            [
                [
                    self._m(Intent.actionless(Interaction(a, o).as_test()))
                    for o in range(pomdp_model.observation_space.n)
                ]
                for a in range(pomdp_model.action_space.n)
            ]
        )

        # (|I|, |A|) array
        self.R = self.V_PI @ pomdp_model.R

        self.discount = pomdp_model.discount
        self.actions = pomdp_model.actions
        self.observations = pomdp_model.observations
        self.start = self.rpsr(pomdp_model.start)

        self.action_space = pomdp_model.action_space
        self.observation_space = pomdp_model.observation_space
        self.reward_set = set((self.V @ self.R).flatten())
        self.reward_range = min(self.reward_set), max(self.reward_set)

        self.rank = self.V.shape[1]

    def outcome(self, intent: Intent):
        return outcome(self.pomdp_model, intent)

    def outcome_matrix(self, intents: FrozenSet[Intent]):
        return outcome_matrix(self.pomdp_model, intents)

    def rpsr(self, belief):
        return belief @ self.V

    @lru_cache(maxsize=None)
    def _m(self, intent):
        # TODO decompose outcome(intent) into individual matrices stuff
        # return self.V_PI @ self.M(intent.test) @ self...
        return self.V_PI @ self.outcome(intent)

    def dynamics(self, state, action, observation):
        M = self.M_aoI[action, observation]
        m = self.m_ao[action, observation]
        return (state @ M) / (state @ m)

    def observation_probs(self, state, action):
        return state @ self.m_ao[action, :, :].T

    def expected_reward(self, state, action):
        return state @ self.R[:, action]

    def R_as_pomdp(self):
        return self.V @ self.R
