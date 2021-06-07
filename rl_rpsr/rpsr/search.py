import logging
from functools import lru_cache
from typing import FrozenSet

import numpy as np
from rl_rpsr.core import Intent, Intents, Test
from rl_rpsr.linalg import linearly_independent
from rl_rpsr.pomdp import POMDP_Model
from rl_rpsr.search import Searcher
from rl_rpsr.util import SearchType, interactions

__all__ = ['searcher_factory']


@lru_cache(maxsize=None)
def outcome(model: POMDP_Model, intent: Intent):
    logger = logging.getLogger(__name__)
    logger.debug('computing outcome vector of %s', intent)

    v = (
        model.R[:, intent.action]
        if intent.action >= 0
        else np.ones(model.state_space.n)
    )
    for interaction in reversed(intent.test.interactions):
        v = model.G[interaction.action, interaction.observation].T @ v

    return v


@lru_cache(maxsize=None)
def outcome_matrix(model: POMDP_Model, intents: FrozenSet[Intent]):
    logger = logging.getLogger(__name__)
    logger.debug('computing outcome matrix of %s', intents)
    return np.column_stack([outcome(model, intent) for intent in intents])


def independent(
    model: POMDP_Model, intent: Intent, I: FrozenSet[Intent]
) -> bool:

    logger = logging.getLogger(__name__)
    logger.debug('checking independence between %s and %s', intent, I)

    vectors = [outcome(model, intent) for intent in I]
    vector = outcome(model, intent)

    ret = linearly_independent(vectors, vector)
    logger.debug('independence result %s', ret)

    return ret


def searcher_factory(search_type: SearchType) -> Searcher:
    if search_type == SearchType.BFS:
        return BFS_RPSR_Searcher()

    if search_type == SearchType.DFS:
        return DFS_RPSR_Searcher()

    raise ValueError(f'No implementation for search type {search_type}')


class BFS_RPSR_Searcher(Searcher):
    def search(self, model: POMDP_Model) -> Intents:
        I: FrozenSet[Intent] = frozenset()

        for z in range(-1, model.action_space.n):
            intent = Intent(Test.empty(), z)
            if independent(model, intent, I):
                I = I.union([intent])

        added = True
        while added:
            added = False
            for intent in I:
                for interaction in interactions(
                    model.action_space, model.observation_space
                ):
                    intent_extended = intent.prepend(interaction)
                    if independent(model, intent_extended, I):
                        I = I.union([intent_extended])
                        added = True

        return Intents(tuple(I))


class DFS_RPSR_Searcher(Searcher):
    def search(self, model: POMDP_Model) -> Intents:
        I = _search_dfs(model, Test.empty(), frozenset())
        return Intents(tuple(I))


def _search_dfs(
    model: POMDP_Model, test: Test, I: FrozenSet[Intent]
) -> FrozenSet[Intent]:

    for z in range(-1, model.action_space.n):
        intent = Intent(test, z)
        if independent(model, intent, I):
            for interaction in interactions(
                model.action_space, model.observation_space
            ):
                test_extended = test.prepend(interaction)
                I_extended = I.union([intent])
                I = _search_dfs(model, test_extended, I_extended)

    return I
