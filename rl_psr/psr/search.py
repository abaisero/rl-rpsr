import logging
from typing import FrozenSet

import numpy as np

from rl_psr.core import Test, Tests
from rl_psr.linalg import linearly_independent
from rl_psr.pomdp import POMDP_Model
from rl_psr.search import Searcher
from rl_psr.util import SearchType, interactions

__all__ = ['searcher_factory']


def outcome(model: POMDP_Model, test: Test):
    logger = logging.getLogger(__name__)
    logger.debug('computing outcome vector of %s', test)

    u = np.ones(model.state_space.n)
    for interaction in reversed(test.interactions):
        G = model.G[interaction.action, interaction.observation]
        u = G.T @ u

    return u


def outcome_matrix(model: POMDP_Model, tests: FrozenSet[Test]):
    logger = logging.getLogger(__name__)
    logger.debug('computing outcome matrix of %s', tests)
    return np.column_stack([outcome(model, test) for test in tests])


def independent(model: POMDP_Model, test: Test, Q: FrozenSet[Test]) -> bool:
    logger = logging.getLogger(__name__)
    logger.debug('checking independence between %s and %s', test, Q)

    vectors = [outcome(model, test) for test in Q]
    vector = outcome(model, test)

    ret = linearly_independent(vectors, vector)
    logger.debug('independence result %s', ret)

    return ret


def searcher_factory(search_type: SearchType) -> Searcher:
    if search_type == SearchType.BFS:
        return BFS_PSR_Searcher()

    if search_type == SearchType.DFS:
        return DFS_PSR_Searcher()

    raise ValueError(f'No implementation for search type {search_type}')


class BFS_PSR_Searcher(Searcher):
    def search(self, model: POMDP_Model) -> Tests:
        Q: FrozenSet[Test] = frozenset()

        for interaction in interactions(
            model.action_space, model.observation_space
        ):
            test = interaction.as_test()
            if independent(model, test, Q):
                Q = Q.union([test])

        added = True
        while added:
            added = False
            for test in Q:
                for interaction in interactions(
                    model.action_space, model.observation_space
                ):
                    test_extended = test.prepend(interaction)
                    if independent(model, test_extended, Q):
                        Q = Q.union([test_extended])
                        added = True

        return Tests(tuple(Q))


class DFS_PSR_Searcher(Searcher):
    def search(self, model: POMDP_Model) -> Tests:
        Q = _search_dfs(model, Test.empty(), frozenset())
        return Tests(tuple(Q))


def _search_dfs(
    model: POMDP_Model, test: Test, Q: FrozenSet[Test]
) -> FrozenSet[Test]:

    for interaction in interactions(
        model.action_space, model.observation_space
    ):
        test_extended = test.prepend(interaction)
        if independent(model, test_extended, Q):
            Q_extended = Q.union({test_extended})
            Q = _search_dfs(model, test_extended, Q_extended)

    return Q
