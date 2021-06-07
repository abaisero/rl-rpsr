from __future__ import annotations

import abc
import logging

import numpy as np
from rl_rpsr.linalg import max_bigraph_distance
from rl_rpsr.value_function import ValueFunction

__all__ = ['AlphaVF_Metric', 'BellmanAtStartVF_Metric']


class VF_Metric(metaclass=abc.ABCMeta):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @abc.abstractmethod
    def distance(self, x: ValueFunction, y: ValueFunction) -> float:
        raise NotImplementedError

    @staticmethod
    def factory(name, **kwargs) -> VF_Metric:
        if name == 'alpha':
            return AlphaVF_Metric()

        if name == 'bellman-at-start':
            start = kwargs['start']
            return BellmanAtStartVF_Metric(start)

        raise ValueError(f'invalid metric name `{name}`')


class AlphaVF_Metric(VF_Metric):
    def distance(self, x: ValueFunction, y: ValueFunction) -> float:
        x_actions = set(alpha.action for alpha in x.alphas)
        self.logger.debug('actions of x %s', x_actions)

        y_actions = set(alpha.action for alpha in y.alphas)
        self.logger.debug('actions of y %s', y_actions)

        if x_actions != y_actions:
            self.logger.debug('actions do not match -> infinite distance')
            # some actions don't have vectors in both VFs -> infinite distance
            return float('inf')

        distance = max(
            (self._action_distance(x, y, action) for action in x_actions),
            default=float('-inf'),
        )
        self.logger.debug(f'distance {distance}')
        assert distance >= 0.0
        return distance

    def _action_distance(
        self, x: ValueFunction, y: ValueFunction, action: int
    ) -> float:

        x_vectors = np.stack(
            [alpha.vector for alpha in x.alphas if alpha.action == action]
        )
        y_vectors = np.stack(
            [alpha.vector for alpha in y.alphas if alpha.action == action]
        )

        distance = max_bigraph_distance(x_vectors, y_vectors)
        self.logger.debug(f'action {action} distance {distance}')
        return distance


class BellmanAtStartVF_Metric(VF_Metric):
    def __init__(self, start):
        super().__init__()
        self.start = start

    def distance(self, x: ValueFunction, y: ValueFunction) -> float:
        x_value = x.value(self.start)
        self.logger.debug(f'x value at start {x_value}')

        y_value = y.value(self.start)
        self.logger.debug(f'y value at start {y_value}')

        distance = abs(x_value - y_value)
        self.logger.debug(f'distance {distance}')
        return distance
