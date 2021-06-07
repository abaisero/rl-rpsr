from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List

import numpy as np
import yaml


@dataclass
class Alpha(yaml.YAMLObject):
    yaml_tag = u'!Alpha'

    action: int
    vector: np.ndarray


class ValueFunction(yaml.YAMLObject):
    yaml_tag = u'!ValueFunction'

    def __init__(self, alphas: Iterable[Alpha], horizon: int):
        self.alphas = self.standardize(alphas)
        self.horizon = horizon

        self.__matrix = None

    def __len__(self):
        return len(self.alphas)

    @staticmethod
    def standardize(alphas: Iterable[Alpha]) -> List[Alpha]:
        return sorted(alphas, key=lambda alpha: alpha.vector.tolist())

    @property
    def matrix(self):
        if self.__matrix is None:
            self.__matrix = np.column_stack(
                [alpha.vector for alpha in self.alphas]
            )
        return self.__matrix

    def value(self, state) -> float:
        return (self.matrix.T @ state).max()

    def policy(self, state) -> int:
        idx = (self.matrix.T @ state).argmax()
        return self.alphas[idx].action
