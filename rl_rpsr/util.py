import enum
from typing import Iterator

from rl_rpsr.core import Interaction

__all__ = ['SearchType', 'VI_Type', 'interactions']


class SearchType(enum.Enum):
    BFS = enum.auto()
    DFS = enum.auto()


class VI_Type(enum.Enum):
    """Value iteration types."""

    ENUM = enum.auto()
    INC_PRUNING = enum.auto()
    TRUE_INC_PRUNING = enum.auto()


def interactions(action_space, observation_space) -> Iterator[Interaction]:
    """Generator of every action-observation interaction."""

    for a in range(action_space.n):
        for o in range(observation_space.n):
            yield Interaction(a, o)
