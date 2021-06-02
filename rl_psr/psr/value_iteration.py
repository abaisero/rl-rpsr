import itertools as itt
from typing import List, Optional, Sequence

from rl_psr.linalg import cross_sum
from rl_psr.pruning import inc_prune, purge
from rl_psr.util import VI_Type
from rl_psr.value_function import Alpha, ValueFunction
from rl_psr.value_iteration import VI_Algo

from .model import PSR_Model

__all__ = ['vi_factory', 'VI_Enum', 'VI_IncPruning']


def _bootstrap(model: PSR_Model, action, observation, vector):
    return model.discount * model.M_aoQ[action, observation] @ vector


def _make_alpha(
    model: PSR_Model, action, next_alphas: Optional[Sequence[Alpha]] = None
) -> Alpha:

    vector = model.R[:, action].copy()

    if next_alphas is not None:
        for observation, alpha in enumerate(next_alphas):
            vector += _bootstrap(model, action, observation, alpha.vector)

    return Alpha(action, vector)


def vi_factory(vi_type: VI_Type, **kwargs) -> VI_Algo:
    if vi_type == VI_Type.ENUM:
        return VI_Enum()

    if vi_type == VI_Type.INC_PRUNING:
        return VI_IncPruning(true_inc_pruning=False)

    if vi_type == VI_Type.TRUE_INC_PRUNING:
        return VI_IncPruning(true_inc_pruning=True)

    raise ValueError(f'No implementation for VI type {vi_type}')


class VI_Enum(VI_Algo):
    def iterate(
        self, model: PSR_Model, vf: ValueFunction, **kwargs
    ) -> ValueFunction:

        alphas = vf.alphas
        alphas = [
            _make_alpha(model, a, next_alphas)
            for a in range(model.action_space.n)
            for next_alphas in itt.product(
                alphas, repeat=model.observation_space.n
            )
        ]

        alphas = purge(
            alphas, model.U, key=lambda alpha: alpha.vector, **kwargs
        )
        return ValueFunction(alphas, vf.horizon + 1)


class VI_IncPruning(VI_Algo):
    def __init__(self, true_inc_pruning=True):
        super().__init__()
        self.true_inc_pruning = true_inc_pruning

    def iterate(
        self, model: PSR_Model, vf: ValueFunction, **kwargs
    ) -> ValueFunction:

        alphas = vf.alphas

        S: List[Alpha] = []
        for a in range(model.action_space.n):
            R_over_O = model.R[:, a] / model.observation_space.n

            S_ao = []
            for o in range(model.observation_space.n):
                vectors = [
                    R_over_O + _bootstrap(model, a, o, alpha.vector)
                    for alpha in alphas
                ]
                self.logger.debug('purging S_ao a=%d o=%d', a, o)
                vectors = purge(vectors, model.U, **kwargs)
                S_ao.append(vectors)

            if self.true_inc_pruning:
                S_a = inc_prune(S_ao, model.U, **kwargs)
            else:
                self.logger.debug('cross_sum S_a a=%d', a)
                S_a = cross_sum(S_ao)
                self.logger.debug('purging S_a a=%d', a)
                S_a = purge(S_a, model.U, **kwargs)

            S.extend(Alpha(a, vector) for vector in S_a)

        self.logger.debug('purging S')
        alphas = purge(S, model.U, key=lambda alpha: alpha.vector, **kwargs)
        return ValueFunction(alphas, vf.horizon + 1)
