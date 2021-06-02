from functools import lru_cache

import numpy as np


@lru_cache(maxsize=None)
def R(env):
    """Return the expected rewards matrix R_{ij} = \\mathbb{E}\\left[ r \\mid s=i, a=j \\right]."""
    return np.einsum('sat,sato,sato->sa', env.T, env.O, env.R)


@lru_cache(maxsize=None)
def G(env):
    """Return the generative matrix G_{ij} = \\Pr(s'=i, o \\mid s=j, a)."""
    return np.einsum('sat,sato->aots', env.T, env.O)


@lru_cache(maxsize=None)
def D(env):
    """Return the dynamics matrix D_{ij} = \\Pr(s'=i \\mid s=j, a, o)."""
    G_ = G(env)
    with np.errstate(invalid='ignore'):
        return np.nan_to_num(G_ / G_.sum(2, keepdims=True))
