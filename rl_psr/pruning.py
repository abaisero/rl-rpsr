import logging
from typing import Any, Callable, List, Optional, TypeVar

import cvxpy as cp
import numpy as np
from scipy.optimize import linprog

from rl_psr.linalg import cross_sum

try:
    from cylp.cy import CyClpSimplex
    from cylp.py.modeling.CyLPModel import CyLPArray, CyLPModel
except (ImportError, ModuleNotFoundError):
    pass


ArrayKey = Callable[[Any], np.ndarray]


T = TypeVar('T', Any, np.ndarray)


def purge(
    objects: List[T], U, *args, key: Optional[ArrayKey] = None, **kwargs
) -> List[T]:

    logger = logging.getLogger(__name__)
    logger.debug('purging %d vectors', len(objects))

    objects_old, objects = objects, dominationCheck(objects, U, key=key)
    logger.debug('purging %d vectors (dominated removed)', len(objects))

    vectors = objects if key is None else list(map(key, objects))
    indices = _purge_indices(vectors, U, *args, **kwargs)
    objects_new = [objects[i] for i in indices]

    logger.debug(
        'purging result %d -> %d -> %d',
        len(objects_old),
        len(objects),
        len(objects_new),
    )
    return objects_new


def dominationCheck(
    objects: List[T], U, key: Optional[ArrayKey] = None
) -> List[T]:
    # function from page 54 of Cassandra's thesis

    vectors = objects if key is None else list(map(key, objects))
    vectors = [vector @ U.T for vector in vectors]

    if len(objects) < 2:
        return objects

    indices: List[int] = []
    for i in range(len(objects)):
        if not any(np.all(vectors[j] >= vectors[i]) for j in indices):
            indices = [
                j for j in indices if not np.all(vectors[i] >= vectors[j])
            ]
            indices.append(i)

    return [objects[i] for i in indices]


def _purge_indices(F, U, *args, **kwargs):
    alphas = np.row_stack(F)
    indices_F = set(range(len(F)))
    # this has a problem with lexicographic order
    # both alpha vectors will be chosen [[0, 0, 1], [0 1 1]]
    # indices_W = set(np.argmax(alphas @ U.T, axis=0))
    k = max(range(len(F)), key=lambda i: (F[i] @ U.T).tolist())
    indices_W = set([k])
    indices_F.difference_update(indices_W)

    while indices_F:
        # print(f'{len(indices_F)} alphas left')
        k = next(iter(indices_F))
        alpha = F[k]
        alphas = np.row_stack([F[i] for i in indices_W if i != k])

        x = dominate(alpha, alphas, U, *args, **kwargs)

        if x is None:
            # alpha is redundant by alphas
            indices_F.difference_update([k])
        else:
            # alpha is NOT redundant by alphas
            indices_F_list = list(indices_F)
            alphas = np.row_stack([F[i] for i in indices_F_list])
            k = np.argmax(alphas @ x)
            k = indices_F_list[k]
            indices_W.add(k)
            indices_F.difference_update([k])

    return indices_W


def inc_prune(
    object_lists: List[List[np.ndarray]], U, **kwargs,
) -> List[np.ndarray]:

    # Figure 3: The incremental pruning method.  from "Incremental Pruning: A
    # Simple, Fast, Exact Method for Partially  Observable Markov  Decision
    # Processes" (Cassandra et al.)

    W = purge(cross_sum(object_lists[:2]), U, **kwargs)
    for S in object_lists[2:]:
        W = purge(cross_sum([W, S]), U, **kwargs)

    return W


def dominate_scipy(alpha, A, U, *, eps=0.0):
    if eps < 0.0:
        raise ValueError('Negative epsilon ({eps})')

    N = U.shape[0]

    # x = (b, d)

    # max `d`
    c = np.zeros(N + 1)
    c[-1] = -1

    # b @ 1 = 1
    A_eq = np.ones((1, N + 1))
    A_eq[0, -1] = 0
    b_eq = 1

    # A b + d <= a b
    A_ub = np.zeros((A.shape[0], N + 1))
    A_ub[:, :-1] = (A - alpha) @ U.T
    A_ub[:, -1] = 1
    b_ub = np.zeros(A.shape[0])

    try:
        result = linprog(c, A_ub, b_ub, A_eq, b_eq)
    except ValueError:
        logger = logging.getLogger(__name__)
        logger.exception('linprog raise error')

        return None

    if result.status != 0:
        return None

    b, d = result.x[:-1], result.x[-1]

    # print(d, d == 0, eps, d <= eps)
    if d <= eps:
        return None

    return U.T @ b


def dominate_cvxpy(alpha, A, U, *, eps=0.0):
    if eps < 0.0:
        raise ValueError('Negative epsilon ({eps})')

    N = U.shape[0]

    b = cp.Variable(N)
    d = cp.Variable()

    objective = cp.Maximize(d)
    constraints = [
        d >= 0,
        b >= 0,
        cp.sum(b) == 1.0,
        ((A - alpha) @ U.T) * b + d <= 0,
    ]
    problem = cp.Problem(objective, constraints)

    # problem.solve()
    problem.solve(cp.CBC)
    # problem.solve(cp.SCS)
    # problem.solve(cp.OSQP)
    # problem.solve(cp.ECOS)
    # problem.solve(cp.ECOS_BB)
    # try:
    #     problem.solve()
    #     # problem.solve(verbose=True, solver=cp.CBC)
    #     # problem.solve(solver=cp.CBC)
    #     # problem.solve(solver=cp.GLPK_MI)
    # except cp.SolverError:
    #     return None

    if problem.status != 'optimal':
        # print(problem.status)
        return None
        # raise Exception(f'Unknown problem status {problem.status}')

    b, d = b.value, d.value

    # print(d, d == 0, eps, d <= eps)
    if d <= eps:
        return None

    return U.T @ b


def dominate_cylp(alpha, A, U, *, eps=0.0):
    # TODO fix / cleanup this method

    if eps < 0.0:
        raise ValueError('Negative epsilon ({eps})')

    N = U.shape[0]

    # TODO check whether this is a minimization or maximization problem
    s = CyClpSimplex()

    b = s.addVariable('b', (1, 2))
    d = s.addVariable('d', 1)

    s.objective = d

    s += d >= 0
    s += b >= 0
    s += b.sum() == 1
    s += np.matrix((A - alpha) @ U.T) * b + d <= 0

    status = s.primal()

    if status != 'optimal':
        raise Exception(f'Unknown problem status {status}')

    b = s.primalVariableSolution['b']
    d = s.primalVariableSolution['d']

    if d.item() <= eps:
        return None

    return U.T @ b


# dominate = dominate_scipy
dominate = dominate_cvxpy
# dominate = dominate_cylp
