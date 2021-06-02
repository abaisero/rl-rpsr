import itertools as itt
import logging
from typing import Iterable, List

import numpy as np
import numpy.linalg as la
from scipy.spatial import distance_matrix

__all__ = ['cross_sum', 'max_bigraph_distance', 'linearly_independent']


def cross_sum(vectors_list: Iterable[List[np.ndarray]]) -> List[np.ndarray]:
    return [np.sum(vectors, axis=0) for vectors in itt.product(*vectors_list)]


def max_bigraph_distance(x: np.ndarray, y: np.ndarray):
    logger = logging.getLogger(__name__)

    if x.ndim != 2:
        msg = f'input `x` should have 2 dimensions, instead got shape {x.shape}'
        logger.error(msg)
        raise ValueError(msg)

    if y.ndim != 2:
        msg = f'input `y` should have 2 dimensions, instead got shape {x.shape}'
        logger.error(msg)
        raise ValueError(msg)

    d_matrix = distance_matrix(x, y)

    distance_x2y = d_matrix.min(0).max()
    logger.debug(f'distance x -> y {distance_x2y}')
    distance_y2x = d_matrix.min(1).max()
    logger.debug(f'distance y -> x {distance_y2x}')

    return max(distance_x2y, distance_y2x)


def linearly_independent_pinv(
    vectors: List[np.ndarray], vector: np.ndarray
) -> bool:
    matrix = np.stack(vectors)

    logger = logging.getLogger(__name__)

    try:
        assert la.matrix_rank(matrix) == len(vectors)
    except AssertionError as e:
        logger.exception('Input vectors are not already linearly independent')
        raise e

    vector_recon = la.pinv(matrix) @ matrix @ vector
    return not np.allclose(vector_recon, vector)


def linearly_independent_rank(
    vectors: List[np.ndarray], vector: np.ndarray
) -> bool:

    logger = logging.getLogger(__name__)

    matrix = np.stack(vectors + [vector])
    rank = la.matrix_rank(matrix[:-1, :]) if vectors else 0

    try:
        assert rank == len(vectors)
    except AssertionError as e:
        logger.exception('Input vectors are not already linearly independent')
        raise e

    rank_with_vector = la.matrix_rank(matrix)
    logger.debug('ranks: %d, %d', rank, rank_with_vector)

    if not (rank_with_vector - rank in [0, 1]):
        logger.warning(
            'ranks (%d, %d) should either be the same or adjacent',
            rank,
            rank_with_vector,
        )

    return rank_with_vector > rank


def linearly_independent_lstsq(
    vectors: List[np.ndarray], vector: np.ndarray
) -> bool:

    logger = logging.getLogger(__name__)

    if np.allclose(vector, 0.0):
        # zero vector is always linearly dependent
        return False

    if not vectors:
        # no vectors makes the vector always linearly independent
        return True

    matrix = np.stack(vectors, axis=1)

    try:
        assert la.matrix_rank(matrix) == len(vectors)
    except AssertionError as e:
        logger.exception('Input vectors are not already linearly independent')
        raise e

    _, residuals, rank, _ = la.lstsq(matrix, vector, rcond=None)

    if rank != len(vectors):
        logger.warning('input `vectors` are not already linearly independent')

    return not np.allclose(residuals, 0.0)


linearly_independent = linearly_independent_lstsq
