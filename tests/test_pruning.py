import unittest

import numpy as np
import numpy.random as rnd

from rl_psr.pruning import purge


class TestPurge(unittest.TestCase):
    def assertContainerEqual(self, first, second, msg=None):
        ids_first = map(id, first)
        ids_second = map(id, second)
        self.assertCountEqual(ids_first, ids_second, msg)

    def assertContainerSubset(self, first, second, msg=None):
        ids_first = list(map(id, first))
        ids_second = list(map(id, second))
        self.assertTrue(
            all(id_first in ids_second for id_first in ids_first), msg
        )

    def test_nopurge(self):
        I = np.eye(5)

        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
        ]
        vectors_purged = purge(vectors, I)
        vectors_target = [vectors[i] for i in [0, 1, 2, 3, 4]]
        self.assertContainerEqual(vectors_purged, vectors_target)

    def test_nopurge_2(self):
        I = np.eye(5)

        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([0.9, 0.9, 0.9, 0.9, 0.9]),
        ]
        vectors_purged = purge(vectors, I)
        vectors_target = [vectors[i] for i in [0, 1, 2, 3, 4, 5]]
        self.assertContainerEqual(vectors_purged, vectors_target)

    def test_dominated(self):
        I = np.eye(5)

        vectors = [
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
        ]
        vectors_purged = purge(vectors, I)
        vectors_target = [vectors[i] for i in [0]]
        self.assertContainerEqual(vectors_purged, vectors_target)

    def test_dominated_2(self):
        I = np.eye(5)

        vectors = [
            np.array([0.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([1.0, 1.0, 1.0, 1.0, 1.0]),
        ]
        vectors_purged = purge(vectors, I)
        vectors_target = [vectors[i] for i in [1]]
        self.assertContainerEqual(vectors_purged, vectors_target)

    def test_redundant(self):
        I = np.eye(5)

        vectors = [
            np.array([1.0, 0.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 1.0, 0.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 1.0, 0.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 1.0, 0.0]),
            np.array([0.0, 0.0, 0.0, 0.0, 1.0]),
            np.array([0.1, 0.1, 0.1, 0.1, 0.1]),
        ]
        vectors_purged = purge(vectors, I)
        vectors_target = [vectors[i] for i in [0, 1, 2, 3, 4]]
        self.assertContainerEqual(vectors_purged, vectors_target)

    def test_large(self):
        ndim = 10
        I = np.eye(ndim)

        vectors_hi = [rnd.randn(ndim) + 100 for _ in range(200)]
        vectors_hi.append(np.zeros(ndim))
        vectors_lo = [rnd.randn(ndim) - 100 for _ in range(200)]
        vectors_purged = purge(vectors_hi + vectors_lo, I)

        self.assertContainerSubset(vectors_purged, vectors_hi)


if __name__ == '__main__':
    unittest.main()
