import unittest

import numpy as np

from rl_psr.linalg import (
    linearly_independent_lstsq,
    linearly_independent_pinv,
    linearly_independent_rank,
)


class TestLinearlyIndependent(unittest.TestCase):
    def test_pinv(self):
        vectors = list(np.eye(5))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertFalse(linearly_independent_pinv(vectors, vector))
        self.assertTrue(linearly_independent_pinv(vectors[:-1], vector))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        self.assertFalse(linearly_independent_pinv(vectors, vector))
        self.assertFalse(linearly_independent_pinv(vectors[:-1], vector))

    def test_rank(self):
        vectors = list(np.eye(5))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertFalse(linearly_independent_rank(vectors, vector))
        self.assertTrue(linearly_independent_rank(vectors[:-1], vector))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        self.assertFalse(linearly_independent_rank(vectors, vector))
        self.assertFalse(linearly_independent_rank(vectors[:-1], vector))

    def test_lstsq(self):
        vectors = list(np.eye(5))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        self.assertFalse(linearly_independent_lstsq(vectors, vector))
        self.assertTrue(linearly_independent_lstsq(vectors[:-1], vector))

        vector = np.array([1.0, 2.0, 3.0, 4.0, 0.0])
        self.assertFalse(linearly_independent_lstsq(vectors, vector))
        self.assertFalse(linearly_independent_lstsq(vectors[:-1], vector))


if __name__ == '__main__':
    unittest.main()
