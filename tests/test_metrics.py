import copy
import unittest

import numpy as np
import numpy.random as rnd
from rl_rpsr import testing
from rl_rpsr.metrics import AlphaVF_Metric, BellmanAtStartVF_Metric


class TestAlphaVF_Metric(unittest.TestCase):
    def test_positive(self):
        metric = AlphaVF_Metric()

        num_alphas, num_actions, num_dim = 10, 5, 5
        vf1 = testing.random_value_function(num_alphas, num_actions, num_dim)
        vf2 = testing.random_value_function(num_alphas, num_actions, num_dim)

        distance = metric.distance(vf1, vf2)
        self.assertGreaterEqual(distance, 0.0)

    def test_infinite(self):
        metric = AlphaVF_Metric()

        num_alphas, num_dim = 10, 5
        num_actions = 1
        vf1 = testing.random_value_function(num_alphas, num_actions, num_dim)
        num_actions = 1_000
        vf2 = testing.random_value_function(num_alphas, num_actions, num_dim)

        distance = metric.distance(vf1, vf2)
        self.assertEqual(distance, float('inf'))

    def test_identity(self):
        metric = AlphaVF_Metric()

        num_alphas, num_actions, num_dim = 10, 5, 5
        vf = testing.random_value_function(num_alphas, num_actions, num_dim)

        distance = metric.distance(vf, vf)
        self.assertEqual(distance, 0.0)

    def test_perturbation(self):
        metric = AlphaVF_Metric()

        num_alphas, num_actions, num_dim = 10, 5, 5
        vf = testing.random_value_function(num_alphas, num_actions, num_dim)

        vf_perturbed = copy.deepcopy(vf)
        for alpha in vf_perturbed.alphas:
            alpha.vector += 0.1 * rnd.randn(*alpha.vector.shape)

        distance = metric.distance(vf, vf_perturbed)
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 1.0)


class TestBellmanAtStartVF_Metric(unittest.TestCase):
    def test_positive(self):
        num_dim = 5
        start = np.ones(num_dim) / num_dim
        metric = BellmanAtStartVF_Metric(start)

        num_alphas, num_actions = 10, 5
        vf1 = testing.random_value_function(num_alphas, num_actions, num_dim)
        vf2 = testing.random_value_function(num_alphas, num_actions, num_dim)

        distance = metric.distance(vf1, vf2)
        self.assertGreaterEqual(distance, 0.0)

    def test_identity(self):
        num_dim = 5
        start = np.ones(num_dim) / num_dim
        metric = BellmanAtStartVF_Metric(start)

        num_alphas, num_actions = 10, 5
        vf = testing.random_value_function(num_alphas, num_actions, num_dim)

        distance = metric.distance(vf, vf)
        self.assertEqual(distance, 0.0)

    def test_perturbation(self):
        num_dim = 5
        start = np.ones(num_dim) / num_dim
        metric = BellmanAtStartVF_Metric(start)

        num_alphas, num_actions = 10, 5
        vf = testing.random_value_function(num_alphas, num_actions, num_dim)

        vf_perturbed = copy.deepcopy(vf)
        for alpha in vf_perturbed.alphas:
            alpha.vector += 0.1 * rnd.randn(*alpha.vector.shape)

        distance = metric.distance(vf, vf_perturbed)
        self.assertGreater(distance, 0.0)
        self.assertLess(distance, 1.0)
