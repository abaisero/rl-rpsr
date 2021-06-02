import unittest

import rl_psr.testing as testing
from rl_psr.core import Intent, Intents, Interaction, Test, Tests
from rl_psr.value_function import Alpha, ValueFunction


class TestTesting(unittest.TestCase):
    def test_random_action(self):
        num_actions = 2
        action = testing.random_action(num_actions)

        self.assertIsInstance(action, int)
        self.assertIn(action, range(num_actions))

    def test_random_action_extended(self):
        num_actions = 2
        action = testing.random_action(num_actions, extended=True)

        self.assertIsInstance(action, int)
        self.assertIn(action, range(-1, num_actions))

    def test_random_observation(self):
        num_observations = 2
        observation = testing.random_observation(num_observations)

        self.assertIsInstance(observation, int)
        self.assertIn(observation, range(num_observations))

    def test_random_interaction(self):
        num_actions, num_observations = 2, 2
        interaction = testing.random_interaction(num_actions, num_observations)

        self.assertIsInstance(interaction, Interaction)
        self.assertIn(interaction.action, range(num_actions))
        self.assertIn(interaction.observation, range(num_observations))

    def test_random_test(self):
        num_interactions = 5
        num_actions, num_observations = 2, 2
        test = testing.random_test(
            num_interactions, num_actions, num_observations
        )

        self.assertIsInstance(test, Test)
        self.assertEqual(len(test), num_interactions)
        for interaction in test:
            self.assertIn(interaction.action, range(num_actions))
            self.assertIn(interaction.observation, range(num_observations))

    def test_random_intent(self):
        num_interactions = 5
        num_actions, num_observations = 2, 2
        intent = testing.random_intent(
            num_interactions, num_actions, num_observations
        )

        self.assertIsInstance(intent, Intent)
        self.assertEqual(len(intent.test), num_interactions)
        for interaction in intent.test:
            self.assertIn(interaction.action, range(num_actions))
            self.assertIn(interaction.observation, range(num_observations))
        self.assertIn(intent.action, range(-1, num_actions))

    def test_random_tests(self):
        num_tests, num_interactions = 3, 5
        num_actions, num_observations = 2, 2
        tests = testing.random_tests(
            num_tests, num_interactions, num_actions, num_observations
        )

        self.assertIsInstance(tests, Tests)
        self.assertEqual(len(tests), num_tests)
        for test in tests.tests:
            self.assertIsInstance(test, Test)

    def test_random_interactions(self):
        num_intents, num_interactions = 3, 5
        num_actions, num_observations = 2, 2
        intents = testing.random_intents(
            num_intents, num_interactions, num_actions, num_observations
        )

        self.assertIsInstance(intents, Intents)
        self.assertEqual(len(intents), num_intents)
        for intent in intents.intents:
            self.assertIsInstance(intent, Intent)

    def test_random_alpha(self):
        num_actions, num_dim = 2, 2
        alpha = testing.random_alpha(num_actions, num_dim)

        self.assertIsInstance(alpha, Alpha)
        self.assertIn(alpha.action, range(num_actions))
        self.assertTupleEqual(alpha.vector.shape, (num_dim,))

    def test_random_value_function(self):
        num_alphas, num_actions, num_dim = 10, 2, 2
        vf = testing.random_value_function(num_alphas, num_actions, num_dim)

        self.assertIsInstance(vf, ValueFunction)
        self.assertEqual(len(vf), num_alphas)
        for alpha in vf.alphas:
            self.assertIsInstance(alpha, Alpha)


if __name__ == '__main__':
    unittest.main()
