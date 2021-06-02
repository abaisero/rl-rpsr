import unittest

import rl_psr.testing as testing
from rl_psr.core import Intent, Test


class TestInteraction(unittest.TestCase):
    def test_as_test(self):
        interaction = testing.random_interaction(10, 10)
        test = interaction.as_test()

        self.assertEqual(len(test), 1)
        self.assertEqual(test.interactions[0], interaction)


class TestTest(unittest.TestCase):
    def test_empty(self):
        test = Test.empty()

        self.assertEqual(len(test), 0)

    def test_prepend(self):
        test = Test.empty()
        interaction = testing.random_interaction(10, 10)
        test_prepended = test.prepend(interaction)

        self.assertEqual(len(test_prepended), 1)
        self.assertEqual(test_prepended.interactions[0], interaction)

        test = testing.random_test(10, 10, 10)
        interaction = testing.random_interaction(10, 10)
        test_prepended = test.prepend(interaction)

        self.assertEqual(len(test_prepended), len(test) + 1)
        self.assertEqual(test_prepended.interactions[0], interaction)
        self.assertEqual(test_prepended.interactions[1:], test.interactions)


class TestIntent(unittest.TestCase):
    def test_testless(self):
        action = testing.random_action(10, extended=True)
        intent = Intent.testless(action)

        self.assertEqual(intent.test, Test.empty())
        self.assertEqual(intent.action, action)

    def test_actionless(self):
        test = testing.random_test(10, 10, 10)
        intent = Intent.actionless(test)

        self.assertEqual(intent.test, test)
        self.assertEqual(intent.action, -1)


if __name__ == '__main__':
    unittest.main()
