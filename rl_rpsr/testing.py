import random

import numpy.random as rnd
from rl_rpsr.core import Intent, Intents, Interaction, Test, Tests
from rl_rpsr.value_function import Alpha, ValueFunction


def random_action(num_actions, extended=False) -> int:
    action_min = -1 if extended else 0
    return random.randint(action_min, num_actions - 1)


def random_observation(num_observations) -> int:
    return random.randint(0, num_observations - 1)


def random_interaction(num_actions, num_observations) -> Interaction:
    action = random_action(num_actions)
    observation = random_observation(num_observations)
    return Interaction(action, observation)


def random_test(num_interactions, num_actions, num_observations) -> Test:
    interactions = tuple(
        random_interaction(num_actions, num_observations)
        for _ in range(num_interactions)
    )
    return Test(interactions)


def random_intent(num_interactions, num_actions, num_observations) -> Intent:
    test = random_test(num_interactions, num_actions, num_observations)
    action = random_action(num_actions, extended=True)
    return Intent(test, action)


def random_tests(
    num_tests, num_interactions, num_actions, num_observations
) -> Tests:

    tests = frozenset(
        random_test(num_interactions, num_actions, num_observations)
        for _ in range(num_tests)
    )
    return Tests(tuple(tests))


def random_intents(
    num_intents, num_interactions, num_actions, num_observations
) -> Intents:

    intents = frozenset(
        random_intent(num_interactions, num_actions, num_observations)
        for _ in range(num_intents)
    )
    return Intents(tuple(intents))


def random_alpha(num_actions, num_dim) -> Alpha:
    action = random_action(num_actions)
    vector = rnd.randn(num_dim)
    return Alpha(action, vector)


def random_value_function(num_alphas, num_actions, num_dim) -> ValueFunction:
    alphas = [random_alpha(num_actions, num_dim) for _ in range(num_alphas)]
    return ValueFunction(alphas, 0)
