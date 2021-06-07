from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import FrozenSet, Iterator, Tuple

import yaml

__all__ = ['Interaction', 'Test', 'Intent', 'Tests', 'Intents']


@dataclass(eq=True, frozen=True)
class Interaction(yaml.YAMLObject):
    yaml_tag = u'!Interaction'

    action: int
    observation: int

    def as_test(self) -> Test:
        return Test((self,))


@dataclass(eq=True, frozen=True)
class Test(yaml.YAMLObject):
    yaml_tag = u'!Test'

    interactions: Tuple[Interaction, ...]

    @staticmethod
    def empty():
        return Test(())

    def prepend(self, interaction) -> Test:
        return Test((interaction,) + deepcopy(self.interactions))

    def __len__(self):
        return len(self.interactions)

    def __iter__(self) -> Iterator[Interaction]:
        yield from self.interactions


@dataclass(eq=True, frozen=True)
class Intent(yaml.YAMLObject):
    yaml_tag = u'!Intent'

    test: Test
    action: int

    @staticmethod
    def testless(action) -> Intent:
        return Intent(Test.empty(), action)

    @staticmethod
    def actionless(test) -> Intent:
        return Intent(test, -1)

    def prepend(self, interaction) -> Intent:
        return Intent(self.test.prepend(interaction), self.action)


@dataclass(frozen=True)
class Tests(yaml.YAMLObject):
    yaml_tag = u'!Tests'

    tests: Tuple[Test, ...]

    def __len__(self):
        return len(self.tests)

    def __iter__(self) -> Iterator[Test]:
        yield from self.tests


@dataclass(frozen=True)
class Intents(yaml.YAMLObject):
    yaml_tag = u'!Intents'

    intents: Tuple[Intent, ...]

    def __len__(self):
        return len(self.intents)

    def __iter__(self) -> Iterator[Intent]:
        yield from self.intents
