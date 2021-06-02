import abc
import sys
from typing import Any, Union

import numpy as np
import yaml

from rl_psr.core import Intents, Tests
from rl_psr.value_function import ValueFunction

__all__ = [
    'TestsSerializer',
    'IntentsSerializer',
    'CoreSerializer',
    'VF_Serializer',
    'AlphaSerializer',
]


class Serializer(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def dump(self, filename: str, obj):
        raise NotImplementedError

    @abc.abstractmethod
    def load(self, filename: str) -> Any:
        raise NotImplementedError


class TestsSerializer(Serializer):
    def dump(self, filename: str, obj: Tests):
        if not isinstance(obj, Tests):
            raise TypeError(f'object is of type {type(obj)}; expected Tests')

        with open(filename, 'w') as f:
            yaml.dump(obj, f)

    def load(self, filename: str) -> Tests:  # pylint: disable=no-self-use
        with open(filename) as f:
            obj = yaml.load(f, Loader=yaml.Loader)

        if not isinstance(obj, Tests):
            raise TypeError(
                f'loaded object is of type {type(obj)}; expected Tests'
            )

        return obj


class IntentsSerializer(Serializer):
    def dump(self, filename: str, obj: Intents):
        if not isinstance(obj, Intents):
            raise TypeError(f'object is of type {type(obj)}; expected Intents')

        with open(filename, 'w') as f:
            yaml.dump(obj, f)

    def load(self, filename: str) -> Intents:  # pylint: disable=no-self-use
        with open(filename) as f:
            obj = yaml.load(f, Loader=yaml.Loader)

        if not isinstance(obj, Intents):
            raise TypeError(
                f'loaded object is of type {type(obj)}; expected Intents'
            )

        return obj


class CoreSerializer(Serializer):
    def dump(self, filename: str, obj: Union[Tests, Intents]):
        if not isinstance(obj, (Tests, Intents)):
            raise TypeError(
                f'object is of type {type(obj)}; expected Tests or Intents'
            )

        with open(filename, 'w') as f:
            yaml.dump(obj, f)

    def load(  # pylint: disable=no-self-use
        self, filename: str
    ) -> Union[Tests, Intents]:

        with open(filename) as f:
            obj = yaml.load(f, Loader=yaml.Loader)

        if not isinstance(obj, (Tests, Intents)):
            raise TypeError(
                f'loaded object is of type {type(obj)}; expected Tests or Intents'
            )

        return obj


class VF_Serializer(Serializer):
    def dump(self, filename: str, obj: ValueFunction):
        if not isinstance(obj, ValueFunction):
            raise TypeError(
                f'object is of type {type(obj)}; expected ValueFunction'
            )

        with open(filename, 'w') as f:
            yaml.dump(obj, f)

    def load(  # pylint: disable=no-self-use
        self, filename: str
    ) -> ValueFunction:

        with open(filename) as f:
            obj = yaml.load(f, Loader=yaml.Loader)

        if not isinstance(obj, ValueFunction):
            raise TypeError(
                f'loaded object is of type {type(obj)}; expected ValueFunction'
            )

        return obj


class AlphaSerializer(Serializer):
    def dump(self, filename: str, obj: ValueFunction):
        if not isinstance(obj, ValueFunction):
            raise TypeError(
                f'object is of type {type(obj)}; expected ValueFunction'
            )

        with open(filename, 'w') as f:
            for alpha in obj.alphas:
                print(alpha.action, file=f)
                with np.printoptions(
                    suppress=True,
                    threshold=sys.maxsize,
                    linewidth=sys.maxsize,
                    precision=16,
                    floatmode='fixed',
                ):
                    string = str(alpha.vector)[1:-1]  # remove brackets
                    print(string, file=f)
                print(file=f)

    def load(self, filename: str) -> ValueFunction:
        raise NotImplementedError('AlphaSerializer does not support load.')
