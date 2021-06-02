import abc
from typing import Union

from rl_psr.core import Intents, Tests
from rl_psr.pomdp import POMDP_Model


class Searcher(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def search(self, model: POMDP_Model) -> Union[Tests, Intents]:
        raise NotImplementedError
