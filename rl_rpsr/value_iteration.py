import abc
import logging

import numpy as np
from rl_rpsr.value_function import Alpha, ValueFunction


class VI_Algo(metaclass=abc.ABCMeta):
    def __init__(self):
        self.logger = logging.getLogger(__name__)

    @staticmethod
    def init(model) -> ValueFunction:
        return ValueFunction([Alpha(-1, np.zeros(model.rank))], 0)

    @abc.abstractmethod
    def iterate(self, model, vf: ValueFunction, **kwargs) -> ValueFunction:
        raise NotImplementedError
