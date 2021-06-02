from __future__ import annotations

import abc


class Policy(metaclass=abc.ABCMeta):
    @abc.abstractmethod
    def reset(self) -> int:
        raise NotImplementedError

    @abc.abstractmethod
    def step(self, action: int, observation: int) -> int:
        raise NotImplementedError


class RandomPolicy(Policy):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def _action(self) -> int:
        return self.model.action_space.sample()

    def reset(self) -> int:
        return self._action()

    def step(self, action: int, observation: int) -> int:
        return self._action()


class ModelPolicy(Policy):
    def __init__(self, model, vf):
        super().__init__()
        self.model = model
        self.vf = vf

        self.state = None

    def _action(self) -> int:
        return self.vf.policy(self.state)

    def reset(self) -> int:
        self.state = self.model.start.copy()
        return self._action()

    def step(self, action: int, observation: int) -> int:
        self.state = self.model.dynamics(self.state, action, observation)
        return self._action()
