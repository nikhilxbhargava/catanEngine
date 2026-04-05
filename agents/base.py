"""Abstract agent interface."""

from __future__ import annotations

from abc import ABC, abstractmethod

from catan.actions import Action
from catan.state import GameState


class Agent(ABC):
    @abstractmethod
    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        raise NotImplementedError

    def __repr__(self) -> str:
        return self.__class__.__name__
