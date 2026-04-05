"""Uniformly random agent — baseline for benchmarking."""

from __future__ import annotations

import random as _random
from typing import Optional

from agents.base import Agent
from catan.actions import Action
from catan.state import GameState


class RandomAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = _random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        return self.rng.choice(legal_actions)
