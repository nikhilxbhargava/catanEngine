"""Simple priority-based heuristic agent for benchmarking."""

from __future__ import annotations

import random as _random
from typing import Optional

from agents.base import Agent
from catan.actions import Action, ActionType
from catan.state import GameState

# Priority order for main-turn actions
_ACTION_PRIORITY = {
    ActionType.BUILD_CITY: 0,
    ActionType.BUILD_SETTLEMENT: 1,
    ActionType.BUY_DEV_CARD: 2,
    ActionType.BUILD_ROAD: 3,
    ActionType.PLAY_KNIGHT: 4,
    ActionType.PLAY_YEAR_OF_PLENTY: 5,
    ActionType.PLAY_MONOPOLY: 6,
    ActionType.PLAY_ROAD_BUILDING: 7,
    ActionType.BANK_TRADE: 8,
    ActionType.END_TURN: 100,
}


class HeuristicAgent(Agent):
    def __init__(self, seed: Optional[int] = None):
        self.rng = _random.Random(seed)

    def choose_action(self, state: GameState, legal_actions: list[Action]) -> Action:
        if len(legal_actions) == 1:
            return legal_actions[0]

        # For setup, discard, robber, steal — pick randomly
        phase_random = {
            ActionType.PLACE_INITIAL_SETTLEMENT,
            ActionType.PLACE_INITIAL_ROAD,
            ActionType.DISCARD,
            ActionType.MOVE_ROBBER,
            ActionType.STEAL,
            ActionType.PLACE_FREE_ROAD,
        }
        if legal_actions[0].action_type in phase_random:
            return self.rng.choice(legal_actions)

        # Roll dice if that's the only non-knight option
        if any(a.action_type == ActionType.ROLL_DICE for a in legal_actions):
            # Prefer rolling (play knight only sometimes)
            roll_actions = [a for a in legal_actions if a.action_type == ActionType.ROLL_DICE]
            return roll_actions[0]

        # Main turn: pick highest priority action
        best = min(
            legal_actions,
            key=lambda a: _ACTION_PRIORITY.get(a.action_type, 50),
        )
        return best
