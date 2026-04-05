"""Gymnasium-compatible Catan environment.

Wraps the game engine for use with standard RL libraries.
Handles action encoding/decoding and observation generation.
"""

from __future__ import annotations

from typing import Any, Optional

from catan.actions import Action, get_legal_actions
from catan.board import Board
from catan.game import Game, apply_action_mutate
from catan.state import GameState
from training.features import state_to_features


class CatanEnv:
    """Catan environment with a dict-based interface.

    If gymnasium is installed, this can be wrapped into a full Gym env.
    Works standalone without gymnasium as a dependency.

    The environment manages one player's perspective. Other players
    use a provided opponent_policy function.
    """

    def __init__(
        self,
        num_players: int = 4,
        opponent_policy=None,
        seed: Optional[int] = None,
    ):
        self.num_players = num_players
        self.opponent_policy = opponent_policy  # callable(state, legal_actions) -> action
        self.seed = seed
        self.player_index = 0  # the RL agent is always player 0
        self.game: Optional[Game] = None
        self._step_count = 0

    def reset(self, seed: Optional[int] = None) -> dict[str, Any]:
        """Reset environment, return initial observation."""
        s = seed if seed is not None else self.seed
        board = Board.build(seed=s)
        self.game = Game(board, self.num_players, seed=s)
        self._step_count = 0

        # Advance through opponent turns until it's the RL agent's turn
        self._advance_to_agent_turn()

        return self._make_obs()

    def step(self, action: Action) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """Take an action, return (obs, reward, terminated, truncated, info).

        Args:
            action: An Action object (from legal_actions).

        Returns:
            Tuple of (observation, reward, terminated, truncated, info).
        """
        assert self.game is not None
        assert self.game.state.current_player_index == self.player_index

        self.game.apply(action)
        self._step_count += 1

        # Check if game ended
        if self.game.is_over():
            return self._make_obs(), self._compute_reward(), True, False, self._make_info()

        # Advance through opponent turns
        self._advance_to_agent_turn()

        terminated = self.game.is_over()
        truncated = self._step_count > 10000
        reward = self._compute_reward() if terminated else 0.0

        return self._make_obs(), reward, terminated, truncated, self._make_info()

    def legal_actions(self) -> list[Action]:
        """Return legal actions for the RL agent."""
        assert self.game is not None
        return get_legal_actions(self.game.state)

    def _advance_to_agent_turn(self) -> None:
        """Play opponent turns until it's the RL agent's turn or game ends."""
        assert self.game is not None
        while (
            not self.game.is_over()
            and self.game.state.current_player_index != self.player_index
        ):
            state = self.game.state
            actions = get_legal_actions(state)
            if not actions:
                break

            if self.opponent_policy is not None:
                action = self.opponent_policy(state, actions)
            else:
                # Default: random
                import random
                action = random.choice(actions)

            self.game.apply(action)

        # Also handle non-agent discard phases etc.
        # If it's a discard phase and the current discard player isn't us,
        # advance past it
        if (
            not self.game.is_over()
            and self.game.state.phase.name == "DISCARD"
            and self.game.state.discard_players
            and self.game.state.discard_players[0] != self.player_index
        ):
            self._advance_to_agent_turn()

    def _make_obs(self) -> dict[str, Any]:
        assert self.game is not None
        features = state_to_features(self.game.state)
        return {
            "features": features,
            "legal_actions": self.legal_actions() if not self.game.is_over() else [],
        }

    def _compute_reward(self) -> float:
        assert self.game is not None
        if self.game.winner() == self.player_index:
            return 1.0
        elif self.game.winner() is not None:
            return -1.0
        return 0.0

    def _make_info(self) -> dict[str, Any]:
        assert self.game is not None
        return {
            "turn": self.game.state.turn_number,
            "winner": self.game.winner(),
            "vp": [p.actual_victory_points for p in self.game.state.players],
        }
