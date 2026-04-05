"""Self-play training loop.

Runs games where all players use the same (or different) policies,
collecting trajectories for training.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Optional

from catan.actions import Action, get_legal_actions
from catan.board import Board
from catan.game import Game
from catan.state import GameState
from training.features import state_to_features


@dataclass
class Transition:
    """A single (s, a, r, s') transition."""
    features: list[float]
    action: Action
    reward: float
    next_features: Optional[list[float]]
    done: bool
    player_index: int


@dataclass
class GameResult:
    """Result of a completed game."""
    winner: Optional[int]
    turns: int
    vp: list[int]
    trajectories: dict[int, list[Transition]] = field(default_factory=dict)


Policy = Callable[[GameState, list[Action]], Action]


def play_game(
    policies: list[Policy],
    num_players: int = 4,
    seed: Optional[int] = None,
    collect_trajectories: bool = True,
    max_turns: int = 3000,
) -> GameResult:
    """Play a single game with given policies, collecting trajectories.

    Args:
        policies: One policy per player. policy(state, legal_actions) -> action
        num_players: Number of players (3 or 4).
        seed: Random seed for board and game.
        collect_trajectories: Whether to record state transitions.
        max_turns: Safety limit.

    Returns:
        GameResult with winner, turn count, VP, and optional trajectories.
    """
    board = Board.build(seed=seed)
    game = Game(board, num_players, seed=seed)

    trajectories: dict[int, list[Transition]] = {i: [] for i in range(num_players)}

    while not game.is_over() and game.state.turn_number < max_turns:
        actions = game.get_legal_actions()
        if not actions:
            break

        pidx = game.state.current_player_index
        state = game.state

        if collect_trajectories:
            features = state_to_features(state)

        action = policies[pidx](state, actions)
        game.apply(action)

        if collect_trajectories:
            next_features = state_to_features(game.state) if not game.is_over() else None
            trajectories[pidx].append(Transition(
                features=features,
                action=action,
                reward=0.0,  # filled in below
                next_features=next_features,
                done=game.is_over(),
                player_index=pidx,
            ))

    # Assign terminal rewards
    winner = game.winner()
    if collect_trajectories and winner is not None:
        for pidx, traj in trajectories.items():
            if traj:
                traj[-1].reward = 1.0 if pidx == winner else -1.0

    return GameResult(
        winner=winner,
        turns=game.state.turn_number,
        vp=[p.actual_victory_points for p in game.state.players],
        trajectories=trajectories if collect_trajectories else {},
    )


def run_benchmark(
    policies: list[Policy],
    num_games: int = 100,
    num_players: int = 4,
    base_seed: int = 0,
) -> dict[str, any]:
    """Run many games and report statistics.

    Returns:
        Dict with win_counts, win_rates, avg_turns, avg_vp.
    """
    win_counts = [0] * num_players
    total_turns = 0
    total_vp = [0] * num_players
    completed = 0

    for g in range(num_games):
        result = play_game(
            policies,
            num_players=num_players,
            seed=base_seed + g,
            collect_trajectories=False,
        )
        if result.winner is not None:
            win_counts[result.winner] += 1
            completed += 1
        total_turns += result.turns
        for i, vp in enumerate(result.vp):
            total_vp[i] += vp

    return {
        "win_counts": win_counts,
        "win_rates": [w / max(completed, 1) for w in win_counts],
        "avg_turns": total_turns / max(num_games, 1),
        "avg_vp": [v / max(num_games, 1) for v in total_vp],
        "games_completed": completed,
        "games_total": num_games,
    }
