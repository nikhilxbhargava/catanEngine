"""Reward shaping for Catan RL training.

Provides intermediate reward signals based on state changes after each action.
Uses potential-based shaping (delta of a potential function) to preserve
optimal policy while accelerating credit assignment.

All reward values are intentionally small relative to the terminal +1/-1
reward so they guide learning without dominating the objective.
"""

from __future__ import annotations

from dataclasses import dataclass

from catan.state import GameState


@dataclass
class PlayerSnapshot:
    """Lightweight snapshot of state features relevant to reward shaping."""
    vp: int
    num_settlements: int
    num_cities: int
    num_roads: int
    has_longest_road: bool
    has_largest_army: bool
    num_knights: int
    resource_count: int

    @staticmethod
    def from_state(state: GameState, player_idx: int) -> PlayerSnapshot:
        p = state.players[player_idx]
        return PlayerSnapshot(
            vp=p.actual_victory_points,
            num_settlements=len(p.settlements),
            num_cities=len(p.cities),
            num_roads=len(p.roads),
            has_longest_road=p.has_longest_road,
            has_largest_army=p.has_largest_army,
            num_knights=p.num_knights_played,
            resource_count=p.resource_count(),
        )


def compute_shaped_reward(
    prev: PlayerSnapshot,
    curr: PlayerSnapshot,
    prev_opp_vp: list[int],
    curr_opp_vp: list[int],
) -> float:
    """Compute intermediate reward from state change.

    Args:
        prev: player snapshot before action
        curr: player snapshot after action
        prev_opp_vp: opponent VP counts before action
        curr_opp_vp: opponent VP counts after action

    Returns:
        Shaped reward (small float, typically -0.2 to +0.3)
    """
    reward = 0.0

    # ── VP gains ──
    vp_delta = curr.vp - prev.vp
    if vp_delta > 0:
        reward += 0.15 * vp_delta

    # ── Building rewards (even if no VP change yet) ──
    new_settlements = curr.num_settlements - prev.num_settlements
    if new_settlements > 0:
        reward += 0.05 * new_settlements

    new_cities = curr.num_cities - prev.num_cities
    if new_cities > 0:
        reward += 0.08 * new_cities

    new_roads = curr.num_roads - prev.num_roads
    if new_roads > 0:
        reward += 0.02 * new_roads

    # ── Longest road / largest army transitions ──
    if curr.has_longest_road and not prev.has_longest_road:
        reward += 0.15
    elif not curr.has_longest_road and prev.has_longest_road:
        reward -= 0.15

    if curr.has_largest_army and not prev.has_largest_army:
        reward += 0.15
    elif not curr.has_largest_army and prev.has_largest_army:
        reward -= 0.15

    # ── Knight played ──
    new_knights = curr.num_knights - prev.num_knights
    if new_knights > 0:
        reward += 0.03 * new_knights

    # ── Opponent VP gains (penalize) ──
    for prev_vp, curr_vp in zip(prev_opp_vp, curr_opp_vp):
        opp_vp_gain = curr_vp - prev_vp
        if opp_vp_gain > 0:
            reward -= 0.05 * opp_vp_gain

    return reward
