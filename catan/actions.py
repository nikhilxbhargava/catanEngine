"""Action definitions and legal action enumeration.

Every possible move in Catan is represented as an Action dataclass.
get_legal_actions(state) returns all valid actions for the current game phase.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

from catan.decks import (
    CITY_COST, DEVELOPMENT_CARD_COST, ROAD_COST, SETTLEMENT_COST,
    RESOURCE_INDEXES, deck_contains, deck_total,
)
from catan.enums import (
    KNIGHT, MONOPOLY, RESOURCES, ROAD_BUILDING, VICTORY_POINT,
    YEAR_OF_PLENTY, GamePhase,
)
from catan.map import EdgeId, VertexId
from catan.state import GameState


class ActionType(IntEnum):
    ROLL_DICE = 0
    BUILD_SETTLEMENT = 1
    BUILD_CITY = 2
    BUILD_ROAD = 3
    BUY_DEV_CARD = 4
    PLAY_KNIGHT = 5
    PLAY_YEAR_OF_PLENTY = 6
    PLAY_MONOPOLY = 7
    PLAY_ROAD_BUILDING = 8
    BANK_TRADE = 9
    DISCARD = 10
    END_TURN = 11
    PLACE_INITIAL_SETTLEMENT = 12
    PLACE_INITIAL_ROAD = 13
    PLACE_FREE_ROAD = 14
    MOVE_ROBBER = 15
    STEAL = 16


@dataclass(frozen=True)
class Action:
    action_type: ActionType
    vertex: Optional[VertexId] = None
    edge: Optional[EdgeId] = None
    tile: Optional[int] = None  # tile index for robber
    steal_from: Optional[int] = None  # player index
    resource1: Optional[str] = None
    resource2: Optional[str] = None
    give_resource: Optional[str] = None
    get_resource: Optional[str] = None
    discard_resources: Optional[tuple[int, ...]] = None  # freqdeck as tuple

    def __repr__(self) -> str:
        parts = [self.action_type.name]
        if self.vertex is not None:
            parts.append(f"v={self.vertex}")
        if self.edge is not None:
            parts.append(f"e={self.edge}")
        if self.tile is not None:
            parts.append(f"tile={self.tile}")
        if self.steal_from is not None:
            parts.append(f"steal={self.steal_from}")
        if self.resource1 is not None:
            parts.append(f"r1={self.resource1}")
        if self.resource2 is not None:
            parts.append(f"r2={self.resource2}")
        if self.give_resource is not None:
            parts.append(f"give={self.give_resource}")
        if self.get_resource is not None:
            parts.append(f"get={self.get_resource}")
        return f"Action({', '.join(parts)})"


# ── Legal action enumeration ────────────────────────────────────────


def get_legal_actions(state: GameState) -> list[Action]:
    """Return all legal actions for the current game state."""
    phase = state.phase

    if phase == GamePhase.SETUP_FIRST_SETTLEMENT or phase == GamePhase.SETUP_SECOND_SETTLEMENT:
        return _legal_setup_settlements(state)
    elif phase == GamePhase.SETUP_FIRST_ROAD or phase == GamePhase.SETUP_SECOND_ROAD:
        return _legal_setup_roads(state)
    elif phase == GamePhase.ROLL_DICE:
        return _legal_pre_roll(state)
    elif phase == GamePhase.DISCARD:
        return _legal_discards(state)
    elif phase == GamePhase.MOVE_ROBBER:
        return _legal_robber_moves(state)
    elif phase == GamePhase.STEAL:
        return _legal_steals(state)
    elif phase == GamePhase.MAIN_TURN:
        return _legal_main_turn(state)
    elif phase == GamePhase.GAME_OVER:
        return []
    return []


# ── Setup phase ──────────────────────────────────────────────────────


def _valid_settlement_vertices(state: GameState) -> list[VertexId]:
    """Vertices where a settlement can be placed (distance rule)."""
    board = state.board
    valid = []
    for vid in board.get_vertex_ids():
        if vid in state.buildings:
            continue
        # Distance rule: no adjacent vertex has a building
        if any(adj in state.buildings for adj in board.vertex_to_adjacent_vertices[vid]):
            continue
        valid.append(vid)
    return valid


def _legal_setup_settlements(state: GameState) -> list[Action]:
    vertices = _valid_settlement_vertices(state)
    at = ActionType.PLACE_INITIAL_SETTLEMENT
    return [Action(action_type=at, vertex=v) for v in vertices]


def _legal_setup_roads(state: GameState) -> list[Action]:
    """After placing a setup settlement, roads must be adjacent to it."""
    board = state.board
    player = state.current_player()
    # Find the most recently placed settlement for this player
    # (the one that doesn't have any adjacent road from this player yet)
    latest_settlement = None
    for v in player.settlements:
        has_road = any(
            e in player.roads
            for e in board.vertex_to_adjacent_edges[v]
        )
        # During first setup, no roads exist yet, so pick any settlement.
        # During second setup, pick the one without a road.
        if not has_road:
            latest_settlement = v
            break

    if latest_settlement is None:
        # Fallback: use any settlement (shouldn't happen in normal play)
        latest_settlement = next(iter(player.settlements))

    actions = []
    at = ActionType.PLACE_INITIAL_ROAD
    for eid in board.vertex_to_adjacent_edges[latest_settlement]:
        if eid not in state.roads:
            actions.append(Action(action_type=at, edge=eid))
    return actions


# ── Roll phase ───────────────────────────────────────────────────────


def _legal_pre_roll(state: GameState) -> list[Action]:
    """Before rolling, player can play a knight (if they have one)."""
    actions = [Action(action_type=ActionType.ROLL_DICE)]
    player = state.current_player()
    if (
        not state.dev_card_played_this_turn
        and KNIGHT in player.dev_cards
        and KNIGHT not in player.dev_cards_bought_this_turn
    ):
        # Knight: choose tile to move robber to
        for tile_idx in range(len(state.board.land_tiles)):
            if tile_idx != state.robber_tile:
                actions.append(Action(
                    action_type=ActionType.PLAY_KNIGHT,
                    tile=tile_idx,
                ))
    return actions


# ── Discard phase ────────────────────────────────────────────────────


def _legal_discards(state: GameState) -> list[Action]:
    """Generate discard actions for the current player who needs to discard."""
    if not state.discard_players:
        return []

    player_idx = state.discard_players[0]
    player = state.players[player_idx]
    total = sum(player.resources)
    to_discard = total // 2

    # Generate all valid discard combinations
    combos = _discard_combinations(player.resources, to_discard)
    return [
        Action(action_type=ActionType.DISCARD, discard_resources=tuple(c))
        for c in combos
    ]


def _discard_combinations(resources: list[int], amount: int) -> list[list[int]]:
    """Generate all ways to discard exactly `amount` cards from `resources`.

    Each result is a freqdeck of cards to discard.
    """
    results: list[list[int]] = []
    current = [0] * 5

    def backtrack(idx: int, remaining: int) -> None:
        if remaining == 0:
            results.append(current.copy())
            return
        if idx >= 5:
            return
        max_from_this = min(resources[idx], remaining)
        for take in range(max_from_this + 1):
            current[idx] = take
            backtrack(idx + 1, remaining - take)
        current[idx] = 0

    backtrack(0, amount)
    return results


# ── Robber ───────────────────────────────────────────────────────────


def _legal_robber_moves(state: GameState) -> list[Action]:
    return [
        Action(action_type=ActionType.MOVE_ROBBER, tile=i)
        for i in range(len(state.board.land_tiles))
        if i != state.robber_tile
    ]


def _legal_steals(state: GameState) -> list[Action]:
    """After placing the robber, steal from a player on that tile."""
    board = state.board
    tile = board.land_tiles[state.robber_tile]
    current = state.current_player_index

    targets: set[int] = set()
    for v in tile.nodes.values():
        if v in state.buildings:
            owner, _ = state.buildings[v]
            if owner != current and state.players[owner].resource_count() > 0:
                targets.add(owner)

    if not targets:
        # No one to steal from — must still "steal" (pass)
        return [Action(action_type=ActionType.STEAL, steal_from=None)]

    return [
        Action(action_type=ActionType.STEAL, steal_from=t)
        for t in sorted(targets)
    ]


# ── Main turn ────────────────────────────────────────────────────────


def _legal_main_turn(state: GameState) -> list[Action]:
    actions: list[Action] = []
    player = state.current_player()
    board = state.board
    pidx = state.current_player_index

    # Always can end turn
    actions.append(Action(action_type=ActionType.END_TURN))

    # ── Build settlement ──
    if deck_contains(player.resources, SETTLEMENT_COST) and len(player.settlements) < 5:
        for vid in board.get_vertex_ids():
            if _can_build_settlement(state, vid, pidx):
                actions.append(Action(action_type=ActionType.BUILD_SETTLEMENT, vertex=vid))

    # ── Build city ──
    if deck_contains(player.resources, CITY_COST) and len(player.cities) < 4:
        for vid in player.settlements:
            actions.append(Action(action_type=ActionType.BUILD_CITY, vertex=vid))

    # ── Build road ──
    if (
        deck_contains(player.resources, ROAD_COST)
        and len(player.roads) < 15
        and state.free_roads_remaining == 0
    ):
        for eid in _legal_road_edges(state, pidx):
            actions.append(Action(action_type=ActionType.BUILD_ROAD, edge=eid))

    # ── Free roads (road building dev card) ──
    if state.free_roads_remaining > 0:
        for eid in _legal_road_edges(state, pidx):
            actions.append(Action(action_type=ActionType.PLACE_FREE_ROAD, edge=eid))
        # If no legal road placements, allow ending
        if not any(a.action_type == ActionType.PLACE_FREE_ROAD for a in actions):
            state.free_roads_remaining = 0  # auto-clear

    # ── Buy development card ──
    if (
        deck_contains(player.resources, DEVELOPMENT_CARD_COST)
        and len(state.dev_card_deck) > 0
    ):
        actions.append(Action(action_type=ActionType.BUY_DEV_CARD))

    # ── Play development cards ──
    if not state.dev_card_played_this_turn:
        playable = set(player.dev_cards) - set(player.dev_cards_bought_this_turn) - {VICTORY_POINT}
        if KNIGHT in playable:
            for tile_idx in range(len(board.land_tiles)):
                if tile_idx != state.robber_tile:
                    actions.append(Action(action_type=ActionType.PLAY_KNIGHT, tile=tile_idx))

        if YEAR_OF_PLENTY in playable:
            for r1 in RESOURCES:
                for r2 in RESOURCES:
                    if state.resource_bank[RESOURCE_INDEXES[r1]] > 0 and (
                        r1 != r2 or state.resource_bank[RESOURCE_INDEXES[r1]] > 1
                    ):
                        actions.append(Action(
                            action_type=ActionType.PLAY_YEAR_OF_PLENTY,
                            resource1=r1, resource2=r2,
                        ))

        if MONOPOLY in playable:
            for r in RESOURCES:
                actions.append(Action(
                    action_type=ActionType.PLAY_MONOPOLY,
                    resource1=r,
                ))

        if ROAD_BUILDING in playable and len(player.roads) < 15:
            actions.append(Action(action_type=ActionType.PLAY_ROAD_BUILDING))

    # ── Bank / port trades ──
    for give_r in RESOURCES:
        give_idx = RESOURCE_INDEXES[give_r]
        # Determine exchange rate
        rate = 4  # default bank trade
        for vid in player.settlements | player.cities:
            if vid in board.port_vertices:
                port_resource = board.port_vertices[vid]
                if port_resource is None:
                    rate = min(rate, 3)  # 3:1 port
                elif port_resource == give_r:
                    rate = min(rate, 2)  # 2:1 specific port

        if player.resources[give_idx] >= rate:
            for get_r in RESOURCES:
                if get_r != give_r and state.resource_bank[RESOURCE_INDEXES[get_r]] > 0:
                    actions.append(Action(
                        action_type=ActionType.BANK_TRADE,
                        give_resource=give_r,
                        get_resource=get_r,
                    ))

    return actions


def _can_build_settlement(state: GameState, vid: VertexId, player_idx: int) -> bool:
    """Check if player can build a settlement at vid (has adjacent road, distance rule)."""
    if vid in state.buildings:
        return False
    board = state.board
    # Distance rule
    if any(adj in state.buildings for adj in board.vertex_to_adjacent_vertices[vid]):
        return False
    # Must have a road to this vertex
    player = state.players[player_idx]
    return any(
        eid in player.roads
        for eid in board.vertex_to_adjacent_edges[vid]
    )


def _legal_road_edges(state: GameState, player_idx: int) -> list[EdgeId]:
    """All edges where the player can legally place a road."""
    board = state.board
    player = state.players[player_idx]
    legal = []
    for eid in board.get_edge_ids():
        if eid in state.roads:
            continue
        v1, v2 = board.edge_to_vertices[eid]
        # Must connect to player's existing road or building
        connected = False
        for v in (v1, v2):
            # Building at this vertex owned by player
            if v in state.buildings and state.buildings[v][0] == player_idx:
                connected = True
                break
            # Road from this vertex owned by player (and no opponent building blocking)
            if v in state.buildings and state.buildings[v][0] != player_idx:
                continue  # opponent building blocks connection through this vertex
            for adj_eid in board.vertex_to_adjacent_edges[v]:
                if adj_eid in player.roads:
                    connected = True
                    break
            if connected:
                break
        if connected:
            legal.append(eid)
    return legal
