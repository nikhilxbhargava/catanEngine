"""Game engine: applies actions to advance game state.

Provides both a functional interface (apply_action returns new state)
and a mutable Game class for fast simulation.
"""

from __future__ import annotations

import random as _random
from typing import Optional

from catan.actions import Action, ActionType, get_legal_actions
from catan.board import Board
from catan.decks import (
    CITY_COST, DEVELOPMENT_CARD_COST, RESOURCE_INDEXES, ROAD_COST,
    SETTLEMENT_COST, deck_subtract, deck_add, deck_total,
)
from catan.enums import (
    KNIGHT, MONOPOLY, RESOURCES, ROAD_BUILDING, VICTORY_POINT,
    YEAR_OF_PLENTY, GamePhase,
)
from catan.map import EdgeId
from catan.state import GameState


def apply_action(state: GameState, action: Action) -> GameState:
    """Return a new GameState with the action applied (for tree search)."""
    new_state = state.clone()
    apply_action_mutate(new_state, action)
    return new_state


def apply_action_mutate(state: GameState, action: Action, rng: Optional[_random.Random] = None) -> None:
    """Apply an action in-place (for fast simulation)."""
    at = action.action_type

    if at == ActionType.PLACE_INITIAL_SETTLEMENT:
        _do_place_initial_settlement(state, action)
    elif at == ActionType.PLACE_INITIAL_ROAD:
        _do_place_initial_road(state, action)
    elif at == ActionType.ROLL_DICE:
        _do_roll_dice(state, rng)
    elif at == ActionType.DISCARD:
        _do_discard(state, action)
    elif at == ActionType.MOVE_ROBBER:
        _do_move_robber(state, action)
    elif at == ActionType.STEAL:
        _do_steal(state, action, rng)
    elif at == ActionType.BUILD_SETTLEMENT:
        _do_build_settlement(state, action)
    elif at == ActionType.BUILD_CITY:
        _do_build_city(state, action)
    elif at == ActionType.BUILD_ROAD:
        _do_build_road(state, action)
    elif at == ActionType.BUY_DEV_CARD:
        _do_buy_dev_card(state)
    elif at == ActionType.PLAY_KNIGHT:
        _do_play_knight(state, action)
    elif at == ActionType.PLAY_YEAR_OF_PLENTY:
        _do_play_year_of_plenty(state, action)
    elif at == ActionType.PLAY_MONOPOLY:
        _do_play_monopoly(state, action)
    elif at == ActionType.PLAY_ROAD_BUILDING:
        _do_play_road_building(state)
    elif at == ActionType.PLACE_FREE_ROAD:
        _do_place_free_road(state, action)
    elif at == ActionType.BANK_TRADE:
        _do_bank_trade(state, action)
    elif at == ActionType.END_TURN:
        _do_end_turn(state)


# ── Setup actions ────────────────────────────────────────────────────


def _do_place_initial_settlement(state: GameState, action: Action) -> None:
    vid = action.vertex
    pidx = state.current_player_index
    player = state.players[pidx]

    state.buildings[vid] = (pidx, "SETTLEMENT")
    player.settlements.add(vid)

    # On second settlement, collect resources from adjacent tiles
    if state.phase == GamePhase.SETUP_SECOND_SETTLEMENT:
        for tile_idx in state.board.vertex_to_tiles[vid]:
            tile = state.board.land_tiles[tile_idx]
            if tile.resource is not None:
                ridx = RESOURCE_INDEXES[tile.resource]
                if state.resource_bank[ridx] > 0:
                    player.resources[ridx] += 1
                    state.resource_bank[ridx] -= 1

    # Advance to road placement
    if state.phase == GamePhase.SETUP_FIRST_SETTLEMENT:
        state.phase = GamePhase.SETUP_FIRST_ROAD
    else:
        state.phase = GamePhase.SETUP_SECOND_ROAD

    _update_victory_points(state, pidx)


def _do_place_initial_road(state: GameState, action: Action) -> None:
    eid = action.edge
    pidx = state.current_player_index
    player = state.players[pidx]

    state.roads[eid] = pidx
    player.roads.add(eid)

    # Advance setup
    state.setup_step += 1
    if state.setup_step < len(state.setup_order):
        state.current_player_index = state.setup_order[state.setup_step]
        if state.setup_step < state.num_players * 2:
            # First half: first settlement+road, second half: second settlement+road
            if state.setup_step < state.num_players:
                state.phase = GamePhase.SETUP_FIRST_SETTLEMENT
            else:
                state.phase = GamePhase.SETUP_SECOND_SETTLEMENT
    else:
        # Setup complete, start normal play
        state.current_player_index = 0
        state.phase = GamePhase.ROLL_DICE
        state.turn_number = 1


# ── Dice roll ────────────────────────────────────────────────────────


def _do_roll_dice(state: GameState, rng: Optional[_random.Random] = None) -> None:
    if rng is None:
        rng = _random.Random()
    d1 = rng.randint(1, 6)
    d2 = rng.randint(1, 6)
    state.dice_result = (d1, d2)
    total = d1 + d2

    if total == 7:
        # Check who needs to discard (>7 cards)
        state.discard_players = [
            i for i in range(state.num_players)
            if state.players[i].resource_count() > 7
        ]
        if state.discard_players:
            state.phase = GamePhase.DISCARD
        else:
            state.phase = GamePhase.MOVE_ROBBER
    else:
        _distribute_resources(state, total)
        state.phase = GamePhase.MAIN_TURN


def _distribute_resources(state: GameState, roll: int) -> None:
    """Distribute resources based on dice roll."""
    board = state.board
    if roll not in board.tiles_by_number:
        return

    for tile_idx in board.tiles_by_number[roll]:
        if tile_idx == state.robber_tile:
            continue
        tile = board.land_tiles[tile_idx]
        if tile.resource is None:
            continue
        ridx = RESOURCE_INDEXES[tile.resource]

        # Count how many resources are needed
        total_needed = 0
        claims: list[tuple[int, int]] = []  # (player_idx, amount)
        for vid in tile.nodes.values():
            if vid in state.buildings:
                owner, btype = state.buildings[vid]
                amount = 2 if btype == "CITY" else 1
                claims.append((owner, amount))
                total_needed += amount

        # If bank doesn't have enough, no one gets any (per rules)
        if state.resource_bank[ridx] < total_needed:
            continue

        for owner, amount in claims:
            state.players[owner].resources[ridx] += amount
            state.resource_bank[ridx] -= amount


# ── Discard ──────────────────────────────────────────────────────────


def _do_discard(state: GameState, action: Action) -> None:
    player_idx = state.discard_players[0]
    player = state.players[player_idx]
    discard = list(action.discard_resources)

    player.resources = deck_subtract(player.resources, discard)
    state.resource_bank = deck_add(state.resource_bank, discard)

    state.discard_players.pop(0)
    if not state.discard_players:
        state.phase = GamePhase.MOVE_ROBBER


# ── Robber ───────────────────────────────────────────────────────────


def _do_move_robber(state: GameState, action: Action) -> None:
    state.robber_tile = action.tile
    state.phase = GamePhase.STEAL


def _do_steal(state: GameState, action: Action, rng: Optional[_random.Random] = None) -> None:
    if action.steal_from is not None:
        if rng is None:
            rng = _random.Random()
        victim = state.players[action.steal_from]
        if victim.resource_count() > 0:
            # Pick a random resource from the victim
            available = []
            for i, count in enumerate(victim.resources):
                available.extend([i] * count)
            ridx = rng.choice(available)
            victim.resources[ridx] -= 1
            state.current_player().resources[ridx] += 1

    state.phase = GamePhase.MAIN_TURN


# ── Building ─────────────────────────────────────────────────────────


def _do_build_settlement(state: GameState, action: Action) -> None:
    vid = action.vertex
    pidx = state.current_player_index
    player = state.players[pidx]

    player.resources = deck_subtract(player.resources, SETTLEMENT_COST)
    state.resource_bank = deck_add(state.resource_bank, SETTLEMENT_COST)
    state.buildings[vid] = (pidx, "SETTLEMENT")
    player.settlements.add(vid)

    # Settlement can break opponents' roads — recalculate all
    _update_longest_road_all(state)
    _update_victory_points(state, pidx)
    _check_winner(state)


def _do_build_city(state: GameState, action: Action) -> None:
    vid = action.vertex
    pidx = state.current_player_index
    player = state.players[pidx]

    player.resources = deck_subtract(player.resources, CITY_COST)
    state.resource_bank = deck_add(state.resource_bank, CITY_COST)
    state.buildings[vid] = (pidx, "CITY")
    player.settlements.discard(vid)
    player.cities.add(vid)

    _update_victory_points(state, pidx)
    _check_winner(state)


def _do_build_road(state: GameState, action: Action) -> None:
    eid = action.edge
    pidx = state.current_player_index
    player = state.players[pidx]

    player.resources = deck_subtract(player.resources, ROAD_COST)
    state.resource_bank = deck_add(state.resource_bank, ROAD_COST)
    state.roads[eid] = pidx
    player.roads.add(eid)

    # Only this player's road length can change
    _update_longest_road_all(state, changed_players=[pidx])
    _update_victory_points(state, pidx)
    _check_winner(state)


# ── Dev cards ────────────────────────────────────────────────────────


def _do_buy_dev_card(state: GameState) -> None:
    pidx = state.current_player_index
    player = state.players[pidx]

    player.resources = deck_subtract(player.resources, DEVELOPMENT_CARD_COST)
    state.resource_bank = deck_add(state.resource_bank, DEVELOPMENT_CARD_COST)

    card = state.dev_card_deck.pop()
    player.dev_cards.append(card)
    player.dev_cards_bought_this_turn.append(card)

    if card == VICTORY_POINT:
        _update_victory_points(state, pidx)
        _check_winner(state)


def _do_play_knight(state: GameState, action: Action) -> None:
    pidx = state.current_player_index
    player = state.players[pidx]

    player.dev_cards.remove(KNIGHT)
    player.played_dev_cards.append(KNIGHT)
    player.num_knights_played += 1
    state.dev_card_played_this_turn = True

    # Move robber
    state.robber_tile = action.tile

    # Update largest army
    if player.num_knights_played >= 3:
        if state.largest_army_player == -1:
            state.largest_army_player = pidx
            state.largest_army_size = player.num_knights_played
            player.has_largest_army = True
        elif player.num_knights_played > state.largest_army_size:
            if state.largest_army_player != pidx:
                state.players[state.largest_army_player].has_largest_army = False
                _update_victory_points(state, state.largest_army_player)
            state.largest_army_player = pidx
            state.largest_army_size = player.num_knights_played
            player.has_largest_army = True

    _update_victory_points(state, pidx)

    # Now need to steal
    state.phase = GamePhase.STEAL
    _check_winner(state)


def _do_play_year_of_plenty(state: GameState, action: Action) -> None:
    player = state.current_player()
    r1_idx = RESOURCE_INDEXES[action.resource1]
    r2_idx = RESOURCE_INDEXES[action.resource2]

    player.resources[r1_idx] += 1
    state.resource_bank[r1_idx] -= 1
    player.resources[r2_idx] += 1
    state.resource_bank[r2_idx] -= 1

    player.dev_cards.remove(YEAR_OF_PLENTY)
    player.played_dev_cards.append(YEAR_OF_PLENTY)
    state.dev_card_played_this_turn = True


def _do_play_monopoly(state: GameState, action: Action) -> None:
    pidx = state.current_player_index
    player = state.players[pidx]
    ridx = RESOURCE_INDEXES[action.resource1]

    total_stolen = 0
    for i, p in enumerate(state.players):
        if i != pidx:
            total_stolen += p.resources[ridx]
            p.resources[ridx] = 0

    player.resources[ridx] += total_stolen
    player.dev_cards.remove(MONOPOLY)
    player.played_dev_cards.append(MONOPOLY)
    state.dev_card_played_this_turn = True


def _do_play_road_building(state: GameState) -> None:
    player = state.current_player()
    player.dev_cards.remove(ROAD_BUILDING)
    player.played_dev_cards.append(ROAD_BUILDING)
    state.dev_card_played_this_turn = True
    state.free_roads_remaining = min(2, 15 - len(player.roads))


def _do_place_free_road(state: GameState, action: Action) -> None:
    eid = action.edge
    pidx = state.current_player_index
    player = state.players[pidx]

    state.roads[eid] = pidx
    player.roads.add(eid)
    state.free_roads_remaining -= 1

    _update_longest_road_all(state, changed_players=[pidx])
    _update_victory_points(state, pidx)
    _check_winner(state)


# ── Trading ──────────────────────────────────────────────────────────


def _do_bank_trade(state: GameState, action: Action) -> None:
    player = state.current_player()
    board = state.board
    give_idx = RESOURCE_INDEXES[action.give_resource]
    get_idx = RESOURCE_INDEXES[action.get_resource]

    # Determine rate
    rate = 4
    for vid in player.settlements | player.cities:
        if vid in board.port_vertices:
            port_resource = board.port_vertices[vid]
            if port_resource is None:
                rate = min(rate, 3)
            elif port_resource == action.give_resource:
                rate = min(rate, 2)

    player.resources[give_idx] -= rate
    state.resource_bank[give_idx] += rate
    player.resources[get_idx] += 1
    state.resource_bank[get_idx] -= 1


# ── End turn ─────────────────────────────────────────────────────────


def _do_end_turn(state: GameState) -> None:
    player = state.current_player()
    player.dev_cards_bought_this_turn.clear()

    state.current_player_index = (state.current_player_index + 1) % state.num_players
    state.turn_number += 1
    state.dice_result = None
    state.dev_card_played_this_turn = False
    state.free_roads_remaining = 0
    state.phase = GamePhase.ROLL_DICE


# ── Victory / longest road ──────────────────────────────────────────


def _update_victory_points(state: GameState, pidx: int) -> None:
    player = state.players[pidx]
    vp = len(player.settlements) + 2 * len(player.cities)
    vp += player.played_dev_cards.count(VICTORY_POINT)
    vp += sum(1 for c in player.dev_cards if c == VICTORY_POINT)
    if player.has_longest_road:
        vp += 2
    if player.has_largest_army:
        vp += 2
    player.actual_victory_points = vp


def _check_winner(state: GameState) -> None:
    for i, player in enumerate(state.players):
        if player.actual_victory_points >= 10:
            state.winner = i
            state.phase = GamePhase.GAME_OVER
            return


def _update_longest_road_all(state: GameState, changed_players: list[int] | None = None) -> None:
    """Recalculate longest road. Only recomputes for affected players when possible."""
    if changed_players is None:
        changed_players = list(range(state.num_players))

    # Compute lengths only for players who changed (+ current holder)
    players_to_check = set(changed_players)
    if state.longest_road_player >= 0:
        players_to_check.add(state.longest_road_player)

    lengths = {}
    for i in players_to_check:
        lengths[i] = _calculate_longest_road(state, i)

    # Find new longest road holder
    # First check if current holder lost it
    if state.longest_road_player >= 0:
        cur = state.longest_road_player
        cur_len = lengths.get(cur, state.longest_road_length)
        if cur_len < 5:
            state.players[cur].has_longest_road = False
            state.longest_road_player = -1
            state.longest_road_length = 0
            # Need to check all players now
            for i in range(state.num_players):
                if i not in lengths:
                    lengths[i] = _calculate_longest_road(state, i)
        else:
            state.longest_road_length = cur_len

    # Check if anyone beats the current holder
    for i in range(state.num_players):
        length = lengths.get(i)
        if length is None:
            continue
        if length >= 5 and length > state.longest_road_length:
            old = state.longest_road_player
            if old >= 0 and old != i:
                state.players[old].has_longest_road = False
                _update_victory_points(state, old)
            state.longest_road_player = i
            state.longest_road_length = length
            state.players[i].has_longest_road = True

    for i in players_to_check:
        _update_victory_points(state, i)


def _calculate_longest_road(state: GameState, player_idx: int) -> int:
    """DFS to find the longest simple path in a player's road network.

    Opponent settlements/cities break the road at that vertex.
    Uses iterative DFS with explicit stack to avoid Python function call overhead.
    """
    board = state.board
    player = state.players[player_idx]
    roads = player.roads

    if not roads:
        return 0

    # Build adjacency: vertex -> list of (edge, other_vertex)
    road_graph: dict[int, list[tuple[EdgeId, int]]] = {}
    edge_to_verts = board.edge_to_vertices
    for eid in roads:
        v1, v2 = edge_to_verts[eid]
        road_graph.setdefault(v1, []).append((eid, v2))
        road_graph.setdefault(v2, []).append((eid, v1))

    # Vertices blocked by opponent buildings
    blocked: set[int] = set()
    buildings = state.buildings
    for vid in buildings:
        if buildings[vid][0] != player_idx:
            blocked.add(vid)

    best = 0

    # Iterative DFS with explicit stack: (vertex, visited_edges_frozenset, length)
    # For small road networks (<=15 edges), recursive with set is fine but
    # we avoid closure overhead by using nonlocal-free approach
    def dfs(start: int) -> None:
        nonlocal best
        # Stack entries: (vertex, visited_edges, length)
        stack = [(start, frozenset(), 0)]
        while stack:
            vertex, visited, length = stack.pop()
            if length > best:
                best = length
            if vertex not in road_graph:
                continue
            for eid, other in road_graph[vertex]:
                if eid in visited:
                    continue
                if other in blocked:
                    if length + 1 > best:
                        best = length + 1
                    continue
                stack.append((other, visited | {eid}, length + 1))

    for start_vertex in road_graph:
        dfs(start_vertex)

    return best


# ── High-level game runner ───────────────────────────────────────────


class Game:
    """Manages a full game with agents."""

    def __init__(
        self,
        board: Board,
        num_players: int = 4,
        seed: Optional[int] = None,
    ):
        self.rng = _random.Random(seed)
        self.state = GameState(board, num_players, seed=seed)
        self.action_log: list[Action] = []

    def get_legal_actions(self) -> list[Action]:
        return get_legal_actions(self.state)

    def apply(self, action: Action) -> None:
        self.action_log.append(action)
        apply_action_mutate(self.state, action, rng=self.rng)

    def is_over(self) -> bool:
        return self.state.phase == GamePhase.GAME_OVER

    def winner(self) -> Optional[int]:
        return self.state.winner
