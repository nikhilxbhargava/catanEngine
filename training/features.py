"""State -> feature vector conversion for neural network input.

Converts a GameState into a flat numeric vector suitable for NN input.
All features are from the perspective of the current player.
"""

from __future__ import annotations

from catan.decks import RESOURCE_INDEXES
from catan.enums import (
    KNIGHT, MONOPOLY, RESOURCES, ROAD_BUILDING, VICTORY_POINT,
    YEAR_OF_PLENTY, GamePhase,
)
from catan.state import GameState

# Precompute dice probability weights (how likely each number is to be rolled)
# Used to encode tile "value" — a 6 or 8 tile is much more valuable than a 2 or 12
_DICE_PROBS = {
    0: 0.0, 2: 1/36, 3: 2/36, 4: 3/36, 5: 4/36, 6: 5/36,
    7: 0.0, 8: 5/36, 9: 4/36, 10: 3/36, 11: 2/36, 12: 1/36,
}


def state_to_features(state: GameState) -> list[float]:
    """Convert game state to a flat feature vector.

    Feature groups:
    1. Current player resources (5)
    2. Current player building counts (3)
    3. Current player dev card counts (5)
    4. Current player VP + knights (2)
    5. Relative features vs opponents (8)
    6. Per-opponent visible info (3 x 5 = 15)
    7. Per-vertex state (54 x 6 = 324)
    8. Per-edge state (72 x 3 = 216)
    9. Per-tile state (19 x 8 = 152)
    10. Game phase one-hot (10)
    11. Bank resources (5)
    12. Production potential per resource (5)
    13. Turn progress (1)

    Total: ~751 features
    """
    board = state.board
    pidx = state.current_player_index
    player = state.players[pidx]
    features: list[float] = []

    # 1. Current player resources (5)
    for r in player.resources:
        features.append(float(r))

    # 2. Building counts (3)
    features.append(float(len(player.settlements)))
    features.append(float(len(player.cities)))
    features.append(float(len(player.roads)))

    # 3. Dev card counts (5)
    for card_type in [KNIGHT, YEAR_OF_PLENTY, MONOPOLY, ROAD_BUILDING, VICTORY_POINT]:
        features.append(float(player.dev_cards.count(card_type)))

    # 4. VP + knights (2)
    features.append(float(player.actual_victory_points))
    features.append(float(player.num_knights_played))

    # 5. Relative features — how we compare to opponents (8)
    opp_max_vp = 0
    opp_total_resources = 0
    opp_max_roads = 0
    opp_max_knights = 0
    for i in range(state.num_players):
        if i == pidx:
            continue
        opp = state.players[i]
        opp_max_vp = max(opp_max_vp, opp.public_victory_points)
        opp_total_resources += opp.resource_count()
        opp_max_roads = max(opp_max_roads, len(opp.roads))
        opp_max_knights = max(opp_max_knights, opp.num_knights_played)

    features.append(float(player.actual_victory_points - opp_max_vp))  # VP lead
    features.append(float(player.resource_count() - opp_total_resources / max(state.num_players - 1, 1)))  # resource advantage
    features.append(float(len(player.roads) - opp_max_roads))  # road lead
    features.append(float(player.num_knights_played - opp_max_knights))  # army lead
    features.append(1.0 if player.has_longest_road else 0.0)
    features.append(1.0 if player.has_largest_army else 0.0)
    features.append(float(10 - player.actual_victory_points))  # distance to win
    features.append(float(10 - opp_max_vp))  # opponent distance to win

    # 6. Opponent info (15)
    for i in range(state.num_players):
        if i == pidx:
            continue
        opp = state.players[i]
        features.append(float(opp.resource_count()))
        features.append(float(len(opp.dev_cards)))
        features.append(float(opp.public_victory_points))
        features.append(float(opp.num_knights_played))
        features.append(float(len(opp.roads)))

    # 7. Per-vertex (54 x 6 = 324)
    land_vids = board.get_vertex_ids()
    for vid in land_vids:
        is_empty = 1.0 if vid not in state.buildings else 0.0
        has_own_settlement = 0.0
        has_own_city = 0.0
        has_opp_settlement = 0.0
        has_opp_city = 0.0
        if vid in state.buildings:
            owner, btype = state.buildings[vid]
            if owner == pidx:
                if btype == "SETTLEMENT":
                    has_own_settlement = 1.0
                else:
                    has_own_city = 1.0
            else:
                if btype == "SETTLEMENT":
                    has_opp_settlement = 1.0
                else:
                    has_opp_city = 1.0
        has_port = 1.0 if vid in board.port_vertices else 0.0
        features.extend([is_empty, has_own_settlement, has_own_city,
                         has_opp_settlement, has_opp_city, has_port])

    # 8. Per-edge (72 x 3 = 216)
    land_eids = board.get_edge_ids()
    for eid in land_eids:
        is_empty = 1.0 if eid not in state.roads else 0.0
        has_own = 0.0
        has_opp = 0.0
        if eid in state.roads:
            if state.roads[eid] == pidx:
                has_own = 1.0
            else:
                has_opp = 1.0
        features.extend([is_empty, has_own, has_opp])

    # 9. Per-tile (19 x 8 = 152)
    for tile in board.land_tiles:
        for r in RESOURCES:
            features.append(1.0 if tile.resource == r else 0.0)
        features.append(_DICE_PROBS.get(tile.number, 0.0))  # probability, not raw number
        features.append(1.0 if tile.id == state.robber_tile else 0.0)
        features.append(1.0 if tile.resource is None else 0.0)

    # 10. Game phase one-hot (10)
    for phase in GamePhase:
        features.append(1.0 if state.phase == phase else 0.0)

    # 11. Bank resources (5)
    for r in state.resource_bank:
        features.append(float(r) / 19.0)  # normalized

    # 12. Production potential per resource (5)
    # How much of each resource does the current player produce per expected roll?
    production = [0.0] * 5
    for vid in player.settlements | player.cities:
        multiplier = 2.0 if vid in player.cities else 1.0
        for tile_idx in board.vertex_to_tiles[vid]:
            tile = board.land_tiles[tile_idx]
            if tile.resource is not None and tile_idx != state.robber_tile:
                ridx = RESOURCE_INDEXES[tile.resource]
                production[ridx] += multiplier * _DICE_PROBS.get(tile.number, 0.0)
    features.extend(production)

    # 13. Turn progress (1) — normalized, helps the network understand game stage
    features.append(min(state.turn_number / 200.0, 1.0))

    return features


def feature_size(num_players: int = 4) -> int:
    """Return the size of the feature vector."""
    base = 5 + 3 + 5 + 2  # resources + buildings + dev cards + vp/knights = 15
    relative = 8  # relative features
    opp = (num_players - 1) * 5  # opponent info = 15
    vertex = 54 * 6  # 324
    edge = 72 * 3  # 216
    tile = 19 * 8  # 152
    phase = len(GamePhase)  # 10
    bank = 5
    production = 5
    turn = 1
    return base + relative + opp + vertex + edge + tile + phase + bank + production + turn
