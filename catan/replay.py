"""Replay serialization: record games as JSON for the visualization frontend.

Converts action logs + board state into a sequence of frames that can be
stepped through in a web UI. Each frame captures the full visible game
state after an action is applied.
"""

from __future__ import annotations

import json
import os
import time
from typing import Optional

from catan.actions import Action, ActionType
from catan.board import Board
from catan.enums import RESOURCES, Vertex, Edge, GamePhase, Color
from catan.game import Game, apply_action_mutate
from catan.map import LandTile
from catan.state import GameState


def _serialize_action(action: Action) -> dict:
    """Convert an Action to a JSON-safe dict."""
    d = {"type": action.action_type.name}
    if action.vertex is not None:
        d["vertex"] = action.vertex
    if action.edge is not None:
        d["edge"] = list(action.edge)  # tuple -> list for JSON
    if action.tile is not None:
        d["tile"] = action.tile
    if action.steal_from is not None:
        d["steal_from"] = action.steal_from
    if action.resource1 is not None:
        d["resource1"] = action.resource1
    if action.resource2 is not None:
        d["resource2"] = action.resource2
    if action.give_resource is not None:
        d["give_resource"] = action.give_resource
    if action.get_resource is not None:
        d["get_resource"] = action.get_resource
    if action.discard_resources is not None:
        d["discard_resources"] = list(action.discard_resources)
    return d


def _serialize_board(board: Board) -> dict:
    """Serialize the static board layout (tiles, ports, vertices, edges)."""
    tiles = []
    for tile in board.land_tiles:
        t = {
            "id": tile.id,
            "resource": tile.resource,
            "number": tile.number,
            "coordinate": list(tile.coordinate),
            "vertices": {v.name: vid for v, vid in tile.nodes.items()},
            "edges": {e.name: list(eid) for e, eid in tile.edges.items()},
        }
        tiles.append(t)

    ports = []
    for port in board.ports:
        p = {
            "id": port.id,
            "resource": port.resource,
            "direction": port.direction.name,
            "vertices": {v.name: vid for v, vid in port.nodes.items()},
        }
        ports.append(p)

    # Vertex positions: compute pixel coords for each land vertex
    vertex_positions = _compute_vertex_positions(board)

    # Edge to vertices mapping
    edge_map = {}
    for eid, (v1, v2) in board.edge_to_vertices.items():
        edge_map[f"{eid[0]},{eid[1]}"] = [v1, v2]

    # Port vertex mapping
    port_verts = {}
    for vid, resource in board.port_vertices.items():
        port_verts[str(vid)] = resource

    return {
        "tiles": tiles,
        "ports": ports,
        "vertex_positions": vertex_positions,
        "edge_to_vertices": edge_map,
        "port_vertices": port_verts,
        "desert_tile": board.desert_tile_index,
    }


def _compute_vertex_positions(board: Board) -> dict[str, list[float]]:
    """Compute x,y pixel positions for each vertex using cube coordinates.

    The engine uses pointy-top hexagons with cube coords where:
      EAST      = ( 1, -1,  0)
      NORTHEAST = ( 1,  0, -1)
      NORTHWEST = ( 0,  1, -1)

    For pointy-top hexes (vertex at top), the cube-to-pixel mapping is:
      x = size * (3/2 * q)
      y = size * (sqrt(3)/2 * q + sqrt(3) * r)
    (This is actually the flat-top formula applied to a rotated coord system.)

    Vertex offsets from hex center for pointy-top:
      NORTH     = ( 0,          -size)
      NORTHEAST = ( size * 3/4,  -size/2)  -- not quite, need to derive from sharing rules
    """
    import math
    size = 60  # hex radius (center to vertex)
    sqrt3 = math.sqrt(3)

    # Derive hex-center formula from the constraint that shared vertices
    # must coincide. The EAST neighbor at (1,-1,0) shares vertices
    # (NE↔NW, SE↔SW). For pointy-top, NE and SE are on the right edge.
    #
    # The center-to-center distance between adjacent hexes in pointy-top
    # is sqrt(3)*size. Going EAST should shift purely in x by sqrt(3)*size.
    #
    # Using x = A*q + B*r, y = C*q + D*r:
    #   EAST (1,-1,0):      A-B = sqrt3*size,  C-D = 0
    #   NORTHEAST (1,0,-1): pure NE at 60° → dx=sqrt3/2*size*...
    #
    # Solving from direction vectors:
    #   EAST      (1,-1, 0) → (sqrt3*size, 0)
    #   NORTHEAST (1, 0,-1) → (sqrt3/2*size, -3/2*size)  [pointy-top NE]
    #
    #   From EAST: A - B = sqrt3*size, C - D = 0
    #   From NE:   A = sqrt3/2*size, C = -3/2*size
    #   → B = sqrt3/2*size - sqrt3*size = -sqrt3/2*size
    #   → D = C = -3/2*size
    #
    # So: x = sqrt3/2*size * q - sqrt3/2*size * r = sqrt3/2*size*(q-r)
    #     y = -3/2*size * q - 3/2*size * r = -3/2*size*(q+r)
    #
    # Since q+r+s=0 → q+r = -s, so y = -3/2*size*(-s) = 3/2*size*s
    # And q-r = q-r, but also q-r = 2q+s (since r = -q-s)
    #   x = sqrt3/2*size*(2q+s) = sqrt3*size*q + sqrt3/2*size*s

    positions: dict[str, list[float]] = {}

    # Vertex offsets for pointy-top hex (relative to hex center)
    # Verified by checking that EAST neighbor's NW = our NE, etc.
    vertex_offsets = {
        Vertex.NORTH:     (0, -size),
        Vertex.NORTHEAST: (sqrt3 / 2 * size, -size / 2),
        Vertex.SOUTHEAST: (sqrt3 / 2 * size, size / 2),
        Vertex.SOUTH:     (0, size),
        Vertex.SOUTHWEST: (-sqrt3 / 2 * size, size / 2),
        Vertex.NORTHWEST: (-sqrt3 / 2 * size, -size / 2),
    }

    for tile in board.land_tiles:
        q, r, s = tile.coordinate
        # Cube to pixel — derived from direction vector constraints
        cx = sqrt3 / 2 * size * (q - r)
        cy = -3.0 / 2.0 * size * (q + r)

        for v, vid in tile.nodes.items():
            key = str(vid)
            dx, dy = vertex_offsets[v]
            positions[key] = [round(cx + dx, 2), round(cy + dy, 2)]

    return positions


def _serialize_frame(state: GameState, action: Optional[Action], action_idx: int) -> dict:
    """Capture the full visible state as a frame."""
    players = []
    for i, p in enumerate(state.players):
        players.append({
            "color": p.color.name,
            "resources": list(p.resources),
            "resource_count": p.resource_count(),
            "dev_card_count": len(p.dev_cards),
            "settlements": list(p.settlements),
            "cities": list(p.cities),
            "roads": [list(e) for e in p.roads],
            "vp": p.actual_victory_points,
            "public_vp": p.public_victory_points,
            "num_knights": p.num_knights_played,
            "has_longest_road": p.has_longest_road,
            "has_largest_army": p.has_largest_army,
        })

    buildings = {}
    for vid, (owner, btype) in state.buildings.items():
        buildings[str(vid)] = {"owner": owner, "type": btype}

    roads = {}
    for eid, owner in state.roads.items():
        roads[f"{eid[0]},{eid[1]}"] = owner

    frame = {
        "action_idx": action_idx,
        "phase": state.phase.name,
        "turn": state.turn_number,
        "current_player": state.current_player_index,
        "dice": list(state.dice_result) if state.dice_result else None,
        "robber_tile": state.robber_tile,
        "players": players,
        "buildings": buildings,
        "roads": roads,
        "winner": state.winner,
    }

    if action is not None:
        frame["action"] = _serialize_action(action)

    return frame


def record_game(board: Board, action_log: list[Action], seed: Optional[int] = None,
                num_players: int = 4, metadata: Optional[dict] = None) -> dict:
    """Replay a game from its action log and produce a full replay dict.

    Returns a dict with:
      - board: static board layout
      - frames: list of state snapshots (one per action + initial)
      - metadata: game info (winner, num_turns, etc.)
    """
    state = GameState(board, num_players=num_players, seed=seed)
    rng = None
    if seed is not None:
        import random as _random
        rng = _random.Random(seed)

    frames = [_serialize_frame(state, None, -1)]

    for i, action in enumerate(action_log):
        apply_action_mutate(state, action, rng=rng)
        frames.append(_serialize_frame(state, action, i))

    replay = {
        "board": _serialize_board(board),
        "frames": frames,
        "metadata": {
            "num_players": num_players,
            "seed": seed,
            "num_actions": len(action_log),
            "num_turns": state.turn_number,
            "winner": state.winner,
            "timestamp": time.time(),
            **(metadata or {}),
        },
    }
    return replay


def save_replay(replay: dict, path: str) -> None:
    """Save a replay dict to a JSON file."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "w") as f:
        json.dump(replay, f)


def load_replay(path: str) -> dict:
    """Load a replay dict from a JSON file."""
    with open(path) as f:
        return json.load(f)


def record_and_save_game(game: Game, path: str, metadata: Optional[dict] = None) -> dict:
    """Convenience: record a Game object and save to disk."""
    replay = record_game(
        board=game.state.board,
        action_log=game.action_log,
        seed=None,  # RNG state is already consumed; we just replay actions
        num_players=game.state.num_players,
        metadata=metadata,
    )
    save_replay(replay, path)
    return replay
