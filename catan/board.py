"""Board topology: builds the vertex/edge graph from a map template.

The board assigns unique integer IDs to each vertex and edge (as sorted
vertex-pair tuples), and precomputes all adjacency lookups needed by
the game engine.

For a standard Catan board: 54 vertices, 72 edges, 19 land tiles, 9 ports.
"""

from __future__ import annotations

import random as _random
from dataclasses import dataclass, field
from typing import Optional

from catan.coordinate_system import (
    Coordinate, Direction, UNIT_VECTORS, add as coord_add,
)
from catan.enums import Edge, Resources, Vertex
from catan.map import (
    BASE_MAP_TEMPLATE, EdgeId, LandTile, MapTemplate, Port, Water, VertexId,
)


# ── Vertex sharing rules ────────────────────────────────────────────
# When two hexes are neighbors in a given direction, they share 2 vertices.
# Key: direction from hex A to hex B
# Value: list of (vertex_on_A, vertex_on_B) pairs that are the same physical vertex.
VERTEX_SHARING: dict[Direction, list[tuple[Vertex, Vertex]]] = {
    Direction.EAST: [
        (Vertex.NORTHEAST, Vertex.NORTHWEST),
        (Vertex.SOUTHEAST, Vertex.SOUTHWEST),
    ],
    Direction.WEST: [
        (Vertex.NORTHWEST, Vertex.NORTHEAST),
        (Vertex.SOUTHWEST, Vertex.SOUTHEAST),
    ],
    Direction.NORTHEAST: [
        (Vertex.NORTH, Vertex.SOUTHWEST),
        (Vertex.NORTHEAST, Vertex.SOUTH),
    ],
    Direction.SOUTHWEST: [
        (Vertex.SOUTHWEST, Vertex.NORTH),
        (Vertex.SOUTH, Vertex.NORTHEAST),
    ],
    Direction.NORTHWEST: [
        (Vertex.NORTH, Vertex.SOUTHEAST),
        (Vertex.NORTHWEST, Vertex.SOUTH),
    ],
    Direction.SOUTHEAST: [
        (Vertex.SOUTHEAST, Vertex.NORTH),
        (Vertex.SOUTH, Vertex.NORTHWEST),
    ],
}

# Edge sharing: direction -> list of (edge_on_A, edge_on_B) that are the same.
EDGE_SHARING: dict[Direction, list[tuple[Edge, Edge]]] = {
    Direction.EAST: [(Edge.EAST, Edge.WEST)],
    Direction.WEST: [(Edge.WEST, Edge.EAST)],
    Direction.NORTHEAST: [(Edge.NORTHEAST, Edge.SOUTHWEST)],
    Direction.SOUTHWEST: [(Edge.SOUTHWEST, Edge.NORTHEAST)],
    Direction.NORTHWEST: [(Edge.NORTHWEST, Edge.SOUTHEAST)],
    Direction.SOUTHEAST: [(Edge.SOUTHEAST, Edge.NORTHWEST)],
}

# Each edge connects two vertices (within the same hex, by local position).
EDGE_VERTICES: dict[Edge, tuple[Vertex, Vertex]] = {
    Edge.EAST: (Vertex.NORTHEAST, Vertex.SOUTHEAST),
    Edge.NORTHEAST: (Vertex.NORTH, Vertex.NORTHEAST),
    Edge.NORTHWEST: (Vertex.NORTH, Vertex.NORTHWEST),
    Edge.WEST: (Vertex.NORTHWEST, Vertex.SOUTHWEST),
    Edge.SOUTHWEST: (Vertex.SOUTH, Vertex.SOUTHWEST),
    Edge.SOUTHEAST: (Vertex.SOUTH, Vertex.SOUTHEAST),
}


def _make_edge_id(v1: int, v2: int) -> EdgeId:
    return (min(v1, v2), max(v1, v2))


@dataclass
class Board:
    """Immutable board topology + tile/port assignment.

    Constructed once, then shared (not copied) across all GameState clones.
    """

    # Tiles
    land_tiles: list[LandTile] = field(default_factory=list)
    ports: list[Port] = field(default_factory=list)

    # Adjacency lookups (all keyed by integer IDs)
    vertex_to_adjacent_vertices: dict[VertexId, set[VertexId]] = field(default_factory=dict)
    vertex_to_adjacent_edges: dict[VertexId, set[EdgeId]] = field(default_factory=dict)
    vertex_to_tiles: dict[VertexId, list[int]] = field(default_factory=dict)  # tile index in land_tiles
    edge_to_vertices: dict[EdgeId, tuple[VertexId, VertexId]] = field(default_factory=dict)

    # Port access: vertex_id -> port resource (None means 3:1)
    port_vertices: dict[VertexId, Optional[Resources]] = field(default_factory=dict)

    # Dice number -> list of land tile indices
    tiles_by_number: dict[int, list[int]] = field(default_factory=dict)

    # Desert tile index (robber starts here)
    desert_tile_index: int = 0

    num_vertices: int = 0
    num_edges: int = 0

    # Cached sorted lists (computed once in build())
    _land_vertex_ids: list[VertexId] = field(default_factory=list)
    _land_edge_ids: list[EdgeId] = field(default_factory=list)

    @staticmethod
    def build(
        template: MapTemplate = BASE_MAP_TEMPLATE,
        seed: Optional[int] = None,
    ) -> Board:
        """Build a Board from a MapTemplate, shuffling resources and numbers."""
        rng = _random.Random(seed)

        board = Board()

        # Shuffle tile resources and numbers
        tile_resources = list(template.tile_resources)
        rng.shuffle(tile_resources)

        numbers = list(template.numbers)
        rng.shuffle(numbers)

        port_resources = list(template.port_resources)
        rng.shuffle(port_resources)

        # ── Pass 1: Identify land tiles and ports from topology ──
        # We iterate topology in a deterministic order (sorted coords).
        sorted_coords = sorted(
            template.topology.keys(),
            key=lambda c: (abs(c[0]) + abs(c[1]) + abs(c[2]), c),
        )

        # Separate land coords, port coords, water coords
        land_coords: list[Coordinate] = []
        port_entries: list[tuple[Coordinate, Direction]] = []
        water_coords: list[Coordinate] = []

        for coord in sorted_coords:
            entry = template.topology[coord]
            if entry is LandTile:
                land_coords.append(coord)
            elif entry is Water:
                water_coords.append(coord)
            else:
                # Port tuple: (Port, Direction)
                port_entries.append((coord, entry[1]))

        # ── Pass 2: Create LandTile objects ──
        number_idx = 0
        for i, coord in enumerate(land_coords):
            resource = tile_resources[i]
            if resource is None:
                # Desert
                num = 0
                board.desert_tile_index = i
            else:
                num = numbers[number_idx]
                number_idx += 1

            tile = LandTile(id=i, resource=resource, number=num, coordinate=coord)
            board.land_tiles.append(tile)

        # ── Pass 3: Assign vertex IDs with deduplication ──
        # coord_vertex_map: (coordinate, Vertex) -> global vertex ID
        coord_vertex_map: dict[tuple[Coordinate, Vertex], VertexId] = {}
        next_vertex_id = 0

        # All hex coords (land + port + water) participate in vertex sharing
        all_hex_coords = set(template.topology.keys())

        for coord in sorted_coords:
            for v in Vertex:
                if (coord, v) in coord_vertex_map:
                    continue
                # Assign new ID
                vid = next_vertex_id
                next_vertex_id += 1
                coord_vertex_map[(coord, v)] = vid

                # Propagate to neighbors that share this vertex
                _propagate_vertex(coord, v, vid, coord_vertex_map, all_hex_coords)

        board.num_vertices = next_vertex_id

        # ── Pass 4: Assign edge IDs with deduplication ──
        coord_edge_map: dict[tuple[Coordinate, Edge], EdgeId] = {}

        for coord in sorted_coords:
            for e in Edge:
                if (coord, e) in coord_edge_map:
                    continue
                v1_local, v2_local = EDGE_VERTICES[e]
                v1 = coord_vertex_map[(coord, v1_local)]
                v2 = coord_vertex_map[(coord, v2_local)]
                eid = _make_edge_id(v1, v2)
                coord_edge_map[(coord, e)] = eid

                # Propagate to neighbors sharing this edge
                for direction, sharing_pairs in EDGE_SHARING.items():
                    neighbor = coord_add(coord, UNIT_VECTORS[direction])
                    if neighbor not in all_hex_coords:
                        continue
                    for local_e, neighbor_e in sharing_pairs:
                        if local_e == e:
                            coord_edge_map[(neighbor, neighbor_e)] = eid

        # ── Pass 5: Populate tile vertex/edge dicts ──
        for tile in board.land_tiles:
            for v in Vertex:
                tile.nodes[v] = coord_vertex_map[(tile.coordinate, v)]
            for e in Edge:
                tile.edges[e] = coord_edge_map[(tile.coordinate, e)]

        # ── Pass 6: Build adjacency structures ──
        all_edges: set[EdgeId] = set()
        for tile in board.land_tiles:
            for e in Edge:
                eid = tile.edges[e]
                all_edges.add(eid)
                v1_local, v2_local = EDGE_VERTICES[e]
                v1 = tile.nodes[v1_local]
                v2 = tile.nodes[v2_local]
                board.edge_to_vertices[eid] = (v1, v2)

        board.num_edges = len(all_edges)

        # vertex -> adjacent vertices, edges, tiles
        for vid in range(board.num_vertices):
            board.vertex_to_adjacent_vertices[vid] = set()
            board.vertex_to_adjacent_edges[vid] = set()
            board.vertex_to_tiles[vid] = []

        for eid, (v1, v2) in board.edge_to_vertices.items():
            board.vertex_to_adjacent_vertices[v1].add(v2)
            board.vertex_to_adjacent_vertices[v2].add(v1)
            board.vertex_to_adjacent_edges[v1].add(eid)
            board.vertex_to_adjacent_edges[v2].add(eid)

        for idx, tile in enumerate(board.land_tiles):
            for v in Vertex:
                vid = tile.nodes[v]
                if idx not in board.vertex_to_tiles[vid]:
                    board.vertex_to_tiles[vid].append(idx)

        # ── Pass 7: Build tiles_by_number ──
        for idx, tile in enumerate(board.land_tiles):
            if tile.number > 0:
                board.tiles_by_number.setdefault(tile.number, []).append(idx)

        # ── Pass 8: Create Port objects and map port vertices ──
        for i, (coord, direction) in enumerate(port_entries):
            resource = port_resources[i]
            port = Port(id=i, resource=resource, direction=direction)

            # Populate port nodes/edges from coord_vertex_map
            for v in Vertex:
                key = (coord, v)
                if key in coord_vertex_map:
                    port.nodes[v] = coord_vertex_map[key]
            for e in Edge:
                key = (coord, e)
                if key in coord_edge_map:
                    port.edges[e] = coord_edge_map[key]

            # The port faces a direction: the two vertices on that facing edge
            # are the ones that grant port access to settlements.
            facing_edge = _direction_to_edge(direction)
            if facing_edge in port.edges:
                v1_local, v2_local = EDGE_VERTICES[facing_edge]
                for vl in (v1_local, v2_local):
                    if vl in port.nodes:
                        board.port_vertices[port.nodes[vl]] = resource

            board.ports.append(port)

        # Cache sorted vertex/edge ID lists (called many times per game)
        vids: set[VertexId] = set()
        eids: set[EdgeId] = set()
        for tile in board.land_tiles:
            vids.update(tile.nodes.values())
            eids.update(tile.edges.values())
        board._land_vertex_ids = sorted(vids)
        board._land_edge_ids = sorted(eids)

        return board

    def get_vertex_ids(self) -> list[VertexId]:
        """All vertex IDs that border at least one land tile (cached)."""
        return self._land_vertex_ids

    def get_edge_ids(self) -> list[EdgeId]:
        """All edge IDs that border at least one land tile (cached)."""
        return self._land_edge_ids


def _propagate_vertex(
    coord: Coordinate,
    v: Vertex,
    vid: VertexId,
    coord_vertex_map: dict[tuple[Coordinate, Vertex], VertexId],
    all_coords: set[Coordinate],
) -> None:
    """Propagate a vertex ID to all neighboring hexes that share it."""
    for direction, sharing_pairs in VERTEX_SHARING.items():
        neighbor = coord_add(coord, UNIT_VECTORS[direction])
        if neighbor not in all_coords:
            continue
        for local_v, neighbor_v in sharing_pairs:
            if local_v == v:
                key = (neighbor, neighbor_v)
                if key not in coord_vertex_map:
                    coord_vertex_map[key] = vid
                    # Recurse: this neighbor's vertex may also be shared further
                    _propagate_vertex(neighbor, neighbor_v, vid, coord_vertex_map, all_coords)


def _direction_to_edge(direction: Direction) -> Edge:
    """Map a port's facing direction to the edge it connects through."""
    return {
        Direction.EAST: Edge.EAST,
        Direction.WEST: Edge.WEST,
        Direction.NORTHEAST: Edge.NORTHEAST,
        Direction.NORTHWEST: Edge.NORTHWEST,
        Direction.SOUTHEAST: Edge.SOUTHEAST,
        Direction.SOUTHWEST: Edge.SOUTHWEST,
    }[direction]
