from enum import Enum
from typing import Tuple

Coordinate = Tuple[int, int, int]


class Direction(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


UNIT_VECTORS = {
    Direction.NORTHEAST: (1, 0, -1),
    Direction.SOUTHWEST: (-1, 0, 1),
    Direction.NORTHWEST: (0, 1, -1),
    Direction.SOUTHEAST: (0, -1, 1),
    Direction.EAST: (1, -1, 0),
    Direction.WEST: (-1, 1, 0),
}


def add(a: Coordinate, b: Coordinate) -> Coordinate:
    return (a[0] + b[0], a[1] + b[1], a[2] + b[2])


def total_tiles(layer: int) -> int:
    if layer == 0:
        return 1
    return 6 * layer + total_tiles(layer - 1)


def generate_coords(num_layers: int) -> list[Coordinate]:
    """Generate tile coordinates using BFS, layer by layer.
    Returns a list in BFS order (deterministic)."""
    n = total_tiles(num_layers)
    queue = [(0, 0, 0)]
    visited_set: set[Coordinate] = set()
    visited_order: list[Coordinate] = []
    while len(visited_order) < n:
        tile = queue.pop(0)
        if tile in visited_set:
            continue
        visited_set.add(tile)
        visited_order.append(tile)
        for d in Direction:
            neighbor = add(tile, UNIT_VECTORS[d])
            if neighbor not in visited_set:
                queue.append(neighbor)
    return visited_order
