from enum import Enum

# creating a cube coordinate system for hexagons as outlined below. x represents northeast to southwest,
# y represetns northwest to southeast, and z represents east to west. 
# https://math.stackexchange.com/questions/2254655/hexagon-grid-coordinate-system
class Direction(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"


UNIT_VECTORS = {
    # X-axis
    Direction.NORTHEAST: (1, 0, -1),
    Direction.SOUTHWEST: (-1, 0, 1),
    # Y-axis
    Direction.NORTHWEST: (0, 1, -1),
    Direction.SOUTHEAST: (0, -1, 1),
    # Z-axis
    Direction.EAST: (1, -1, 0),
    Direction.WEST: (-1, 1, 0),
}

#adding two coordinates together\
def add(acoord, bcoord):
    (x, y, z) = acoord
    (u, v, w) = bcoord
    return (x + u, y + v, z + w)

#layer 1 represents the center tile.
def total_tiles(layer):
    if layer == 0:
        return 1
    return 6 * layer + total_tiles(layer - 1)

# generate the coordinate system of tiles, given a number of layers (for a standard board it will be 3)
# use BFS so we populate each layer sequentially. 
def generate_coords(num_layers):
    n = total_tiles(num_layers)
    queue = [(0,0,0)]
    visited = set()
    while len(visited) < n:
        tile = queue.pop(0)
        visited.add(tile)
        for d in Direction:
            neighbor = add(tile, UNIT_VECTORS[d])
            print(d)
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
    print(visited)
    return visited