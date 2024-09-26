from typing import Dict, List, Mapping, Set, Tuple, Type, Union
from dataclasses import dataclass
from coordinate_system import Direction, add, UNIT_VECTORS
from enums import (
    Resources,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    Edge,
    Vertex,
)

# for a standard board
NUM_VERTEX = 54
NUM_EDGES = 72
NUM_TILES = 19

EdgeId = Tuple[int, int]
VertexId = int
Coordinate = Tuple[int, int, int]

@dataclass
class Port:
    id: int
    resource: Union[Resources, None]  # None = 3:1 port
    direction: Direction
    nodes: Dict[Vertex, VertexId]
    edges: Dict[Edge, EdgeId]
    
    def __hash__(self):
        return self.id

@dataclass
class LandTile:
    id: int
    resource: Union[Resources, None] # None = Desert
    vertexes: Dict[Vertex, VertexId] 
    edges: Dict[Edge, EdgeId]

    def __hash__(self):
        return self.id
    
@dataclass(frozen=True)
class Water:
    nodes: Dict[Vertex, int]
    edges: Dict[Edge, EdgeId]


Tile = Union[LandTile, Port, Water]

@dataclass(frozen=True)
class MapTemplate:
    numbers: List[int]
    port_resources: List[Union[Resources, None]]
    tile_resources: List[Union[Resources, None]]
    topology: Mapping[
        Coordinate, Union[Type[LandTile], Type[Water], Tuple[Type[Port], Direction]]
    ]


#hard code standard map
BASE_MAP_TEMPLATE = MapTemplate(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    [
        # These are 2:1 ports
        WOOD,
        BRICK,
        SHEEP,
        WHEAT,
        ORE,
        # These represent 3:1 ports
        None,
        None,
        None,
        None,
    ],
    [
        # Four wood tiles
        WOOD,
        WOOD,
        WOOD,
        WOOD,
        # Three brick tiles
        BRICK,
        BRICK,
        BRICK,
        # Four sheep tiles
        SHEEP,
        SHEEP,
        SHEEP,
        SHEEP,
        # Four wheat tiles
        WHEAT,
        WHEAT,
        WHEAT,
        WHEAT,
        # Three ore tiles
        ORE,
        ORE,
        ORE,
        # One desert
        None,
    ],
    # 3 layers, where last layer is water
    {
        # center
        (0, 0, 0): LandTile,
        # first layer
        (1, -1, 0): LandTile,
        (0, -1, 1): LandTile,
        (-1, 0, 1): LandTile,
        (-1, 1, 0): LandTile,
        (0, 1, -1): LandTile,
        (1, 0, -1): LandTile,
        # second layer
        (2, -2, 0): LandTile,
        (1, -2, 1): LandTile,
        (0, -2, 2): LandTile,
        (-1, -1, 2): LandTile,
        (-2, 0, 2): LandTile,
        (-2, 1, 1): LandTile,
        (-2, 2, 0): LandTile,
        (-1, 2, -1): LandTile,
        (0, 2, -2): LandTile,
        (1, 1, -2): LandTile,
        (2, 0, -2): LandTile,
        (2, -1, -1): LandTile,
        # third (water) layer
        (3, -3, 0): (Port, Direction.WEST),
        (2, -3, 1): Water,
        (1, -3, 2): (Port, Direction.NORTHWEST),
        (0, -3, 3): Water,
        (-1, -2, 3): (Port, Direction.NORTHWEST),
        (-2, -1, 3): Water,
        (-3, 0, 3): (Port, Direction.NORTHEAST),
        (-3, 1, 2): Water,
        (-3, 2, 1): (Port, Direction.EAST),
        (-3, 3, 0): Water,
        (-2, 3, -1): (Port, Direction.EAST),
        (-1, 3, -2): Water,
        (0, 3, -3): (Port, Direction.SOUTHEAST),
        (1, 2, -3): Water,
        (2, 1, -3): (Port, Direction.SOUTHWEST),
        (3, 0, -3): Water,
        (3, -1, -2): (Port, Direction.SOUTHWEST),
        (3, -2, -1): Water,
    },
)