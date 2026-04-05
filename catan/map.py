from typing import Dict, List, Mapping, Tuple, Type, Union
from dataclasses import dataclass, field
from catan.coordinate_system import Direction, Coordinate
from catan.enums import (
    Resources, WOOD, BRICK, SHEEP, WHEAT, ORE, Edge, Vertex,
)

NUM_VERTICES = 54
NUM_EDGES = 72
NUM_LAND_TILES = 19

EdgeId = Tuple[int, int]  # sorted pair of vertex ids
VertexId = int


@dataclass
class Port:
    id: int
    resource: Union[Resources, None]  # None = 3:1 port
    direction: Direction
    nodes: Dict[Vertex, VertexId] = field(default_factory=dict)
    edges: Dict[Edge, EdgeId] = field(default_factory=dict)

    def __hash__(self):
        return self.id


@dataclass
class LandTile:
    id: int
    resource: Union[Resources, None]  # None = Desert
    number: int  # dice number (0 for desert)
    coordinate: Coordinate = (0, 0, 0)
    nodes: Dict[Vertex, VertexId] = field(default_factory=dict)
    edges: Dict[Edge, EdgeId] = field(default_factory=dict)

    def __hash__(self):
        return self.id


@dataclass(frozen=True)
class Water:
    coordinate: Coordinate = (0, 0, 0)


Tile = Union[LandTile, Port, Water]


@dataclass(frozen=True)
class MapTemplate:
    numbers: List[int]
    port_resources: List[Union[Resources, None]]
    tile_resources: List[Union[Resources, None]]
    topology: Mapping[
        Coordinate, Union[Type[LandTile], Type[Water], Tuple[Type[Port], Direction]]
    ]


BASE_MAP_TEMPLATE = MapTemplate(
    [2, 3, 3, 4, 4, 5, 5, 6, 6, 8, 8, 9, 9, 10, 10, 11, 11, 12],
    [
        WOOD, BRICK, SHEEP, WHEAT, ORE,
        None, None, None, None,
    ],
    [
        WOOD, WOOD, WOOD, WOOD,
        BRICK, BRICK, BRICK,
        SHEEP, SHEEP, SHEEP, SHEEP,
        WHEAT, WHEAT, WHEAT, WHEAT,
        ORE, ORE, ORE,
        None,  # desert
    ],
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
