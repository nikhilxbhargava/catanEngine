from enum import Enum
from typing import List, Literal, Final

Resources = Literal["WOOD", "BRICK", "SHEEP", "WHEAT", "ORE"]
DevCards = Literal[
    "KNIGHT", "YEAR_OF_PLENTY", "MONOPOLY", "ROAD_BUILDING", "VICTORY_POINT"
]
BuildingTypes = Literal["SETTLEMENT", "CITY", "ROAD"]

# Resources
WOOD: Final = "WOOD"
BRICK: Final = "BRICK"
SHEEP: Final = "SHEEP"
WHEAT: Final = "WHEAT"
ORE: Final = "ORE"
RESOURCES: List[Resources] = [WOOD, BRICK, SHEEP, WHEAT, ORE]

KNIGHT: Final = "KNIGHT"
YEAR_OF_PLENTY: Final = "YEAR_OF_PLENTY"
MONOPOLY: Final = "MONOPOLY"
ROAD_BUILDING: Final = "ROAD_BUILDING"
VICTORY_POINT: Final = "VICTORY_POINT"
DEVELOPMENT_CARDS: List[DevCards] = [
    KNIGHT,
    YEAR_OF_PLENTY,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
]

SETTLEMENT: Final = "SETTLEMENT"
CITY: Final = "CITY"
ROAD: Final = "ROAD"
BUILDING_TYPES: List[BuildingTypes] = [SETTLEMENT, CITY, ROAD]

#       1
#      / \
#   6 /   \ 2
#    |     |
#   5 \   / 3
#      \ /
#       4

# References a vertex from a tile.
class Vertex(Enum):
    NORTH = "NORTH"
    NORTHEAST = "NORTHEAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTH = "SOUTH"
    SOUTHWEST = "SOUTHWEST"
    NORTHWEST = "NORTHWEST"


# References an edge from a tile.
class Edge(Enum):
    EAST = "EAST"
    SOUTHEAST = "SOUTHEAST"
    SOUTHWEST = "SOUTHWEST"
    WEST = "WEST"
    NORTHWEST = "NORTHWEST"
    NORTHEAST = "NORTHEAST"