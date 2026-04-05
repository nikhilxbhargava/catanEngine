from catan.enums import (
    KNIGHT, MONOPOLY, ROAD_BUILDING, VICTORY_POINT, YEAR_OF_PLENTY,
    WOOD, BRICK, SHEEP, WHEAT, ORE,
    DevCards, Resources,
)

RESOURCE_INDEXES = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}

ROAD_COST = [1, 1, 0, 0, 0]
SETTLEMENT_COST = [1, 1, 1, 1, 0]
CITY_COST = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST = [0, 0, 1, 1, 1]


def starting_resource_bank() -> list[int]:
    return [19, 19, 19, 19, 19]


def starting_devcard_bank() -> list[str]:
    return (
        [KNIGHT] * 14
        + [YEAR_OF_PLENTY] * 2
        + [ROAD_BUILDING] * 2
        + [MONOPOLY] * 2
        + [VICTORY_POINT] * 5
    )


def deck_count(deck: list[int], card: Resources) -> int:
    return deck[RESOURCE_INDEXES[card]]


def deck_draw(deck: list[int], amount: int, card: Resources) -> None:
    deck[RESOURCE_INDEXES[card]] -= amount


def freqdeck_replenish(deck: list[int], amount: int, card: Resources) -> None:
    deck[RESOURCE_INDEXES[card]] += amount


def deck_can_draw(deck: list[int], amount: int, card: Resources) -> bool:
    return deck[RESOURCE_INDEXES[card]] >= amount


def deck_add(a: list[int], b: list[int]) -> list[int]:
    return [x + y for x, y in zip(a, b)]


def deck_subtract(a: list[int], b: list[int]) -> list[int]:
    return [x - y for x, y in zip(a, b)]


def deck_contains(a: list[int], b: list[int]) -> bool:
    return all(x >= y for x, y in zip(a, b))


def deck_total(deck: list[int]) -> int:
    return sum(deck)


def empty_freqdeck() -> list[int]:
    return [0, 0, 0, 0, 0]
