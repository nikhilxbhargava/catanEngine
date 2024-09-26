from typing import Iterable, List

from enums import (
    KNIGHT,
    MONOPOLY,
    ROAD_BUILDING,
    VICTORY_POINT,
    YEAR_OF_PLENTY,
    WOOD,
    BRICK,
    SHEEP,
    WHEAT,
    ORE,
    DevCards,
    Resources,
)

RESOURCE_INDEXES = {WOOD: 0, BRICK: 1, SHEEP: 2, WHEAT: 3, ORE: 4}

ROAD_COST = [1, 1, 0, 0, 0]
SETTLEMENT_COST = [1, 1, 1, 1, 0]
CITY_COST = [0, 0, 0, 2, 3]
DEVELOPMENT_CARD_COST = [0, 0, 1, 1, 1]


def starting_resource_bank():
    """Returns freqdeck of resource cards"""
    return [19, 19, 19, 19, 19]

def starting_devcard_bank():
    """Returns listdeck of devcards"""
    return (
        [KNIGHT] * 14
        + [YEAR_OF_PLENTY] * 2
        + [ROAD_BUILDING] * 2
        + [MONOPOLY] * 2
        + [VICTORY_POINT] * 5
    )

#counts the number of cards in the deck
def deck_count(deck, card: Resources):
    return deck[RESOURCE_INDEXES[card]]

#draws a card from the deck
def deck_draw(deck, amount, card: Resources):
    deck[RESOURCE_INDEXES[card]] -= amount

#adds a card in the deck
def freqdeck_replenish(deck, amount, card: Resources):
    deck[RESOURCE_INDEXES[card]] += amount

#returns true if the deck has enough cards to draw
def deck_can_draw(deck, amount, card: Resources):
    return deck[RESOURCE_INDEXES[card]] >= amount

#adds two decks together
def deck_add(list1, list2):
    return [a + b for a, b in zip(list1, list2)]

#subtracts two decks ie creating a building
def deck_subtract(list1, list2):
    return [a - b for a, b in zip(list1, list2)]

#checks if one deck is greater than or equal to another deck
def deck_contains(list1, list2):
    for count1, count2 in zip(list1, list2):
        if count1 < count2:
            return False
    return True