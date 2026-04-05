"""Player state representation — purely data, no decision logic.

Decision-making lives in agents/. This module only tracks what each
player owns and has done.
"""

from __future__ import annotations

from catan.decks import empty_freqdeck
from catan.enums import Color
from catan.map import EdgeId, VertexId


class PlayerState:
    __slots__ = (
        "color",
        "resources",
        "dev_cards",
        "dev_cards_bought_this_turn",
        "played_dev_cards",
        "settlements",
        "cities",
        "roads",
        "num_knights_played",
        "has_longest_road",
        "has_largest_army",
        "actual_victory_points",
    )

    def __init__(self, color: Color):
        self.color = color
        self.resources: list[int] = empty_freqdeck()
        self.dev_cards: list[str] = []
        self.dev_cards_bought_this_turn: list[str] = []
        self.played_dev_cards: list[str] = []
        self.settlements: set[VertexId] = set()
        self.cities: set[VertexId] = set()
        self.roads: set[EdgeId] = set()
        self.num_knights_played: int = 0
        self.has_longest_road: bool = False
        self.has_largest_army: bool = False
        self.actual_victory_points: int = 0

    @property
    def public_victory_points(self) -> int:
        """VP visible to all players (excludes hidden VP dev cards)."""
        vp = len(self.settlements) + 2 * len(self.cities)
        if self.has_longest_road:
            vp += 2
        if self.has_largest_army:
            vp += 2
        return vp

    def copy(self) -> PlayerState:
        p = PlayerState.__new__(PlayerState)
        p.color = self.color
        p.resources = self.resources.copy()
        p.dev_cards = self.dev_cards.copy()
        p.dev_cards_bought_this_turn = self.dev_cards_bought_this_turn.copy()
        p.played_dev_cards = self.played_dev_cards.copy()
        p.settlements = self.settlements.copy()
        p.cities = self.cities.copy()
        p.roads = self.roads.copy()
        p.num_knights_played = self.num_knights_played
        p.has_longest_road = self.has_longest_road
        p.has_largest_army = self.has_largest_army
        p.actual_victory_points = self.actual_victory_points
        return p

    def resource_count(self) -> int:
        return sum(self.resources)

    def __repr__(self) -> str:
        return f"PlayerState({self.color.name}, vp={self.actual_victory_points})"
