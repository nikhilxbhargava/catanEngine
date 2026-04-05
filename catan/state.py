"""Game state — complete snapshot of a Catan game at any point in time.

The Board is shared (immutable topology), everything else is copied
on clone() for tree search.
"""

from __future__ import annotations

from typing import Optional

from catan.board import Board
from catan.decks import starting_devcard_bank, starting_resource_bank
from catan.enums import Color, GamePhase
from catan.map import EdgeId, VertexId
from catan.player import PlayerState


# Setup turn order: 0,1,2,3,3,2,1,0 for 4 players
def _setup_order(num_players: int) -> list[int]:
    return list(range(num_players)) + list(range(num_players - 1, -1, -1))


class GameState:
    __slots__ = (
        "board",
        "players",
        "num_players",
        "current_player_index",
        "resource_bank",
        "dev_card_deck",
        "robber_tile",
        "phase",
        "turn_number",
        "buildings",
        "roads",
        "longest_road_player",
        "longest_road_length",
        "largest_army_player",
        "largest_army_size",
        "dice_result",
        "dev_card_played_this_turn",
        "free_roads_remaining",
        "setup_order",
        "setup_step",
        "discard_players",
        "winner",
    )

    def __init__(
        self,
        board: Board,
        num_players: int = 4,
        seed: Optional[int] = None,
    ):
        import random as _random

        self.board = board
        self.num_players = num_players
        self.players = [PlayerState(Color(i)) for i in range(num_players)]
        self.current_player_index: int = 0
        self.resource_bank: list[int] = starting_resource_bank()
        self.dev_card_deck: list[str] = starting_devcard_bank()

        # Shuffle dev card deck
        rng = _random.Random(seed)
        rng.shuffle(self.dev_card_deck)

        self.robber_tile: int = board.desert_tile_index
        self.phase: GamePhase = GamePhase.SETUP_FIRST_SETTLEMENT
        self.turn_number: int = 0

        # Building placement tracking (global, not per-player)
        self.buildings: dict[VertexId, tuple[int, str]] = {}  # vid -> (player_idx, "SETTLEMENT"/"CITY")
        self.roads: dict[EdgeId, int] = {}  # eid -> player_idx

        # Longest road / largest army
        self.longest_road_player: int = -1
        self.longest_road_length: int = 0
        self.largest_army_player: int = -1
        self.largest_army_size: int = 0

        # Turn state
        self.dice_result: Optional[tuple[int, int]] = None
        self.dev_card_played_this_turn: bool = False
        self.free_roads_remaining: int = 0

        # Setup phase tracking
        self.setup_order: list[int] = _setup_order(num_players)
        self.setup_step: int = 0

        # Discard tracking (for robber 7 rolls)
        self.discard_players: list[int] = []  # player indices that still need to discard

        self.winner: Optional[int] = None

    def current_player(self) -> PlayerState:
        return self.players[self.current_player_index]

    def clone(self) -> GameState:
        """Deep copy for tree search. Board is shared (immutable)."""
        s = GameState.__new__(GameState)
        s.board = self.board  # shared reference
        s.num_players = self.num_players
        s.players = [p.copy() for p in self.players]
        s.current_player_index = self.current_player_index
        s.resource_bank = self.resource_bank.copy()
        s.dev_card_deck = self.dev_card_deck.copy()
        s.robber_tile = self.robber_tile
        s.phase = self.phase
        s.turn_number = self.turn_number
        s.buildings = self.buildings.copy()
        s.roads = self.roads.copy()
        s.longest_road_player = self.longest_road_player
        s.longest_road_length = self.longest_road_length
        s.largest_army_player = self.largest_army_player
        s.largest_army_size = self.largest_army_size
        s.dice_result = self.dice_result
        s.dev_card_played_this_turn = self.dev_card_played_this_turn
        s.free_roads_remaining = self.free_roads_remaining
        s.setup_order = self.setup_order  # immutable after init
        s.setup_step = self.setup_step
        s.discard_players = self.discard_players.copy()
        s.winner = self.winner
        return s

    def __repr__(self) -> str:
        return (
            f"GameState(phase={self.phase.name}, "
            f"turn={self.turn_number}, "
            f"player={self.current_player_index})"
        )
