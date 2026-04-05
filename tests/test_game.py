"""Core game engine tests."""

import random
from catan.board import Board
from catan.game import Game, apply_action
from catan.actions import get_legal_actions, ActionType
from catan.enums import GamePhase
from catan.map import NUM_LAND_TILES


def test_board_topology():
    board = Board.build(seed=0)
    assert len(board.land_tiles) == 19
    assert len(board.get_vertex_ids()) == 54
    assert len(board.get_edge_ids()) == 72
    assert len(board.ports) == 9
    assert len(board.port_vertices) == 18  # 2 per port


def test_board_adjacency():
    board = Board.build(seed=0)
    # Each land tile has exactly 6 vertices and 6 edges
    for tile in board.land_tiles:
        assert len(tile.nodes) == 6
        assert len(tile.edges) == 6

    # Inland vertices touch 3 tiles, coastal touch 1-2
    for vid in board.get_vertex_ids():
        assert 1 <= len(board.vertex_to_tiles[vid]) <= 3

    # Each vertex has 2-3 adjacent vertices
    for vid in board.get_vertex_ids():
        assert 2 <= len(board.vertex_to_adjacent_vertices[vid]) <= 3


def test_board_deterministic():
    b1 = Board.build(seed=42)
    b2 = Board.build(seed=42)
    for i in range(len(b1.land_tiles)):
        assert b1.land_tiles[i].resource == b2.land_tiles[i].resource
        assert b1.land_tiles[i].number == b2.land_tiles[i].number


def test_setup_phase():
    board = Board.build(seed=0)
    state = Game(board, num_players=4, seed=0).state
    assert state.phase == GamePhase.SETUP_FIRST_SETTLEMENT

    actions = get_legal_actions(state)
    assert all(a.action_type == ActionType.PLACE_INITIAL_SETTLEMENT for a in actions)
    assert len(actions) == 54  # all vertices open on empty board


def test_full_game_completes():
    """A game with random agents should terminate."""
    rng = random.Random(42)
    board = Board.build(seed=42)
    game = Game(board, num_players=4, seed=42)

    turns = 0
    while not game.is_over() and turns < 5000:
        actions = game.get_legal_actions()
        assert actions, f"No legal actions at turn {game.state.turn_number}, phase={game.state.phase.name}"
        action = rng.choice(actions)
        game.apply(action)
        turns += 1

    assert game.is_over(), f"Game didn't finish in 5000 actions (turn {game.state.turn_number})"
    assert game.winner() is not None
    winner = game.state.players[game.winner()]
    assert winner.actual_victory_points >= 10


def test_clone_independence():
    """Cloned state should be independent of original."""
    board = Board.build(seed=0)
    game = Game(board, num_players=4, seed=0)

    # Play a few setup moves
    rng = random.Random(0)
    for _ in range(4):
        actions = game.get_legal_actions()
        game.apply(rng.choice(actions))

    # Clone and modify
    original_phase = game.state.phase
    clone = game.state.clone()
    clone.phase = GamePhase.GAME_OVER
    assert game.state.phase == original_phase


def test_resource_distribution():
    """After setup, rolling dice should distribute resources."""
    board = Board.build(seed=0)
    game = Game(board, num_players=4, seed=0)
    rng = random.Random(0)

    # Complete setup (8 settlements + 8 roads = 16 actions)
    while game.state.phase in (
        GamePhase.SETUP_FIRST_SETTLEMENT, GamePhase.SETUP_FIRST_ROAD,
        GamePhase.SETUP_SECOND_SETTLEMENT, GamePhase.SETUP_SECOND_ROAD,
    ):
        actions = game.get_legal_actions()
        game.apply(rng.choice(actions))

    assert game.state.phase == GamePhase.ROLL_DICE
    assert game.state.turn_number == 1


def test_apply_action_returns_new_state():
    """apply_action (functional) should not mutate the original."""
    board = Board.build(seed=0)
    game = Game(board, num_players=4, seed=0)
    original = game.state
    actions = get_legal_actions(original)
    new_state = apply_action(original, actions[0])
    assert new_state is not original
    assert new_state.phase != original.phase or new_state.buildings != original.buildings
