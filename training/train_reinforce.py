"""Train a REINFORCE agent via parallel self-play against random opponents.

Architecture:
  - Main process owns the policy network and optimizer
  - Worker processes play games using a frozen copy of the weights
  - After each batch of games, workers send back EpisodeRecords
  - Main process does a single batched gradient update
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import torch

from agents.reinforce_agent import EpisodeRecord, ReinforceAgent, SimplePolicy
from agents.random_agent import RandomAgent
from catan.actions import get_legal_actions
from catan.board import Board
from catan.game import Game
from training.features import state_to_features, feature_size


# ── Worker function (runs in subprocess) ─────────────────────────────


def _play_one_game(args: tuple) -> EpisodeRecord:
    """Play a single game. Runs in a worker process.

    Args is a tuple of (weights_dict, game_seed, hidden_dim, player_index).
    Returns an EpisodeRecord with the trajectory.
    """
    weights, game_seed, hidden_dim, player_index = args

    # Reconstruct agent in worker (no GPU, CPU only for inference)
    agent = ReinforceAgent(hidden_dim=hidden_dim, device="cpu")
    agent.set_weights(weights)
    agent.training = True
    agent.start_episode()

    opponents = [RandomAgent(seed=game_seed + i + 1) for i in range(3)]

    board = Board.build(seed=game_seed)
    game = Game(board, num_players=4, seed=game_seed + 100)

    while not game.is_over() and game.state.turn_number < 3000:
        actions = game.get_legal_actions()
        if not actions:
            break
        pidx = game.state.current_player_index
        if pidx == player_index:
            action = agent.choose_action(game.state, actions)
        else:
            # Map opponent indices
            opp_idx = pidx - 1 if pidx > player_index else pidx
            action = opponents[opp_idx].choose_action(game.state, actions)
        game.apply(action)

    if game.winner() == player_index:
        reward = 1.0
    elif game.winner() is not None:
        reward = -1.0
    else:
        reward = 0.0

    return agent.finish_episode(reward)


def _eval_one_game(args: tuple) -> int:
    """Evaluate one game (greedy). Returns 1 if agent wins, 0 otherwise."""
    weights, game_seed, hidden_dim, player_index = args

    agent = ReinforceAgent(hidden_dim=hidden_dim, device="cpu")
    agent.set_weights(weights)
    agent.training = False

    opponents = [RandomAgent(seed=game_seed + i + 1) for i in range(3)]

    board = Board.build(seed=game_seed)
    game = Game(board, num_players=4, seed=game_seed + 100)

    while not game.is_over() and game.state.turn_number < 3000:
        actions = game.get_legal_actions()
        if not actions:
            break
        pidx = game.state.current_player_index
        if pidx == player_index:
            action = agent.choose_action(game.state, actions)
        else:
            opp_idx = pidx - 1 if pidx > player_index else pidx
            action = opponents[opp_idx].choose_action(game.state, actions)
        game.apply(action)

    return 1 if game.winner() == player_index else 0


# ── Main training loop ───────────────────────────────────────────────


def train(
    num_batches: int = 200,
    batch_size: int = 32,
    eval_every: int = 10,
    eval_games: int = 40,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    num_workers: int = 0,
    device: str = "cpu",
    save_path: str = "checkpoints/reinforce.pt",
    seed: int = 0,
):
    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    print(f"Training config:")
    print(f"  Batches: {num_batches} x {batch_size} games = {num_batches * batch_size} total games")
    print(f"  Workers: {num_workers}")
    print(f"  Device:  {device}")
    print(f"  LR:      {lr}")
    print(f"  Hidden:  {hidden_dim}")
    print()

    agent = ReinforceAgent(hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)
    best_win_rate = 0.0
    total_games = 0
    start = time.time()

    pool = Pool(processes=num_workers)

    try:
        for batch in range(1, num_batches + 1):
            batch_start = time.time()

            # Get current weights (CPU tensors for pickling to workers)
            weights = agent.get_weights()

            # Dispatch batch of games to workers
            # Rotate player_index so agent learns from all seats
            work_items = [
                (weights, seed + total_games + i, hidden_dim, i % 4)
                for i in range(batch_size)
            ]
            episodes: list[EpisodeRecord] = pool.map(_play_one_game, work_items)
            total_games += batch_size

            # Batched gradient update on main process
            loss = agent.update_from_episodes(episodes)

            batch_wins = sum(1 for ep in episodes if ep.reward > 0)
            batch_time = time.time() - batch_start
            games_per_sec = batch_size / batch_time

            # Logging
            elapsed = time.time() - start
            print(
                f"Batch {batch:4d}/{num_batches} | "
                f"Wins: {batch_wins:2d}/{batch_size} ({batch_wins/batch_size:4.0%}) | "
                f"Loss: {loss:7.4f} | "
                f"{games_per_sec:5.1f} games/s | "
                f"Elapsed: {elapsed:6.1f}s"
            )

            # Evaluate periodically
            if batch % eval_every == 0:
                eval_start = time.time()
                eval_items = [
                    (weights, seed + 999999 + i, hidden_dim, i % 4)
                    for i in range(eval_games)
                ]
                eval_results = pool.map(_eval_one_game, eval_items)
                win_rate = sum(eval_results) / eval_games
                eval_time = time.time() - eval_start

                marker = ""
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    agent.save(save_path)
                    marker = " ** NEW BEST **"

                print(
                    f"  ── Eval: {sum(eval_results)}/{eval_games} wins "
                    f"({win_rate:.0%}) | Best: {best_win_rate:.0%} | "
                    f"Eval time: {eval_time:.1f}s{marker}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted by user.")
    finally:
        pool.terminate()
        pool.join()

    print(f"\nTraining complete.")
    print(f"  Total games: {total_games}")
    print(f"  Best eval win rate: {best_win_rate:.0%}")
    print(f"  Total time: {time.time() - start:.1f}s")
    print(f"  Model saved to: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train REINFORCE Catan agent (parallel)")
    parser.add_argument("--batches", type=int, default=200, help="Number of training batches")
    parser.add_argument("--batch-size", type=int, default=32, help="Games per batch")
    parser.add_argument("--eval-every", type=int, default=10, help="Evaluate every N batches")
    parser.add_argument("--eval-games", type=int, default=40, help="Games per evaluation")
    parser.add_argument("--hidden-dim", type=int, default=256, help="Hidden layer size")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--gamma", type=float, default=0.99, help="Discount factor")
    parser.add_argument("--workers", type=int, default=0, help="Worker processes (0=auto)")
    parser.add_argument("--device", type=str, default="cpu", help="Device (cpu/mps)")
    parser.add_argument("--save-path", type=str, default="checkpoints/reinforce.pt")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    # Auto-detect MPS
    if args.device == "auto":
        if torch.backends.mps.is_available():
            args.device = "mps"
        else:
            args.device = "cpu"

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    train(
        num_batches=args.batches,
        batch_size=args.batch_size,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        hidden_dim=args.hidden_dim,
        lr=args.lr,
        gamma=args.gamma,
        num_workers=args.workers,
        device=args.device,
        save_path=args.save_path,
        seed=args.seed,
    )
