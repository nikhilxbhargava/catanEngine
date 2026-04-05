"""Train a Catan agent via parallel self-play against random opponents.

Supports both REINFORCE and A2C (Advantage Actor-Critic) algorithms.
Supports --resume to continue from a saved checkpoint.

Architecture:
  - Main process owns the network and optimizer
  - Worker processes play games using a frozen copy of the weights
  - After each batch of games, workers send back EpisodeRecords
  - Main process does a single batched gradient update

Usage:
  python -m training.train --algo a2c --batches 500 --batch-size 32
  python -m training.train --algo a2c --batches 1000 --resume  # continue from checkpoint
"""

from __future__ import annotations

import argparse
import os
import time
from multiprocessing import Pool, cpu_count
from typing import Optional

import torch

from agents.random_agent import RandomAgent
from catan.board import Board
from catan.game import Game


# ── Worker functions (run in subprocesses) ───────────────────────────


def _play_one_game(args: tuple):
    """Play a single game in a worker process. Returns an EpisodeRecord."""
    weights, game_seed, hidden_dim, player_index, algo = args

    if algo == "a2c":
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(hidden_dim=hidden_dim, device="cpu")
    else:
        from agents.reinforce_agent import ReinforceAgent
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
            opp_idx = pidx - 1 if pidx > player_index else pidx
            action = opponents[opp_idx].choose_action(game.state, actions)
        game.apply(action)

    reward = 1.0 if game.winner() == player_index else (-1.0 if game.winner() is not None else 0.0)
    return agent.finish_episode(reward)


def _eval_one_game(args: tuple) -> int:
    """Evaluate one game (greedy). Returns 1 if agent wins, 0 otherwise."""
    weights, game_seed, hidden_dim, player_index, algo = args

    if algo == "a2c":
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(hidden_dim=hidden_dim, device="cpu")
    else:
        from agents.reinforce_agent import ReinforceAgent
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
    algo: str = "a2c",
    num_batches: int = 200,
    batch_size: int = 32,
    eval_every: int = 10,
    eval_games: int = 40,
    hidden_dim: int = 256,
    lr: float = 3e-4,
    gamma: float = 0.99,
    num_workers: int = 0,
    device: str = "cpu",
    save_path: str = "checkpoints/a2c.pt",
    seed: int = 0,
    resume: bool = False,
):
    if num_workers <= 0:
        num_workers = max(1, cpu_count() - 1)

    if algo == "a2c":
        from agents.a2c_agent import A2CAgent
        agent = A2CAgent(hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)
    else:
        from agents.reinforce_agent import ReinforceAgent
        agent = ReinforceAgent(hidden_dim=hidden_dim, lr=lr, gamma=gamma, device=device)

    # Resume from checkpoint
    start_batch = 1
    best_win_rate = 0.0
    total_games = 0

    if resume and os.path.exists(save_path):
        print(f"Resuming from {save_path}...")
        metadata = agent.load(save_path)
        if metadata:
            start_batch = metadata.get("batch", 0) + 1
            best_win_rate = metadata.get("best_win_rate", 0.0)
            total_games = metadata.get("total_games", 0)
            print(f"  Resumed at batch {start_batch}, best_win_rate={best_win_rate:.0%}, total_games={total_games}")
        else:
            print(f"  Loaded weights (no metadata — starting from batch 1)")

    print(f"\nTraining config:")
    print(f"  Algorithm: {algo.upper()}")
    print(f"  Batches:   {start_batch}..{num_batches} x {batch_size} games")
    print(f"  Workers:   {num_workers}")
    print(f"  Device:    {device}")
    print(f"  LR:        {lr}")
    print(f"  Hidden:    {hidden_dim}")
    if algo == "a2c":
        print(f"  Value coeff:   {agent.value_coeff}")
        print(f"  Entropy coeff: {agent.entropy_coeff}")
    print()

    start = time.time()

    pool = Pool(processes=num_workers)

    try:
        for batch in range(start_batch, num_batches + 1):
            batch_start = time.time()

            # LR scheduling (cosine decay)
            if algo == "a2c" and hasattr(agent, "adjust_lr"):
                agent.adjust_lr(batch / num_batches)

            weights = agent.get_weights()

            work_items = [
                (weights, seed + total_games + i, hidden_dim, i % 4, algo)
                for i in range(batch_size)
            ]
            episodes = pool.map(_play_one_game, work_items)
            total_games += batch_size

            # Update
            result = agent.update_from_episodes(episodes)

            batch_wins = sum(1 for ep in episodes if ep.reward > 0)
            batch_time = time.time() - batch_start
            games_per_sec = batch_size / batch_time
            elapsed = time.time() - start

            if algo == "a2c":
                print(
                    f"Batch {batch:4d}/{num_batches} | "
                    f"Wins: {batch_wins:2d}/{batch_size} ({batch_wins/batch_size:4.0%}) | "
                    f"P: {result['policy_loss']:7.4f} "
                    f"V: {result['value_loss']:7.4f} "
                    f"H: {result['entropy']:5.2f} | "
                    f"{games_per_sec:5.1f} g/s | "
                    f"{elapsed:6.1f}s"
                )
            else:
                loss = result if isinstance(result, float) else result.get("total_loss", 0)
                print(
                    f"Batch {batch:4d}/{num_batches} | "
                    f"Wins: {batch_wins:2d}/{batch_size} ({batch_wins/batch_size:4.0%}) | "
                    f"Loss: {loss:7.4f} | "
                    f"{games_per_sec:5.1f} g/s | "
                    f"{elapsed:6.1f}s"
                )

            # Evaluate periodically
            if batch % eval_every == 0:
                eval_start = time.time()
                eval_items = [
                    (weights, seed + 999999 + i, hidden_dim, i % 4, algo)
                    for i in range(eval_games)
                ]
                eval_results = pool.map(_eval_one_game, eval_items)
                win_rate = sum(eval_results) / eval_games
                eval_time = time.time() - eval_start

                marker = ""
                if win_rate > best_win_rate:
                    best_win_rate = win_rate
                    marker = " ** BEST **"

                # Always save with metadata for resume
                agent.save(save_path, metadata={
                    "batch": batch,
                    "best_win_rate": best_win_rate,
                    "total_games": total_games,
                    "win_rate": win_rate,
                    "algo": algo,
                    "hidden_dim": hidden_dim,
                    "lr": lr,
                    "seed": seed,
                })

                print(
                    f"  >> Eval: {sum(eval_results)}/{eval_games} wins "
                    f"({win_rate:.0%}) | Best: {best_win_rate:.0%} | "
                    f"Saved | {eval_time:.1f}s{marker}"
                )

    except KeyboardInterrupt:
        print("\nInterrupted — saving checkpoint...")
        agent.save(save_path, metadata={
            "batch": batch,
            "best_win_rate": best_win_rate,
            "total_games": total_games,
            "algo": algo,
            "hidden_dim": hidden_dim,
            "lr": lr,
            "seed": seed,
        })
        print(f"Saved to {save_path}")
    finally:
        pool.terminate()
        pool.join()

    print(f"\nDone. {total_games} games in {time.time() - start:.1f}s")
    print(f"Best eval win rate: {best_win_rate:.0%}")
    print(f"Model: {save_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Catan agent")
    parser.add_argument("--algo", type=str, default="a2c", choices=["reinforce", "a2c"])
    parser.add_argument("--batches", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--eval-every", type=int, default=10)
    parser.add_argument("--eval-games", type=int, default=40)
    parser.add_argument("--hidden-dim", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--workers", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--save-path", type=str, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--resume", action="store_true", help="Resume from existing checkpoint")
    args = parser.parse_args()

    if args.save_path is None:
        args.save_path = f"checkpoints/{args.algo}.pt"

    os.makedirs(os.path.dirname(args.save_path) or ".", exist_ok=True)

    # Auto-detect CUDA
    if args.device == "cpu" and torch.cuda.is_available():
        args.device = "cuda"
        print(f"Auto-detected CUDA GPU!")

    train(
        algo=args.algo,
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
        resume=args.resume,
    )
